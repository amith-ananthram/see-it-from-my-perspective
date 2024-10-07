import re
import os
import sys
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, \
    LlavaForConditionalGeneration, \
    Blip2ForConditionalGeneration, \
    AutoModelForVisualQuestionAnswering, \
    AutoModelForCausalLM, DataCollatorWithPadding

sys.path.append('.')

sys.path.append('other_repos/llava')
from llava import LlavaLlamaForCausalLM as LegacyLlavaForConditionalGeneration
from llava.conversation import conv_templates as legacy_llava_conv_templates
from llava.mm_utils import process_images as legacy_llava_process_images, \
    tokenizer_image_token as legacy_llava_tokenizer_image_token
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN as LEGACY_LLAVA_DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN as LEGACY_LLAVA_DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN as LEGACY_LLAVA_DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN as LEGACY_LLAVA_DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER as LEGACY_LLAVA_IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX as LEGACY_LLAVA_IMAGE_TOKEN_INDEX
)

sys.path.append('other_repos/mllava')
from mllava.mm_utils import process_images as mllava_process_images
from mllava.conversation import conv_templates as mllava_conv_templates
from mllava.mm_utils import tokenizer_image_token as mllava_tokenizer_image_token
from mllava.mm_utils import get_model_name_from_path as mllava_get_model_name_from_path
from mllava.model.builder import load_pretrained_model as mllava_load_pretrained_model
from mllava.constants import IMAGE_TOKEN_INDEX as MLLAVA_IMAGE_TOKEN_INDEX

from constants import LANG_COMMAS, LANG_COLONS, LANG_DELIMITERS
from datasets.dollarstreet import DollarStreet
from datasets.aokvqa import AOKVQA
from datasets.cvqa import CVQA
from datasets.artelingo import ArtELingo
from utils import clean_generation

TARGET_LANGS = ['en', 'zh']


class LegacyLlavaDataset(Dataset):

    def __init__(
            self, delegate, instructions, include_sys, translate_template,
            response_start, image_placement, model, config, tokenizer, image_processor
    ):
        assert not include_sys, "not implemented!"
        assert not translate_template, "not implemented!"

        self.delegate = delegate
        self.instructions = instructions
        self.include_sys = include_sys
        self.translate_template = translate_template
        self.response_start = response_start
        self.image_placement = image_placement
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        idx, image_file, lang, img, prefix_and_label, metadatum = self.delegate[idx]

        if self.image_placement == 'before':
            template = f"{LEGACY_LLAVA_IMAGE_PLACEHOLDER}\n%s"
        else:
            assert self.image_placement == 'after'
            template = f"%s\n{LEGACY_LLAVA_IMAGE_PLACEHOLDER}"

        prefix_text, prompt, label = build_prompt(
            lang, prefix_and_label, self.instructions[lang], template
        )

        image_token_se = LEGACY_LLAVA_DEFAULT_IM_START_TOKEN + \
                         LEGACY_LLAVA_DEFAULT_IMAGE_TOKEN + LEGACY_LLAVA_DEFAULT_IM_END_TOKEN
        if self.config.mm_use_im_start_end:
            prompt = re.sub(LEGACY_LLAVA_IMAGE_PLACEHOLDER, image_token_se, prompt)
        else:
            prompt = re.sub(LEGACY_LLAVA_IMAGE_PLACEHOLDER, LEGACY_LLAVA_DEFAULT_IMAGE_TOKEN, prompt)

        conv = legacy_llava_conv_templates[self.model].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], self.response_start[lang])
        prompt = conv.get_prompt()

        input_ids = legacy_llava_tokenizer_image_token(
            prompt, self.tokenizer, LEGACY_LLAVA_IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        processed = {
            'input_ids': input_ids,
            'attention_mask': torch.ones(len(input_ids))
        }

        img = img.convert('RGB')
        images = legacy_llava_process_images(
            [img],
            self.image_processor,
            self.config
        ).to(dtype=torch.float16).squeeze()
        image_sizes = img.size

        return idx, image_file, lang, metadatum, (prefix_text, prompt, processed, images, image_sizes), label


LLAVA_SYSTEM_MESSAGES = {
    "en": "A chat between a curious user and an artificial intelligence assistant. " +
          "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    "zh": "好奇的用户和人工智能助手之间的聊天。 助理对用户的问题提供有用、详细且礼貌的回答。"
}

LLAVA_ROLES = {
    "en": ["USER", "ASSISTANT"],
    "zh": ["用户", "助理"]
}


class LlavaDataset(Dataset):

    def __init__(
            self, delegate, instructions, include_sys,
            translate_template, response_start, image_placement, processor
    ):
        self.delegate = delegate
        self.instructions = instructions
        self.include_sys = include_sys
        self.translate_template = translate_template
        self.response_start = response_start
        self.image_placement = image_placement
        self.processor = processor

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        idx, image_file, lang, img, prefix_and_label, metadatum = self.delegate[idx]

        template = []
        if self.include_sys:
            if self.translate_template:
                template.append(LLAVA_SYSTEM_MESSAGES[lang])
            else:
                template.append(LLAVA_SYSTEM_MESSAGES["en"])

        if self.translate_template:
            colon = LANG_COLONS[lang]
            user_template, assistant_template = LLAVA_ROLES[lang]
        else:
            colon = LANG_COLONS["en"]
            user_template, assistant_template = LLAVA_ROLES["en"]

        if self.image_placement == 'before':
            template.append(f"{user_template}{colon}<image>\n%s\n{assistant_template}{colon}")
        else:
            assert self.image_placement == 'after'
            template.append(f"{user_template}{colon}%s\n<image>\n{assistant_template}{colon}")

        if self.translate_template:
            template = LANG_DELIMITERS[lang].join(template)
        else:
            template = LANG_DELIMITERS["en"].join(template)

        prefix_text, prompt, label = build_prompt(
            lang, prefix_and_label, self.instructions[lang], template + self.response_start[lang]
        )

        processed = self.processor(
            prompt, img, return_tensors='pt'
        )
        for key in processed:
            processed[key] = processed[key].squeeze()

        return idx, image_file, lang, metadatum, (prefix_text, prompt, processed), label


class MLlavaDataset(Dataset):

    def __init__(
            self, delegate, instructions, include_sys, translate_template,
            response_start, image_placement, config, conv_mode, tokenizer, image_processor
    ):
        assert not include_sys, "mllava was not trained with system text"
        assert not translate_template, "mllava was not trained with natural language templates"

        self.delegate = delegate
        self.instructions = instructions
        self.include_sys = include_sys
        self.translate_template = translate_template
        self.response_start = response_start
        self.image_placement = image_placement
        self.config = config
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        idx, image_file, lang, img, prefix_and_label, metadatum = self.delegate[idx]

        if self.image_placement == 'before':
            template = "<image>\n%s"
        else:
            assert self.image_placement == 'after'
            template = "%s\n<image>"

        conv = mllava_conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], template)
        conv.append_message(conv.roles[1], self.response_start[lang] or None)
        template = conv.get_prompt()

        prefix_text, prompt, label = build_prompt(
            lang, prefix_and_label, metadatum, template
        )

        input_ids = mllava_tokenizer_image_token(
            prompt, self.tokenizer, MLLAVA_IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).squeeze()
        processed = {
            'input_ids': input_ids,
            'attention_mask': torch.ones(input_ids.shape[0])
        }

        images = mllava_process_images(
            [img.convert('RGB')], self.image_processor, self.config
        ).to(dtype=torch.float16).squeeze()

        return idx, image_file, lang, metadatum, (prefix_text, prompt, processed, images, img.size), label


class BlipDataset(Dataset):

    def __init__(
            self, delegate, instructions, include_sys,
            translate_template, response_start, image_placement, processor
    ):
        assert not include_sys, "not implemented!"
        assert not translate_template, "not implemented!"

        self.delegate = delegate
        self.instructions = instructions
        self.include_sys = include_sys
        self.translate_template = translate_template
        self.response_start = response_start
        self.image_placement = image_placement
        self.processor = processor

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        idx, image_file, lang, img, prefix_and_label, metadatum = self.delegate[idx]

        assert self.image_placement == 'before'

        template = "Question: %s"
        prefix_text, prompt, label = build_prompt(
            lang, prefix_and_label, self.instructions[lang],
            LANG_DELIMITERS[lang].join([template, self.response_start[lang]]).strip()
        )

        processed = self.processor(
            images=img, text=prompt, return_tensors='pt'
        )
        for key in processed:
            processed[key] = processed[key].squeeze()

        return idx, image_file, lang, metadatum, (prefix_text, prompt, processed), label


class QwenDataset(Dataset):

    def __init__(
            self, delegate, instructions, include_sys,
            translate_template, response_start, image_placement, tokenizer
    ):
        assert not include_sys, "not implemented!"
        assert not translate_template, "not implemented!"

        self.delegate = delegate
        self.instructions = instructions
        self.include_sys = include_sys
        self.translate_template = translate_template
        self.response_start = response_start
        self.image_placement = image_placement
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        idx, image_file, lang, img, prefix_and_label, metadatum = self.delegate[idx]

        prefix_text, prompt, label = build_prompt(
            lang, prefix_and_label, self.instructions[lang], "%s" + self.response_start[lang]
        )

        if self.image_placement == 'before':
            prompt = self.tokenizer.from_list_format([
                {'image': image_file},
                {'text': prompt}
            ])
        else:
            assert self.image_placement == 'after'
            prompt = self.tokenizer.from_list_format([
                {'text': prompt},
                {'image': image_file}
            ])

        processed = self.tokenizer(
            prompt, return_tensors='pt'
        )
        for key in processed:
            processed[key] = processed[key].squeeze()

        return idx, image_file, lang, metadatum, (prefix_text, prompt, processed), label


def prompt_collator(batch):
    idxs, image_files, langs, metadata, inputs, labels = [], [], [], [], [], []
    for idx, image_file, lang, metadatum, input, label in batch:
        idxs.append(idx)
        image_files.append(image_file)
        langs.append(lang)
        metadata.append(metadatum)
        inputs.append(input)
        labels.append(label)

    prefix_texts = [prompts_and_inputs[0] for prompts_and_inputs in inputs]
    prompts = [prompts_and_inputs[1] for prompts_and_inputs in inputs]
    input_ids = [prompts_and_inputs[2] for prompts_and_inputs in inputs]
    if len(inputs[0]) == 3:
        inputs = (prefix_texts, prompts, input_ids)
    elif len(inputs[0]) == 4:
        images = [input_ids_and_images[3] for input_ids_and_images in inputs]
        inputs = (prefix_texts, prompts, input_ids, torch.stack(images))
    else:
        assert len(inputs[0]) == 5
        images = [input_ids_and_images[3] for input_ids_and_images in inputs]
        image_sizes = [input_ids_and_images[4] for input_ids_and_images in inputs]
        inputs = (prefix_texts, prompts, input_ids, torch.stack(images), image_sizes)

    return idxs, image_files, langs, metadata, inputs, labels


def collect_labels(target_lang, test_set, is_multi):
    labels = set()
    for i in range(len(test_set)):
        example = test_set.examples[i]
        if is_multi:
            labels.update(example['label'].split(LANG_COMMAS[target_lang]))
        else:
            labels.add(example['label'])
    return list(sorted(labels))


def build_prompt(
    lang, prefix_and_label, instruction, template
):
    prefix_text, label = prefix_and_label

    prompt = [prefix_text.strip()]

    if instruction.strip() != '':
        prompt.append(instruction)

    return (
        prefix_text,
        template % (LANG_DELIMITERS[lang].join(prompt).strip()),
        label
    )


def main(args):
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        assert torch.cuda.is_available()
        device = torch.device('cuda', int(args.device))

    # build variant name

    variant = [args.task]

    if args.instructions.strip() != '|':
        variant.append('inst=%s' % args.instructions)
        instructions_en, instructions_zh = args.instructions.split('|')
    else:
        instructions_en, instructions_zh = "", ""
    instructions = {'en': instructions_en, 'zh': instructions_zh}

    if args.include_sys:
        variant.append('sys')

    if args.translate_template:
        variant.append('trans_temp')

    if args.response_start.strip() != '|':
        variant.append(args.response_start)
        response_start_en, response_start_zh = args.response_start.split('|')
    else:
        response_start_en, response_start_zh = "", ""
    response_starts = {'en': response_start_en, 'zh': response_start_zh}

    if args.skip_labels.strip() != '':
        variant.append(args.skip_labels)

    variant.append(args.mlm_variant.split('/')[-1])

    if args.mlm_variant in {
        'llava_v0', 'llava_v1'
    }:
        model_path = os.path.join(
            args.resources_dir, 'llava', args.mlm_variant
        )
        model = LegacyLlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).eval().to(device)
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device_map = {"": device}
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([LEGACY_LLAVA_DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([
                LEGACY_LLAVA_DEFAULT_IM_START_TOKEN, LEGACY_LLAVA_DEFAULT_IM_END_TOKEN
            ], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(dtype=torch.float16).to(device)
        image_processor = vision_tower.image_processor

        mlm_dataset_class = LegacyLlavaDataset
        mlm_dataset_args = {
            'model': args.mlm_variant,
            'config': model.config,
            'tokenizer': tokenizer,
            'image_processor': image_processor
        }
    elif args.mlm_variant in {
        'llava-hf/llava-1.5-7b-hf',
        'llava-hf/llava-1.5-13b-hf',
        'llava-hf/bakLlava-v1-hf'
    }:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.mlm_variant, torch_dtype=torch.float16
        ).eval().to(device)

        processor = AutoProcessor.from_pretrained(
            args.mlm_variant
        )
        tokenizer = processor.tokenizer

        mlm_dataset_class = LlavaDataset
        mlm_dataset_args = {'processor': processor}
    elif args.mlm_variant in {
        'Salesforce/blip2-opt-6.7b',
        'Salesforce/blip2-flan-t5-xxl',
        'Gregor/mblip-mt0-xl',
        'Gregor/mblip-bloomz-7b'
    }:
        assert args.image_placement == 'before'
        if 'mblip' in args.mlm_variant:
            model_class = Blip2ForConditionalGeneration
        else:
            model_class = AutoModelForVisualQuestionAnswering
        model = model_class.from_pretrained(
            args.mlm_variant, torch_dtype=torch.float16
        ).eval().to(device)

        processor = AutoProcessor.from_pretrained(
            args.mlm_variant
        )
        tokenizer = processor.tokenizer

        mlm_dataset_class = BlipDataset
        mlm_dataset_args = {'processor': processor}
    elif args.mlm_variant in {'Qwen/Qwen-VL', 'Qwen/Qwen-VL-Chat'}:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.mlm_variant, trust_remote_code=True
        )

        # to support batching (pulled from generation_config)
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = 151643
        tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.mlm_variant, trust_remote_code=True, torch_dtype=torch.float16
        ).eval().to(device)

        mlm_dataset_class = QwenDataset
        mlm_dataset_args = {'tokenizer': tokenizer}
    else:
        assert 'mllava' in args.mlm_variant

        model_path = os.path.join(args.resources_dir, args.mlm_variant)
        if 'baichuan' in args.mlm_variant:
            conv_mode = 'baichuan_2_chat'
            model_base = 'baichuan-inc/Baichuan2-7B-Chat'
        else:
            assert "llama" in args.mlm_variant
            conv_mode = 'llama_2_chat'
            model_base = 'meta-llama/Llama-2-7b-chat-hf'

        model_name = mllava_get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = mllava_load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        processor = None
        model = model.eval()
        tokenizer.pad_token = tokenizer.unk_token

        assert not model.config.mm_use_im_start_end

        mlm_dataset_class = MLlavaDataset
        mlm_dataset_args = {
            'config': model.config,
            'conv_mode': conv_mode,
            'tokenizer': tokenizer,
            'image_processor': image_processor
        }

    def preprocess(img):
        img.load()
        return img

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def batch_processor(batch):
        idxs, \
        image_files, \
        langs, \
        metadata, \
        inputs, \
        labels = batch

        with torch.inference_mode():
            sampling_args = {
                'do_sample': args.do_sample,
                'num_beams': args.num_beams,
                'length_penalty': args.length_penalty,
                'max_new_tokens': args.max_new_tokens
            }
            if args.do_sample:
                sampling_args['temperature'] = args.temperature
                sampling_args['top_k'] = args.top_k
                sampling_args['top_p'] = args.top_p

            prefix_texts = inputs[0]
            prompts = inputs[1]
            if len(inputs) == 3:
                outputs = model.generate(
                    **(collator(inputs[2]).to(device)),
                    **sampling_args
                )
            elif len(inputs) == 4:
                outputs = model.generate(
                    **(collator(inputs[2]).to(device)),
                    images=inputs[3].to(device),
                    **sampling_args
                )
            else:
                assert len(inputs) == 5
                collated = collator(inputs[2])
                outputs = model.generate(
                    inputs=collated['input_ids'].to(device),
                    attention_mask=collated['attention_mask'].to(device),
                    images=inputs[3].to(device),
                    image_sizes=inputs[4],
                    **sampling_args
                )

            if processor is not None:
                decoder = processor
            else:
                decoder = tokenizer

            generations = [
                clean_generation(generation, prompt, args.mlm_variant)
                for prompt, generation in zip(
                    prompts, decoder.batch_decode(outputs, skip_special_tokens=True)
                )
            ]

        return idxs, image_files, langs, labels, metadata, prefix_texts, prompts, generations

    if args.do_sample:
        variant.append('sample')

        if args.temperature > 0.0:
            variant.append('temp=%.2f' % args.temperature)

        if args.top_k is not None:
            variant.append('top_k=%d' % args.top_k)

        if args.top_p is not None:
            variant.append('top_p=%.2f' % args.top_p)
    elif args.num_beams > 1:
        variant.append('beams=%d' % args.num_beams)

    if args.length_penalty != 1.0:
        variant.append('lenpen=%.2f' % args.length_penalty)

    if args.image_placement != 'before':
        variant.append(args.image_placement)

    if args.max_new_tokens != 200:
        variant.append('max_new_tokens=%d' % args.max_new_tokens)

    variant = '-'.join(variant)
    print("variant= %s" % variant)
    output_dir = os.path.join(args.output_dir, variant)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.task in {'artelingo', 'allartelingo'}:
        assert 'country' in args.prefixes
        if args.task == 'artelingo':
            splits = ['test']
        else:
            assert args.task == 'allartelingo'
            splits = [
                'train', 'val', 'test'
            ]
        test_sets = [
            (
                target_lang,
                ArtELingo(
                    ann_langs=['en', 'zh'], target_langs=[target_lang],
                    skip_labels=[
                        skip_label for skip_label in args.skip_labels.split(',')
                        if skip_label.strip() != ''
                    ],
                    required_agreement=args.required_agreement,
                    splits=splits, preprocess=preprocess, corpus_dir=args.resources_dir
                )
            ) for target_lang in TARGET_LANGS
        ]
    elif args.task == 'dollar':
        test_sets = [
            (
                target_lang,
                DollarStreet(
                    ann_countries=[], target_langs=[target_lang], label_set='dollar',
                    splits=['test', 'train'], preprocess=preprocess, corpus_dir=args.resources_dir
                )
            ) for target_lang in TARGET_LANGS
        ]
    elif args.task == 'aokvqa':
        test_sets = [
            (
                target_lang,
                AOKVQA(target_lang, preprocess)
            ) for target_lang in TARGET_LANGS
        ]
    elif args.task == 'cvqa':
        test_sets = [
            (
                target_lang,
                CVQA(target_lang, preprocess)
            ) for target_lang in TARGET_LANGS
        ]
    else:
        raise Exception("Unsupported task: %s" % args.task)

    for (target_lang, test_set) in test_sets:
        output_file = os.path.join(output_dir, '%s_results.json' % target_lang)
        if os.path.exists(output_file):
            print("skipping %s %s..." % (target_lang, args.task))
            continue

        label_set = collect_labels(
            target_lang, test_set, is_multi='dollar' in args.task
        )

        print("loaded %d labels for %s %s..." % (len(label_set), target_lang, args.task))

        test_iterator = DataLoader(
            mlm_dataset_class(
                test_set, instructions, args.include_sys, args.translate_template,
                response_starts, args.image_placement, **mlm_dataset_args
            ),
            batch_size=args.batch_size,
            num_workers=8,
            collate_fn=prompt_collator
        )
        num_items = len(test_iterator)

        test_files, test_prefixes, test_prompts, test_gens, test_labels = \
            defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for batch in tqdm(
                test_iterator, desc='evaluating %s %s' % (target_lang, args.task), total=num_items
        ):
            idxs, image_files, langs, labels, metadata, prefix_texts, prompts, generations = batch_processor(
                batch
            )

            for i in range(len(idxs)):
                metadatum = metadata[i]
                test_files[metadatum].append(image_files[i])
                test_prefixes[metadatum].append(prefix_texts[i])
                test_prompts[metadatum].append(prompts[i])
                test_gens[metadatum].append(generations[i])
                test_labels[metadatum].append(labels[i])

        results = {
            'label_set': label_set,
            'test_files': test_files,
            'test_prefixes': test_prefixes,
            'test_prompts': test_prompts,
            'test_gens': test_gens,
            'test_labels': test_labels
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Produce East / West generations for the specified VLM & task."
    )
    parser.add_argument(
        '--task', choices=[
            'dollar',
            'aokvqa',
            'cvqa',
            'artelingo',
            'allartelingo'
        ], required=True
    )
    parser.add_argument(
        '--prefix-text', type=str
    )
    parser.add_argument(
        '--response-start', type=str, default='|'
    )
    parser.add_argument(
        '--instructions', type=str, default='|'
    )
    parser.add_argument(
        '--include-sys', action='store_true', default=False
    )
    parser.add_argument(
        '--translate-template', action='store_true', default=False
    )
    parser.add_argument('--skip-labels', type=str, default="")
    parser.add_argument('--required-agreement', type=int, default=None)
    parser.add_argument('--mlm-variant', type=str, default=None, choices=[
        'llava_v0', 'llava_v1',
        'llava-hf/llava-1.5-7b-hf',
        'llava-hf/llava-1.5-13b-hf',
        'llava-hf/bakLlava-v1-hf',
        'Salesforce/blip2-opt-6.7b',
        'Salesforce/blip2-flan-t5-xxl',
        'Qwen/Qwen-VL', 'Qwen/Qwen-VL-Chat',
        'Gregor/mblip-mt0-xl',
        'Gregor/mblip-bloomz-7b',
        'mllava/baichuan2-en',
        'mllava/baichuan2-zh',
        'mllava/baichuan2-en_zh',
        'mllava/llama2-en',
        'mllava/llama2-zh',
        'mllava/llama2-en_zh'
    ])
    parser.add_argument('--image-placement', type=str, choices=['before', 'after'])
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--length-penalty', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str)
    parser.add_argument('--resources-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    main(args)
