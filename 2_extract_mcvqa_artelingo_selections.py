import os
import json
import glob
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
from datasets import load_dataset, Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from benchmarks.artelingo import EMOTION_LABELS

OUTPUT_KEY = "test_preds_mistral7b_qprompt_en"
MISTRAL_TEMPLATE = "<s>[INST]%s[/INST]"

QA_PROMPT = "\"Question: %s, Answer: %s\" Which of the following answers is closest to the one in the previous text? 1) %s 2) %s 3) %s or 4) %s. Respond concisely with a single number."

ARTELINGO_CANDIDATES = EMOTION_LABELS['en'][:-1]
ARTELINGO_PROMPT = f"\"%s\" Which of the following emotions is closest to the one in the previous text: {', '.join(ARTELINGO_CANDIDATES)}. Answer in a single word."


def parse_selection(output, answer_choices):
    found_choices = []
    for choice in answer_choices:
        if choice.lower() in output.lower():
            found_choices.append((
                output.lower().index(choice.lower()), choice
            ))

    selected = None
    if len(found_choices) > 0:
        selected = answer_choices.index(
            min(found_choices)[1]
        )
    else:
        for choice_idx in range(1, len(answer_choices) + 1):
            if str(choice_idx) in output:
                selected = choice_idx - 1

    if selected is None:
        return random.choice(list(range(len(answer_choices))))
    else:
        return selected

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    aokvqa = load_dataset(
        'HuggingFaceM4/A-OKVQA', split='validation'
    )
    cvqa = load_dataset(
        'afaji/cvqa', split='test',
        revision='0f90214047e51d2e5f3f4dc9865107215553096a'
    )

    pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device=torch.device('cuda', args.device)
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    to_process = []
    for benchmark in ['cvqa', 'aokvqa', 'artelingo', 'allartelingo']:
        for dirnum, dirname in enumerate(sorted(glob.glob('%s/%s*' % (args.output_dir, benchmark)))):
            for lang in ['en', 'zh']:
                results_file = os.path.join(
                    dirname, '%s_results.json' % lang
                )
                if not os.path.exists(results_file):
                    continue

                with open(results_file, 'r') as f:
                    results = json.load(f)

                if OUTPUT_KEY in results:
                    print("%s done, skipping..." % results_file)
                    continue

                if lang == 'en' or 'artelingo' in benchmark:
                    results_key = 'test_gens'
                else:
                    results_key = 'test_gens_en_trans'

                if results_key not in results:
                    print("%s is missing results_key: %s" % (
                        results_file, results_key
                    ))
                    continue

                to_process.append((benchmark, dirname, lang, results_file))

    to_process = [
        to_process[i] for i in range(len(to_process))
        if i % args.num_threads == args.offset
    ]

    for filenum, (benchmark, dirname, lang, results_file) in enumerate(to_process):
        print("processing %s..." % results_file)

        with open(results_file, 'r') as f:
            results = json.load(f)

        if lang == 'en' or 'artelingo' in benchmark:
            results_key = 'test_gens'
        else:
            results_key = 'test_gens_en_trans'

        strata, prompts, answer_choice_sets = [], [], defaultdict(list)
        for stratum in results[results_key].keys():
            for identifier, generation in zip(
                results['test_files'][stratum],
                results[results_key][stratum]
            ):
                if 'vqa' in benchmark:
                    if benchmark == 'aokvqa':
                        question_idx = aokvqa['question_id'].index(identifier)
                        en_question = aokvqa['question'][question_idx]
                        en_answers = aokvqa['choices'][question_idx]
                    else:
                        assert benchmark == 'cvqa', benchmark
                        question_idx = cvqa['ID'].index(identifier)
                        en_question = cvqa['Translated Question'][question_idx]
                        en_answers = cvqa['Translated Options'][question_idx]

                    prompt = MISTRAL_TEMPLATE % (QA_PROMPT % (
                        en_question, generation, *en_answers
                    ))
                else:
                    assert 'artelingo' in benchmark

                    prompt = MISTRAL_TEMPLATE % (ARTELINGO_PROMPT % generation)
                    en_answers = ARTELINGO_CANDIDATES

                strata.append(stratum)
                prompts.append(prompt)
                answer_choice_sets[stratum].append(en_answers)

        dataset = Dataset.from_dict({
            'text': prompts
        })

        idx, outputs_by_strata = 0, defaultdict(list)
        for batch_output in tqdm(
            pipe(
                KeyDataset(dataset, 'text'),
                max_new_tokens=250,
                return_full_text=False, batch_size=args.bs,
                pad_token_id=pipe.model.config.eos_token_id
            ),
            desc='file %s/%s' % (filenum, len(to_process)),
            total=len(dataset) // args.bs
        ):
            for output in batch_output:
                outputs_by_strata[strata[idx]].append(output)
                idx += 1

        assert idx == len(strata) == len(prompts)

        test_stratum_preds = {}
        for stratum in outputs_by_strata.keys():
            selected_idxs = []
            for output, answer_choices in zip(
                outputs_by_strata[stratum], answer_choice_sets[stratum]
            ):
                selected_idxs.append(
                    parse_selection(
                        output['generated_text'], answer_choices
                    )
                )
            test_stratum_preds[stratum] = selected_idxs

        results[OUTPUT_KEY] = test_stratum_preds

        with open(results_file, 'w') as f:
            json.dump(results, f)

