import os
import json
import warnings
from PIL import Image
from torch.utils.data import Dataset

from utils import get_en_translations

COCO_DIR = 'coco'
COCO_CAPTIONS_PATH = os.path.join(COCO_DIR, 'annotations')
VQAV2_QUESTION_PATH = os.path.join(
    'vqav2', 'v2_OpenEnded_mscoco_val2014_questions_sample.json'
)
VQAV2_ANSWER_PATH = os.path.join(
    'vqav2', 'v2_mscoco_val2014_annotations_sample.json'
)

MARVL_DIR = 'marvl'
MARVL_VQA_PATH = os.path.join('vlue', 'vqa_vlue_test.json')

TRANSLATIONS = get_en_translations(
    {lang: 'translations/vqa/%s_questions_answers.docx' % lang for lang in {'en', 'zh'}}
)


def load_vqa_examples(target_langs, corpus_dir):
    coco_ids_to_file_names = {}
    with open(os.path.join(corpus_dir, COCO_CAPTIONS_PATH, 'captions_val2014.json'), 'r') as f:
        for image_info in json.load(f)['images']:
            assert image_info['id'] not in coco_ids_to_file_names
            coco_ids_to_file_names[image_info['id']] = image_info['file_name']

    with open(os.path.join(corpus_dir, VQAV2_QUESTION_PATH), 'r') as f:
        questions = json.load(f)

    question_ids_to_questions = {}
    for question in questions['questions']:
        question_ids_to_questions[
            question['question_id']
        ] = question['question'].strip()

    with open(os.path.join(corpus_dir, VQAV2_ANSWER_PATH), 'r') as f:
        answers = json.load(f)

    examples = []
    for answer in answers['annotations']:
        image_file = os.path.join(
            corpus_dir, COCO_DIR, 'val2014', coco_ids_to_file_names[answer['image_id']]
        )

        assert os.path.exists(image_file), image_file

        question_id = answer['question_id']
        en_question = question_ids_to_questions[question_id]
        en_answer = answer['multiple_choice_answer'].strip()

        for target_lang in target_langs:
            if target_lang == 'en':
                question = en_question
                answer = en_answer
            else:
                zh_question_answer = TRANSLATIONS[
                    '%s %s' % (en_question, en_answer)
                ][target_lang]
                assert '？' in zh_question_answer, (en_question, en_answer, zh_question_answer)

                question = zh_question_answer.split('？')[0].strip() + '？'
                answer = '？'.join(zh_question_answer.split('？')[1:]).strip()

            examples.append({
                'image_file': image_file,
                'lang': target_lang,
                'prefix_text': question,
                'label': answer,
                'metadata': 'us'
            })

    return examples


def load_marvl_examples(target_langs, corpus_dir):
    with open(os.path.join(corpus_dir, MARVL_VQA_PATH), 'r') as f:
        marvl = json.load(f)

    examples = []
    for example in marvl:
        if example['image'].split('/')[0] != "zh":
            continue

        image_file = os.path.join(
            corpus_dir, MARVL_DIR, example["image"]
        )

        assert os.path.exists(image_file), image_file

        en_question = example['question'].strip()
        en_question = en_question[0].capitalize() + en_question[1:]
        en_question = en_question.replace('??', '?')
        if not (en_question.endswith('?') or en_question.endswith('？')):
            en_question = en_question + '?'
        en_answer = example['answer'].strip()

        for target_lang in target_langs:
            if target_lang == 'en':
                question = en_question
                answer = en_answer
            else:
                zh_question_answer = TRANSLATIONS[
                    '%s %s' % (en_question, en_answer)
                    ][target_lang]
                assert '？' in zh_question_answer

                question, answer = zh_question_answer.split('？')
                question = question.strip() + '？'

            examples.append({
                'image_file': image_file,
                'lang': target_lang,
                'prefix_text': question,
                'label': answer,
                'metadata': 'cn'
            })

    return examples


class VQA(Dataset):

    def __init__(
            self, ann_langs, target_langs, preprocess, corpus_dir
    ):
        super().__init__()

        assert len(set(ann_langs) - {'en', 'zh'}) == 0, ann_langs
        assert len(set(target_langs) - {'en', 'zh'}) == 0, target_langs

        self.ann_langs = ann_langs
        self.target_langs = target_langs
        self.preprocess = preprocess
        self.corpus_dir = corpus_dir

        self.examples = []
        if 'en' in self.ann_langs:
            self.examples.extend(
                load_vqa_examples(self.target_langs, self.corpus_dir)
            )

        if 'zh' in self.ann_langs:
            self.examples.extend(
                load_marvl_examples(self.target_langs, self.corpus_dir)
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Image.DecompressionBombWarning)
            with Image.open(example['image_file']) as img:
                img_t = self.preprocess(img)

        return (
            idx if idx >= 0 else len(self) + idx,
            example['image_file'],
            example['lang'],
            img_t,
            (example['prefix_text'], example['label']),
            example['metadata']
        )
