import warnings
from PIL import Image

from datasets import load_dataset
from torch.utils.data import Dataset

from utils import get_en_translations, format_multiple_choice_response

SUPPORTED_LANGS = {'en', 'zh'}
TRANSLATIONS_PATH = 'translations/aokvqa/%s_questions_answers.docx'


class AOKVQA(Dataset):

    def __init__(
        self, target_lang, preprocess
    ):
        super().__init__()

        assert target_lang in SUPPORTED_LANGS

        self.target_lang = target_lang
        self.preprocess = preprocess
        self.aokvqa = load_dataset('HuggingFaceM4/A-OKVQA', split='validation')
        self.aokvqa_translations = get_en_translations(
            {lang: TRANSLATIONS_PATH % lang for lang in SUPPORTED_LANGS}
        )

        # this is to collect candidate labels within
        # run_vlm_cultural_bias_inference.py

        self.examples = []
        for candidates in self.cvqa['choices']:
            for candidate in candidates:
                self.examples.append({
                    'label': candidate
                })

    def __len__(self):
        return len(self.aokvqa)

# 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x569 at 0x7FA99A1805B0>
# 'question_id': '22jbM6gDxdaMaunuzgrsBB',
# 'question': "What is in the motorcyclist's mouth?",
# 'choices': ['toothpick', 'food', 'popsicle stick', 'cigarette'],
# 'correct_choice_idx': 3,
# 'direct_answers': "['cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette']",
# 'difficult_direct_answer': False,
# 'rationales': ["He's smoking while riding.", 'The motorcyclist has a lit cigarette in his mouth while he rides on the street.', 'The man is smoking.']

    def __getitem__(self, idx):
        example = self.aokvqa[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Image.DecompressionBombWarning)
            img_t = self.preprocess(example['image'])

        en_q = example['question'].strip()
        en_q = en_q[0].upper() + en_q[1:]

        en_candidate_answers = example['choices']
        en_formatted_q = "%s %s" % (
            en_q, format_multiple_choice_response(
                "en", en_candidate_answers, include_numbers=False
            )
        )

        if self.target_lang == 'en':
            formatted_q = en_formatted_q
        else:
            formatted_q = self.aokvqa_translations[en_formatted_q][self.target_lang]

        return (
            idx,
            example['question_id'],
            self.target_lang,
            img_t,
            (
                formatted_q,
                en_candidate_answers[example['correct_choice_idx']]
            ),
            'west'
        )
