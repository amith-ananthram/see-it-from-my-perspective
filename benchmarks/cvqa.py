import warnings
from PIL import Image

from datasets import load_dataset
from torch.utils.data import Dataset

from utils import get_en_translations, format_multiple_choice_response

SUPPORTED_LANGS = {'en', 'zh'}
TRANSLATIONS_PATH = 'translations/cvqa/%s_questions_answers.docx'

ALREADY_MC = {
    '5964281096652575511_1', '5919991134274940944_2', '5919991134271265432_1',
    '5948825686369792424_1', '5919991134278992434_0', '5948825686368125351_2'
}
Q_WORDS = {
    'what', 'which', 'who', 'during', 'in which', 'where', 'when',
    'in the picture on what object', 'how many', 'the leaves of which tree',
    'from the picutre which',
}


class CVQA(Dataset):

    def __init__(
            self, target_lang, preprocess
    ):
        super().__init__()

        assert target_lang in SUPPORTED_LANGS

        self.target_lang = target_lang
        self.preprocess = preprocess
        self.cvqa = load_dataset(
            'afaji/cvqa', revision='0f90214047e51d2e5f3f4dc9865107215553096a'
        )['test']
        self.cvqa_translations = get_en_translations(
            {lang: TRANSLATIONS_PATH % lang for lang in SUPPORTED_LANGS}
        )

        # this is to collect candidate labels within
        # run_vlm_cultural_bias_inference.py

        self.examples = []
        for candidates in self.cvqa['Options']:
            for candidate in candidates:
                self.examples.append({
                    'label': candidate
                })

    def __len__(self):
        return len(self.cvqa)

    # 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=6016x4000 at 0x7F7CFBC460E0>,
    # 'ID': '5920942824275514756_0', 'Subset': "('Telugu', 'India')",
    # 'Question': 'ఈ యోగా భంగిమ పేరు ____ పువ్వు నుండి వచ్చింది',
    # 'Translated Question': 'The name of this yoga pose has its origins from ____ flower',
    # 'Options': ['గులాబీ', 'కమలం', ' మల్లెపూవు', 'మందార'],
    # 'Translated Options': ['Rose', 'Lotus', 'Jasmine', 'Hibiscus'],
    # 'Label': -1,
    # 'Category': 'People and everyday life',
    # 'Image Type': 'External',
    # 'Image Source': 'https://commons.wikimedia.org/wiki/File:Dabbawala.jpg',
    # 'License': 'CC BY 2.0'

    def __getitem__(self, idx):
        example = self.cvqa[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Image.DecompressionBombWarning)
            img_t = self.preprocess(example['image'])

        en_q = example['Translated Question'].strip().strip('"')
        en_q = en_q[0].upper() + en_q[1:]

        if not en_q.endswith('?') and example['ID'] not in ALREADY_MC:
            updated = False
            for q_word in Q_WORDS:
                if en_q.lower().startswith(q_word):
                    en_q = "%s?" % en_q
                    updated = True
                    break

            if not updated:
                en_q = "%s." % en_q

        en_candidate_answers = example['Translated Options']
        en_formatted_q = "%s %s" % (
            en_q, format_multiple_choice_response(
                "en", en_candidate_answers, include_numbers=False
            )
        )

        if self.target_lang == 'en':
            formatted_q = en_formatted_q
        else:
            formatted_q = self.cvqa_translations[en_formatted_q][self.target_lang]

        return (
            idx,
            example['ID'],
            self.target_lang,
            img_t,
            (formatted_q, "NO_LABEL"),
            example['Subset']
        )
