import os
import warnings
import pandas as pd
from PIL import Image
from ast import literal_eval

from torch.utils.data import Dataset

from constants import LANG_COMMAS
from utils import get_en_translations

DOLLAR_STREET_DIR = 'dollar/dataset_dollarstreet'
ANNOTATIONS_PATH = os.path.join(DOLLAR_STREET_DIR, 'images_v2_imagenet_%s.csv')

TRANSLATION_PATHS = {
    lang: 'translations/dollarstreet/%s_classes.docx' % lang
    for lang in ['en', 'zh']
}

VALID_COUNTRIES = {
    'at', 'bd', 'bf', 'bi', 'bo', 'br', 'ca', 'ch', 'ci', 'cm', 'cn', 'co', 'cz', 'dk', 'eg', 'es',
    'et', 'fr', 'gb', 'gh', 'gt', 'ht', 'id', 'in', 'ir', 'it', 'jo', 'ke', 'kg', 'kh', 'kr', 'kz',
    'lb', 'lk', 'lr', 'mm', 'mn', 'mw', 'mx', 'ng', 'nl', 'np', 'pe', 'pg', 'ph', 'pk', 'ps', 'ro',
    'rs', 'ru', 'rw', 'se', 'so', 'tg', 'th', 'tn', 'tr', 'tz', 'ua', 'us', 'vn', 'za', 'zw'
}
VALID_LABEL_SET = {
    'dollar',
    'imagenet'
}
VALID_SPLITS = {
    'train', 'test'
}

TASK_TEXTS = {
    'en': 'What object is in this image?',
    'zh': '这张图片中是什么物体？'
}


class DollarStreet(Dataset):

    def __init__(
        self, ann_countries, target_langs, label_set, splits, preprocess, corpus_dir
    ):
        super().__init__()

        assert len(set(ann_countries) - VALID_COUNTRIES) == 0, ann_countries
        assert len(set(target_langs) - {'en', 'zh'}) == 0, target_langs
        assert label_set in VALID_LABEL_SET, label_set
        assert all(split in VALID_SPLITS for split in splits), splits

        self.ann_countries = ann_countries
        self.target_langs = target_langs
        self.splits = splits
        self.preprocess = preprocess
        self.corpus_dir = corpus_dir

        self.examples, translations = [], get_en_translations(TRANSLATION_PATHS)
        for split in splits:
            split_df = pd.read_csv(os.path.join(corpus_dir, ANNOTATIONS_PATH % split))
            split_df['topics'] = split_df['topics'].apply(literal_eval)
            split_df['imagenet_synonyms'] = split_df['imagenet_synonyms'].apply(literal_eval)

            for _, row in split_df.iterrows():
                ann_country = row['country.id']
                if len(self.ann_countries) > 0 and ann_country not in self.ann_countries:
                    continue

                image_file = os.path.join(
                    os.path.join(corpus_dir, DOLLAR_STREET_DIR), row['imageRelPath']
                )
                assert os.path.exists(image_file), image_file

                if label_set == 'dollar':
                    en_labels = row['topics']
                else:
                    assert label_set == 'imagenet'
                    en_labels = row['imagenet_synonyms']

                for target_lang in target_langs:
                    if target_lang == 'en':
                        labels = en_labels
                    else:
                        labels = [
                            translations[en_label][target_lang] for en_label in en_labels
                        ]

                    self.examples.append({
                        'image_file': image_file,
                        'lang': target_lang,
                        'prefix_text': TASK_TEXTS[target_lang],
                        'label': LANG_COMMAS[target_lang].join(list(sorted(labels))),
                        'metadata': ann_country
                    })

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
