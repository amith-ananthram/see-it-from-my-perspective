import os
import warnings
import unicodedata
import pandas as pd
from PIL import Image
from collections import defaultdict, Counter

from torch.utils.data import Dataset

from constants import LANGS

WIKI_ART_DIR = 'wikiart'
ANNOTATIONS_FILE = 'artelingo/artelingo_release.csv'

VALID_SPLITS = {
    'train', 'val', 'test'
}

EMOTION_LABELS = {
    'en': [
        'amusement', 'anger', 'awe', 'contentment', 'disgust',
        'excitement', 'fear', 'sadness', 'something else'
    ],
    'zh': [
        '好笑', '愤怒', '惊叹', '满意', '嫌恶',
        '兴奋', '恐惧', '伤感', '其它情感'
    ]
}

TASK_TEXTS = {
    'en': 'What emotion does this work of art evoke?',
    'zh': '这件艺术作品唤起了什么样的情感？'
}


class ArtELingo(Dataset):

    def __init__(
            self, ann_langs, target_langs, skip_labels,
            required_agreement, splits, preprocess, corpus_dir
    ):
        super().__init__()

        assert len(set(ann_langs) - LANGS.keys()) == 0, ann_langs
        assert len(set(target_langs) - LANGS.keys()) == 0, target_langs
        assert len(set(skip_labels) - set(EMOTION_LABELS['en'])) == 0, skip_labels
        assert all(split in VALID_SPLITS for split in splits), splits

        self.ann_langs = ann_langs
        self.target_langs = target_langs
        self.skip_labels = skip_labels
        self.required_agreement = required_agreement
        self.splits = splits
        self.preprocess = preprocess
        self.corpus_dir = corpus_dir

        self.grouped = defaultdict(list)
        for _, row in pd.read_csv(os.path.join(corpus_dir, ANNOTATIONS_FILE)).iterrows():
            if row['split'] not in splits:
                continue

            if row['language'] not in {LANGS[ann_lang] for ann_lang in ann_langs}:
                continue

            self.grouped[
                (LANGS.inverse[row['language']], row['image_file'])
            ].append(row)

        self.examples = []
        for example_id, (ann_lang, image_file) in enumerate(sorted(self.grouped.keys())):
            emotion_counts = Counter()
            for row in self.grouped[(ann_lang, image_file)]:
                emotion_counts[row['emotion']] += 1

            max_emotion = emotion_counts.most_common(1)[0][0]

            if max_emotion in self.skip_labels:
                continue

            if self.required_agreement and emotion_counts[max_emotion] < self.required_agreement:
                continue

            image_file = unicodedata.normalize('NFC', image_file).replace(
                'YOUR/PATH/TO/WIKIART', os.path.join(corpus_dir, WIKI_ART_DIR)
            )
            assert os.path.exists(image_file), '%s: %s does not exist!' % (example_id, image_file)

            for target_lang in self.target_langs:
                label = max_emotion
                if target_lang != 'en':
                    label_idx = EMOTION_LABELS['en'].index(label)
                    label = EMOTION_LABELS[target_lang][label_idx]

                self.examples.append({
                    'image_file': image_file,
                    'lang': target_lang,
                    'prefix_text': TASK_TEXTS[target_lang],
                    'label': label,
                    'metadata': ann_lang,
                    'agreement': emotion_counts[max_emotion]
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
