import unittest

from datasets.dollarstreet import DollarStreet


class TestDatasetsDollarStreet(unittest.TestCase):

    def test_in_en(self):
        dollarstreet = DollarStreet(
            ann_countries=['us', 'cn'], target_langs=['en'], label_set='dollar',
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(5, len(dollarstreet))

        idx, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[0]
        self.assertEqual(idx, 0)
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4bf2b4cf0b3a0f3f358be6/5d4bf2b4cf0b3a0f3f358be6.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, 'What object is in this image?')
        self.assertEqual(label, 'cooking utensils')
        self.assertEqual(metadata, 'cn')

        idx, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[1]
        self.assertEqual(idx, 1)
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4be643cf0b3a0f3f3435a9/5d4be643cf0b3a0f3f3435a9.JPG'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, 'What object is in this image?')
        self.assertEqual(label, 'front door, lock on front door')
        self.assertEqual(metadata, 'cn')

        idx, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[-1]
        self.assertEqual(idx, 4)
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4bdec1cf0b3a0f3f33684d/5d4bdec1cf0b3a0f3f33684d.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, 'What object is in this image?')
        self.assertEqual(label, 'dish racks')
        self.assertEqual(metadata, 'us')

    def test_gen_zh_no_prefixes(self):
        dollarstreet = DollarStreet(
            ann_countries=['us', 'cn'], target_langs=['zh'], label_set='dollar',
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(5, len(dollarstreet))

        _, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[0]
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4bf2b4cf0b3a0f3f358be6/5d4bf2b4cf0b3a0f3f358be6.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, '这张图片中是什么物体？')
        self.assertEqual(label, '炊具')
        self.assertEqual(metadata, 'cn')

        _, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[1]
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4be643cf0b3a0f3f3435a9/5d4be643cf0b3a0f3f3435a9.JPG'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, '这张图片中是什么物体？')
        self.assertEqual(label, '前门、锁上前门')
        self.assertEqual(metadata, 'cn')

        _, image_file, lang, img, (prefix_text, label), metadata = dollarstreet[-1]
        self.assertEqual(
            image_file,
            'tests/fixtures/dollar/dataset_dollarstreet/assets/5d4bdec1cf0b3a0f3f33684d/5d4bdec1cf0b3a0f3f33684d.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(prefix_text, '这张图片中是什么物体？')
        self.assertEqual(label, '碗碟架')
        self.assertEqual(metadata, 'us')
