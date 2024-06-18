import unittest

from datasets.vqa import VQA


class TestDatasetsVQA(unittest.TestCase):

    def test_en_anns_in_en(self):
        vqa = VQA(
            ann_langs=['en'], target_langs=['en'],
            preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(4, len(vqa))

        idx, image_file, lang, img, (question, answer), metadata = vqa[0]
        self.assertEqual(idx, 0)
        self.assertEqual(
            image_file,
            'tests/fixtures/coco/val2014/COCO_val2014_000000341397.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, 'Is that a walk-in shower?')
        self.assertEqual(answer, 'no')
        self.assertEqual(metadata, 'us')

        idx, image_file, lang, img, (question, answer), metadata = vqa[-1]
        self.assertEqual(idx, 3)
        self.assertEqual(
            image_file,
            'tests/fixtures/coco/val2014/COCO_val2014_000000341397.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, 'What shape are all the tiles?')
        self.assertEqual(answer, 'square')
        self.assertEqual(metadata, 'us')

    def test_en_anns_in_zh(self):
        vqa = VQA(
            ann_langs=['en'], target_langs=['zh'],
            preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(4, len(vqa))

        idx, image_file, lang, img, (question, answer), metadata = vqa[0]
        self.assertEqual(idx, 0)
        self.assertEqual(
            image_file,
            'tests/fixtures/coco/val2014/COCO_val2014_000000341397.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, '这是步入式淋浴间吗？')
        self.assertEqual(answer, '不')
        self.assertEqual(metadata, 'us')

        idx, image_file, lang, img, (question, answer), metadata = vqa[-1]
        self.assertEqual(idx, 3)
        self.assertEqual(
            image_file,
            'tests/fixtures/coco/val2014/COCO_val2014_000000341397.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, '所有的瓷砖都是什么形状的？')
        self.assertEqual(answer, '正方形')
        self.assertEqual(metadata, 'us')

    def test_zh_anns_in_en(self):
        vqa = VQA(
            ann_langs=['zh'], target_langs=['en'],
            preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(11, len(vqa))

        idx, image_file, lang, img, (question, answer), metadata = vqa[0]
        self.assertEqual(idx, 0)
        self.assertEqual(
            image_file,
            'tests/fixtures/marvl/zh/images/18-牡丹/18-1.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, 'Does there only one color of peony here?')
        self.assertEqual(answer, 'yes')
        self.assertEqual(metadata, 'cn')

        idx, image_file, lang, img, (question, answer), metadata = vqa[-1]
        self.assertEqual(idx, 10)
        self.assertEqual(
            image_file,
            'tests/fixtures/marvl/zh/images/46-T恤/46-7.jpg'
        )
        self.assertEqual(lang, 'en')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, 'What\'s on the T-shirt?')
        self.assertEqual(answer, 'print')
        self.assertEqual(metadata, 'cn')

    def test_zh_anns_in_zh(self):
        vqa = VQA(
            ann_langs=['zh'], target_langs=['zh'],
            preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(11, len(vqa))

        idx, image_file, lang, img, (question, answer), metadata = vqa[0]
        self.assertEqual(idx, 0)
        self.assertEqual(
            image_file,
            'tests/fixtures/marvl/zh/images/18-牡丹/18-1.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, '这里的牡丹只有一种颜色吗？')
        self.assertEqual(answer, '是的')
        self.assertEqual(metadata, 'cn')

        idx, image_file, lang, img, (question, answer), metadata = vqa[-1]
        self.assertEqual(idx, 10)
        self.assertEqual(
            image_file,
            'tests/fixtures/marvl/zh/images/46-T恤/46-7.jpg'
        )
        self.assertEqual(lang, 'zh')
        self.assertEqual(image_file, img.filename)
        self.assertEqual(question, 'T恤上写的是什么？')
        self.assertEqual(answer, '打印')
        self.assertEqual(metadata, 'cn')
