import unittest

from benchmarks.artelingo import ArtELingo


class TestDatasetsArtELingo(unittest.TestCase):

    def test_splits(self):
        artelingo = ArtELingo(
            ann_langs=['en'], target_langs=['en'], skip_labels=[], required_agreement=0,
            splits=['test'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(0, len(artelingo))

    def test_en_anns_in_en(self):
        artelingo = ArtELingo(
            ann_langs=['en'], target_langs=['en'], skip_labels=[], required_agreement=0,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(3, len(artelingo))

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(0, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Expressionism/toyen_crying.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('sadness', label)
        self.assertEqual('en', metadata)

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[1]
        self.assertEqual(1, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Impressionism/willard-metcalf_havana-harbor.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('awe', label)
        self.assertEqual('en', metadata)

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[2]
        self.assertEqual(2, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Romanticism/viktor-vasnetsov_ivan-tsarevich-riding-the-grey-wolf-1889.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('awe', label)
        self.assertEqual('en', metadata)

    def test_en_anns_in_zh(self):
        artelingo = ArtELingo(
            ann_langs=['en'], target_langs=['zh'], skip_labels=[], required_agreement=0,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(3, len(artelingo))

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(
            'tests/fixtures/wikiart/Expressionism/toyen_crying.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('伤感', label)
        self.assertEqual('en', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[1]
        self.assertEqual(
            'tests/fixtures/wikiart/Impressionism/willard-metcalf_havana-harbor.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('惊叹', label)
        self.assertEqual('en', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[2]
        self.assertEqual(
            'tests/fixtures/wikiart/Romanticism/viktor-vasnetsov_ivan-tsarevich-riding-the-grey-wolf-1889.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('惊叹', label)
        self.assertEqual('en', metadata)

    def test_zh_anns_in_zh(self):
        artelingo = ArtELingo(
            ann_langs=['zh'], target_langs=['zh'], skip_labels=[], required_agreement=0,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(3, len(artelingo))

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(
            'tests/fixtures/wikiart/Expressionism/toyen_crying.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('伤感', label)
        self.assertEqual('zh', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[1]
        self.assertEqual(
            'tests/fixtures/wikiart/Impressionism/willard-metcalf_havana-harbor.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('满意', label)
        self.assertEqual('zh', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[2]
        self.assertEqual(
            'tests/fixtures/wikiart/Romanticism/viktor-vasnetsov_ivan-tsarevich-riding-the-grey-wolf-1889.jpg', image_file
        )
        self.assertEqual('zh', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('这件艺术作品唤起了什么样的情感？', prefix_text)
        self.assertEqual('满意', label)
        self.assertEqual('zh', metadata)

    def test_zh_anns_in_en(self):
        artelingo = ArtELingo(
            ann_langs=['zh'], target_langs=['en'], skip_labels=[], required_agreement=0,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(3, len(artelingo))

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(
            'tests/fixtures/wikiart/Expressionism/toyen_crying.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('sadness', label)
        self.assertEqual('zh', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[1]
        self.assertEqual(
            'tests/fixtures/wikiart/Impressionism/willard-metcalf_havana-harbor.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('contentment', label)
        self.assertEqual('zh', metadata)

        _, image_file, lang, img, (prefix_text, label), metadata = artelingo[2]
        self.assertEqual(
            'tests/fixtures/wikiart/Romanticism/viktor-vasnetsov_ivan-tsarevich-riding-the-grey-wolf-1889.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('contentment', label)
        self.assertEqual('zh', metadata)

    def test_required_agreement(self):
        artelingo = ArtELingo(
            ann_langs=['en'], target_langs=['en'], skip_labels=[], required_agreement=5,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(1, len(artelingo))

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(0, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Expressionism/toyen_crying.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('sadness', label)
        self.assertEqual('en', metadata)

    def test_skip_labels(self):
        artelingo = ArtELingo(
            ann_langs=['en'], target_langs=['en'], skip_labels=['sadness'], required_agreement=0,
            splits=['train'], preprocess=lambda img: img, corpus_dir='tests/fixtures'
        )
        self.assertEqual(2, len(artelingo))

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[0]
        self.assertEqual(0, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Impressionism/willard-metcalf_havana-harbor.jpg', image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('awe', label)
        self.assertEqual('en', metadata)

        idx, image_file, lang, img, (prefix_text, label), metadata = artelingo[1]
        self.assertEqual(1, idx)
        self.assertEqual(
            'tests/fixtures/wikiart/Romanticism/viktor-vasnetsov_ivan-tsarevich-riding-the-grey-wolf-1889.jpg',
            image_file
        )
        self.assertEqual('en', lang)
        self.assertEqual(image_file, img.filename)
        self.assertEqual('What emotion does this work of art evoke?', prefix_text)
        self.assertEqual('awe', label)
        self.assertEqual('en', metadata)