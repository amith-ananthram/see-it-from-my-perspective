import docx


def get_en_translations(sentence_paths):
    sentences = {}
    for lang in sentence_paths:
        doc = docx.Document(sentence_paths[lang])
        sentences[lang] = [
            paragraph.text.replace("‚Äô", "’").replace("‚Ä∫", "›").replace("√©", "é")
            for paragraph in doc.paragraphs if paragraph.text.strip() != ''
        ]

    assert len({len(sentences[lang]) for lang in sentences}) == 1

    translations = {}
    for idx, en_sentence in enumerate(sentences['en']):
        translations[en_sentence] = {}
        for other_lang in sentences:
            if other_lang == 'en':
                continue
            translations[en_sentence][other_lang] = sentences[other_lang][idx]

    return translations
