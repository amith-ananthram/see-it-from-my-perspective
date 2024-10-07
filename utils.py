import docx
from constants import LANG_COMMAS, MULTI_CHOICE_INSTRUCTION

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


def format_multiple_choice_response(lang, candidates, include_numbers=True):
    choices = LANG_COMMAS[lang].join(
        [
            '%s) %s' % (i + 1, candidate) if include_numbers else candidate
            for i, candidate in enumerate(candidates)
        ]
    )
    return MULTI_CHOICE_INSTRUCTION[lang] % choices


def clean_generation(generation, prefix, model_type):
    if prefix is None or (
            model_type in {'llava_v0', 'llava_v1'} or 'blip' in model_type.lower() or 'mllava' in model_type.lower()):
        return generation.strip().lower()
    else:
        if "<img>" in prefix and "</img" in prefix:
            prefix = prefix.replace("<img>", "")
            prefix = prefix.replace("</img>", "")

        if "[/INST]" in generation:
            prefix = "[/INST]"

        if "ASSISTANT:" in generation:
            prefix = "ASSISTANT:"

        if "助理：" in generation:
            prefix = "助理："

        if "<reserved_107>" in generation:
            prefix = "<reserved_107>"

        assert prefix in generation, (prefix, generation)

        return generation[
               generation.index(prefix) + len(prefix):
               ].strip()