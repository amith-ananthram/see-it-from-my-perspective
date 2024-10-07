from bidict import bidict

LANGS = bidict({
    'en': 'english',
    'zh': 'chinese',
    'ar': 'arabic'
})

LANGS_TO_COUNTRIES = bidict({
    'en': 'us',
    'zh': 'cn'
})

LANG_PERIODS = {
    'en': '.',
    'zh': '。'
}
LANG_COMMAS = {
    'en': ', ',
    'zh': '、'
}
LANG_COLONS = {
    'en': ': ',
    'zh': '：'
}
LANG_DELIMITERS = {
    'en': ' ',
    'zh': ''
}

MULTI_CHOICE_INSTRUCTION = {
    'en': 'Choose exactly one from %s.',
    'zh': '从%s中准确选择一个。'
}