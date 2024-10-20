import requests
from translate import translate, Translator
from translate.providers import MyMemoryProvider


class FasterMyMemoryProvider(MyMemoryProvider):
    """
    @FasterMyMemoryProvider: This is a faster integration with Translated MyMemory API.
    Use session pool to speedup connect.
    Follow Information:
      Website: https://mymemory.translated.net/
      Documentation: https://mymemory.translated.net/doc/spec.php

    Usage Tips: Use a valid email instead of the default.
        With a valid email you get 10 times more words/day to translate.
    For further information checkout:
    http://mymemory.translated.net/doc/usagelimits.php
    """
    name = 'FasterMyMemory'
    base_url = 'http://api.mymemory.translated.net/get'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = requests.Session()

    def _make_request(self, text):
        params = {'q': text, 'langpair': self.languages}
        if self.email:
            params['de'] = self.email

        response = self.session.get(self.base_url, params=params, headers=self.headers)
        return response.json()

    def get_translation(self, text):
        data = self._make_request(text)

        translation = data['responseData']['translatedText']
        if translation:
            return translation
        else:
            matches = data['matches']
            next_best_match = next(match for match in matches)
            return next_best_match['translation']


translate.PROVIDERS_CLASS[FasterMyMemoryProvider.name] = FasterMyMemoryProvider

def get_translator() -> Translator:
    return Translator(to_lang="en", from_lang='zh-TW', provider=FasterMyMemoryProvider.name)