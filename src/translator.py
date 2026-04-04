"""
Module dịch thuật dùng deep-translator (Google Translate, miễn phí)
"""

from deep_translator import GoogleTranslator


class Translator:
    def __init__(self):
        self._cache: dict[tuple, str] = {}           # (src, tgt, text) → translated
        self._translator: GoogleTranslator | None = None
        self._src = ""
        self._tgt = ""

    def _get_translator(self, source: str, target: str) -> GoogleTranslator | None:
        if source != self._src or target != self._tgt:
            try:
                new_tr = GoogleTranslator(source=source, target=target)
                self._src = source
                self._tgt = target
                self._translator = new_tr
            except Exception as e:
                print(f"\033[91m[ERROR] Translate error (init): {e}\033[0m")
                return None
        return self._translator

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Dịch text từ source → target.
        Returns text gốc nếu source == target hoặc target rỗng.
        """
        # Fallback manual fixes
        if source == "zh": source = "zh-CN"
        if target == "zh": target = "zh-CN"

        if not text or not target or source == target:
            return ""

        # Check cache
        key = (source, target, text)
        if key in self._cache:
            return self._cache[key]

        try:
            tr = self._get_translator(source, target)
            if not tr:
                return ""

            result = tr.translate(text)
            self._cache[key] = result
            # Giới hạn cache size
            if len(self._cache) > 500:
                keys = list(self._cache.keys())
                for k in keys[:200]:
                    del self._cache[k]
            return result or ""
        except Exception as e:
            print(f"\033[91m[ERROR] Translate error: {e}\033[0m")
            return ""
