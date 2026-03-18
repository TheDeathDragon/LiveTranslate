import locale
import yaml
from pathlib import Path

_strings: dict = {}
_lang = "en"
_dir = Path(__file__).parent / "i18n"


def _detect_system_lang() -> str:
    """Return 'zh' if system locale is Chinese, else 'en'."""
    try:
        lang_code = locale.getdefaultlocale()[0] or ""
        if lang_code.startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


def set_lang(lang: str):
    global _lang, _strings
    _lang = lang
    f = _dir / f"{lang}.yaml"
    if not f.exists():
        f = _dir / "en.yaml"
    _strings = yaml.safe_load(f.read_text("utf-8")) or {}


def get_lang() -> str:
    return _lang


def t(key: str) -> str:
    return _strings.get(key, key)


# Detect system language on import
set_lang(_detect_system_lang())

# Shared language list: (code, native_name)
LANGUAGES = [
    ("auto", None),  # display name comes from t("asr_lang_auto")
    ("ja", "日本語"),
    ("en", "English"),
    ("zh", "中文"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("ar", "العربية"),
    ("th", "ไทย"),
    ("vi", "Tiếng Việt"),
    ("id", "Bahasa Indonesia"),
    ("ms", "Bahasa Melayu"),
    ("hi", "हिन्दी"),
    ("uk", "Українська"),
    ("cs", "Čeština"),
    ("ro", "Română"),
    ("el", "Ελληνικά"),
    ("hu", "Magyar"),
    ("sv", "Svenska"),
    ("da", "Dansk"),
    ("fi", "Suomi"),
    ("no", "Norsk"),
    ("he", "עברית"),
]

# Common languages shown directly in tray menu (no submenu)
COMMON_LANG_CODES = {"auto", "ja", "en", "zh", "ko", "fr", "de", "es", "ru"}
