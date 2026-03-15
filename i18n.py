import yaml
from pathlib import Path

_strings: dict = {}
_lang = "en"
_dir = Path(__file__).parent / "i18n"


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


# Load English by default
set_lang("en")
