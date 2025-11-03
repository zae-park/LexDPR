# lex_dpr/models/textnorm.py
import unicodedata, re

_ws = re.compile(r"\s+")
def normalize_text(s: str, lower=False) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")       # zero-width space
    s = _ws.sub(" ", s).strip()
    return s.lower() if lower else s
