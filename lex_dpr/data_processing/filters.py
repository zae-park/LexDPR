# lex_dpr/data_processing/filters.py

import re, unicodedata
from typing import Any

_DELETED_PAT = re.compile(r"(삭제\s*됨?|^\s*\(삭제\)\s*$)", re.IGNORECASE)

def _coerce_to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # 문자열만 취해 합치기
        return " ".join([s for s in x if isinstance(s, str)])
    # dict 등 기타 타입은 일단 비움(원하면 여기서 키 추출 로직 추가)
    return ""

def is_deleted_clause(text: Any) -> bool:
    t = normalize_whitespace(text)
    return (t == "" or bool(_DELETED_PAT.search(t)) or t.strip() in {"삭제", "(삭제)"})

def normalize_whitespace(text: Any) -> str:
    s = _coerce_to_str(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def dedup_texts(items: list[dict], text_key: str = "text") -> list[dict]:
    seen = set()
    out = []
    for it in items:
        txt = normalize_whitespace(it.get(text_key, ""))
        if not txt or txt in seen:
            continue
        it[text_key] = txt
        seen.add(txt)
        out.append(it)
    return out
