import re, unicodedata

_DELETED_PAT = re.compile(r"(삭제\s*됨?|^\s*\(삭제\)\s*$)", re.IGNORECASE)

def is_deleted_clause(text: str | None) -> bool:
    if not text:
        return True  # 빈 본문은 노이즈로 취급
    t = normalize_whitespace(text)
    return bool(_DELETED_PAT.search(t)) or t.strip() in {"삭제", "(삭제)"}

def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

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
