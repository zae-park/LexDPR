# lex_dpr/utils/data.py
from typing import Dict, Any
from .utils.io import read_jsonl

def load_passages(corpus_path: str) -> Dict[str, Dict[str, Any]]:
    passages: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(corpus_path):
        pid = row["id"]
        passages[pid] = row
    return passages

def validate_pairs_exist(passages: Dict[str, dict], pairs_path: str) -> None:
    missing = []
    for row in read_jsonl(pairs_path):
        for pid in row.get("positive_passages", []):
            if pid not in passages: missing.append(pid)
        for nid in row.get("hard_negatives", []):
            if nid not in passages: missing.append(nid)
    if missing:
        uniq = sorted(set(missing))
        raise ValueError(f"pairs 내 ID {len(uniq)}개가 corpus에 없음. 예시: {uniq[:5]} ...")
