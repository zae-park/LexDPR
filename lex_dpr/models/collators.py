# lex_dpr/models/collators.py
from typing import List, Tuple
from sentence_transformers import InputExample

def mnr_collate(batch: List[Tuple[str, str]]):
    # (q, p)
    return [InputExample(texts=[q, p]) for (q, p) in batch]

def mnr_with_hn_collate(batch: List[Tuple[str, str, list]]):
    # (q, p_pos, [p_neg...]) -> texts=[q, p_pos, *p_neg]
    ex = []
    for (q, pos, negs) in batch:
        ex.append(InputExample(texts=[q, pos] + (negs or [])))
    return ex
