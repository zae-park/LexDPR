# lex_dpr/models/datasets.py
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from ..utils.io import read_jsonl
from .templates import TemplateMode, tq, tp
from ..utils import text as _text  # (없다면 건너뛰어도 됨)

class PairDataset(Dataset):
    """
    기본: (q, p_pos) 샘플 → MNRLoss(in-batch negatives).
    옵션: use_hard_negatives=True 이면 (q, p_pos, p_neg1, ...) triplet-ish 샘플 생성.
    """
    def __init__(self, pairs_path: str, passages: Dict[str, Dict],
                 use_bge_template: bool = True,
                 use_hard_negatives: bool = False,
                 normalize_text: bool = False):
        self.passages = passages
        self.mode = TemplateMode.BGE if use_bge_template else TemplateMode.NONE
        self.use_hn = use_hard_negatives
        self.normalize_text = normalize_text

        self.samples: List[Tuple] = []
        miss = 0
        for r in read_jsonl(pairs_path):
            q = r["query_text"]
            if self.normalize_text:
                q = _text.normalize_text(q)
            q = tq(q, self.mode)

            pos_list = [pid for pid in r.get("positive_passages", []) if pid in passages]
            if not pos_list:
                miss += 1; continue

            if not self.use_hn:
                for pid in pos_list:
                    p = passages[pid]["text"]
                    if self.normalize_text:
                        p = _text.normalize_text(p)
                    p = tp(p, self.mode)
                    self.samples.append((q, p))
            else:
                # (q, p_pos, [p_neg...]) 형태
                neg_ids = [nid for nid in r.get("hard_negatives", []) if nid in passages]
                neg_texts = []
                for nid in neg_ids:
                    n = passages[nid]["text"]
                    if self.normalize_text:
                        n = _text.normalize_text(n)
                    neg_texts.append(tp(n, self.mode))
                for pid in pos_list:
                    p = passages[pid]["text"]
                    if self.normalize_text:
                        p = _text.normalize_text(p)
                    p = tp(p, self.mode)
                    self.samples.append((q, p, neg_texts))

        if miss:
            print(f"[PairDataset] missing positives: {miss}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
