# lex_dpr/data_processing/make_pairs.py
from __future__ import annotations
import argparse, random
from typing import Dict, Any, List, Iterable, Tuple, Optional
from .utils_io import read_jsonl, write_jsonl


# =========================
# Query builders (type-wise)
# =========================
def build_query_law(p: Dict[str, Any]) -> str:
    law_name = p.get("law_name", "").strip()
    article = p.get("article", "").strip()
    title = (p.get("title") or "").strip()
    if article and title and title not in article:
        return f"{law_name} {article}({title})의 내용은 무엇인가?"
    if article:
        return f"{law_name} {article}의 내용은 무엇인가?"
    return f"{law_name} 관련 내용은 무엇인가?"

def build_query_admin(p: Dict[str, Any]) -> str:
    rule = p.get("rule_name", "").strip()
    article = (p.get("article") or "").strip()
    title = (p.get("title") or "").strip()
    annex = (p.get("annex_title") or p.get("appendix_title") or "").strip()

    if article:
        if title and title not in article:
            return f"{rule} {article}({title})의 내용은 무엇인가?"
        return f"{rule} {article}의 내용은 무엇인가?"
    if annex:
        return f"{rule}의 '{annex}' 별표 내용은 무엇인가?"
    return f"{rule} 관련 내용은 무엇인가?"

def build_query_prec(p: Dict[str, Any]) -> str:
    title = (p.get("title") or p.get("headnote") or "").strip()
    if title:
        return f"{title}의 요지는 무엇인가?"
    return "이 판례의 요지는 무엇인가?"


# =========================
# Hard negative utilities
# =========================
def _sample_hard_negatives(
    target: Dict[str, Any],
    pool: List[Dict[str, Any]],
    n: int,
    group_key: Optional[str],
) -> List[str]:
    """
    1) 같은 group_key(예: 같은 law_name/rule_name/court_name)에서 우선 추출
    2) 부족하면 동일 타입 전체에서 보충
    """
    if n <= 0:
        return []

    target_id = target.get("id")
    same_group_ids: List[str] = []
    if group_key:
        gval = (target.get(group_key) or "").strip()
        if gval:
            same_group_ids = [
                x["id"]
                for x in pool
                if x.get(group_key, "").strip() == gval and x.get("id") != target_id
            ]

    random.shuffle(same_group_ids)
    hn: List[str] = same_group_ids[:n]

    # 보충
    if len(hn) < n:
        rest = [x["id"] for x in pool if x.get("id") not in set(hn) and x.get("id") != target_id]
        random.shuffle(rest)
        need = n - len(hn)
        hn.extend(rest[:need])

    return hn[:n]


# =========================
# Builders for each type
# =========================
def build_pairs_from_law(law: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in law:
        q = build_query_law(p)
        pos = [p["id"]]
        # 같은 법령명에서 우선 추출, 이후 부족분 전체에서 보충
        hn = _sample_hard_negatives(p, law, hn_per_q, group_key="law_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn
        })
    return rows

def build_pairs_from_admin(admin: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in admin:
        q = build_query_admin(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, admin, hn_per_q, group_key="rule_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn
        })
    return rows

def build_pairs_from_prec(prec: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in prec:
        q = build_query_prec(p)
        pos = [p["id"]]
        # 같은 법원명(또는 같은 법원/사건군)으로 우선 추출
        hn = _sample_hard_negatives(p, prec, hn_per_q, group_key="court_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn
        })
    return rows


# =========================
# Main maker
# =========================
def make_pairs(
    law_path: Optional[str],
    admin_path: Optional[str],
    prec_path: Optional[str],
    out_path: str,
    hn_per_q: int = 2,
    seed: int = 42,
) -> None:
    random.seed(seed)

    law = list(read_jsonl(law_path)) if law_path else []
    admin = list(read_jsonl(admin_path)) if admin_path else []
    prec = list(read_jsonl(prec_path)) if prec_path else []

    rows: List[Dict[str, Any]] = []
    rows.extend(build_pairs_from_law(law, hn_per_q) if law else [])
    rows.extend(build_pairs_from_admin(admin, hn_per_q) if admin else [])
    rows.extend(build_pairs_from_prec(prec, hn_per_q) if prec else [])

    # assign query_id sequentially for stability
    for i, r in enumerate(rows, 1):
        r["query_id"] = f"Q_{i:05d}"

    write_jsonl(out_path, rows)
    print(f"[make_pairs] queries: {len(rows)} → {out_path}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", required=False, help="law_passages.jsonl")
    ap.add_argument("--admin", required=False, help="admin_passages.jsonl")
    ap.add_argument("--prec", required=False, help="prec_passages.jsonl")
    ap.add_argument("--out", required=True, help="pairs_train.jsonl")
    ap.add_argument("--hn_per_q", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    make_pairs(
        law_path=args.law,
        admin_path=args.admin,
        prec_path=args.prec,
        out_path=args.out,
        hn_per_q=args.hn_per_q,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
