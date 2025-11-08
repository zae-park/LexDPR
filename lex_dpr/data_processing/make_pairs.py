# lex_dpr/data_processing/make_pairs.py
from __future__ import annotations
import argparse, random, re
from typing import Dict, Any, List, Optional, Tuple
from .utils_io import read_jsonl, write_jsonl


# =========================
# Helpers: text normalize
# =========================
def _one_line(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:max_len]

def _short(s: Optional[str], n: int = 80) -> str:
    return (s or "").strip()[:n]

def _valid_passage(p: Dict[str, Any], min_len: int = 50) -> bool:
    return len((p.get("text") or "").strip()) >= min_len


# =========================
# Query builders (type-wise)
# =========================
def build_query_law(p: Dict[str, Any]) -> str:
    law_name = _short(p.get("law_name"), 60)
    article = _short(p.get("article"), 40)
    title = _short(p.get("title"), 60)
    if article and title and title not in article:
        return f"{law_name} {article}({_short(title,40)})의 내용은 무엇인가?"
    if article:
        return f"{law_name} {article}의 내용은 무엇인가?"
    return f"{law_name} 관련 내용은 무엇인가?"

def build_query_admin(p: Dict[str, Any]) -> str:
    rule = _short(p.get("rule_name"), 60)
    article = _short(p.get("article"), 40)
    title = _short(p.get("title"), 60)
    annex = _short(p.get("annex_title") or p.get("appendix_title"), 60)

    if article:
        if title and title not in article:
            return f"{rule} {article}({_short(title,40)})의 내용은 무엇인가?"
        return f"{rule} {article}의 내용은 무엇인가?"
    if annex:
        return f"{rule}의 '{annex}' 별표 내용은 무엇인가?"
    return f"{rule} 관련 내용은 무엇인가?"

def build_query_prec(p: Dict[str, Any]) -> str:
    title = (p.get("title") or "").strip()
    if title:
        return f"{_one_line(title, 120)}의 요지는 무엇인가?"
    # fallback: headnote/summary에서 한 줄
    hs = (p.get("headnote") or p.get("summary") or "").strip()
    hs = _one_line(hs, 120)
    return f"{hs}의 요지는 무엇인가?" if hs else "이 판례의 요지는 무엇인가?"


# =========================
# Hard negative utilities
# =========================
def _sample_hard_negatives(
    target: Dict[str, Any],
    pool: List[Dict[str, Any]],
    n: int,
    group_key: Optional[str],
    avoid_same_parent: bool = True,
) -> List[str]:
    """
    1) 같은 group_key(예: 같은 law_name/rule_name/court_name)에서 우선 추출
    2) 부족하면 동일 타입 전체에서 보충
    3) 같은 parent_id(동일 문서의 다른 청크)는 제외하여 in-document leakage 방지
    """
    if n <= 0:
        return []

    tid = target.get("id")
    tparent = target.get("parent_id")

    def ok(x: Dict[str, Any]) -> bool:
        if x.get("id") == tid:
            return False
        if avoid_same_parent and tparent and x.get("parent_id") == tparent:
            return False
        return True

    same_group_ids: List[str] = []
    if group_key:
        gval = (target.get(group_key) or "").strip()
        if gval:
            same_group_ids = [
                x["id"] for x in pool
                if ok(x) and (x.get(group_key, "").strip() == gval)
            ]

    random.shuffle(same_group_ids)
    hn: List[str] = same_group_ids[:n]

    if len(hn) < n:
        rest = [x["id"] for x in pool if ok(x) and x.get("id") not in set(hn)]
        random.shuffle(rest)
        hn.extend(rest[: (n - len(hn))])

    return hn[:n]


# =========================
# Builders for each type
#  - meta 보존
#  - very short passage 필터
# =========================
def build_pairs_from_law(law: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    law = [p for p in law if _valid_passage(p)]
    for p in law:
        q = build_query_law(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, law, hn_per_q, group_key="law_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "law",
                "law_name": p.get("law_name"),
                "article": p.get("article"),
                "parent_id": p.get("parent_id"),
            },
        })
    return rows

def build_pairs_from_admin(admin: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    admin = [p for p in admin if _valid_passage(p)]
    for p in admin:
        q = build_query_admin(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, admin, hn_per_q, group_key="rule_name")
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "admin",
                "rule_name": p.get("rule_name"),
                "article": p.get("article"),
                "parent_id": p.get("parent_id"),
            },
        })
    return rows

def build_pairs_from_prec(prec: List[Dict[str, Any]], hn_per_q: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prec = [p for p in prec if _valid_passage(p)]
    for p in prec:
        q = build_query_prec(p)
        pos = [p["id"]]
        hn = _sample_hard_negatives(p, prec, hn_per_q, group_key="court_name")
        # 판례의 headnote/summary/text를 meta에 보관 → cross positive 추출에 활용
        meta_source = " ".join([
            (p.get("headnote") or ""),
            (p.get("summary") or ""),
            (p.get("text") or ""),
        ]).strip()
        rows.append({
            "query_text": q,
            "positive_passages": pos,
            "hard_negatives": hn,
            "meta": {
                "type": "prec",
                "court_name": p.get("court_name"),
                "case_number": p.get("case_number"),
                "parent_id": p.get("parent_id"),
                "source_text": _one_line(meta_source, 400),
            },
        })
    return rows


# =========================
# Cross-type positives (prec → law)
#  - 판례 요지/본문에서 "○○법 제n조(의m)" 인용 탐지
#  - 해당 법령 passage를 positive에 추가 (상한 2개)
# =========================
LAW_MENTION = re.compile(
    r"([가-힣A-Za-z0-9·\s]+법)\s*제?\s*([0-9]+)\s*조(?:\s*의\s*([0-9]+))?",
    flags=re.UNICODE
)

def _law_index_by_name(law_passages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for lp in law_passages:
        name = (lp.get("law_name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name, []).append(lp)
    return by_name

def _article_has_number(art: str, num: str) -> bool:
    """article('제536조의2')에 num('536')이 포함되는지 간단 판정"""
    art = (art or "").replace(" ", "")
    return num in re.sub(r"[^0-9]", "", art)

def attach_cross_positives(rows: List[Dict[str, Any]], law_passages: List[Dict[str, Any]], max_add: int = 2) -> None:
    if not rows or not law_passages:
        return
    law_by_name = _law_index_by_name(law_passages)

    for r in rows:
        meta = r.get("meta") or {}
        if meta.get("type") != "prec":
            continue
        src = meta.get("source_text") or r.get("query_text", "")
        adds: List[str] = []

        # 여러 인용 가능 → 좌측부터 탐색
        for m in LAW_MENTION.finditer(src):
            law_name = _one_line(m.group(1), 80)
            num = (m.group(2) or "").strip()
            # 의조 번호(예: 조의2)는 여기선 우선 num만 사용
            cands = law_by_name.get(law_name)
            if not cands:
                continue
            for lp in cands:
                if _article_has_number(lp.get("article") or "", num):
                    adds.append(lp["id"])
                    if len(adds) >= max_add:
                        break
            if len(adds) >= max_add:
                break

        if adds:
            # 기존 positive와 합치되 중복 제거(순서 유지)
            existing = r.get("positive_passages", [])
            merged = list(dict.fromkeys(existing + adds))
            r["positive_passages"] = merged


# =========================
# Dedup by query_text
# =========================
def dedup_by_query(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = r.get("query_text", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


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
    enable_cross_positive: bool = True,
) -> None:
    random.seed(seed)

    law = list(read_jsonl(law_path)) if law_path else []
    admin = list(read_jsonl(admin_path)) if admin_path else []
    prec = list(read_jsonl(prec_path)) if prec_path else []

    rows: List[Dict[str, Any]] = []
    rows.extend(build_pairs_from_law(law, hn_per_q) if law else [])
    rows.extend(build_pairs_from_admin(admin, hn_per_q) if admin else [])
    rows.extend(build_pairs_from_prec(prec, hn_per_q) if prec else [])

    # 판례 → 법령 cross positive 부여
    if enable_cross_positive and law:
        attach_cross_positives(rows, law, max_add=2)

    # dedup by query_text
    rows = dedup_by_query(rows)

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
    ap.add_argument("--no_cross", action="store_true", help="disable prec→law cross positives")
    args = ap.parse_args()

    make_pairs(
        law_path=args.law,
        admin_path=args.admin,
        prec_path=args.prec,
        out_path=args.out,
        hn_per_q=args.hn_per_q,
        seed=args.seed,
        enable_cross_positive=(not args.no_cross),
    )

if __name__ == "__main__":
    main()
