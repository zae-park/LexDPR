from __future__ import annotations
import argparse, random
from typing import Dict, Any, List
from .utils_io import read_jsonl, write_jsonl

def index_corpus(corpus_path: str) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(corpus_path)}

def law_title_query(p: Dict[str, Any]) -> str:
    # 조문 제목이 없을 수 있어 article로 대체
    law_name = p.get("law_name", "")
    article = p.get("article", "")
    return f"{law_name} {article}의 내용은 무엇인가?"

def prec_title_query(p: Dict[str, Any]) -> str:
    title = p.get("title") or ""
    if title:
        return f"{title}의 요지는 무엇인가?"
    return "이 판례의 요지는 무엇인가?"

def admin_query_for_rule(p):
    # 조문 passage
    rule = p.get("rule_name","")
    article = p.get("article","")
    title = p.get("title","")
    if article:
        if title:
            return f"{rule} {article}({title})의 내용은 무엇인가?"
        return f"{rule} {article}의 내용은 무엇인가?"
    # 별표 passage
    annex = p.get("annex_title") or p.get("appendix_title") or ""
    if annex:
        return f"{rule}의 '{annex}' 별표 내용은 무엇인가?"
    # fallback
    return f"{rule} 관련 내용은 무엇인가?"


def make_pairs(law_path: str, prec_path: str, out_path: str, hn_per_q: int = 2) -> None:
    law = list(read_jsonl(law_path)) if law_path else []
    prec = list(read_jsonl(prec_path)) if prec_path else []

    law_ids = [r["id"] for r in law]
    prec_ids = [r["id"] for r in prec]

    rows = []
    qid = 1

    # 법령: 같은 법령명 + 조문을 query로, 해당 본문을 positive
    for p in law:
        q = law_title_query(p)
        pos = [p["id"]]
        # 간단 hard negative: 같은 law_name이지만 다른 article
        hn_pool = [x["id"] for x in law if x.get("law_name")==p.get("law_name") and x.get("article")!=p.get("article")]
        random.shuffle(hn_pool)
        rows.append({"query_id": f"Q_{qid:05d}", "query_text": q, "positive_passages": pos, "hard_negatives": hn_pool[:hn_per_q]})
        qid += 1

    # 판례: 제목 기반 질의
    for p in prec:
        q = prec_title_query(p)
        pos = [p["id"]]
        # 간단 hard negative: 같은 court_name이지만 다른 케이스
        hn_pool = [x["id"] for x in prec if x.get("court_name")==p.get("court_name") and x["parent_id"]!=p["parent_id"]]
        random.shuffle(hn_pool)
        rows.append({"query_id": f"Q_{qid:05d}", "query_text": q, "positive_passages": pos, "hard_negatives": hn_pool[:hn_per_q]})
        qid += 1

    write_jsonl(out_path, rows)
    print(f"[make_pairs] queries: {len(rows)} → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", help="law_passages.jsonl")
    ap.add_argument("--prec", help="prec_passages.jsonl")
    ap.add_argument("--out", required=True, help="pairs_train.jsonl")
    ap.add_argument("--hn_per_q", type=int, default=2)
    args = ap.parse_args()
    make_pairs(args.law, args.prec, args.out, hn_per_q=args.hn_per_q)

if __name__ == "__main__":
    main()
