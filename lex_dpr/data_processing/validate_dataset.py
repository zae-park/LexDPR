from __future__ import annotations
import argparse
from .utils_io import read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pairs", required=True)
    args = ap.parse_args()

    corpus_ids = {row["id"] for row in read_jsonl(args.corpus)}
    missing = set()
    queries = set()
    empty_text = 0

    for row in read_jsonl(args.pairs):
        qid = row.get("query_id")
        if qid in queries:
            print(f"[warn] duplicated query_id: {qid}")
        queries.add(qid)

        for pid in row.get("positive_passages", []):
            if pid not in corpus_ids:
                missing.add(pid)
        for pid in row.get("hard_negatives", []):
            if pid not in corpus_ids:
                missing.add(pid)

    if missing:
        ex = list(missing)[:10]
        print(f"[validate] missing passage ids: {len(missing)} e.g., {ex}")
    else:
        print("[validate] all passage ids exist in corpus ✅")

if __name__ == "__main__":
    main()
