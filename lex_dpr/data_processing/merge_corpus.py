from __future__ import annotations
import argparse
from .utils_io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", required=False, help="law_passages.jsonl")
    ap.add_argument("--prec", required=False, help="prec_passages.jsonl")
    ap.add_argument("--out", required=True, help="merged_corpus.jsonl")
    args = ap.parse_args()

    ids = set()
    merged = []
    for path in [args.law, args.prec]:
        if not path: continue
        for row in read_jsonl(path):
            if row["id"] in ids:
                continue
            ids.add(row["id"])
            merged.append(row)

    write_jsonl(args.out, merged)
    print(f"[merge_corpus] merged: {len(merged)} → {args.out}")

if __name__ == "__main__":
    main()
