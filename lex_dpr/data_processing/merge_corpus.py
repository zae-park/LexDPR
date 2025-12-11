from __future__ import annotations
import argparse
from ..utils.io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", required=False, help="data/processed/law_passages.jsonl")
    ap.add_argument("--admin", required=False, help="data/processed/admin_passages.jsonl")
    ap.add_argument("--prec", required=False, help="data/processed/prec_passages.jsonl 또는 prec_fallback_passages.jsonl")
    ap.add_argument("--out", required=True, help="merged_corpus.jsonl")
    args = ap.parse_args()

    ids = set()
    merged = []
    passage_counts = {}
    
    for path_name, path in [("law", args.law), ("admin", args.admin), ("prec", args.prec)]:
        if not path: 
            continue
        count = 0
        for row in read_jsonl(path):
            if row["id"] in ids:
                continue
            ids.add(row["id"])
            merged.append(row)
            count += 1
        if count > 0:
            passage_counts[path_name] = count
    
    write_jsonl(args.out, merged)
    
    print(f"[merge_corpus] 병합 완료:")
    for name, count in passage_counts.items():
        print(f"  - {name}: {count:,}개")
    print(f"  총: {len(merged):,}개 → {args.out}")

if __name__ == "__main__":
    main()
