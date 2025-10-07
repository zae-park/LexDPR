
import argparse
import json
from pathlib import Path
import numpy as np
import faiss

def recall_at_k(preds, gold, k=10):
    hit = 0
    for pid_list, g in zip(preds, gold):
        hit += int(any(p in pid_list[:k] for p in g))
    return hit / max(1, len(gold))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoint/passage_embeds.npy")
    ap.add_argument("--index_dir", type=str, default="index")
    ap.add_argument("--queries", type=str, default="data/queries/queries.jsonl")
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    index = faiss.read_index(str(Path(args.index_dir) / "lexdpr.faiss"))
    passage_ids = np.load(Path(args.index_dir) / "passage_ids.npy", allow_pickle=True).tolist()

    # load queries
    queries = []
    gold = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append(row["question"])
            gold.append(row.get("positive_ids", []))

    # NOTE: This skeleton expects you to pre-encode queries and supply them as embeddings.
    # For simplicity, we mock random query vectors. Replace with real encoding before use.
    d = index.d
    Q = np.random.randn(len(queries), d).astype(np.float32)

    scores, idx = index.search(Q, args.top_k)
    preds = [[passage_ids[i] for i in row] for row in idx]
    r10 = recall_at_k(preds, gold, k=10)
    print(json.dumps({"Recall@10": r10}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
