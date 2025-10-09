
import argparse, numpy as np, faiss
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="checkpoint")
    ap.add_argument("--output", type=str, default="index")
    ap.add_argument("--factory", type=str, default="Flat")
    ap.add_argument("--metric", type=str, default="dot", choices=["dot","cosine"])
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    P = np.load(in_dir/"passage_embeds.npy")
    d = P.shape[1]
    if args.metric == "cosine":
        P = P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-8)

    index = faiss.IndexFlatIP(d) if args.factory=="Flat" else faiss.index_factory(d, args.factory)
    index.add(P.astype(np.float32))
    faiss.write_index(index, str(out_dir/"lexdpr.faiss"))

    ids = np.load(in_dir/"passage_ids.npy", allow_pickle=True)
    np.save(out_dir/"passage_ids.npy", ids)
    print(f"Index built at {out_dir} with {len(ids)} passages")

if __name__ == "__main__":
    main()
