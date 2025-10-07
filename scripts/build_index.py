
import argparse
from pathlib import Path
import numpy as np
import faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="directory with passage embeddings and ids (checkpoint/)")
    ap.add_argument("--output", type=str, required=True, help="output index directory")
    ap.add_argument("--factory", type=str, default="Flat", help='FAISS factory string, e.g. "IVF512,Flat"')
    ap.add_argument("--metric", type=str, default="dot", choices=["dot","cosine"])
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    P = np.load(in_dir / "passage_embeds.npy")
    d = P.shape[1]
    if args.metric == "cosine":
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
        P = P / norms
    if args.factory == "Flat":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.index_factory(d, args.factory)
    index.add(P.astype(np.float32))

    faiss.write_index(index, str(out_dir / "lexdpr.faiss"))
    np.save(out_dir / "passage_ids.npy", np.load(in_dir / "passage_ids.npy"))
    print(f"Index built at {out_dir}")

if __name__ == "__main__":
    main()
