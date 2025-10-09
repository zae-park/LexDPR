
import argparse, numpy as np, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import ujson
from tqdm import tqdm

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield ujson.loads(line)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="checkpoint")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","cls"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModel.from_pretrained(args.model).to(device).eval()

    ids, texts = [], []
    for row in read_jsonl(Path(args.input)):
        ids.append(row["id"])
        texts.append(row["text"])

    embs = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i+args.batch_size]
        toks = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**toks).last_hidden_state
            if args.pooling == "cls":
                rep = out[:,0]
            else:
                rep = mean_pool(out, toks["attention_mask"])
        embs.append(rep.detach().cpu().numpy())
    E = np.concatenate(embs, axis=0).astype(np.float32)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir/"passage_ids.npy", np.array(ids, dtype=object))
    np.save(outdir/"passage_embeds.npy", E)
    print(f"Saved {len(ids)} passage embeddings to {outdir}")

if __name__ == "__main__":
    main()
