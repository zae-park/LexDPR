
import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from models.encoder import TextEncoder
from models.utils import read_jsonl
from omegaconf import OmegaConf

def load_corpus(processed_jsonl: Path):
    ids, texts = [], []
    for row in read_jsonl(processed_jsonl):
        ids.append(row["id"])
        texts.append(row["text"])
    return ids, texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to model/configs yaml")
    args = ap.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
        OmegaConf.load("configs/data.yaml")
    )

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    q_enc = TextEncoder(cfg.encoders.query.pretrained_model_name_or_path, cfg.encoders.query.pooling)
    p_enc = TextEncoder(cfg.encoders.passage.pretrained_model_name_or_path, cfg.encoders.passage.pooling)

    processed_path = Path(cfg.paths.processed_dir) / "corpus.jsonl"
    if not processed_path.exists():
        raise FileNotFoundError(f"{processed_path} not found. Run preprocess_acts.py first.")
    p_ids, p_texts = load_corpus(processed_path)

    print("Encoding passages...")
    P = p_enc.encode(p_texts, device=device, batch_size=64).numpy()
    Path("checkpoint").mkdir(parents=True, exist_ok=True)
    np.save("checkpoint/passage_ids.npy", np.array(p_ids))
    np.save("checkpoint/passage_embeds.npy", P.astype(np.float32))
    print("Saved passage embeddings to checkpoint/.")

    # NOTE: this is a skeleton. For real DPR, implement contrastive training with positive/negative pairs.
    print("Skeleton training complete. Implement contrastive fine-tuning as needed.")

if __name__ == "__main__":
    main()
