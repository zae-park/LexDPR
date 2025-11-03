# lex_dpr/embed/build_index.py
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, os
from ..utils.io import read_jsonl
from ..utils.logging import Logger

def build_faiss(
    corpus_path: str,
    model_dir: str,
    out_dir: str,
    index_name: str = "faiss.index",
    batch_size: int = 256
):
    log = Logger()
    model = SentenceTransformer(model_dir)
    texts, ids = [], []
    for row in read_jsonl(corpus_path):
        ids.append(row["id"])
        texts.append(row["text"])

    # 임베딩
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    # FAISS
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, index_name))
    with open(os.path.join(out_dir, "docids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids))
    log.info(f"FAISS saved -> {os.path.join(out_dir, index_name)} with {len(ids)} vectors")
