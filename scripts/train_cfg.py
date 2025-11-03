#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, random
from typing import Dict, List
import math
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from omegaconf import OmegaConf

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_passages(paths: List[str]) -> Dict[str, dict]:
    idx = {}
    for p in paths:
        for row in read_jsonl(p):
            idx[row["id"]] = row
    return idx

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_ir_evaluator(passages: dict, eval_pairs_path: str, k_vals=None):
    # 1) k 리스트 정규화
    if k_vals is None:
        k_vals = [1, 3, 5, 10]
    elif isinstance(k_vals, int):
        k_vals = [k_vals]

    # 2) corpus / queries / relevant_docs 구성
    corpus = {pid: row["text"] for pid, row in passages.items()}

    queries, relevant_docs = {}, {}
    for row in read_jsonl(eval_pairs_path):
        qid = row.get("query_id") or row["query_text"]
        queries[qid] = row["query_text"]
        relevant_docs[qid] = set(row["positive_passages"])

    # 3) 코퍼스 크기에 맞춰 k 잘라내기 (빈 리스트 방지)
    max_k = max(1, len(corpus))  # 최소 1
    k_vals = [k for k in k_vals if 1 <= k <= max_k]
    if not k_vals:
        k_vals = [1]

    # 4) evaluator 생성 (모든 @k 인자에 비지 않은 리스트 전달)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=list(k_vals),
        ndcg_at_k=list(k_vals),
        map_at_k=list(k_vals),
        accuracy_at_k=list(k_vals),         # ★ 빈 리스트 금지
        precision_recall_at_k=[],           # 원하면 여기도 k_vals 가능
        show_progress_bar=False,
        name="val",
    )
    return evaluator, k_vals

def eval_recall_at_k(model: SentenceTransformer, passages: Dict[str, dict], eval_pairs_path: str, k: int = 10) -> float:
    eval_pairs = list(read_jsonl(eval_pairs_path))
    corpus_texts = [row["text"] for row in passages.values()]
    corpus_ids = list(passages.keys())
    with torch.no_grad():
        corpus_emb = model.encode(corpus_texts, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    hit = 0
    for row in eval_pairs:
        q = row["query_text"]
        pos_ids = set(row["positive_passages"])
        q_emb = model.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, corpus_emb)[0]
        topk = torch.topk(scores, k=k).indices.tolist()
        top_ids = {corpus_ids[i] for i in topk}
        if top_ids & pos_ids:
            hit += 1
    return hit / max(1, len(eval_pairs))

def train_bi(cfg):
    set_seed(cfg.seed)
    passages = load_passages(cfg.data.passages)
    pairs = list(read_jsonl(cfg.data.pairs))

    model = SentenceTransformer(cfg.model.bi_model)

    examples = []
    for row in pairs:
        q = row["query_text"]
        pos_ids = row["positive_passages"]
        neg_ids = row.get("hard_negatives", [])
        for pid in pos_ids:
            if pid not in passages: 
                continue
            p_text = passages[pid]["text"]
            neg_texts = [passages[n]["text"] for n in neg_ids if n in passages]
            examples.append(InputExample(texts=[q, p_text] + neg_texts))
    
    examples *= 128
    
    n_ex = len(examples)
    if n_ex == 0:
        raise ValueError(
            "[train_bi] No training examples. "
            "pairs/positive_passages가 passages의 id와 매칭되는지, 경로(configs/data.yaml)가 맞는지 확인하세요."
        )

    # 배치보다 적으면 줄이고, drop_last는 False로
    if n_ex < cfg.data.batches.bi:
        print(f"[train_bi] Reducing batch size: {cfg.data.batches.bi} -> {n_ex}")
        cfg.data.batches.bi = n_ex

    loader = DataLoader(examples, batch_size=cfg.data.batches.bi, shuffle=True, drop_last=False)
    loss = losses.MultipleNegativesRankingLoss(model, scale=cfg.trainer.temperature)
    
    evaluator = None
    if cfg.trainer.eval_pairs:
        evaluator, _ = build_ir_evaluator(passages, cfg.trainer.eval_pairs, k_vals=cfg.trainer.k_values)

    
    steps_per_epoch = max(1, math.ceil(len(examples) / cfg.data.batches.bi))
    total_steps = steps_per_epoch * cfg.trainer.epochs
    warmup = max(10, int(total_steps * 0.1))

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=cfg.trainer.epochs,
        warmup_steps=warmup,
        scheduler="warmupcosine",
        optimizer_params={"lr": cfg.trainer.lr},
        use_amp=bool(cfg.trainer.use_amp),
        show_progress_bar=True,
        evaluator=evaluator,                    # ★ 추가
        evaluation_steps=cfg.trainer.eval_steps # ★ 추가 (예: 500)
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save(os.path.join(cfg.out_dir, "bi_encoder"))

    if cfg.trainer.eval_pairs:
        recall = eval_recall_at_k(model, passages, cfg.trainer.eval_pairs, k=cfg.trainer.k)
        print(f"[Eval] Recall@{cfg.trainer.k}: {recall:.4f}")

def train_ce(cfg):
    set_seed(cfg.seed)
    passages = load_passages(cfg.data.passages)
    pairs = list(read_jsonl(cfg.data.pairs))

    samples = []
    for row in pairs:
        q = row["query_text"]
        for pid in row["positive_passages"]:
            if pid in passages:
                samples.append((q, passages[pid]["text"], 1.0))
        for nid in row.get("hard_negatives", []):
            if nid in passages:
                samples.append((q, passages[nid]["text"], 0.0))
    
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError(
            "[train_ce] No (q,p) samples. positives/hard_negatives가 passages와 매칭되는지 확인하세요."
        )
    if n_samples < cfg.data.batches.ce:
        print(f"[train_ce] Reducing ce batch size: {cfg.data.batches.ce} -> {n_samples}")
        cfg.data.batches.ce = n_samples
        
    model = CrossEncoder(
        cfg.model.ce_model,
        num_labels=1,
        max_length=cfg.cross_trainer.max_len,
        automodel_kwargs={"trust_remote_code": True}
    )

    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.95))
    train_data = samples[:split]
    dev_data = samples[split:]

    evaluator = None
    if dev_data:
        eval_samples = [(q, p, float(l)) for (q, p, l) in dev_data]
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_samples)

    model.fit(
        train_dataloader=DataLoader(train_data, batch_size=cfg.data.batches.ce, shuffle=True),
        evaluator=evaluator,
        epochs=cfg.cross_trainer.epochs,
        learning_rate=cfg.cross_trainer.lr,
        warmup_steps=100,
        output_path=os.path.join(cfg.out_dir, "cross_encoder"),
        show_progress_bar=True
    )

def main():
    base = OmegaConf.load("configs/base.yaml")
    data = OmegaConf.load("configs/data.yaml")
    model = OmegaConf.load("configs/model.yaml")
    cfg = OmegaConf.merge(base, {"data": data, "model": model})

    # dotlist overrides: e.g., model.bi_model=... trainer.epochs=4
    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "bi":
        train_bi(cfg)
    else:
        train_ce(cfg)

if __name__ == "__main__":
    main()
