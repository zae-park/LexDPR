# scripts/train_cfg.py

import os, sys, math
from typing import Dict
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from omegaconf import OmegaConf

from lex_dpr.models.encoders import BiEncoder
from lex_dpr.utils.io import read_jsonl
from lex_dpr.data import load_passages
from lex_dpr.utils.seed import set_seed
from lex_dpr.models.factory import get_bi_encoder
from lex_dpr.models.templates import TemplateMode, tq, tp

BGE_TEMPLATE_QUERY = TemplateMode.BGE

# -------------------------
# Evaluator (IR) 빌더
# -------------------------
def build_ir_evaluator(passages: dict, eval_pairs_path: str, k_vals=None, use_bge_tmpl: bool = True):
    # 1) k 리스트 정규화
    if k_vals is None:
        k_vals = [1, 3, 5, 10]
    elif isinstance(k_vals, int):
        k_vals = [k_vals]

    # 2) corpus / queries / relevant_docs 구성 (템플릿 동일 적용)
    corpus = {pid: tp(row["text"], BGE_TEMPLATE_QUERY) for pid, row in passages.items()}

    queries, relevant_docs = {}, {}
    for row in read_jsonl(eval_pairs_path):
        qid = row.get("query_id") or row["query_text"]
        queries[qid] = tq(row["query_text"], BGE_TEMPLATE_QUERY)
        relevant_docs[qid] = set(row["positive_passages"])

    # 3) 코퍼스 크기에 맞춰 k 잘라내기 (빈 리스트 방지)
    max_k = max(1, len(corpus))  # 최소 1
    k_vals = [k for k in k_vals if 1 <= k <= max_k] or [1]

    # 4) evaluator 생성 (모든 @k 인자에 비지 않은 리스트 전달)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=list(k_vals),
        ndcg_at_k=list(k_vals),
        map_at_k=list(k_vals),
        accuracy_at_k=list(k_vals),     # 빈 리스트 금지
        precision_recall_at_k=[],       # 필요 시 list(k_vals)로 교체 가능
        show_progress_bar=False,
        name="val",
    )
    return evaluator, k_vals


# -------------------------
# 간단 Recall@k 평가 (템플릿 적용)
# -------------------------
def eval_recall_at_k(
    model: BiEncoder,
    passages: Dict[str, dict],
    eval_pairs_path: str,
    k: int = 10,
    use_bge_tmpl: bool = True
) -> float:
    eval_pairs = list(read_jsonl(eval_pairs_path))
    corpus_ids = list(passages.keys())
    corpus_texts = [tp(passages[i]["text"], BGE_TEMPLATE_QUERY) for i in corpus_ids]

    with torch.no_grad():
        corpus_emb = model.encode(
            corpus_texts, batch_size=128, convert_to_tensor=True, normalize_embeddings=True
        )

    hit = 0
    for row in eval_pairs:
        q = tq(row["query_text"], BGE_TEMPLATE_QUERY)
        pos_ids = set(row["positive_passages"])
        q_emb = model.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, corpus_emb)[0]
        topk = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
        top_ids = {corpus_ids[i] for i in topk}
        if top_ids & pos_ids:
            hit += 1
    return hit / max(1, len(eval_pairs))


# -------------------------
# BI 학습 루틴
# -------------------------
def train_bi(cfg):
    set_seed(cfg.seed)

    # 모델 템플릿 사용 여부 (모델 설정에 없으면 True 기본)
    use_bge_tmpl = bool(getattr(cfg.model, "use_bge_template", True))

    passages = load_passages(cfg.data.passages)
    pairs = list(read_jsonl(cfg.data.pairs))

    # 모델 초기화
    model = get_bi_encoder(cfg.model.bi_model)
    # max_len 가드 (있으면 적용)
    max_len = int(getattr(cfg.model, "max_len", 0) or 0)
    if max_len > 0:
        model.max_seq_length = max_len
        print(f"[train_bi] set model.max_seq_length = {model.max_seq_length}")

    # (q, p) 페어만 구성 → in-batch negatives
    examples = []
    miss_pos = 0
    for row in pairs:
        q = tq(row["query_text"], BGE_TEMPLATE_QUERY)
        for pid in row["positive_passages"]:
            if pid not in passages:
                miss_pos += 1
                continue
            p_text = tp(passages[pid]["text"], BGE_TEMPLATE_QUERY)
            examples.append(InputExample(texts=[q, p_text]))

    if miss_pos:
        print(f"[train_bi] skipped positives not in corpus: {miss_pos}")

    # 소량 데이터 증폭 옵션 (선택)
    if int(getattr(cfg.data, "multiply", 0) or 0) > 0:
        examples *= int(cfg.data.multiply)

    n_ex = len(examples)
    if n_ex == 0:
        raise ValueError(
            "[train_bi] No training examples. pairs/positives가 corpus id와 매칭되는지 확인하세요."
        )

    # 배치보다 적으면 축소
    if n_ex < cfg.data.batches.bi:
        print(f"[train_bi] Reducing batch size: {cfg.data.batches.bi} -> {n_ex}")
        cfg.data.batches.bi = n_ex

    loader = DataLoader(examples, batch_size=cfg.data.batches.bi, shuffle=True, drop_last=False)
    loss = losses.MultipleNegativesRankingLoss(model, scale=cfg.trainer.temperature)

    evaluator = None
    if cfg.trainer.eval_pairs:
        evaluator, _ = build_ir_evaluator(
            passages, cfg.trainer.eval_pairs, k_vals=cfg.trainer.k_values, use_bge_tmpl=use_bge_tmpl
        )

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
        evaluator=evaluator,
        evaluation_steps=cfg.trainer.eval_steps if evaluator else None,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save(os.path.join(cfg.out_dir, "bi_encoder"))

    if cfg.trainer.eval_pairs:
        recall = eval_recall_at_k(
            model, passages, cfg.trainer.eval_pairs, k=cfg.trainer.k, use_bge_tmpl=use_bge_tmpl
        )
        print(f"[Eval] Recall@{cfg.trainer.k}: {recall:.4f}")


# -------------------------
# Entrypoint
# -------------------------
def main():
    base = OmegaConf.load("configs/base.yaml")
    data = OmegaConf.load("configs/data.yaml")
    model = OmegaConf.load("configs/model.yaml")
    cfg = OmegaConf.merge(base, {"data": data, "model": model})

    # dotlist overrides: e.g., model.bi_model=... trainer.epochs=4
    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))
    train_bi(cfg)


if __name__ == "__main__":
    main()
