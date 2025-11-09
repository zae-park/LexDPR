# lex_dpr/utils/eval.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from .models.encoders import BiEncoder
from .models.templates import TemplateMode, tq, tp


def _normalize_k_values(k_vals: Sequence[int] | int | None, corpus_size: int) -> List[int]:
    if k_vals is None:
        values = [1, 3, 5, 10]
    elif isinstance(k_vals, int):
        values = [k_vals]
    else:
        values = list(k_vals)

    corpus_size = max(1, corpus_size)
    return [k for k in values if 1 <= k <= corpus_size] or [1]


def build_ir_evaluator(
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    read_jsonl_fn: Callable[[str], Iterable[Dict]],
    *,
    k_vals: Sequence[int] | int | None = None,
    template: TemplateMode = TemplateMode.BGE,
) -> Tuple[InformationRetrievalEvaluator, List[int]]:
    """
    IR 평가자를 생성한다. 템플릿 모드는 BGE 프롬프트 적용 여부를 제어한다.
    """
    corpus = {pid: tp(row["text"], template) for pid, row in passages.items()}

    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}
    for row in read_jsonl_fn(eval_pairs_path):
        qid = row.get("query_id") or row["query_text"]
        queries[qid] = tq(row["query_text"], template)
        relevant_docs[qid] = set(row["positive_passages"])

    normalized_k = _normalize_k_values(k_vals, len(corpus))

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=list(normalized_k),
        ndcg_at_k=list(normalized_k),
        map_at_k=list(normalized_k),
        accuracy_at_k=list(normalized_k),  # 비어 있으면 안 됨
        precision_recall_at_k=[],  # 필요하면 list(normalized_k)로 변경 가능
        show_progress_bar=False,
        name="val",
    )
    return evaluator, normalized_k


def eval_recall_at_k(
    encoder: BiEncoder,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    read_jsonl_fn: Callable[[str], Iterable[Dict]],
    *,
    k: int = 10,
) -> float:
    """
    간단한 Recall@k 평가. encoder는 BiEncoder 래퍼를 기대한다.
    """
    eval_pairs = list(read_jsonl_fn(eval_pairs_path))
    if not eval_pairs:
        return 0.0

    corpus_ids = list(passages.keys())
    corpus_texts = [passages[i]["text"] for i in corpus_ids]

    with torch.no_grad():
        corpus_emb = encoder.encode_passages(corpus_texts, batch_size=128)
    corpus_tensor = torch.from_numpy(corpus_emb).float()

    hits = 0
    for row in eval_pairs:
        q_emb = encoder.encode_queries([row["query_text"]], batch_size=1)
        q_tensor = torch.from_numpy(q_emb).float()
        scores = q_tensor @ corpus_tensor.T
        topk = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
        top_ids = {corpus_ids[i] for i in topk}
        if top_ids & set(row["positive_passages"]):
            hits += 1

    return hits / max(1, len(eval_pairs))
