# lex_dpr/utils/eval.py
from typing import Dict, Tuple, List, Set
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def build_ir_evaluator(
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    read_jsonl_fn,
    k_vals: List[int] | int | None = None,
):
    if k_vals is None:
        k_vals = [1, 3, 5, 10]
    elif isinstance(k_vals, int):
        k_vals = [k_vals]

    corpus = {pid: row["text"] for pid, row in passages.items()}

    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}
    for row in read_jsonl_fn(eval_pairs_path):
        qid = row.get("query_id") or row["query_text"]
        queries[qid] = row["query_text"]
        relevant_docs[qid] = set(row["positive_passages"])

    max_k = max(1, len(corpus))
    k_vals = [k for k in k_vals if 1 <= k <= max_k] or [1]

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=list(k_vals),
        ndcg_at_k=list(k_vals),
        map_at_k=list(k_vals),
        accuracy_at_k=list(k_vals),  # 빈 리스트 금지
        precision_recall_at_k=[],    # 필요시 list(k_vals)
        show_progress_bar=False,
        name="val",
    )
    return evaluator, k_vals
