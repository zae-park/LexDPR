# lex_dpr/eval.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import losses

from .models.encoders import BiEncoder
from .models.templates import TemplateMode, tq, tp
from .utils.io import read_jsonl


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
        accuracy_at_k=list(normalized_k),  # Recall@k와 유사 (상위 k개 중 정답 포함 여부)
        precision_recall_at_k=list(normalized_k),  # Precision@k, Recall@k 명시적으로 계산
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
        # torch.topk는 (values, indices) 튜플을 반환
        # scores는 2D 텐서이므로 squeeze하여 1D로 만든 후 topk 적용
        scores_1d = scores.squeeze(0)
        topk_result = torch.topk(scores_1d, k=min(k, len(corpus_ids)))
        topk_indices = topk_result.indices.tolist()
        top_ids = {corpus_ids[i] for i in topk_indices}
        if top_ids & set(row["positive_passages"]):
            hits += 1

    return hits / max(1, len(eval_pairs))


class ValidationLossEvaluator:
    """
    Validation loss를 계산하는 evaluator
    MultipleNegativesRankingLoss를 사용하여 validation set에 대한 loss를 계산합니다.
    """
    
    def __init__(
        self,
        model: SentenceTransformer,
        passages: Dict[str, Dict],
        eval_pairs_path: str,
        read_jsonl_fn: Callable[[str], Iterable[Dict]],
        temperature: float = 0.05,
        template: TemplateMode = TemplateMode.BGE,
        batch_size: int = 32,
    ):
        self.model = model
        self.passages = passages
        self.eval_pairs_path = eval_pairs_path
        self.read_jsonl_fn = read_jsonl_fn
        self.temperature = temperature
        self.template = template
        self.batch_size = batch_size
        
        # Loss 함수 생성
        self.loss_fn = losses.MultipleNegativesRankingLoss(model, scale=temperature)
        
        # Evaluation 데이터 준비
        self.eval_examples = []
        for row in read_jsonl_fn(eval_pairs_path):
            q_text = tq(row["query_text"], template)
            pos_ids = [pid for pid in row.get("positive_passages", []) if pid in passages]
            for pid in pos_ids:
                p_text = tp(passages[pid]["text"], template)
                from sentence_transformers import InputExample
                self.eval_examples.append(InputExample(texts=[q_text, p_text]))
    
    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """Validation loss 계산"""
        if not self.eval_examples:
            return {"val_loss": 0.0}
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 배치로 나누어 처리
        from torch.utils.data import DataLoader
        loader = DataLoader(self.eval_examples, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                # MultipleNegativesRankingLoss는 배치 내에서 자동으로 negative sampling을 수행
                # 하지만 evaluator에서는 직접 loss를 계산해야 함
                # sentence-transformers의 loss 함수는 forward pass를 통해 loss를 계산
                try:
                    # 임베딩 생성
                    texts = [example.texts for example in batch]
                    query_texts = [t[0] for t in texts]
                    passage_texts = [t[1] for t in texts]
                    
                    query_embeddings = model.encode(query_texts, convert_to_tensor=True, normalize_embeddings=True)
                    passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, normalize_embeddings=True)
                    
                    # Loss 계산: MultipleNegativesRankingLoss는 배치 내에서 각 query에 대해 
                    # positive passage와의 유사도를 높이고, 다른 passage들을 negative로 사용
                    # Cross-entropy loss 형태로 계산
                    # scores: [batch_size, batch_size] - 각 query와 각 passage 간의 유사도
                    scores = torch.mm(query_embeddings, passage_embeddings.t()) / self.temperature
                    # 각 query에 대해 대각선 원소(positive)가 정답
                    labels = torch.arange(len(query_texts), device=scores.device)
                    batch_loss = torch.nn.functional.cross_entropy(scores, labels)
                    total_loss += batch_loss.item()
                    num_batches += 1
                except Exception as e:
                    # Loss 계산 실패 시 스킵
                    import logging
                    logging.warning(f"Validation loss 계산 실패: {e}")
                    continue
        
        model.train()
        
        avg_loss = total_loss / max(1, num_batches)
        return {"val_loss": avg_loss, "val_cosine_loss": avg_loss}  # cosine_loss도 추가하여 호환성 유지
