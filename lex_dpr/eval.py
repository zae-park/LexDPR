# lex_dpr/eval.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import torch
import numpy as np
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


class BatchedInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    쿼리를 배치로 처리하는 InformationRetrievalEvaluator
    
    기본 InformationRetrievalEvaluator는 쿼리를 하나씩 처리하지만,
    이 클래스는 쿼리를 배치로 인코딩하여 평가 속도를 향상시킵니다.
    """
    
    def __init__(self, *args, query_batch_size: int = 32, **kwargs):
        # corpus 딕셔너리를 먼저 저장 (super().__init__() 전에)
        corpus = kwargs.get('corpus', None)
        if corpus is None and len(args) > 1:
            # 위치 인자로 전달된 경우
            corpus = args[1] if len(args) > 1 else None
        
        super().__init__(*args, **kwargs)
        self.query_batch_size = query_batch_size
        
        # InformationRetrievalEvaluator는 corpus를 딕셔너리로 받지만, 
        # 내부적으로 corpus_ids와 corpus_texts로 변환할 수 있음
        # 원본 corpus 딕셔너리를 보존하기 위해 저장
        if corpus is not None and isinstance(corpus, dict):
            self._corpus_dict = corpus.copy()
        else:
            self._corpus_dict = None
        
        # primary_metric이 None인 경우 기본값 설정 (SequentialEvaluator 호환성)
        # InformationRetrievalEvaluator는 보통 ndcg@k를 primary_metric으로 사용
        if not hasattr(self, 'primary_metric') or self.primary_metric is None:
            # ndcg_at_k가 있으면 가장 큰 k를 primary로 설정
            if hasattr(self, 'ndcg_at_k') and self.ndcg_at_k:
                max_k = max(self.ndcg_at_k)
                self.primary_metric = f"ndcg@{max_k}"
            # 없으면 mrr_at_k 사용
            elif hasattr(self, 'mrr_at_k') and self.mrr_at_k:
                max_k = max(self.mrr_at_k)
                self.primary_metric = f"mrr@{max_k}"
            # 그것도 없으면 accuracy_at_k 사용
            elif hasattr(self, 'accuracy_at_k') and self.accuracy_at_k:
                max_k = max(self.accuracy_at_k)
                self.primary_metric = f"accuracy@{max_k}"
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """쿼리를 배치로 처리하여 평가"""
        # Corpus를 먼저 인코딩 (한 번만)
        # InformationRetrievalEvaluator는 corpus를 딕셔너리로 받지만, 내부적으로 리스트로 변환할 수 있음
        # corpus_ids와 corpus_texts 속성이 있는지 확인
        if hasattr(self, 'corpus_ids') and hasattr(self, 'corpus_texts'):
            # InformationRetrievalEvaluator가 내부적으로 변환한 경우
            corpus_ids = self.corpus_ids
            corpus_texts = self.corpus_texts
        elif self._corpus_dict is not None:
            # 원본 딕셔너리 사용
            corpus_ids = list(self._corpus_dict.keys())
            corpus_texts = list(self._corpus_dict.values())
        elif isinstance(self.corpus, dict):
            corpus_ids = list(self.corpus.keys())
            corpus_texts = list(self.corpus.values())
        else:
            # 리스트인 경우 (일반적으로는 발생하지 않지만 안전성을 위해)
            corpus_ids = list(range(len(self.corpus)))
            corpus_texts = list(self.corpus)
        
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        corpus_embeddings = torch.from_numpy(corpus_embeddings).float()
        
        # 쿼리를 배치로 인코딩
        if isinstance(self.queries, dict):
            query_ids = list(self.queries.keys())
            query_texts = [self.queries[qid] for qid in query_ids]
        else:
            # 리스트인 경우
            query_ids = list(range(len(self.queries)))
            query_texts = list(self.queries)
        
        # 배치로 쿼리 인코딩 (프로그레스 바 비활성화)
        query_embeddings_list = []
        # tqdm 출력 억제를 위해 임시로 환경 변수 설정
        import os
        from tqdm import tqdm
        import sys
        from io import StringIO
        
        # tqdm 출력을 임시로 억제
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            for i in range(0, len(query_texts), self.query_batch_size):
                batch_texts = query_texts[i:i + self.query_batch_size]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),  # 배치 크기를 실제 텍스트 수로 설정하여 단일 배치로 처리
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                query_embeddings_list.append(torch.from_numpy(batch_embeddings).float())
        finally:
            # stdout 복원
            sys.stdout = old_stdout
        
        query_embeddings = torch.cat(query_embeddings_list, dim=0)
        
        # 모든 쿼리에 대해 점수 계산
        scores = query_embeddings @ corpus_embeddings.T  # [n_queries, n_corpus]
        
        # 각 쿼리에 대해 메트릭 계산
        results = {}
        all_k_values = list(set(self.ndcg_at_k + self.mrr_at_k + self.map_at_k + self.accuracy_at_k + self.precision_recall_at_k))
        for k in all_k_values:
            results[k] = {}
        
        for idx, qid in enumerate(query_ids):
            query_scores = scores[idx].cpu().numpy()
            max_k = max(all_k_values) if all_k_values else 10
            top_k_indices = np.argsort(query_scores)[::-1][:max_k]
            top_k_ids = [corpus_ids[i] for i in top_k_indices]
            
            relevant_docs = self.relevant_docs.get(qid, set())
            
            # 각 k 값에 대해 메트릭 계산
            for k in all_k_values:
                top_k = top_k_ids[:k]
                hits = len(set(top_k) & relevant_docs)
                
                # Accuracy@k (Recall@k와 유사)
                if k in self.accuracy_at_k:
                    results[k].setdefault('accuracy', []).append(1.0 if hits > 0 else 0.0)
                
                # Precision@k
                if k in self.precision_recall_at_k:
                    precision = hits / k if k > 0 else 0.0
                    results[k].setdefault('precision', []).append(precision)
                
                # Recall@k
                if k in self.precision_recall_at_k:
                    recall = hits / len(relevant_docs) if relevant_docs else 0.0
                    results[k].setdefault('recall', []).append(recall)
                
                # MRR@k
                if k in self.mrr_at_k:
                    mrr = 0.0
                    for rank, doc_id in enumerate(top_k, 1):
                        if doc_id in relevant_docs:
                            mrr = 1.0 / rank
                            break
                    results[k].setdefault('mrr', []).append(mrr)
                
                # NDCG@k (간단 버전)
                if k in self.ndcg_at_k:
                    dcg = 0.0
                    for rank, doc_id in enumerate(top_k, 1):
                        if doc_id in relevant_docs:
                            dcg += 1.0 / np.log2(rank + 1)
                    # Ideal DCG는 모든 relevant가 상위에 있을 때
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    results[k].setdefault('ndcg', []).append(ndcg)
                
                # MAP@k (간단 버전)
                if k in self.map_at_k:
                    ap = 0.0
                    relevant_found = 0
                    for rank, doc_id in enumerate(top_k, 1):
                        if doc_id in relevant_docs:
                            relevant_found += 1
                            ap += relevant_found / rank
                    ap = ap / len(relevant_docs) if relevant_docs else 0.0
                    results[k].setdefault('map', []).append(ap)
        
        # 평균 계산 및 결과 반환
        # InformationRetrievalEvaluator의 메트릭 키 형식에 맞춤
        final_results = {}
        for k in all_k_values:
            if k in self.accuracy_at_k and 'accuracy' in results[k]:
                final_results[f"{self.name}_accuracy@{k}"] = np.mean(results[k]['accuracy'])
            if k in self.precision_recall_at_k:
                if 'precision' in results[k]:
                    final_results[f"{self.name}_precision@{k}"] = np.mean(results[k]['precision'])
                if 'recall' in results[k]:
                    final_results[f"{self.name}_recall@{k}"] = np.mean(results[k]['recall'])
            if k in self.mrr_at_k and 'mrr' in results[k]:
                final_results[f"{self.name}_mrr@{k}"] = np.mean(results[k]['mrr'])
            if k in self.ndcg_at_k and 'ndcg' in results[k]:
                final_results[f"{self.name}_ndcg@{k}"] = np.mean(results[k]['ndcg'])
            if k in self.map_at_k and 'map' in results[k]:
                final_results[f"{self.name}_map@{k}"] = np.mean(results[k]['map'])
        
        # primary_metric이 있는 경우 해당 키도 추가 (SequentialEvaluator 호환성)
        # InformationRetrievalEvaluator는 보통 ndcg@k를 primary_metric으로 사용
        # SequentialEvaluator가 evaluator.primary_metric 키를 찾으므로 추가
        # primary_metric이 None이 아닌 경우에만 처리
        primary_key = getattr(self, 'primary_metric', None)
        if primary_key is not None:
            # primary_metric이 결과에 없으면 찾아서 추가
            if primary_key not in final_results:
                # primary_metric 형식에 맞는 키 찾기
                # 예: "ndcg@10" -> "val_ndcg@10" 또는 "val_cosine_ndcg@10"
                found = False
                for key in final_results.keys():
                    # primary_metric이 "ndcg@10" 형식이면 "val_ndcg@10" 또는 "val_cosine_ndcg@10" 찾기
                    if primary_key.replace('@', '_at_') in key or primary_key.replace('@', '@') in key:
                        final_results[primary_key] = final_results[key]
                        found = True
                        break
                # 찾지 못한 경우 첫 번째 ndcg 메트릭 사용
                if not found:
                    for key in final_results.keys():
                        if 'ndcg' in key.lower():
                            final_results[primary_key] = final_results[key]
                            break
        else:
            # primary_metric이 None인 경우, 기본적으로 가장 큰 k의 ndcg를 primary로 설정
            # SequentialEvaluator가 primary_metric을 찾지 못하면 에러가 발생하므로
            # 기본값을 설정해야 함
            if self.ndcg_at_k:
                max_k = max(self.ndcg_at_k)
                primary_key = f"ndcg@{max_k}"
                # 해당 키가 있으면 primary_metric으로 설정
                for key in final_results.keys():
                    if f"ndcg@{max_k}" in key or f"ndcg_at_{max_k}" in key:
                        final_results[primary_key] = final_results[key]
                        # primary_metric 속성도 설정 (다음 호출을 위해)
                        self.primary_metric = primary_key
                        break
        
        return final_results


def build_ir_evaluator(
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    read_jsonl_fn: Callable[[str], Iterable[Dict]],
    *,
    k_vals: Sequence[int] | int | None = None,
    template: TemplateMode = TemplateMode.BGE,
    batch_size: int = 32,  # 평가 시 배치 크기 (메모리 절약)
    use_batched_queries: bool = True,  # 쿼리를 배치로 처리할지 여부 (기본값: True)
) -> Tuple[InformationRetrievalEvaluator, List[int]]:
    """
    IR 평가자를 생성한다. 템플릿 모드는 BGE 프롬프트 적용 여부를 제어한다.
    
    Args:
        batch_size: 평가 시 사용할 배치 크기 (기본값: 32, 메모리 절약을 위해 작게 설정)
        use_batched_queries: 쿼리를 배치로 처리할지 여부 (기본값: True, 평가 속도 향상)
    """
    corpus = {pid: tp(row["text"], template) for pid, row in passages.items()}

    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}
    for row in read_jsonl_fn(eval_pairs_path):
        qid = row.get("query_id") or row["query_text"]
        queries[qid] = tq(row["query_text"], template)
        relevant_docs[qid] = set(row["positive_passages"])

    normalized_k = _normalize_k_values(k_vals, len(corpus))

    # 기본 InformationRetrievalEvaluator 사용 (BatchedInformationRetrievalEvaluator는 버그가 있어서 임시로 비활성화)
    # TODO: BatchedInformationRetrievalEvaluator의 메트릭 계산 로직 수정 필요
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=list(normalized_k),
        ndcg_at_k=list(normalized_k),
        map_at_k=list(normalized_k),
        accuracy_at_k=list(normalized_k),
        precision_recall_at_k=list(normalized_k),
        show_progress_bar=False,
        name="val",
        batch_size=batch_size,
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
        # 모든 passages를 한 번에 임베딩
        corpus_emb = encoder.encode_passages(corpus_texts, batch_size=128)
        corpus_tensor = torch.from_numpy(corpus_emb).float()
        
        # 모든 쿼리를 한 번에 임베딩 (배치 처리로 로그 감소)
        query_texts = [row["query_text"] for row in eval_pairs]
        query_emb = encoder.encode_queries(query_texts, batch_size=128)
        query_tensor = torch.from_numpy(query_emb).float()
        
        # 모든 쿼리에 대해 한 번에 계산
        scores = query_tensor @ corpus_tensor.T  # [n_queries, n_passages]
        
        hits = 0
        for i, row in enumerate(eval_pairs):
            # 각 쿼리에 대해 top-k 검색
            scores_1d = scores[i]  # i번째 쿼리의 모든 passage와의 유사도
            topk_result = torch.topk(scores_1d, k=min(k, len(corpus_ids)))
            topk_indices = topk_result.indices.tolist()
            top_ids = {corpus_ids[j] for j in topk_indices}
            if top_ids & set(row["positive_passages"]):
                hits += 1

    return hits / max(1, len(eval_pairs))


class ValidationLossEvaluator:
    """
    Validation loss를 계산하는 evaluator
    전체 corpus에서 negative를 샘플링하여 실제 검색 시나리오를 모방합니다.
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
        use_full_corpus_negatives: bool = True,
        num_negatives_per_query: int = 1000,  # 각 query당 샘플링할 negative 개수
    ):
        self.model = model
        self.passages = passages
        self.eval_pairs_path = eval_pairs_path
        self.read_jsonl_fn = read_jsonl_fn
        self.temperature = temperature
        self.template = template
        self.batch_size = batch_size
        self.use_full_corpus_negatives = use_full_corpus_negatives
        self.num_negatives_per_query = num_negatives_per_query
        
        # 전체 corpus 텍스트 준비 (negative 샘플링용)
        if self.use_full_corpus_negatives:
            self.corpus_ids = list(passages.keys())
            self.corpus_texts = [tp(passages[pid]["text"], template) for pid in self.corpus_ids]
            import random
            self.negative_indices = list(range(len(self.corpus_ids)))  # negative 샘플링용 인덱스
        
        # Evaluation 데이터 준비
        self.eval_examples = []
        for row in read_jsonl_fn(eval_pairs_path):
            q_text = tq(row["query_text"], template)
            pos_ids = [pid for pid in row.get("positive_passages", []) if pid in passages]
            for pid in pos_ids:
                p_text = tp(passages[pid]["text"], template)
                self.eval_examples.append((q_text, p_text, pos_ids))  # pos_ids도 함께 저장
    
    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """Validation loss 계산 - 전체 corpus에서 negative 샘플링 (최적화 버전)"""
        if not self.eval_examples:
            return {"val_loss": 0.0}
        
        model.eval()
        total_loss = 0.0
        num_queries = 0
        
        import random
        import logging
        
        with torch.no_grad():
            if self.use_full_corpus_negatives:
                # 최적화: 전체 corpus를 한 번만 인코딩하고 재사용
                # 메모리 절약을 위해 배치로 나누어 인코딩하고 CPU로 옮김
                logging.info(f"전체 corpus 인코딩 중... (총 {len(self.corpus_texts):,}개 passage)")
                corpus_embeddings = []
                
                # 배치로 나누어 인코딩 (GPU 메모리 절약)
                for i in range(0, len(self.corpus_texts), self.batch_size):
                    batch_texts = self.corpus_texts[i:i + self.batch_size]
                    batch_embs = model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    # CPU로 옮겨서 메모리 절약 (필요시 GPU로 다시 옮김)
                    corpus_embeddings.append(batch_embs.cpu())
                
                # 전체 corpus 임베딩 결합 (CPU에 저장)
                corpus_embeddings = torch.cat(corpus_embeddings, dim=0)  # [n_corpus, dim]
                logging.info(f"Corpus 인코딩 완료: {corpus_embeddings.shape}")
                
                # 쿼리별로 처리 (이미 인코딩된 corpus 임베딩 재사용)
                for q_text, p_text, pos_ids in self.eval_examples:
                    try:
                        query_emb = model.encode([q_text], convert_to_tensor=True, normalize_embeddings=True)
                        pos_emb = model.encode([p_text], convert_to_tensor=True, normalize_embeddings=True)
                        
                        # 전체 corpus에서 negative 샘플링 (positive 제외)
                        pos_set = set(pos_ids)
                        negative_candidates = [
                            idx for idx in self.negative_indices 
                            if self.corpus_ids[idx] not in pos_set
                        ]
                        
                        # 샘플링할 negative 개수 결정
                        num_samples = min(self.num_negatives_per_query, len(negative_candidates))
                        sampled_indices = random.sample(negative_candidates, num_samples)
                        
                        # 이미 인코딩된 negative 임베딩 추출 (CPU → GPU)
                        neg_embs = corpus_embeddings[sampled_indices].to(query_emb.device)  # [num_negatives, dim]
                        
                        # Positive와 negative 결합
                        all_passage_embs = torch.cat([pos_emb, neg_embs], dim=0)  # [1 + num_negatives, dim]
                        
                        # Loss 계산: query와 모든 passage (positive + negatives) 간의 유사도
                        scores = torch.mm(query_emb, all_passage_embs.t()) / self.temperature  # [1, 1 + num_negatives]
                        labels = torch.zeros(1, dtype=torch.long, device=scores.device)  # 첫 번째(positive)가 정답
                        query_loss = torch.nn.functional.cross_entropy(scores, labels)
                        
                        total_loss += query_loss.item()
                        num_queries += 1
                        
                    except Exception as e:
                        logging.warning(f"Validation loss 계산 실패: {e}")
                        continue
                
                # 메모리 정리
                del corpus_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                # 기존 방식: 배치 내에서만 negative 사용 (하위 호환성)
                for q_text, p_text, pos_ids in self.eval_examples:
                    try:
                        query_emb = model.encode([q_text], convert_to_tensor=True, normalize_embeddings=True)
                        pos_emb = model.encode([p_text], convert_to_tensor=True, normalize_embeddings=True)
                        
                        scores = torch.mm(query_emb, pos_emb.t()) / self.temperature
                        labels = torch.zeros(1, dtype=torch.long, device=scores.device)
                        query_loss = torch.nn.functional.cross_entropy(scores, labels)
                        
                        total_loss += query_loss.item()
                        num_queries += 1
                        
                    except Exception as e:
                        logging.warning(f"Validation loss 계산 실패: {e}")
                        continue
        
        model.train()
        
        avg_loss = total_loss / max(1, num_queries)
        return {"val_loss": avg_loss, "val_cosine_loss": avg_loss}
