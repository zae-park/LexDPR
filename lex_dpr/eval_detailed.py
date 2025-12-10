# lex_dpr/eval_detailed.py
"""
상세 평가 분석 모듈

기본 메트릭 외에 다음 분석을 제공:
- 쿼리별 성능 분석
- 소스별 성능 분석
- 실패 케이스 분석
- 쿼리/Passage 길이별 성능 분석
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from lex_dpr.data import load_passages
from lex_dpr.models.encoders import BiEncoder
from lex_dpr.models.templates import TemplateMode, tq, tp
from lex_dpr.utils.io import read_jsonl


def detect_query_source(query_meta: Dict) -> str:
    """쿼리의 소스 타입 감지"""
    qtype = query_meta.get("type", "")
    if qtype in ["법령", "law"]:
        return "법령"
    elif qtype in ["행정규칙", "admin"]:
        return "행정규칙"
    elif qtype in ["판례", "prec"]:
        return "판례"
    return "기타"


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 추정 (공백 기준 단어 수)"""
    return len(text.split())


class DetailedEvaluationResult:
    """상세 평가 결과를 담는 클래스"""
    
    def __init__(self):
        # 기본 메트릭
        self.metrics: Dict[str, float] = {}
        
        # 쿼리별 상세 결과
        self.query_results: List[Dict] = []
        
        # 소스별 통계
        self.source_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "mrr": [],
            "ndcg": [],
            "recall": [],
        })
        
        # 실패 케이스
        self.failed_queries: List[Dict] = []
        
        # 길이별 통계
        self.length_stats: Dict[str, Dict] = defaultdict(lambda: {
            "mrr": [],
            "ndcg": [],
            "recall": [],
        })


def evaluate_detailed(
    model: SentenceTransformer,
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    k_values: List[int] = [1, 3, 5, 10],
    template: TemplateMode = TemplateMode.BGE,
    batch_size: int = 64,
) -> DetailedEvaluationResult:
    """
    상세 평가 수행
    
    Args:
        model: 평가할 SentenceTransformer 모델
        passages: Passage 딕셔너리 {id: {text, ...}}
        eval_pairs_path: 평가용 쌍 JSONL 경로
        k_values: 평가할 k 값 목록
        template: 템플릿 모드
        batch_size: 배치 크기
    
    Returns:
        DetailedEvaluationResult 객체
    """
    result = DetailedEvaluationResult()
    
    # 평가 쌍 로드
    eval_pairs = list(read_jsonl(eval_pairs_path))
    if not eval_pairs:
        return result
    
    # Corpus 임베딩 생성
    corpus_ids = list(passages.keys())
    corpus_texts = [passages[pid]["text"] for pid in corpus_ids]
    corpus_texts_templated = [tp(text, template) for text in corpus_texts]
    
    print(f"[평가] Corpus 임베딩 생성 중... ({len(corpus_ids)}개)")
    with torch.no_grad():
        corpus_embeddings = model.encode(
            corpus_texts_templated,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    corpus_tensor = torch.from_numpy(corpus_embeddings).float()
    
    # 쿼리별 평가
    print(f"[평가] 쿼리별 평가 중... ({len(eval_pairs)}개)")
    query_embeddings_list = []
    query_texts = []
    query_metas = []
    
    for pair in eval_pairs:
        query_text = pair["query_text"]
        query_texts.append(query_text)
        query_metas.append(pair.get("meta", {}))
        query_embeddings_list.append(tq(query_text, template))
    
    with torch.no_grad():
        query_embeddings = model.encode(
            query_embeddings_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    
    # 각 쿼리별로 검색 및 평가
    max_k = max(k_values)
    all_mrr = defaultdict(list)
    all_ndcg = defaultdict(list)
    all_recall = defaultdict(list)
    all_precision = defaultdict(list)
    
    for idx, pair in enumerate(eval_pairs):
        query_id = pair.get("query_id", f"Q_{idx}")
        query_text = pair["query_text"]
        positive_ids = set(pair["positive_passages"])
        query_meta = pair.get("meta", {})
        
        # 검색 수행
        q_emb = torch.from_numpy(query_embeddings[idx:idx+1]).float()
        scores = (q_emb @ corpus_tensor.T).squeeze(0)
        top_indices = torch.topk(scores, k=min(max_k, len(corpus_ids))).indices.tolist()
        top_ids = [corpus_ids[i] for i in top_indices]
        
        # 각 k 값에 대해 메트릭 계산
        query_result = {
            "query_id": query_id,
            "query_text": query_text,
            "positive_passages": list(positive_ids),
            "source": detect_query_source(query_meta),
            "query_length_chars": len(query_text),
            "query_length_tokens": count_tokens(query_text),
            "positive_count": len(positive_ids),
            "ranks": {},
            "metrics": {},
        }
        
        # 각 k에 대해 순위 및 메트릭 계산
        for k in k_values:
            top_k_ids = set(top_ids[:k])
            
            # 첫 번째 정답의 순위 찾기
            first_rank = None
            for rank, pid in enumerate(top_ids[:k], 1):
                if pid in positive_ids:
                    first_rank = rank
                    break
            
            query_result["ranks"][f"first_positive_rank@{k}"] = first_rank
            
            # Recall@k
            recall = len(top_k_ids & positive_ids) / len(positive_ids) if positive_ids else 0.0
            all_recall[k].append(recall)
            query_result["metrics"][f"recall@{k}"] = recall
            
            # Precision@k
            precision = len(top_k_ids & positive_ids) / k if k > 0 else 0.0
            all_precision[k].append(precision)
            query_result["metrics"][f"precision@{k}"] = precision
            
            # MRR@k
            mrr = 1.0 / first_rank if first_rank else 0.0
            all_mrr[k].append(mrr)
            query_result["metrics"][f"mrr@{k}"] = mrr
            
            # NDCG@k (간단 버전: 첫 번째 정답만 고려)
            ndcg = 1.0 / np.log2(first_rank + 1) if first_rank else 0.0
            all_ndcg[k].append(ndcg)
            query_result["metrics"][f"ndcg@{k}"] = ndcg
        
        # 실패 케이스 체크 (상위 max_k에 정답이 없는 경우)
        if not (set(top_ids[:max_k]) & positive_ids):
            result.failed_queries.append({
                "query_id": query_id,
                "query_text": query_text[:200] + ("..." if len(query_text) > 200 else ""),
                "positive_passages": list(positive_ids)[:5],  # 처음 5개만
                "source": detect_query_source(query_meta),
                "top_5_retrieved": top_ids[:5],
            })
        
        # 소스별 통계 수집
        source = detect_query_source(query_meta)
        result.source_stats[source]["count"] += 1
        for k in k_values:
            result.source_stats[source]["mrr"].append(query_result["metrics"][f"mrr@{k}"])
            result.source_stats[source]["ndcg"].append(query_result["metrics"][f"ndcg@{k}"])
            result.source_stats[source]["recall"].append(query_result["metrics"][f"recall@{k}"])
        
        # 길이별 통계 수집
        query_len_bucket = _get_length_bucket(query_result["query_length_tokens"])
        for k in k_values:
            result.length_stats[f"query_{query_len_bucket}"]["mrr"].append(query_result["metrics"][f"mrr@{k}"])
            result.length_stats[f"query_{query_len_bucket}"]["ndcg"].append(query_result["metrics"][f"ndcg@{k}"])
            result.length_stats[f"query_{query_len_bucket}"]["recall"].append(query_result["metrics"][f"recall@{k}"])
        
        result.query_results.append(query_result)
    
    # 전체 평균 메트릭 계산
    for k in k_values:
        result.metrics[f"MRR@{k}"] = statistics.mean(all_mrr[k]) if all_mrr[k] else 0.0
        result.metrics[f"NDCG@{k}"] = statistics.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        result.metrics[f"Recall@{k}"] = statistics.mean(all_recall[k]) if all_recall[k] else 0.0
        result.metrics[f"Precision@{k}"] = statistics.mean(all_precision[k]) if all_precision[k] else 0.0
    
    return result


def _get_length_bucket(token_count: int) -> str:
    """토큰 수를 구간으로 분류"""
    if token_count < 10:
        return "very_short"
    elif token_count < 20:
        return "short"
    elif token_count < 50:
        return "medium"
    elif token_count < 100:
        return "long"
    else:
        return "very_long"


def print_detailed_report(
    result: DetailedEvaluationResult,
    output_file: Optional[str] = None,
    k_values: List[int] = [1, 3, 5, 10],
):
    """상세 평가 리포트 출력"""
    lines = []
    
    def add_line(s: str = ""):
        lines.append(s)
    
    add_line("=" * 80)
    add_line("LexDPR 상세 평가 리포트")
    add_line("=" * 80)
    add_line()
    
    # 기본 메트릭
    add_line("📊 전체 평균 메트릭")
    add_line("-" * 80)
    for k in k_values:
        add_line(f"  k={k}:")
        add_line(f"    MRR@{k}:      {result.metrics.get(f'MRR@{k}', 0.0):.4f}")
        add_line(f"    NDCG@{k}:     {result.metrics.get(f'NDCG@{k}', 0.0):.4f}")
        add_line(f"    Recall@{k}:   {result.metrics.get(f'Recall@{k}', 0.0):.4f}")
        add_line(f"    Precision@{k}: {result.metrics.get(f'Precision@{k}', 0.0):.4f}")
    add_line()
    
    # 소스별 통계
    add_line("📚 소스별 성능 분석")
    add_line("-" * 80)
    for source in sorted(result.source_stats.keys()):
        stats = result.source_stats[source]
        count = stats["count"]
        if count == 0:
            continue
        
        add_line(f"\n  [{source}] (총 {count}개 쿼리)")
        if stats["mrr"]:
            avg_mrr = statistics.mean(stats["mrr"])
            add_line(f"    평균 MRR:  {avg_mrr:.4f}")
        if stats["ndcg"]:
            avg_ndcg = statistics.mean(stats["ndcg"])
            add_line(f"    평균 NDCG: {avg_ndcg:.4f}")
        if stats["recall"]:
            avg_recall = statistics.mean(stats["recall"])
            add_line(f"    평균 Recall: {avg_recall:.4f}")
    add_line()
    
    # 실패 케이스
    add_line("❌ 실패 케이스 분석")
    add_line("-" * 80)
    add_line(f"  상위 {max(k_values)}개에 정답이 없는 쿼리: {len(result.failed_queries)}개")
    if result.failed_queries:
        add_line(f"\n  상위 10개 실패 케이스:")
        for i, failed in enumerate(result.failed_queries[:10], 1):
            add_line(f"\n  [{i}] {failed['query_id']} ({failed['source']})")
            add_line(f"      질의: {failed['query_text']}")
            add_line(f"      예상 정답: {', '.join(failed['positive_passages'][:3])}")
            add_line(f"      상위 5개 검색 결과: {', '.join(failed['top_5_retrieved'][:5])}")
    add_line()
    
    # 길이별 통계
    add_line("📏 쿼리 길이별 성능 분석")
    add_line("-" * 80)
    length_order = ["very_short", "short", "medium", "long", "very_long"]
    for bucket in length_order:
        if bucket not in result.length_stats or not result.length_stats[bucket]["mrr"]:
            continue
        
        stats = result.length_stats[bucket]
        bucket_name = {
            "very_short": "매우 짧음 (<10 토큰)",
            "short": "짧음 (10-19 토큰)",
            "medium": "중간 (20-49 토큰)",
            "long": "김 (50-99 토큰)",
            "very_long": "매우 김 (≥100 토큰)",
        }.get(bucket, bucket)
        
        add_line(f"\n  [{bucket_name}]")
        if stats["mrr"]:
            avg_mrr = statistics.mean(stats["mrr"])
            add_line(f"    평균 MRR:  {avg_mrr:.4f}")
        if stats["recall"]:
            avg_recall = statistics.mean(stats["recall"])
            add_line(f"    평균 Recall: {avg_recall:.4f}")
    add_line()
    
    # 쿼리별 성능 분포
    add_line("📈 쿼리별 성능 분포")
    add_line("-" * 80)
    if result.query_results:
        mrr_values = []
        recall_values = []
        for qr in result.query_results:
            mrr_values.append(qr["metrics"].get(f"mrr@{max(k_values)}", 0.0))
            recall_values.append(qr["metrics"].get(f"recall@{max(k_values)}", 0.0))
        
        if mrr_values:
            add_line(f"  MRR@{max(k_values)} 분포:")
            add_line(f"    최소: {min(mrr_values):.4f}")
            add_line(f"    최대: {max(mrr_values):.4f}")
            add_line(f"    평균: {statistics.mean(mrr_values):.4f}")
            add_line(f"    중앙값: {statistics.median(mrr_values):.4f}")
            add_line(f"    표준편차: {statistics.stdev(mrr_values) if len(mrr_values) > 1 else 0.0:.4f}")
        
        if recall_values:
            add_line(f"\n  Recall@{max(k_values)} 분포:")
            add_line(f"    최소: {min(recall_values):.4f}")
            add_line(f"    최대: {max(recall_values):.4f}")
            add_line(f"    평균: {statistics.mean(recall_values):.4f}")
            add_line(f"    중앙값: {statistics.median(recall_values):.4f}")
            add_line(f"    표준편차: {statistics.stdev(recall_values) if len(recall_values) > 1 else 0.0:.4f}")
    add_line()
    
    add_line("=" * 80)
    
    # 출력
    report_text = "\n".join(lines)
    print(report_text)
    
    # 파일 저장
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"\n✅ 리포트 저장: {output_path}")


def compare_models(
    model_paths: List[str],
    passages: Dict[str, Dict],
    eval_pairs_path: str,
    k_values: List[int] = [1, 3, 5, 10],
    template: TemplateMode = TemplateMode.BGE,
    output_file: Optional[str] = None,
    batch_size: int = 16,
) -> Dict:
    """
    여러 모델의 성능을 비교
    
    Args:
        model_paths: 평가할 모델 경로 목록
        passages: Passage 딕셔너리
        eval_pairs_path: 평가용 쌍 JSONL 경로
        k_values: 평가할 k 값 목록
        template: 템플릿 모드
        output_file: 비교 리포트 저장 경로
    
    Returns:
        비교 결과 딕셔너리
    """
    comparison_results = []
    
    for model_path in model_paths:
        print(f"\n[비교 평가] 모델 평가 중: {model_path}")
        model = SentenceTransformer(model_path)
        try:
            result = evaluate_detailed(
                model=model,
                passages=passages,
                eval_pairs_path=eval_pairs_path,
                k_values=k_values,
                template=template,
                batch_size=batch_size,
            )
            
            comparison_results.append({
                "model_path": model_path,
                "metrics": result.metrics,
                "source_stats": dict(result.source_stats),
                "failed_count": len(result.failed_queries),
            })
        finally:
            # 메모리 정리
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    # 비교 리포트 생성
    lines = []
    lines.append("=" * 80)
    lines.append("모델 성능 비교 리포트")
    lines.append("=" * 80)
    lines.append("")
    
    # 메트릭별 비교 테이블
    for k in k_values:
        lines.append(f"📊 k={k} 메트릭 비교")
        lines.append("-" * 80)
        lines.append(f"{'모델':<50} {'MRR@{k}':<12} {'NDCG@{k}':<12} {'Recall@{k}':<12} {'Precision@{k}':<12}")
        lines.append("-" * 80)
        
        for comp in comparison_results:
            model_name = Path(comp["model_path"]).name
            mrr = comp["metrics"].get(f"MRR@{k}", 0.0)
            ndcg = comp["metrics"].get(f"NDCG@{k}", 0.0)
            recall = comp["metrics"].get(f"Recall@{k}", 0.0)
            precision = comp["metrics"].get(f"Precision@{k}", 0.0)
            lines.append(f"{model_name:<50} {mrr:<12.4f} {ndcg:<12.4f} {recall:<12.4f} {precision:<12.4f}")
        
        # 최고 성능 모델 표시
        best_mrr = max(comp["metrics"].get(f"MRR@{k}", 0.0) for comp in comparison_results)
        best_model = next(
            comp["model_path"] for comp in comparison_results
            if comp["metrics"].get(f"MRR@{k}", 0.0) == best_mrr
        )
        lines.append(f"\n  최고 MRR@{k}: {best_mrr:.4f} ({Path(best_model).name})")
        lines.append("")
    
    # 소스별 비교
    lines.append("📚 소스별 성능 비교")
    lines.append("-" * 80)
    sources = set()
    for comp in comparison_results:
        sources.update(comp["source_stats"].keys())
    
    for source in sorted(sources):
        lines.append(f"\n  [{source}]")
        lines.append(f"{'모델':<50} {'평균 MRR':<15} {'평균 Recall':<15}")
        lines.append("-" * 80)
        
        for comp in comparison_results:
            model_name = Path(comp["model_path"]).name
            source_stat = comp["source_stats"].get(source, {})
            if source_stat.get("mrr"):
                avg_mrr = statistics.mean(source_stat["mrr"])
                avg_recall = statistics.mean(source_stat["recall"]) if source_stat.get("recall") else 0.0
                lines.append(f"{model_name:<50} {avg_mrr:<15.4f} {avg_recall:<15.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    print(report_text)
    
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"\n✅ 비교 리포트 저장: {output_path}")
    
    return {
        "comparison_results": comparison_results,
        "k_values": k_values,
    }

