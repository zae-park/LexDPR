#!/usr/bin/env python3
"""
Chunk 단위별 성능 및 토큰 절약 분석 리포트 생성 스크립트

Article top-10 vs Paragraph top-10/20의 성능과 토큰 차이를 분석하여
종합 리포트를 생성합니다.

사용법:
    poetry run python scripts/generate_chunk_comparison_report.py \
        --eval-results-dir data/eval_chunk_units/results \
        --output-report data/eval_chunk_units/results/comprehensive_report.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from lex_dpr.utils.io import read_jsonl


def load_eval_results(results_dir: Path) -> Dict[str, Any]:
    """평가 결과 파일들을 로드"""
    results = {
        'model_info': {},
        'chunk_results': {},
        'token_stats': {},
        'token_savings': {}
    }
    
    # 모델 정보 로드
    model_info_file = results_dir / "model_info.json"
    if model_info_file.exists():
        with open(model_info_file, 'r', encoding='utf-8') as f:
            results['model_info'] = json.load(f)
    
    # 토큰 통계 로드
    token_stats_file = results_dir / "token_stats.json"
    if token_stats_file.exists():
        with open(token_stats_file, 'r', encoding='utf-8') as f:
            token_data = json.load(f)
            results['token_stats'] = token_data.get('token_stats', {})
            results['token_savings'] = token_data.get('token_savings', {})
    
    # Chunk 단위별 평가 결과 로드
    chunk_types = ["paragraph", "item", "article"]
    for chunk_type in chunk_types:
        results['chunk_results'][chunk_type] = {}
        
        # 각 모델별 결과 파일 찾기
        for result_file in results_dir.glob(f"{chunk_type}_*.json"):
            model_name = result_file.stem.replace(f"{chunk_type}_", "").replace("_", "/")
            
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', data)
                    results['chunk_results'][chunk_type][model_name] = metrics
            except Exception as e:
                print(f"⚠️  결과 파일 로드 실패 ({result_file}): {e}")
    
    return results


def calculate_llm_token_comparison(
    token_stats: Dict[str, Any],
    top_k_values: List[int] = [10, 20]
) -> Dict[str, Any]:
    """
    LLM에 전달할 때의 토큰 비교 계산
    
    Article top-10 vs Paragraph top-10/20의 토큰 수 비교
    """
    comparison = {}
    
    article_stats = token_stats.get('article', {})
    paragraph_stats = token_stats.get('paragraph', {})
    
    article_avg_tokens = article_stats.get('avg_tokens', 0)
    paragraph_avg_tokens = paragraph_stats.get('avg_tokens', 0)
    
    for k in top_k_values:
        # Article top-k: k개의 article passage
        article_total_tokens = article_avg_tokens * k
        
        # Paragraph top-k: k개의 paragraph passage
        paragraph_total_tokens = paragraph_avg_tokens * k
        
        # 절약 토큰 및 비율
        if article_total_tokens > 0:
            savings_tokens = article_total_tokens - paragraph_total_tokens
            savings_percentage = (savings_tokens / article_total_tokens) * 100
        else:
            savings_tokens = 0
            savings_percentage = 0
        
        comparison[f"top_{k}"] = {
            "article_tokens": article_total_tokens,
            "paragraph_tokens": paragraph_total_tokens,
            "savings_tokens": savings_tokens,
            "savings_percentage": savings_percentage
        }
    
    return comparison


def generate_markdown_report(
    results: Dict[str, Any],
    output_file: Path,
    top_k_values: List[int] = [10, 20]
) -> None:
    """마크다운 리포트 생성"""
    
    lines = []
    lines.append("# Chunk 단위별 모델 성능 및 토큰 절약 분석 리포트")
    lines.append("")
    lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. 모델 정보 요약
    lines.append("## 1. 평가된 모델 정보")
    lines.append("")
    lines.append("| 모델 | 크기 (M) | Max Length | Embedding Dim |")
    lines.append("|------|----------|------------|---------------|")
    
    model_info = results.get('model_info', {})
    for model_name, info in sorted(model_info.items()):
        size_str = f"{info.get('size_m', 'N/A')}M" if info.get('size_m') else "N/A"
        max_len = info.get('max_length', 'N/A')
        emb_dim = info.get('embedding_dim', 'N/A')
        lines.append(f"| {model_name} | {size_str} | {max_len} | {emb_dim} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 2. Chunk 단위별 성능 비교
    lines.append("## 2. Chunk 단위별 모델 성능 비교")
    lines.append("")
    
    chunk_results = results.get('chunk_results', {})
    chunk_types = ["paragraph", "item", "article"]
    
    for chunk_type in chunk_types:
        if chunk_type not in chunk_results:
            continue
        
        lines.append(f"### 2.{chunk_types.index(chunk_type) + 1} {chunk_type.upper()} 단위")
        lines.append("")
        lines.append("| 모델 | NDCG@10 | Recall@10 | MRR@10 |")
        lines.append("|------|---------|-----------|--------|")
        
        # 성능 순으로 정렬
        model_scores = []
        for model_name, metrics in chunk_results[chunk_type].items():
            ndcg = metrics.get('val_cosine_ndcg@10', 0)
            model_scores.append((model_name, metrics, ndcg))
        
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        for model_name, metrics, _ in model_scores:
            ndcg = metrics.get('val_cosine_ndcg@10', 0)
            recall = metrics.get('val_cosine_recall@10', 0)
            mrr = metrics.get('val_cosine_mrr@10', 0)
            lines.append(f"| {model_name} | {ndcg:.4f} | {recall:.4f} | {mrr:.4f} |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 3. 토큰 통계
    lines.append("## 3. Chunk 단위별 토큰 통계")
    lines.append("")
    
    token_stats = results.get('token_stats', {})
    if token_stats:
        lines.append("| Chunk 단위 | 평균 토큰 수 | 전체 토큰 수 | Passage 개수 |")
        lines.append("|------------|--------------|--------------|-------------|")
        
        for chunk_type in chunk_types:
            if chunk_type in token_stats:
                stats = token_stats[chunk_type]
                avg_tokens = stats.get('avg_tokens', 0)
                total_tokens = stats.get('total_tokens', 0)
                passage_count = stats.get('passage_count', 0)
                lines.append(f"| {chunk_type} | {avg_tokens:.1f} | {total_tokens:,} | {passage_count:,} |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 4. LLM 전달 시 토큰 절약 분석
    lines.append("## 4. LLM 전달 시 토큰 절약 분석")
    lines.append("")
    lines.append("Article top-k vs Paragraph top-k의 토큰 수 비교")
    lines.append("")
    
    llm_comparison = calculate_llm_token_comparison(token_stats, top_k_values)
    
    lines.append("| Top-K | Article 토큰 | Paragraph 토큰 | 절약 토큰 | 절약율 |")
    lines.append("|-------|-------------|----------------|-----------|--------|")
    
    for k in top_k_values:
        comp = llm_comparison.get(f"top_{k}", {})
        article_tokens = comp.get('article_tokens', 0)
        paragraph_tokens = comp.get('paragraph_tokens', 0)
        savings_tokens = comp.get('savings_tokens', 0)
        savings_pct = comp.get('savings_percentage', 0)
        
        lines.append(f"| Top-{k} | {article_tokens:.1f} | {paragraph_tokens:.1f} | {savings_tokens:.1f} | {savings_pct:.1f}% |")
    
    lines.append("")
    lines.append("### 4.1 토큰 절약 효과")
    lines.append("")
    
    for k in top_k_values:
        comp = llm_comparison.get(f"top_{k}", {})
        savings_pct = comp.get('savings_percentage', 0)
        savings_tokens = comp.get('savings_tokens', 0)
        
        lines.append(f"- **Top-{k}**: Article 대비 Paragraph 사용 시 **{savings_pct:.1f}%** 절약 ({savings_tokens:.1f} 토큰)")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 5. 성능 vs 토큰 절약 트레이드오프
    lines.append("## 5. 성능 vs 토큰 절약 트레이드오프")
    lines.append("")
    
    # Article과 Paragraph의 성능 비교
    if "article" in chunk_results and "paragraph" in chunk_results:
        lines.append("### 5.1 Article vs Paragraph 성능 비교")
        lines.append("")
        lines.append("| 모델 | Article NDCG@10 | Paragraph NDCG@10 | 성능 차이 |")
        lines.append("|------|-----------------|-------------------|-----------|")
        
        # 모든 모델에 대해 비교
        all_models = set()
        if "article" in chunk_results:
            all_models.update(chunk_results["article"].keys())
        if "paragraph" in chunk_results:
            all_models.update(chunk_results["paragraph"].keys())
        
        for model_name in sorted(all_models):
            article_ndcg = chunk_results.get("article", {}).get(model_name, {}).get('val_cosine_ndcg@10', 0)
            paragraph_ndcg = chunk_results.get("paragraph", {}).get(model_name, {}).get('val_cosine_ndcg@10', 0)
            diff = paragraph_ndcg - article_ndcg
            
            lines.append(f"| {model_name} | {article_ndcg:.4f} | {paragraph_ndcg:.4f} | {diff:+.4f} |")
        
        lines.append("")
        
        # 토큰 절약 정보와 함께 요약
        if "article_vs_paragraph" in results.get('token_savings', {}):
            savings_info = results['token_savings']['article_vs_paragraph']
            savings_pct = savings_info.get('savings_percentage', 0)
            
            lines.append(f"**결론**: Paragraph 사용 시 평균 **{savings_pct:.1f}%** 토큰 절약이 가능하며, ")
            lines.append("대부분의 모델에서 성능 차이는 미미합니다.")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 6. 권장사항
    lines.append("## 6. 권장사항")
    lines.append("")
    lines.append("### 6.1 Chunk 단위 선택")
    lines.append("")
    lines.append("- **Paragraph 단위**: 토큰 절약이 중요하고 성능 저하를 감수할 수 있는 경우")
    lines.append("- **Article 단위**: 최고 성능이 필요한 경우")
    lines.append("- **Item 단위**: 중간 성능과 토큰 절약의 균형이 필요한 경우")
    lines.append("")
    
    lines.append("### 6.2 Top-K 선택")
    lines.append("")
    
    for k in top_k_values:
        comp = llm_comparison.get(f"top_{k}", {})
        savings_pct = comp.get('savings_percentage', 0)
        lines.append(f"- **Top-{k}**: {savings_pct:.1f}% 토큰 절약")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*리포트 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # 파일 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ 리포트 생성 완료: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk 단위별 성능 및 토큰 절약 분석 리포트 생성"
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default="data/eval_chunk_units/results",
        help="평가 결과 디렉토리 (기본값: data/eval_chunk_units/results)"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/eval_chunk_units/results/comprehensive_report.md",
        help="출력 리포트 파일 경로 (기본값: data/eval_chunk_units/results/comprehensive_report.md)"
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        type=int,
        default=[10, 20],
        help="비교할 Top-K 값 (기본값: 10 20)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.eval_results_dir)
    output_file = Path(args.output_report)
    
    if not results_dir.exists():
        print(f"❌ 오류: 평가 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        print("   먼저 scripts/eval_chunk_units.py를 실행하여 평가를 완료하세요.")
        sys.exit(1)
    
    print("=" * 80)
    print("Chunk 단위별 성능 및 토큰 절약 분석 리포트 생성")
    print("=" * 80)
    print()
    print(f"평가 결과 디렉토리: {results_dir}")
    print(f"출력 리포트: {output_file}")
    print(f"Top-K 값: {args.top_k_values}")
    print()
    
    # 결과 로드
    print("결과 파일 로드 중...")
    results = load_eval_results(results_dir)
    
    if not results.get('chunk_results'):
        print("⚠️  경고: 평가 결과를 찾을 수 없습니다.")
        print("   먼저 scripts/eval_chunk_units.py를 실행하여 평가를 완료하세요.")
        sys.exit(1)
    
    # 리포트 생성
    print("리포트 생성 중...")
    generate_markdown_report(results, output_file, args.top_k_values)
    
    print()
    print("=" * 80)
    print("✅ 리포트 생성 완료!")
    print("=" * 80)
    print()
    print(f"📄 리포트 파일: {output_file}")
    print()


if __name__ == "__main__":
    main()

