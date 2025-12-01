#!/usr/bin/env python3
"""
데이터 품질 분석 스크립트: train/valid/test 데이터셋의 통계 및 분포 분석

분석 항목:
- 데이터셋 크기 (train/valid/test)
- Positive/Negative 비율 및 분포
- 쿼리 타입별 분포 (law, admin, prec)
- 질의(query) 토큰 길이 분포
- Passage 토큰 길이 분포 (positive passages)
- 기타 통계 정보
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[warn] transformers가 설치되지 않았습니다. 토큰 길이는 문자 수로 대체됩니다.")


def count_tokens(text: str, tokenizer: Optional[object] = None) -> int:
    """텍스트의 토큰 수 계산"""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # 폴백: 공백 기준 단어 수 (대략적인 추정)
    return len(text.split())


def analyze_dataset(
    pairs_path: str,
    passages_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    dataset_name: str = "dataset",
) -> Dict:
    """단일 데이터셋 분석"""
    from lex_dpr.utils.io import read_jsonl
    
    pairs = list(read_jsonl(pairs_path))
    if not pairs:
        return {"error": f"데이터셋이 비어있습니다: {pairs_path}"}
    
    # Passage 로드 (토큰 길이 계산용)
    passages = {}
    if passages_path:
        try:
            passages = {row["id"]: row for row in read_jsonl(passages_path)}
        except Exception as e:
            print(f"[warn] Passage 파일 로드 실패: {e}")
    
    # 토크나이저 초기화
    tokenizer = None
    if HAS_TRANSFORMERS and tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"[warn] 토크나이저 로드 실패 ({tokenizer_name}): {e}")
    
    # 기본 통계
    total_queries = len(pairs)
    total_positives = sum(len(p.get("positive_passages", [])) for p in pairs)
    total_negatives = sum(len(p.get("hard_negatives", [])) for p in pairs)
    
    # Positive/Negative 분포
    pos_counts = [len(p.get("positive_passages", [])) for p in pairs]
    neg_counts = [len(p.get("hard_negatives", [])) for p in pairs]
    
    # 쿼리 타입별 분포
    type_counter = Counter()
    for p in pairs:
        meta = p.get("meta", {})
        query_type = meta.get("type", "unknown")
        type_counter[query_type] += 1
    
    # 질의 토큰 길이 분포
    query_lengths = []
    for p in pairs:
        query_text = p.get("query_text", "")
        if query_text:
            query_lengths.append(count_tokens(query_text, tokenizer))
    
    # Passage 토큰 길이 분포 (positive passages만)
    passage_lengths = []
    passage_ids_seen = set()
    for p in pairs:
        for pid in p.get("positive_passages", []):
            if pid in passage_ids_seen:
                continue  # 중복 제거
            passage_ids_seen.add(pid)
            if pid in passages:
                passage_text = passages[pid].get("text", "")
                if passage_text:
                    passage_lengths.append(count_tokens(passage_text, tokenizer))
    
    # 통계 계산
    def calc_stats(values: List[float]) -> Dict:
        if not values:
            return {}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p25": statistics.quantiles(values, n=4)[0] if len(values) > 1 else values[0],
            "p75": statistics.quantiles(values, n=4)[2] if len(values) > 1 else values[0],
        }
    
    return {
        "dataset_name": dataset_name,
        "file_path": pairs_path,
        "basic_stats": {
            "total_queries": total_queries,
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "avg_positives_per_query": total_positives / total_queries if total_queries > 0 else 0,
            "avg_negatives_per_query": total_negatives / total_queries if total_queries > 0 else 0,
            "pos_neg_ratio": total_positives / total_negatives if total_negatives > 0 else float("inf"),
        },
        "positive_distribution": calc_stats(pos_counts),
        "negative_distribution": calc_stats(neg_counts),
        "query_type_distribution": dict(type_counter),
        "query_length_stats": calc_stats(query_lengths),
        "passage_length_stats": calc_stats(passage_lengths) if passage_lengths else {},
        "sample_queries": [
            {
                "query_id": p.get("query_id", "N/A"),
                "query_text": p.get("query_text", "")[:100] + ("..." if len(p.get("query_text", "")) > 100 else ""),
                "num_positives": len(p.get("positive_passages", [])),
                "num_negatives": len(p.get("hard_negatives", [])),
                "type": p.get("meta", {}).get("type", "unknown"),
            }
            for p in pairs[:5]  # 처음 5개 샘플
        ],
    }


def print_analysis_report(results: Dict, output_file: Optional[str] = None):
    """분석 결과를 포맷팅하여 출력"""
    lines = []
    
    def add_section(title: str):
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  {title}")
        lines.append("=" * 80)
    
    def add_subsection(title: str):
        lines.append("")
        lines.append(f"--- {title} ---")
    
    def add_stat(name: str, value, fmt: str = "{}"):
        if isinstance(value, float):
            lines.append(f"  {name:40s}: {fmt.format(value)}")
        else:
            lines.append(f"  {name:40s}: {fmt.format(value)}")
    
    # 전체 요약
    add_section("데이터 품질 분석 리포트")
    lines.append(f"생성 시간: {Path(__file__).stat().st_mtime}")
    
    # 각 데이터셋별 분석
    for dataset_name in ["train", "valid", "test"]:
        if dataset_name not in results:
            continue
        
        result = results[dataset_name]
        if "error" in result:
            add_section(f"{dataset_name.upper()} 데이터셋")
            lines.append(f"  오류: {result['error']}")
            continue
        
        add_section(f"{dataset_name.upper()} 데이터셋")
        add_subsection("기본 통계")
        add_stat("총 질의 수", result["basic_stats"]["total_queries"], "{:,}")
        add_stat("총 Positive 개수", result["basic_stats"]["total_positives"], "{:,}")
        add_stat("총 Negative 개수", result["basic_stats"]["total_negatives"], "{:,}")
        add_stat("질의당 평균 Positive", result["basic_stats"]["avg_positives_per_query"], "{:.2f}")
        add_stat("질의당 평균 Negative", result["basic_stats"]["avg_negatives_per_query"], "{:.2f}")
        add_stat("Positive/Negative 비율", result["basic_stats"]["pos_neg_ratio"], "{:.3f}")
        
        add_subsection("Positive 분포")
        if result["positive_distribution"]:
            pos_dist = result["positive_distribution"]
            add_stat("  최소값", pos_dist["min"], "{:.0f}")
            add_stat("  최대값", pos_dist["max"], "{:.0f}")
            add_stat("  평균", pos_dist["mean"], "{:.2f}")
            add_stat("  중앙값", pos_dist["median"], "{:.2f}")
            add_stat("  표준편차", pos_dist["stdev"], "{:.2f}")
            add_stat("  25% 분위수", pos_dist["p25"], "{:.2f}")
            add_stat("  75% 분위수", pos_dist["p75"], "{:.2f}")
        
        add_subsection("Negative 분포")
        if result["negative_distribution"]:
            neg_dist = result["negative_distribution"]
            add_stat("  최소값", neg_dist["min"], "{:.0f}")
            add_stat("  최대값", neg_dist["max"], "{:.0f}")
            add_stat("  평균", neg_dist["mean"], "{:.2f}")
            add_stat("  중앙값", neg_dist["median"], "{:.2f}")
            add_stat("  표준편차", neg_dist["stdev"], "{:.2f}")
            add_stat("  25% 분위수", neg_dist["p25"], "{:.2f}")
            add_stat("  75% 분위수", neg_dist["p75"], "{:.2f}")
        
        add_subsection("쿼리 타입별 분포")
        type_dist = result["query_type_distribution"]
        total = sum(type_dist.values())
        for qtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            add_stat(f"  {qtype}", f"{count:,} ({pct:.1f}%)", "{}")
        
        add_subsection("질의 토큰 길이 분포")
        if result["query_length_stats"]:
            qlen = result["query_length_stats"]
            add_stat("  최소값", qlen["min"], "{:.0f}")
            add_stat("  최대값", qlen["max"], "{:.0f}")
            add_stat("  평균", qlen["mean"], "{:.1f}")
            add_stat("  중앙값", qlen["median"], "{:.1f}")
            add_stat("  표준편차", qlen["stdev"], "{:.1f}")
            add_stat("  25% 분위수", qlen["p25"], "{:.1f}")
            add_stat("  75% 분위수", qlen["p75"], "{:.1f}")
        
        add_subsection("Passage 토큰 길이 분포 (Positive)")
        if result["passage_length_stats"]:
            plen = result["passage_length_stats"]
            add_stat("  최소값", plen["min"], "{:.0f}")
            add_stat("  최대값", plen["max"], "{:.0f}")
            add_stat("  평균", plen["mean"], "{:.1f}")
            add_stat("  중앙값", plen["median"], "{:.1f}")
            add_stat("  표준편차", plen["stdev"], "{:.1f}")
            add_stat("  25% 분위수", plen["p25"], "{:.1f}")
            add_stat("  75% 분위수", plen["p75"], "{:.1f}")
        else:
            lines.append("  Passage 정보를 로드할 수 없습니다.")
        
        add_subsection("샘플 질의 (처음 5개)")
        for i, sample in enumerate(result["sample_queries"], 1):
            lines.append(f"  [{i}] {sample['query_id']} ({sample['type']})")
            lines.append(f"      질의: {sample['query_text']}")
            lines.append(f"      Positive: {sample['num_positives']}, Negative: {sample['num_negatives']}")
    
    # 전체 요약 비교
    if all(d in results and "error" not in results[d] for d in ["train", "valid", "test"]):
        add_section("전체 데이터셋 비교")
        lines.append(f"{'데이터셋':<15} {'질의 수':>12} {'Positive':>12} {'Negative':>12} {'P/N 비율':>12}")
        lines.append("-" * 80)
        for dataset_name in ["train", "valid", "test"]:
            if dataset_name in results:
                r = results[dataset_name]
                stats = r["basic_stats"]
                lines.append(
                    f"{dataset_name:<15} {stats['total_queries']:>12,} "
                    f"{stats['total_positives']:>12,} {stats['total_negatives']:>12,} "
                    f"{stats['pos_neg_ratio']:>12.3f}"
                )
        
        # 전체 합계
        total_q = sum(results[d]["basic_stats"]["total_queries"] for d in ["train", "valid", "test"] if d in results)
        total_p = sum(results[d]["basic_stats"]["total_positives"] for d in ["train", "valid", "test"] if d in results)
        total_n = sum(results[d]["basic_stats"]["total_negatives"] for d in ["train", "valid", "test"] if d in results)
        total_ratio = total_p / total_n if total_n > 0 else float("inf")
        lines.append("-" * 80)
        lines.append(f"{'합계':<15} {total_q:>12,} {total_p:>12,} {total_n:>12,} {total_ratio:>12.3f}")
    
    # 출력
    report_text = "\n".join(lines)
    print(report_text)
    
    # 파일 저장
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n리포트가 저장되었습니다: {output_file}")
        
        # JSON 형식으로도 저장
        json_path = Path(output_file).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON 리포트가 저장되었습니다: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="데이터 품질 분석 스크립트: train/valid/test 데이터셋 통계 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 분석 (train/valid/test 자동 감지)
  python scripts/analyze_pairs.py --pairs-dir data

  # 특정 파일 지정
  python scripts/analyze_pairs.py \\
    --train data/pairs_train.jsonl \\
    --valid data/pairs_train_valid.jsonl \\
    --test data/pairs_train_test.jsonl \\
    --passages data/processed/merged_corpus.jsonl \\
    --tokenizer BAAI/bge-m3 \\
    --output analysis_report.txt
        """,
    )
    
    parser.add_argument(
        "--pairs-dir",
        type=str,
        help="pairs 파일들이 있는 디렉토리 (자동으로 train/valid/test 파일 찾기)",
    )
    parser.add_argument(
        "--train",
        type=str,
        help="Train 데이터셋 경로 (pairs_train.jsonl)",
    )
    parser.add_argument(
        "--valid",
        type=str,
        help="Valid 데이터셋 경로 (pairs_train_valid.jsonl)",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test 데이터셋 경로 (pairs_train_test.jsonl)",
    )
    parser.add_argument(
        "--passages",
        type=str,
        help="Passage 코퍼스 경로 (merged_corpus.jsonl) - 토큰 길이 계산용",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BAAI/bge-m3",
        help="토크나이저 모델 이름 (기본값: BAAI/bge-m3). None이면 단어 수로 계산",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="분석 리포트 출력 파일 경로 (텍스트 + JSON)",
    )
    
    args = parser.parse_args()
    
    # 파일 경로 결정
    train_path = None
    valid_path = None
    test_path = None
    
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        train_path = pairs_dir / "pairs_train.jsonl"
        valid_path = pairs_dir / "pairs_train_valid.jsonl"
        test_path = pairs_dir / "pairs_train_test.jsonl"
        
        # 파일 존재 확인
        if not train_path.exists():
            train_path = None
        if not valid_path.exists():
            valid_path = None
        if not test_path.exists():
            test_path = None
    else:
        train_path = args.train
        valid_path = args.valid
        test_path = args.test
    
    if not any([train_path, valid_path, test_path]):
        parser.error("분석할 데이터셋 파일을 찾을 수 없습니다. --pairs-dir 또는 --train/--valid/--test를 지정하세요.")
    
    # 토크나이저 설정
    tokenizer_name = args.tokenizer if args.tokenizer.lower() != "none" else None
    
    # 분석 실행
    results = {}
    
    if train_path and Path(train_path).exists():
        print(f"[분석 중] Train 데이터셋: {train_path}")
        results["train"] = analyze_dataset(
            str(train_path),
            passages_path=args.passages,
            tokenizer_name=tokenizer_name,
            dataset_name="train",
        )
    
    if valid_path and Path(valid_path).exists():
        print(f"[분석 중] Valid 데이터셋: {valid_path}")
        results["valid"] = analyze_dataset(
            str(valid_path),
            passages_path=args.passages,
            tokenizer_name=tokenizer_name,
            dataset_name="valid",
        )
    
    if test_path and Path(test_path).exists():
        print(f"[분석 중] Test 데이터셋: {test_path}")
        results["test"] = analyze_dataset(
            str(test_path),
            passages_path=args.passages,
            tokenizer_name=tokenizer_name,
            dataset_name="test",
        )
    
    if not results:
        parser.error("분석할 데이터가 없습니다.")
    
    # 리포트 출력
    print_analysis_report(results, output_file=args.output)


if __name__ == "__main__":
    main()

