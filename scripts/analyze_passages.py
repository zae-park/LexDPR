#!/usr/bin/env python3
"""
Passage Corpus 분석 스크립트: passage corpus의 품질 및 통계 분석

분석 항목:
- 총 passage 개수 및 소스별 분포
- 중복 passage 탐지 및 통계
- 길이 분포 분석 (문자 수, 토큰 수)
- 소스별(법령/행정규칙/판례) 통계
- 중복 제거 제안
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[warn] transformers가 설치되지 않았습니다. 토큰 길이는 문자 수로 대체됩니다.")


def normalize_text(text: str) -> str:
    """텍스트 정규화 (중복 탐지용)"""
    if not text:
        return ""
    # 공백 정규화 및 소문자 변환 (선택적)
    return " ".join(text.split())


def get_text_hash(text: str) -> str:
    """텍스트의 해시값 계산"""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def count_tokens(text: str, tokenizer: Optional[object] = None) -> int:
    """텍스트의 토큰 수 계산"""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # 폴백: 공백 기준 단어 수 (대략적인 추정)
    return len(text.split())


def detect_source_type(passage: Dict) -> str:
    """Passage의 소스 타입 감지"""
    pid = passage.get("id", "")
    ptype = passage.get("type", "")
    
    # ID 기반 감지
    if pid.startswith("LAW_"):
        return "법령"
    elif pid.startswith("ADMIN_"):
        return "행정규칙"
    elif pid.startswith("PREC_"):
        return "판례"
    
    # type 필드 기반 감지
    if ptype:
        type_map = {
            "법령": "법령",
            "행정규칙": "행정규칙",
            "판례": "판례",
            "law": "법령",
            "admin": "행정규칙",
            "prec": "판례",
        }
        return type_map.get(ptype, "기타")
    
    return "기타"


def analyze_passages(
    corpus_path: str,
    tokenizer_name: Optional[str] = None,
    min_text_length: int = 10,
) -> Dict:
    """Passage corpus 분석"""
    import sys
    from pathlib import Path
    # scripts 디렉토리에서 실행될 때를 대비해 경로 추가
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from lex_dpr.utils.io import read_jsonl
    
    passages = list(read_jsonl(corpus_path))
    if not passages:
        return {"error": f"Corpus가 비어있습니다: {corpus_path}"}
    
    # 토크나이저 초기화
    tokenizer = None
    if HAS_TRANSFORMERS and tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"[warn] 토크나이저 로드 실패 ({tokenizer_name}): {e}")
    
    # 기본 통계
    total_passages = len(passages)
    
    # 소스별 분포
    source_counter = Counter()
    source_passages = defaultdict(list)
    
    # 길이 통계
    char_lengths = []
    token_lengths = []
    
    # 중복 탐지
    text_hash_to_ids: Dict[str, List[str]] = defaultdict(list)
    text_hash_to_text: Dict[str, str] = {}
    duplicate_groups: List[Dict] = []
    
    # 빈 텍스트/짧은 텍스트 탐지
    empty_texts = 0
    short_texts = 0
    
    for passage in passages:
        pid = passage.get("id", "")
        text = passage.get("text", "").strip()
        
        # 소스 타입 감지
        source_type = detect_source_type(passage)
        source_counter[source_type] += 1
        source_passages[source_type].append(passage)
        
        # 빈 텍스트 체크
        if not text:
            empty_texts += 1
            continue
        
        # 짧은 텍스트 체크
        if len(text) < min_text_length:
            short_texts += 1
        
        # 길이 통계
        char_len = len(text)
        char_lengths.append(char_len)
        
        token_len = count_tokens(text, tokenizer)
        token_lengths.append(token_len)
        
        # 중복 탐지 (해시 기반)
        text_hash = get_text_hash(text)
        text_hash_to_ids[text_hash].append(pid)
        text_hash_to_text[text_hash] = text
    
    # 중복 그룹 생성
    for text_hash, ids in text_hash_to_ids.items():
        if len(ids) > 1:
            duplicate_groups.append({
                "text_hash": text_hash,
                "passage_ids": ids,
                "count": len(ids),
                "sample_text": text_hash_to_text[text_hash][:200] + ("..." if len(text_hash_to_text[text_hash]) > 200 else ""),
            })
    
    # 중복 통계
    total_duplicates = sum(len(ids) - 1 for ids in text_hash_to_ids.values() if len(ids) > 1)
    unique_passages = len(text_hash_to_ids)
    duplicate_ratio = total_duplicates / total_passages if total_passages > 0 else 0.0
    
    # 통계 계산 함수
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
    
    # 소스별 상세 통계
    source_stats = {}
    for source_type, source_passage_list in source_passages.items():
        source_char_lengths = []
        source_token_lengths = []
        
        for passage in source_passage_list:
            text = passage.get("text", "").strip()
            if text:
                source_char_lengths.append(len(text))
                source_token_lengths.append(count_tokens(text, tokenizer))
        
        source_stats[source_type] = {
            "count": len(source_passage_list),
            "char_length_stats": calc_stats(source_char_lengths),
            "token_length_stats": calc_stats(source_token_lengths),
        }
    
    # 중복 그룹 정렬 (가장 많은 중복부터)
    duplicate_groups.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "file_path": corpus_path,
        "basic_stats": {
            "total_passages": total_passages,
            "unique_passages": unique_passages,
            "duplicate_passages": total_duplicates,
            "duplicate_ratio": duplicate_ratio,
            "empty_texts": empty_texts,
            "short_texts": short_texts,
        },
        "source_distribution": dict(source_counter),
        "source_stats": source_stats,
        "char_length_stats": calc_stats(char_lengths),
        "token_length_stats": calc_stats(token_lengths),
        "duplicate_groups": duplicate_groups[:20],  # 상위 20개만 저장
        "duplicate_summary": {
            "total_groups": len(duplicate_groups),
            "max_duplicates_in_group": max((g["count"] for g in duplicate_groups), default=0),
            "avg_duplicates_per_group": statistics.mean([g["count"] for g in duplicate_groups]) if duplicate_groups else 0.0,
        },
    }


def print_analysis_report(results: Dict, output_file: Optional[str] = None):
    """분석 결과 리포트 출력"""
    lines = []
    
    def add_line(s: str = ""):
        lines.append(s)
    
    add_line("=" * 80)
    add_line("Passage Corpus 분석 리포트")
    add_line("=" * 80)
    add_line()
    
    if "error" in results:
        add_line(f"❌ 오류: {results['error']}")
        if output_file:
            Path(output_file).write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return
    
    # 기본 통계
    basic = results["basic_stats"]
    add_line("📊 기본 통계")
    add_line("-" * 80)
    add_line(f"총 Passage 개수: {basic['total_passages']:,}")
    add_line(f"고유 Passage 개수: {basic['unique_passages']:,}")
    add_line(f"중복 Passage 개수: {basic['duplicate_passages']:,}")
    add_line(f"중복 비율: {basic['duplicate_ratio']:.2%}")
    add_line(f"빈 텍스트: {basic['empty_texts']:,}")
    add_line(f"짧은 텍스트 (< {basic.get('min_text_length', 10)}자): {basic['short_texts']:,}")
    add_line()
    
    # 소스별 분포
    add_line("📚 소스별 분포")
    add_line("-" * 80)
    source_dist = results["source_distribution"]
    for source_type, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
        ratio = count / basic["total_passages"] * 100 if basic["total_passages"] > 0 else 0
        add_line(f"  {source_type}: {count:,} ({ratio:.1f}%)")
    add_line()
    
    # 길이 분포
    add_line("📏 길이 분포 (문자 수)")
    add_line("-" * 80)
    char_stats = results["char_length_stats"]
    if char_stats:
        add_line(f"  최소: {char_stats['min']:,}자")
        add_line(f"  최대: {char_stats['max']:,}자")
        add_line(f"  평균: {char_stats['mean']:.1f}자")
        add_line(f"  중앙값: {char_stats['median']:.1f}자")
        add_line(f"  표준편차: {char_stats['stdev']:.1f}자")
        add_line(f"  25% 백분위: {char_stats['p25']:.1f}자")
        add_line(f"  75% 백분위: {char_stats['p75']:.1f}자")
    add_line()
    
    add_line("📏 길이 분포 (토큰 수)")
    add_line("-" * 80)
    token_stats = results["token_length_stats"]
    if token_stats:
        add_line(f"  최소: {token_stats['min']:,}토큰")
        add_line(f"  최대: {token_stats['max']:,}토큰")
        add_line(f"  평균: {token_stats['mean']:.1f}토큰")
        add_line(f"  중앙값: {token_stats['median']:.1f}토큰")
        add_line(f"  표준편차: {token_stats['stdev']:.1f}토큰")
        add_line(f"  25% 백분위: {token_stats['p25']:.1f}토큰")
        add_line(f"  75% 백분위: {token_stats['p75']:.1f}토큰")
    add_line()
    
    # 소스별 상세 통계
    add_line("📊 소스별 상세 통계")
    add_line("-" * 80)
    source_stats = results["source_stats"]
    for source_type in sorted(source_stats.keys()):
        stats = source_stats[source_type]
        add_line(f"\n  [{source_type}]")
        add_line(f"    개수: {stats['count']:,}")
        if stats["char_length_stats"]:
            char_s = stats["char_length_stats"]
            add_line(f"    문자 길이: 평균 {char_s['mean']:.1f}자, 중앙값 {char_s['median']:.1f}자")
        if stats["token_length_stats"]:
            token_s = stats["token_length_stats"]
            add_line(f"    토큰 길이: 평균 {token_s['mean']:.1f}토큰, 중앙값 {token_s['median']:.1f}토큰")
    add_line()
    
    # 중복 요약
    dup_summary = results["duplicate_summary"]
    add_line("🔄 중복 Passage 요약")
    add_line("-" * 80)
    add_line(f"  중복 그룹 수: {dup_summary['total_groups']:,}")
    add_line(f"  그룹당 최대 중복 수: {dup_summary['max_duplicates_in_group']}")
    add_line(f"  그룹당 평균 중복 수: {dup_summary['avg_duplicates_per_group']:.2f}")
    add_line()
    
    # 상위 중복 그룹
    duplicate_groups = results.get("duplicate_groups", [])
    if duplicate_groups:
        add_line("🔍 상위 중복 그룹 (최대 10개)")
        add_line("-" * 80)
        for i, group in enumerate(duplicate_groups[:10], 1):
            add_line(f"\n  [{i}] {group['count']}개 중복")
            add_line(f"      Passage IDs: {', '.join(group['passage_ids'][:5])}{' ...' if len(group['passage_ids']) > 5 else ''}")
            add_line(f"      샘플 텍스트: {group['sample_text']}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Passage Corpus 품질 분석 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 분석
  python scripts/analyze_passages.py --corpus data/merged_corpus.jsonl

  # 토크나이저 지정 및 JSON 출력
  python scripts/analyze_passages.py \\
    --corpus data/merged_corpus.jsonl \\
    --tokenizer BAAI/bge-m3 \\
    --output report.txt \\
    --json-output report.json

  # 최소 텍스트 길이 설정
  python scripts/analyze_passages.py \\
    --corpus data/merged_corpus.jsonl \\
    --min-text-length 20
        """
    )
    
    parser.add_argument(
        "--corpus",
        required=True,
        help="분석할 passage corpus JSONL 파일 경로",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="토큰 길이 계산용 토크나이저 (예: BAAI/bge-m3). 지정하지 않으면 단어 수로 추정",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="짧은 텍스트로 간주할 최소 길이 (기본값: 10)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="텍스트 리포트 저장 경로 (선택사항)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="JSON 리포트 저장 경로 (선택사항)",
    )
    
    args = parser.parse_args()
    
    # 분석 실행
    print(f"[analyze_passages] Corpus 분석 중: {args.corpus}")
    print()
    
    results = analyze_passages(
        corpus_path=args.corpus,
        tokenizer_name=args.tokenizer,
        min_text_length=args.min_text_length,
    )
    
    # 리포트 출력
    print_analysis_report(results, output_file=args.output)
    
    # JSON 출력
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ JSON 리포트 저장: {json_path}")


if __name__ == "__main__":
    main()

