#!/usr/bin/env python3
"""
조문, 항, 호 단위로 passage 생성 및 pre-trained 모델 평가 스크립트

사용법:
    poetry run python scripts/eval_chunk_units.py
    poetry run python scripts/eval_chunk_units.py --models jhgan/ko-sroberta-multitask dragonkue/BGE-m3-ko
    poetry run python scripts/eval_chunk_units.py --law-src-dir data/laws --output-dir data/eval_chunk_units
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from lex_dpr.utils.io import read_jsonl, write_jsonl


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """명령어 실행"""
    print(f"실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"에러: {result.stderr}", file=sys.stderr)
    return result


def create_article_level_passages(law_passages_file: Path, output_file: Path):
    """법령 passage를 조문 단위로 병합"""
    print(f"조문 단위로 변환 중: {law_passages_file} -> {output_file}")
    
    law_passages = list(read_jsonl(law_passages_file))
    article_dict = {}
    
    for p in law_passages:
        article_key = p.get('article', '')
        if not article_key:
            continue
        
        # 조문 ID 생성 (parent_id 또는 article 기반)
        article_id = p.get('parent_id') or p.get('id', '').rsplit('_', 1)[0] if '_' in p.get('id', '') else p.get('id', '')
        
        if article_key not in article_dict:
            article_dict[article_key] = {
                'id': article_id,
                'parent_id': article_id,
                'type': p.get('type', '법령'),
                'law_id': p.get('law_id'),
                'law_name': p.get('law_name'),
                'article': article_key,
                'effective_date': p.get('effective_date'),
                'text': p.get('text', '').strip(),
            }
        else:
            # 같은 조문의 다른 항/호를 합침
            existing_text = article_dict[article_key]['text']
            new_text = p.get('text', '').strip()
            if new_text and new_text not in existing_text:
                article_dict[article_key]['text'] += '\n' + new_text
    
    # 조문 단위 passage 저장
    article_passages = list(article_dict.values())
    write_jsonl(output_file, article_passages)
    print(f"✅ 조문 단위 passage 생성: {len(article_passages)}개")
    return article_passages


def main():
    parser = argparse.ArgumentParser(
        description="조문, 항, 호 단위로 passage 생성 및 pre-trained 모델 평가"
    )
    parser.add_argument(
        "--law-src-dir",
        type=str,
        default="data/laws",
        help="법령 소스 디렉토리 (기본값: data/laws)"
    )
    parser.add_argument(
        "--admin-src-dir",
        type=str,
        default="data/admin_rules",
        help="행정규칙 소스 디렉토리 (기본값: data/admin_rules)"
    )
    parser.add_argument(
        "--prec-json-dir",
        type=str,
        default="data/precedents",
        help="판례 JSON 디렉토리 (기본값: data/precedents)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval_chunk_units",
        help="출력 디렉토리 (기본값: data/eval_chunk_units)"
    )
    parser.add_argument(
        "--eval-pairs",
        type=str,
        default="data/processed/pairs_train_valid.jsonl",
        help="평가 쌍 파일 (기본값: data/processed/pairs_train_valid.jsonl)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["jhgan/ko-sroberta-multitask", "dragonkue/BGE-m3-ko"],
        help="평가할 모델 목록 (기본값: jhgan/ko-sroberta-multitask dragonkue/BGE-m3-ko)"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="평가할 K 값 (기본값: 1 3 5 10 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="배치 크기 (기본값: 8)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="결과를 WandB에 로깅 (기본값: False, 로컬 파일만 생성)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lexdpr-eval-chunk-units",
        help="WandB 프로젝트 이름 (기본값: lexdpr-eval-chunk-units)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="WandB entity 이름 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 경로 설정
    law_src_dir = Path(args.law_src_dir)
    admin_src_dir = Path(args.admin_src_dir)
    output_base_dir = Path(args.output_dir)
    eval_pairs = Path(args.eval_pairs)
    results_dir = output_base_dir / "results"
    
    # 출력 디렉토리 생성
    output_base_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Passage Chunk 단위별 Pre-trained 모델 평가")
    print("=" * 80)
    print()
    print("설정:")
    print(f"  법령 소스 디렉토리: {law_src_dir}")
    print(f"  행정규칙 소스 디렉토리: {admin_src_dir}")
    print(f"  출력 디렉토리: {output_base_dir}")
    print(f"  평가 쌍 파일: {eval_pairs}")
    print(f"  평가 모델: {args.models}")
    print()
    
    # 평가 쌍 파일 확인
    if not eval_pairs.exists():
        print(f"⚠️  경고: 평가 쌍 파일을 찾을 수 없습니다: {eval_pairs}")
        print("   먼저 make_pairs를 실행하여 평가 쌍을 생성하세요.")
        sys.exit(1)
    
    # ==========================================
    # 1. 항 단위로 passage 생성 (기본값)
    # ==========================================
    print("[1/6] 항 단위 passage 생성 중...")
    paragraph_dir = output_base_dir / "paragraph"
    paragraph_dir.mkdir(exist_ok=True)
    
    # 행정규칙 전처리
    if admin_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(paragraph_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (항 단위)
    if law_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(law_src_dir),
            "--out-law", str(paragraph_dir / "law_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 코퍼스 병합
    law_file = paragraph_dir / "law_passages.jsonl"
    admin_file = paragraph_dir / "admin_passages.jsonl"
    merged_file = paragraph_dir / "merged_corpus.jsonl"
    
    if law_file.exists() and admin_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--out", str(merged_file)
        ])
    
    print(f"✅ 항 단위 passage 생성 완료: {merged_file}")
    print()
    
    # ==========================================
    # 2. 호 단위로 passage 생성
    # ==========================================
    print("[2/6] 호 단위 passage 생성 중...")
    item_dir = output_base_dir / "item"
    item_dir.mkdir(exist_ok=True)
    
    # 행정규칙 전처리
    if admin_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(item_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (호 단위)
    if law_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(law_src_dir),
            "--out-law", str(item_dir / "law_passages.jsonl"),
            "--include-items",
            "--glob", "**/*.json"
        ], check=False)
    
    # 코퍼스 병합
    law_file = item_dir / "law_passages.jsonl"
    admin_file = item_dir / "admin_passages.jsonl"
    merged_file = item_dir / "merged_corpus.jsonl"
    
    if law_file.exists() and admin_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--out", str(merged_file)
        ])
    
    print(f"✅ 호 단위 passage 생성 완료: {merged_file}")
    print()
    
    # ==========================================
    # 3. 조문 단위로 passage 생성
    # ==========================================
    print("[3/6] 조문 단위 passage 생성 중...")
    article_dir = output_base_dir / "article"
    article_dir.mkdir(exist_ok=True)
    
    # 행정규칙 전처리 (이미 조문 단위)
    if admin_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(article_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (항 단위로 먼저 생성 후 조문 단위로 병합)
    if law_src_dir.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(law_src_dir),
            "--out-law", str(article_dir / "law_passages_temp.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
        
        # 조문 단위로 변환
        temp_file = article_dir / "law_passages_temp.jsonl"
        article_file = article_dir / "law_passages_article.jsonl"
        
        if temp_file.exists():
            create_article_level_passages(temp_file, article_file)
    
    # 코퍼스 병합
    law_file = article_dir / "law_passages_article.jsonl"
    admin_file = article_dir / "admin_passages.jsonl"
    merged_file = article_dir / "merged_corpus.jsonl"
    
    if law_file.exists() and admin_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            "poetry", "run", "python", "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--out", str(merged_file)
        ])
    
    print(f"✅ 조문 단위 passage 생성 완료: {merged_file}")
    print()
    
    # ==========================================
    # 4. 각 chunk 단위별로 모델 평가
    # ==========================================
    print("[4/6] 각 chunk 단위별 모델 평가 시작...")
    print()
    
    # 평가 결과 저장
    all_results = {}
    
    for chunk_type in ["paragraph", "item", "article"]:
        chunk_dir = output_base_dir / chunk_type
        corpus_file = chunk_dir / "merged_corpus.jsonl"
        
        if not corpus_file.exists():
            print(f"⚠️  경고: {chunk_type} 코퍼스 파일을 찾을 수 없습니다: {corpus_file}")
            continue
        
        # Passage 개수 확인
        passage_count = len(list(read_jsonl(corpus_file)))
        print(f"Chunk 단위: {chunk_type} (Passage 개수: {passage_count:,})")
        
        all_results[chunk_type] = {}
        
        for model in args.models:
            model_name = model.replace("/", "_").replace("-", "_")
            result_file = results_dir / f"{chunk_type}_{model_name}.json"
            report_file = results_dir / f"{chunk_type}_{model_name}.txt"
            
            print(f"  평가 중: {model}")
            
            # 평가 실행
            k_values_str = " ".join(str(k) for k in args.k_values)
            eval_cmd = [
                "poetry", "run", "lex-dpr", "eval",
                "--model", model,
                "--corpus", str(corpus_file),
                "--eval-pairs", str(eval_pairs),
                "--output", str(result_file),
                "--report", str(report_file),
                "--k-values", *[str(k) for k in args.k_values],
                "--batch-size", str(args.batch_size),
            ]
            
            # WandB 옵션 추가
            if args.wandb:
                eval_cmd.append("--wandb")
                eval_cmd.extend(["--wandb-project", args.wandb_project])
                eval_cmd.extend(["--wandb-name", f"{chunk_type}_{model_name}"])
                if args.wandb_entity:
                    eval_cmd.extend(["--wandb-entity", args.wandb_entity])
            else:
                eval_cmd.append("--no-wandb")
            
            try:
                run_command(eval_cmd)
                
                # 결과 로드
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        all_results[chunk_type][model] = results
                        print(f"    ✅ 완료: NDCG@10={results.get('val_cosine_ndcg@10', 0):.4f}")
            except subprocess.CalledProcessError as e:
                print(f"    ⚠️  평가 실패: {e}")
        
        print()
    
    # ==========================================
    # 5. 결과 비교 및 출력
    # ==========================================
    print("[5/6] 결과 비교 중...")
    
    comparison_file = results_dir / "comparison.txt"
    summary_file = results_dir / "summary.txt"
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Passage Chunk 단위별 모델 성능 비교\n")
        f.write("=" * 100 + "\n\n")
        
        for model in args.models:
            f.write(f"\n모델: {model}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Chunk 단위':<20} {'NDCG@10':<15} {'Recall@10':<15} {'MRR@10':<15} {'Passage 수':<15}\n")
            f.write("-" * 100 + "\n")
            
            for chunk_type in ["paragraph", "item", "article"]:
                if chunk_type in all_results and model in all_results[chunk_type]:
                    results = all_results[chunk_type][model]
                    corpus_file = output_base_dir / chunk_type / "merged_corpus.jsonl"
                    passage_count = len(list(read_jsonl(corpus_file))) if corpus_file.exists() else 0
                    
                    f.write(f"{chunk_type:<20} "
                           f"{results.get('val_cosine_ndcg@10', 0):<15.4f} "
                           f"{results.get('val_cosine_recall@10', 0):<15.4f} "
                           f"{results.get('val_cosine_mrr@10', 0):<15.4f} "
                           f"{passage_count:<15,}\n")
            
            f.write("\n")
    
    # 요약 파일 생성
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Passage Chunk 단위별 Pre-trained 모델 평가 결과\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"평가 일시: {__import__('datetime').datetime.now()}\n")
        f.write(f"평가 쌍 파일: {eval_pairs}\n")
        f.write(f"평가 모델: {', '.join(args.models)}\n\n")
        
        for chunk_type in ["paragraph", "item", "article"]:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Chunk 단위: {chunk_type}\n")
            f.write(f"{'=' * 80}\n")
            
            corpus_file = output_base_dir / chunk_type / "merged_corpus.jsonl"
            if corpus_file.exists():
                passage_count = len(list(read_jsonl(corpus_file)))
                f.write(f"Passage 개수: {passage_count:,}\n\n")
            
            for model in args.models:
                if chunk_type in all_results and model in all_results[chunk_type]:
                    results = all_results[chunk_type][model]
                    f.write(f"모델: {model}\n")
                    for key in ['val_cosine_ndcg@10', 'val_cosine_recall@10', 'val_cosine_mrr@10']:
                        if key in results:
                            f.write(f"  {key}: {results[key]:.4f}\n")
                    f.write("\n")
    
    # ==========================================
    # 6. 최종 요약 출력
    # ==========================================
    print("[6/6] 최종 요약")
    print()
    print("=" * 80)
    print("평가 완료!")
    print("=" * 80)
    print()
    print("📊 결과 파일:")
    print(f"  - 요약: {summary_file}")
    print(f"  - 비교: {comparison_file}")
    print()
    print("📈 비교 결과:")
    print()
    with open(comparison_file, 'r', encoding='utf-8') as f:
        print(f.read())
    print("=" * 80)


if __name__ == "__main__":
    main()

