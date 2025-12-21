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
from typing import List, Dict, Optional, Any

from lex_dpr.utils.io import read_jsonl, write_jsonl
from lex_dpr.models.factory import ALIASES


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """명령어 실행"""
    print(f"실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"에러: {result.stderr}", file=sys.stderr)
    return result


def calculate_token_stats(corpus_file: Path, model_name: str = "jhgan/ko-sroberta-multitask") -> Dict[str, Any]:
    """
    각 chunk 단위별 토큰 통계를 계산합니다.
    
    Returns:
        dict: {
            'avg_tokens': 평균 토큰 수,
            'total_tokens': 전체 토큰 수,
            'passage_count': passage 개수,
            'avg_chars': 평균 문자 수
        }
    """
    try:
        from lex_dpr.models.encoders import BiEncoder
        
        # 모델 이름을 실제 경로로 변환 (alias 처리)
        real_model_name = ALIASES.get(model_name, model_name)
        
        # BiEncoder를 초기화하여 토크나이저 가져오기
        encoder = BiEncoder(real_model_name, template="bge")
        tokenizer = encoder.model.tokenizer
        
        passages = list(read_jsonl(corpus_file))
        if not passages:
            return {
                'avg_tokens': 0,
                'total_tokens': 0,
                'passage_count': 0,
                'avg_chars': 0
            }
        
        total_tokens = 0
        total_chars = 0
        
        for passage in passages:
            text = passage.get('text', '')
            total_chars += len(text)
            
            # 토큰 수 계산
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
        
        # 메모리 정리
        del encoder
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'avg_tokens': total_tokens / len(passages) if passages else 0,
            'total_tokens': total_tokens,
            'passage_count': len(passages),
            'avg_chars': total_chars / len(passages) if passages else 0
        }
    except Exception as e:
        print(f"⚠️  토큰 통계 계산 실패: {e}")
        return {
            'avg_tokens': 0,
            'total_tokens': 0,
            'passage_count': 0,
            'avg_chars': 0
        }


def get_model_info(model_name: str) -> Dict[str, Optional[Any]]:
    """
    모델의 크기와 max length 정보를 가져옵니다.
    
    Returns:
        dict: {
            'size_m': 모델 크기 (Million 파라미터),
            'max_length': 최대 시퀀스 길이,
            'embedding_dim': 임베딩 차원
        }
    """
    try:
        from lex_dpr.models.encoders import BiEncoder
        from sentence_transformers import SentenceTransformer
        
        # 모델 이름을 실제 경로로 변환 (alias 처리)
        real_model_name = ALIASES.get(model_name, model_name)
        
        # BiEncoder를 초기화하여 모델 정보 가져오기
        encoder = BiEncoder(real_model_name, template="bge")
        
        # Max length 가져오기
        max_length = encoder.model.max_seq_length
        if hasattr(encoder.model, 'tokenizer') and hasattr(encoder.model.tokenizer, 'model_max_length'):
            original_max_length = encoder.model.tokenizer.model_max_length
            if original_max_length:
                max_length = original_max_length
        
        # 모델 크기 계산 (파라미터 개수)
        try:
            total_params = sum(p.numel() for p in encoder.model.parameters())
            size_m = total_params / 1_000_000  # Million 단위
        except:
            size_m = None
        
        # 임베딩 차원 가져오기
        embedding_dim = None
        try:
            # SentenceTransformer의 첫 번째 모듈에서 임베딩 차원 추출
            if hasattr(encoder.model, 'get_sentence_embedding_dimension'):
                embedding_dim = encoder.model.get_sentence_embedding_dimension()
            elif hasattr(encoder.model, '_modules'):
                for module in encoder.model._modules.values():
                    if hasattr(module, 'get_sentence_embedding_dimension'):
                        embedding_dim = module.get_sentence_embedding_dimension()
                        break
                    elif hasattr(module, 'config'):
                        # Transformer 모델의 경우 config에서 가져오기
                        config = module.config
                        if hasattr(config, 'hidden_size'):
                            embedding_dim = config.hidden_size
                            break
        except:
            embedding_dim = None
        
        # 메모리 정리
        del encoder
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'size_m': round(size_m, 2) if size_m else None,
            'max_length': max_length,
            'embedding_dim': embedding_dim
        }
    except Exception as e:
        print(f"⚠️  모델 정보 가져오기 실패 ({model_name}): {e}")
        return {
            'size_m': None,
            'max_length': None,
            'embedding_dim': None
        }


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
        default=None,
        help="평가할 모델 목록 (기본값: 모든 사용 가능한 모델)"
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
    
    # 모델 목록 설정 (기본값: 모든 사용 가능한 모델)
    if args.models is None:
        # ALIASES의 모든 모델 + 주요 모델들
        default_models = list(ALIASES.keys()) + [
            "BAAI/bge-m3",
            "dragonkue/BGE-m3-ko",
            "jhgan/ko-sroberta-multitask",
        ]
        # 중복 제거 및 정렬
        models_to_eval = sorted(list(set(default_models)))
    else:
        models_to_eval = args.models
    
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
    print(f"  평가 모델 ({len(models_to_eval)}개): {', '.join(models_to_eval)}")
    print()
    
    # 모델 정보 수집
    print("[0/7] 모델 정보 수집 중...")
    model_info_dict = {}
    for model in models_to_eval:
        print(f"  정보 수집 중: {model}")
        model_info_dict[model] = get_model_info(model)
        if model_info_dict[model]['size_m']:
            print(f"    크기: {model_info_dict[model]['size_m']}M 파라미터")
        if model_info_dict[model]['max_length']:
            print(f"    Max Length: {model_info_dict[model]['max_length']}")
        if model_info_dict[model]['embedding_dim']:
            print(f"    Embedding Dim: {model_info_dict[model]['embedding_dim']}")
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
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(paragraph_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (항 단위)
    if law_src_dir.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
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
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
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
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(item_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (호 단위)
    if law_src_dir.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
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
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
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
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
            "--src-dir", str(admin_src_dir),
            "--out-admin", str(article_dir / "admin_passages.jsonl"),
            "--glob", "**/*.json"
        ], check=False)
    
    # 법령 전처리 (항 단위로 먼저 생성 후 조문 단위로 병합)
    if law_src_dir.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.preprocess_auto",
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
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--admin", str(admin_file),
            "--out", str(merged_file)
        ])
    elif law_file.exists():
        run_command([
            sys.executable, "-m", "lex_dpr.data_processing.merge_corpus",
            "--law", str(law_file),
            "--out", str(merged_file)
        ])
    
    print(f"✅ 조문 단위 passage 생성 완료: {merged_file}")
    print()
    
    # ==========================================
    # 4. 각 chunk 단위별로 모델 평가
    # ==========================================
    print("[4/7] 각 chunk 단위별 모델 평가 시작...")
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
        
        for model in models_to_eval:
            model_name = model.replace("/", "_").replace("-", "_")
            result_file = results_dir / f"{chunk_type}_{model_name}.json"
            report_file = results_dir / f"{chunk_type}_{model_name}.txt"
            
            print(f"  평가 중: {model}")
            
            # 평가 실행
            k_values_str = " ".join(str(k) for k in args.k_values)
            eval_cmd = [
                "lex-dpr", "eval",
                "--model", model,
                "--passages", str(corpus_file),
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
            
            try:
                run_command(eval_cmd)
                
                # 결과 로드
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        # metrics가 중첩된 경우 처리
                        metrics = results.get('metrics', results)
                        # 모델 정보 추가
                        metrics['model_info'] = model_info_dict.get(model, {})
                        all_results[chunk_type][model] = metrics
                        print(f"    ✅ 완료: NDCG@10={metrics.get('val_cosine_ndcg@10', 0):.4f}")
            except subprocess.CalledProcessError as e:
                print(f"    ⚠️  평가 실패: {e}")
        
        print()
    
    # ==========================================
    # 5. 결과 비교 및 출력
    # ==========================================
    print("[5/7] 결과 비교 중...")
    
    comparison_file = results_dir / "comparison.txt"
    summary_file = results_dir / "summary.txt"
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("Passage Chunk 단위별 모델 성능 비교\n")
        f.write("=" * 120 + "\n\n")
        
        for model in models_to_eval:
            model_info = model_info_dict.get(model, {})
            size_str = f"{model_info.get('size_m', 'N/A')}M" if model_info.get('size_m') else "N/A"
            max_len_str = str(model_info.get('max_length', 'N/A'))
            
            f.write(f"\n모델: {model}\n")
            f.write(f"  크기: {size_str} 파라미터, Max Length: {max_len_str}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Chunk 단위':<20} {'NDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12} {'Passage 수':<15} {'Size(M)':<10} {'Max Len':<10}\n")
            f.write("-" * 120 + "\n")
            
            for chunk_type in ["paragraph", "item", "article"]:
                if chunk_type in all_results and model in all_results[chunk_type]:
                    metrics = all_results[chunk_type][model]
                    corpus_file = output_base_dir / chunk_type / "merged_corpus.jsonl"
                    passage_count = len(list(read_jsonl(corpus_file))) if corpus_file.exists() else 0
                    
                    f.write(f"{chunk_type:<20} "
                           f"{metrics.get('val_cosine_ndcg@10', 0):<12.4f} "
                           f"{metrics.get('val_cosine_recall@10', 0):<12.4f} "
                           f"{metrics.get('val_cosine_mrr@10', 0):<12.4f} "
                           f"{passage_count:<15,} "
                           f"{size_str:<10} "
                           f"{max_len_str:<10}\n")
            
            f.write("\n")
        
        # 모델별 비교 테이블 (Chunk 단위별)
        f.write("\n" + "=" * 120 + "\n")
        f.write("Chunk 단위별 모델 비교\n")
        f.write("=" * 120 + "\n\n")
        
        for chunk_type in ["paragraph", "item", "article"]:
            f.write(f"\n[{chunk_type.upper()}]\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'모델':<40} {'Size(M)':<10} {'Max Len':<10} {'NDCG@10':<12} {'Recall@10':<12} {'MRR@10':<12}\n")
            f.write("-" * 120 + "\n")
            
            # 성능 순으로 정렬
            model_scores = []
            for model in models_to_eval:
                if chunk_type in all_results and model in all_results[chunk_type]:
                    metrics = all_results[chunk_type][model]
                    model_info = model_info_dict.get(model, {})
                    ndcg = metrics.get('val_cosine_ndcg@10', 0)
                    model_scores.append((model, model_info, metrics, ndcg))
            
            # NDCG@10 기준 내림차순 정렬
            model_scores.sort(key=lambda x: x[3], reverse=True)
            
            for model, model_info, metrics, _ in model_scores:
                size_str = f"{model_info.get('size_m', 'N/A')}M" if model_info.get('size_m') else "N/A"
                max_len_str = str(model_info.get('max_length', 'N/A'))
                
                f.write(f"{model:<40} "
                       f"{size_str:<10} "
                       f"{max_len_str:<10} "
                       f"{metrics.get('val_cosine_ndcg@10', 0):<12.4f} "
                       f"{metrics.get('val_cosine_recall@10', 0):<12.4f} "
                       f"{metrics.get('val_cosine_mrr@10', 0):<12.4f}\n")
            
            f.write("\n")
    
    # 요약 파일 생성
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Passage Chunk 단위별 Pre-trained 모델 평가 결과\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"평가 일시: {__import__('datetime').datetime.now()}\n")
        f.write(f"평가 쌍 파일: {eval_pairs}\n")
        f.write(f"평가 모델 ({len(models_to_eval)}개): {', '.join(models_to_eval)}\n\n")
        
        for chunk_type in ["paragraph", "item", "article"]:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Chunk 단위: {chunk_type}\n")
            f.write(f"{'=' * 80}\n")
            
            corpus_file = output_base_dir / chunk_type / "merged_corpus.jsonl"
            if corpus_file.exists():
                passage_count = len(list(read_jsonl(corpus_file)))
                f.write(f"Passage 개수: {passage_count:,}\n\n")
            
            for model in models_to_eval:
                if chunk_type in all_results and model in all_results[chunk_type]:
                    metrics = all_results[chunk_type][model]
                    model_info = model_info_dict.get(model, {})
                    
                    f.write(f"모델: {model}\n")
                    if model_info.get('size_m'):
                        f.write(f"  크기: {model_info['size_m']}M 파라미터\n")
                    if model_info.get('max_length'):
                        f.write(f"  Max Length: {model_info['max_length']}\n")
                    if model_info.get('embedding_dim'):
                        f.write(f"  Embedding Dim: {model_info['embedding_dim']}\n")
                    
                    for key in ['val_cosine_ndcg@10', 'val_cosine_recall@10', 'val_cosine_mrr@10']:
                        if key in metrics:
                            f.write(f"  {key}: {metrics[key]:.4f}\n")
                    f.write("\n")
    
    # ==========================================
    # 6. 토큰 통계 계산 및 절약 비율 분석
    # ==========================================
    print("[6/7] 토큰 통계 계산 중...")
    
    # 각 chunk 단위별 토큰 통계 계산
    token_stats = {}
    reference_model = models_to_eval[0] if models_to_eval else "jhgan/ko-sroberta-multitask"
    
    for chunk_type in ["paragraph", "item", "article"]:
        chunk_dir = output_base_dir / chunk_type
        corpus_file = chunk_dir / "merged_corpus.jsonl"
        
        if corpus_file.exists():
            print(f"  계산 중: {chunk_type}")
            token_stats[chunk_type] = calculate_token_stats(corpus_file, reference_model)
            print(f"    평균 토큰 수: {token_stats[chunk_type]['avg_tokens']:.1f}")
            print(f"    전체 토큰 수: {token_stats[chunk_type]['total_tokens']:,}")
            print(f"    Passage 개수: {token_stats[chunk_type]['passage_count']:,}")
    
    # 토큰 절약 비율 계산 (article 대비 paragraph)
    token_savings = {}
    if "article" in token_stats and "paragraph" in token_stats:
        article_avg = token_stats["article"]["avg_tokens"]
        paragraph_avg = token_stats["paragraph"]["avg_tokens"]
        
        if article_avg > 0:
            savings_ratio = (article_avg - paragraph_avg) / article_avg * 100
            token_savings["article_vs_paragraph"] = {
                "article_avg_tokens": article_avg,
                "paragraph_avg_tokens": paragraph_avg,
                "savings_tokens": article_avg - paragraph_avg,
                "savings_percentage": savings_ratio
            }
            print(f"\n  📊 토큰 절약 분석 (Article vs Paragraph):")
            print(f"    Article 평균 토큰: {article_avg:.1f}")
            print(f"    Paragraph 평균 토큰: {paragraph_avg:.1f}")
            print(f"    절약 토큰: {article_avg - paragraph_avg:.1f} ({savings_ratio:.1f}%)")
    
    if "article" in token_stats and "item" in token_stats:
        article_avg = token_stats["article"]["avg_tokens"]
        item_avg = token_stats["item"]["avg_tokens"]
        
        if article_avg > 0:
            savings_ratio = (article_avg - item_avg) / article_avg * 100
            token_savings["article_vs_item"] = {
                "article_avg_tokens": article_avg,
                "item_avg_tokens": item_avg,
                "savings_tokens": article_avg - item_avg,
                "savings_percentage": savings_ratio
            }
            print(f"\n  📊 토큰 절약 분석 (Article vs Item):")
            print(f"    Article 평균 토큰: {article_avg:.1f}")
            print(f"    Item 평균 토큰: {item_avg:.1f}")
            print(f"    절약 토큰: {article_avg - item_avg:.1f} ({savings_ratio:.1f}%)")
    
    # 토큰 통계 저장
    token_stats_file = results_dir / "token_stats.json"
    with open(token_stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "token_stats": token_stats,
            "token_savings": token_savings
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 토큰 통계 저장: {token_stats_file}")
    print()
    
    # ==========================================
    # 7. 모델 정보 JSON 저장
    # ==========================================
    print("[7/8] 모델 정보 저장 중...")
    model_info_file = results_dir / "model_info.json"
    with open(model_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ 모델 정보 저장: {model_info_file}")
    print()
    
    # ==========================================
    # 8. 최종 요약 출력 (토큰 절약 정보 포함)
    # ==========================================
    print("[8/8] 최종 요약")
    print()
    print("=" * 80)
    print("평가 완료!")
    print("=" * 80)
    print()
    print("📊 결과 파일:")
    print(f"  - 요약: {summary_file}")
    print(f"  - 비교: {comparison_file}")
    print(f"  - 모델 정보: {model_info_file}")
    print(f"  - 토큰 통계: {token_stats_file}")
    print()
    
    # 토큰 절약 정보 출력
    if token_savings:
        print("💰 토큰 절약 분석:")
        if "article_vs_paragraph" in token_savings:
            savings = token_savings["article_vs_paragraph"]
            print(f"  Article → Paragraph:")
            print(f"    절약율: {savings['savings_percentage']:.1f}%")
            print(f"    절약 토큰: {savings['savings_tokens']:.1f} 토큰/passage")
        if "article_vs_item" in token_savings:
            savings = token_savings["article_vs_item"]
            print(f"  Article → Item:")
            print(f"    절약율: {savings['savings_percentage']:.1f}%")
            print(f"    절약 토큰: {savings['savings_tokens']:.1f} 토큰/passage")
        print()
    print("📈 비교 결과:")
    print()
    with open(comparison_file, 'r', encoding='utf-8') as f:
        print(f.read())
    print("=" * 80)


if __name__ == "__main__":
    main()

