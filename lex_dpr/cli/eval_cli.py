"""
LexDPR 평가 CLI 모듈

학습된 Bi-Encoder(SentenceTransformer) 체크포인트를 이용해
MRR@k, NDCG@k, MAP@k, Precision/Recall@k 등을 계산한다.

상세 분석 기능:
- 쿼리별 성능 분석
- 소스별 성능 분석
- 실패 케이스 분석
- 쿼리/Passage 길이별 성능 분석
- 여러 모델 비교 분석
"""

import argparse
import json
import logging
import os
import statistics
from pathlib import Path
from typing import List, Optional, Sequence

from sentence_transformers import SentenceTransformer

from lex_dpr.data import load_passages
from lex_dpr.eval import build_ir_evaluator
from lex_dpr.eval_detailed import (
    compare_models,
    evaluate_detailed,
    print_detailed_report,
)
from lex_dpr.models.templates import TemplateMode
from lex_dpr.utils.io import read_jsonl
from lex_dpr.utils.web_logging import WebLogger

logger = logging.getLogger("lex_dpr.cli.eval")


def _parse_k_values(values: Sequence[str] | None) -> List[int]:
    if not values:
        return [1, 3, 5, 10]
    parsed: List[int] = []
    for v in values:
        try:
            parsed.append(int(v))
        except ValueError:
            continue
    return parsed or [1, 3, 5, 10]


def _normalize_metric_name(key: str) -> str:
    """메트릭 이름을 WandB 형식으로 정규화"""
    # "val_cosine_ndcg@10" -> "ndcg_at_10"
    metric_name = key
    # "val_" 제거
    if metric_name.startswith("val_"):
        metric_name = metric_name[4:]
    # "cosine_" 제거
    if metric_name.startswith("cosine_"):
        metric_name = metric_name[7:]
    # "@" 기호를 "_at_"로 변경 (WandB는 @ 기호를 허용하지 않음)
    metric_name = metric_name.replace("@", "_at_")
    return metric_name


def _log_metrics_to_wandb(metrics: dict, model_name: str, web_logger: WebLogger, k_values: List[int]):
    """메트릭을 WandB에 로깅"""
    if not web_logger or not web_logger.is_active:
        return
    
    # 메트릭 이름 정규화 및 로깅
    wandb_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalized_key = _normalize_metric_name(key)
            wandb_metrics[normalized_key] = float(value)
    
    # 모델 이름을 태그로 추가
    try:
        import wandb
        if hasattr(wandb.run, 'tags'):
            tags = list(wandb.run.tags) if wandb.run.tags else []
            if model_name not in tags:
                tags.append(model_name)
            wandb.run.tags = tags
    except Exception:
        pass
    
    if wandb_metrics:
        web_logger.log_metrics(wandb_metrics, step=0)
        logger.info(f"WandB에 메트릭 로깅 완료: {len(wandb_metrics)}개 메트릭")


def _create_report_file(metrics: dict, model_name: str, k_values: List[int], output_path: Path):
    """평가 결과 리포트 파일 생성"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"LexDPR 평가 리포트: {model_name}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 메트릭 그룹화
    metric_groups = {
        "MRR": [],
        "NDCG": [],
        "MAP": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": [],
    }
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            for group_name in metric_groups.keys():
                if group_name.lower() in key.lower():
                    metric_groups[group_name].append((key, value))
                    break
    
    # 각 그룹별 출력
    for group_name, group_metrics in metric_groups.items():
        if group_metrics:
            report_lines.append(f"\n{group_name} 메트릭:")
            report_lines.append("-" * 80)
            for key, value in sorted(group_metrics):
                normalized_key = _normalize_metric_name(key)
                report_lines.append(f"  {normalized_key:30s}: {value:.4f}")
    
    report_lines.append("\n" + "=" * 80)
    
    # 파일 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"평가 리포트 저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LexDPR Bi-Encoder 평가 스크립트 (Sentence-Transformers 기반)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoint/lexdpr/bi_encoder",
        help="학습된 SentenceTransformer/Bi-Encoder 체크포인트 경로 (기본: checkpoint/lexdpr/bi_encoder)",
    )
    parser.add_argument(
        "--passages",
        type=str,
        default="data/processed/merged_corpus.jsonl",
        help="Passage 코퍼스 JSONL 경로 (기본: data/processed/merged_corpus.jsonl)",
    )
    parser.add_argument(
        "--eval-pairs",
        type=str,
        default="data/pairs_eval.jsonl",
        help="평가용 쿼리-패시지 쌍 JSONL 경로 (기본: data/pairs_eval.jsonl)",
    )
    parser.add_argument(
        "--k-values",
        nargs="*",
        help="평가할 k 값 목록 (예: --k-values 1 3 5 10, 기본: 1 3 5 10)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="bge",
        choices=["bge", "none"],
        help="템플릿 모드: 'bge' 또는 'none' (기본: bge)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="평가 시 배치 크기 (기본: 16, 메모리 절약을 위해 작게 설정)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="결과를 저장할 JSON 파일 경로 (비우면 stdout에만 출력)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="상세 분석 리포트 생성 (쿼리별, 소스별, 실패 케이스 분석 포함)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="상세 리포트를 저장할 텍스트 파일 경로 (--detailed 옵션과 함께 사용)",
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        help="여러 모델을 비교 평가 (모델 경로들을 공백으로 구분)",
    )
    parser.add_argument(
        "--compare-output",
        type=str,
        default="",
        help="모델 비교 리포트 저장 경로 (--compare-models와 함께 사용)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="결과를 WandB에 로깅",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lexdpr-eval",
        help="WandB 프로젝트 이름 (기본: lexdpr-eval)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="",
        help="WandB run 이름 (기본: 모델 이름 또는 'eval')",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="WandB entity 이름 (선택사항)",
    )

    args = parser.parse_args()

    passages_path = Path(args.passages)
    eval_pairs_path = Path(args.eval_pairs)

    if not passages_path.exists():
        raise FileNotFoundError(f"passages 파일을 찾을 수 없습니다: {passages_path}")
    if not eval_pairs_path.exists():
        raise FileNotFoundError(f"eval-pairs 파일을 찾을 수 없습니다: {eval_pairs_path}")

    # 1) Passage 로드
    passages = load_passages(str(passages_path))
    
    # WandB 로거 초기화 (옵션이 활성화된 경우)
    web_logger = None
    if args.wandb:
        wandb_token = os.getenv("WANDB_API_KEY")
        if not wandb_token:
            logger.warning("WANDB_API_KEY 환경 변수가 설정되지 않았습니다. WandB 로깅을 건너뜁니다.")
        else:
            try:
                web_logger = WebLogger(
                    service="wandb",
                    token=wandb_token,
                    project=args.wandb_project,
                    name=args.wandb_name or "eval",
                    entity=args.wandb_entity if args.wandb_entity else None,
                )
            except Exception as e:
                logger.warning(f"WandB 초기화 실패: {e}. WandB 로깅을 건너뜁니다.")

    # 2) 여러 모델 비교 모드
    if args.compare_models:
        k_vals = _parse_k_values(args.k_values)
        template_mode = TemplateMode.BGE if args.template == "bge" else TemplateMode.NONE
        
        # compare_models 함수가 batch_size를 지원하는지 확인 필요
        # 일단 기본 InformationRetrievalEvaluator를 사용하도록 수정
        # compare_models 함수 사용 (더 효율적)
        from lex_dpr.eval_detailed import compare_models
        compare_result_dict = compare_models(
            model_paths=args.compare_models,
            passages=passages,
            eval_pairs_path=str(eval_pairs_path),
            k_values=k_vals,
            template=template_mode,
            output_file=args.compare_output or None,
            batch_size=args.batch_size,
        )
        
        # compare_models는 딕셔너리를 반환하므로 결과 추출
        comparison_results = compare_result_dict.get("comparison_results", [])
        compare_result = {}
        for item in comparison_results:
            model_path = item.get("model_path", "")
            compare_result[model_path] = item.get("metrics", {})
        
        # 비교 리포트 생성
        if args.compare_output:
            compare_output_path = Path(args.compare_output)
            compare_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("모델 비교 평가 리포트")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            for model_path, metrics in compare_result.items():
                model_name = Path(model_path).name if model_path else "unknown"
                report_lines.append(f"\n모델: {model_name}")
                report_lines.append("-" * 80)
                for key, value in sorted(metrics.items()):
                    if isinstance(value, (int, float)):
                        normalized_key = _normalize_metric_name(key)
                        report_lines.append(f"  {normalized_key:30s}: {value:.4f}")
            
            compare_output_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"비교 리포트 저장: {compare_output_path}")
        
        # WandB에 각 모델 결과 로깅
        if web_logger and web_logger.is_active:
            for model_path, model_metrics in compare_result.items():
                model_name = Path(model_path).name if model_path else "unknown"
                # WandB run 이름 업데이트
                try:
                    import wandb
                    if args.wandb_name:
                        wandb.run.name = f"{args.wandb_name}-{model_name}"
                    else:
                        wandb.run.name = f"eval-{model_name}"
                except Exception:
                    pass
                _log_metrics_to_wandb(model_metrics, model_name, web_logger, k_vals)
        
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(compare_result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        if web_logger:
            web_logger.finish()
        return

    # 3) 상세 분석 모드
    if args.detailed:
        k_vals = _parse_k_values(args.k_values)
        template_mode = TemplateMode.BGE if args.template == "bge" else TemplateMode.NONE
        
        # GPU 메모리 정리
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        model = SentenceTransformer(args.model)
        
        try:
            detailed_result = evaluate_detailed(
                model=model,
                passages=passages,
                eval_pairs_path=str(eval_pairs_path),
                k_values=k_vals,
                template=template_mode,
                batch_size=args.batch_size,
            )
        finally:
            # 메모리 정리
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 모델 이름 추출
        model_name = Path(args.model).name if args.model else "unknown"
        
        # WandB 로깅
        if web_logger and web_logger.is_active:
            try:
                import wandb
                if args.wandb_name:
                    wandb.run.name = args.wandb_name
                else:
                    wandb.run.name = f"eval-detailed-{model_name}"
                # 모델 경로를 파라미터로 로깅
                web_logger.log_params({"model": args.model, "template": args.template})
            except Exception:
                pass
            _log_metrics_to_wandb(detailed_result.metrics, model_name, web_logger, k_vals)
        
        # 리포트 출력
        report_path = args.report or None
        print_detailed_report(
            result=detailed_result,
            output_file=report_path,
            k_values=k_vals,
        )
        
        # 리포트를 WandB 아티팩트로 업로드
        if report_path and web_logger and web_logger.is_active:
            web_logger.log_artifact(str(report_path), artifact_path="detailed_eval_report.txt")
        
        # JSON 출력
        if args.output:
            result_dict = {
                "k_values": k_vals,
                "metrics": detailed_result.metrics,
                "source_stats": {
                    source: {
                        "count": stats["count"],
                        "avg_mrr": statistics.mean(stats["mrr"]) if stats["mrr"] else 0.0,
                        "avg_ndcg": statistics.mean(stats["ndcg"]) if stats["ndcg"] else 0.0,
                        "avg_recall": statistics.mean(stats["recall"]) if stats["recall"] else 0.0,
                    }
                    for source, stats in detailed_result.source_stats.items()
                },
                "failed_count": len(detailed_result.failed_queries),
                "failed_queries": detailed_result.failed_queries[:20],  # 상위 20개만
            }
            
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result_dict, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        if web_logger:
            web_logger.finish()
        return

    # 4) 기본 평가 모드 (기존 동작)
    k_vals = _parse_k_values(args.k_values)
    template_mode = TemplateMode.BGE if args.template == "bge" else TemplateMode.NONE
    
    # GPU 메모리 정리
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    evaluator, normalized_k = build_ir_evaluator(
        passages=passages,
        eval_pairs_path=str(eval_pairs_path),
        read_jsonl_fn=read_jsonl,
        k_vals=k_vals,
        template=template_mode,
        batch_size=args.batch_size,
    )

    model = SentenceTransformer(args.model)
    try:
        metrics = evaluator(model, output_path=None)
    finally:
        # 메모리 정리
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    result = {
        "k_values": normalized_k,
        "metrics": metrics,
    }

    # 모델 이름 추출
    model_name = Path(args.model).name if args.model else "unknown"
    
    # WandB 로깅
    if web_logger and web_logger.is_active:
        # WandB run 이름 설정
        try:
            import wandb
            if args.wandb_name:
                wandb.run.name = args.wandb_name
            else:
                wandb.run.name = f"eval-{model_name}"
            # 모델 경로를 파라미터로 로깅
            web_logger.log_params({"model": args.model, "template": args.template})
        except Exception:
            pass
        _log_metrics_to_wandb(metrics, model_name, web_logger, normalized_k)
    
    # 리포트 생성
    if args.report:
        report_path = Path(args.report)
        _create_report_file(metrics, model_name, normalized_k, report_path)
        # 리포트를 WandB 아티팩트로 업로드
        if web_logger and web_logger.is_active:
            web_logger.log_artifact(str(report_path), artifact_path="eval_report.txt")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # WandB 종료
    if web_logger:
        web_logger.finish()


if __name__ == "__main__":
    main()


