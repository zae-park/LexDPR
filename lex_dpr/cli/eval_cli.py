"""
LexDPR 평가 CLI 모듈

학습된 Bi-Encoder(SentenceTransformer) 체크포인트를 이용해
MRR@k, NDCG@k, MAP@k, Precision/Recall@k 등을 계산한다.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

from lex_dpr.data import load_passages
from lex_dpr.eval import build_ir_evaluator
from lex_dpr.models.templates import TemplateMode
from lex_dpr.utils.io import read_jsonl


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
        "--output",
        type=str,
        default="",
        help="결과를 저장할 JSON 파일 경로 (비우면 stdout에만 출력)",
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

    # 2) Evaluator 생성
    k_vals = _parse_k_values(args.k_values)
    template_mode = TemplateMode.BGE if args.template == "bge" else TemplateMode.NONE
    evaluator, normalized_k = build_ir_evaluator(
        passages=passages,
        eval_pairs_path=str(eval_pairs_path),
        read_jsonl_fn=read_jsonl,
        k_vals=k_vals,
        template=template_mode,
    )

    # 3) 모델 로드 (SentenceTransformer)
    model = SentenceTransformer(args.model)

    # 4) 평가 실행
    metrics = evaluator(model, output_path=None)

    # 5) 결과 정리 및 출력
    result = {
        "k_values": normalized_k,
        "metrics": metrics,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


