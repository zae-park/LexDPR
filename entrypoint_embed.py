#!/usr/bin/env python
"""
임베딩 추출 엔트리포인트

사용 예시:
  # Passages 임베딩 추출
  poetry run python entrypoint_embed.py \
    --model checkpoint/lexdpr/bi_encoder \
    --input data/processed/law_passages.jsonl \
    --outdir embeds \
    --prefix passages \
    --type passage

  # Queries 임베딩 추출
  poetry run python entrypoint_embed.py \
    --model checkpoint/lexdpr/bi_encoder \
    --input data/queries/queries.jsonl \
    --outdir embeds \
    --prefix queries \
    --type query
"""

import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from lex_dpr.models.encoders import BiEncoder
from lex_dpr.models.templates import TemplateMode
from lex_dpr.utils.io import read_jsonl


def _save_embeddings(
    outdir: Path,
    prefix: str,
    ids: List[str],
    embeddings: np.ndarray,
    *,
    fmt: str = "npz",
) -> None:
    """임베딩을 파일로 저장"""
    outdir.mkdir(parents=True, exist_ok=True)
    ids_array = np.asarray(ids, dtype=object)
    emb_array = embeddings.astype(np.float32, copy=False)

    if fmt in {"npz", "both"}:
        np.savez(outdir / f"{prefix}.npz", ids=ids_array, embeddings=emb_array)
        print(f"[embed] Saved {len(ids)} embeddings to {outdir / f'{prefix}.npz'}")
    if fmt in {"npy", "both"}:
        np.save(outdir / f"{prefix}_ids.npy", ids_array)
        np.save(outdir / f"{prefix}_embeds.npy", emb_array)
        print(f"[embed] Saved {len(ids)} embeddings to {outdir / f'{prefix}_ids.npy'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings from trained Bi-Encoder model")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", required=True, help="Input JSONL file (passages or queries)")
    parser.add_argument("--outdir", required=True, help="Output directory for embeddings")
    parser.add_argument("--prefix", required=True, help="Prefix for output files (e.g., 'passages', 'queries')")
    parser.add_argument(
        "--type",
        choices=["passage", "query"],
        required=True,
        help="Type of embeddings to extract: 'passage' or 'query'",
    )
    parser.add_argument("--id-field", default="id", help="Field name for ID in input JSONL")
    parser.add_argument("--text-field", default="text", help="Field name for text in input JSONL")
    parser.add_argument(
        "--template",
        default="bge",
        choices=["bge", "none"],
        help="Template mode (should match training config)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--max-len", type=int, default=0, help="Max sequence length (0 = use model default)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument(
        "--output-format",
        default="npz",
        choices=["npz", "npy", "both"],
        help="Output format",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to encode (for testing)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    parser.add_argument("--peft-adapter", type=str, default=None, help="Path to PEFT adapter (usually auto-detected)")

    args = parser.parse_args()

    # 템플릿 모드 파싱
    template_mode = TemplateMode.BGE if args.template == "bge" else TemplateMode.NONE
    normalize = not args.no_normalize
    max_len = args.max_len if args.max_len > 0 else None

    # 모델 로드
    print(f"[embed] Loading model from {args.model}")
    encoder = BiEncoder(
        args.model,
        template=template_mode,
        normalize=normalize,
        max_seq_length=max_len,
        peft_adapter_path=args.peft_adapter,
    )
    if args.device:
        encoder.model.to(args.device)
        print(f"[embed] Using device: {args.device}")

    # 데이터 로드
    print(f"[embed] Loading data from {args.input}")
    ids: List[str] = []
    texts: List[str] = []
    for row in read_jsonl(Path(args.input)):
        ids.append(str(row[args.id_field]))
        texts.append(str(row[args.text_field]))
        if args.limit is not None and len(ids) >= args.limit:
            break

    if not ids:
        raise ValueError(f"No rows loaded from {args.input}")

    print(f"[embed] Loaded {len(ids)} items")

    # 임베딩 생성
    print(f"[embed] Generating {args.type} embeddings...")
    if args.type == "query":
        embeddings = encoder.encode_queries(texts, batch_size=args.batch_size)
    else:  # passage
        embeddings = encoder.encode_passages(texts, batch_size=args.batch_size)

    print(f"[embed] Generated embeddings shape: {embeddings.shape}")

    # 저장
    outdir = Path(args.outdir)
    _save_embeddings(outdir, args.prefix, ids, embeddings, fmt=args.output_format)
    print(f"[embed] Completed! Saved {len(ids)} {args.type} embeddings to {outdir}")


if __name__ == "__main__":
    main()

