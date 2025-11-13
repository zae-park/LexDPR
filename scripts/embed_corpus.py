#!/usr/bin/env python
"""
Embed passages or queries using a fine-tuned Bi-Encoder checkpoint.

Examples
--------
poetry run python scripts/embed_corpus.py \
  --model checkpoint/lexdpr/bi_encoder \
  --input data/processed/law_passages.jsonl \
  --outdir embeds \
  --prefix passages
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from lex_dpr.models.encoders import BiEncoder
from lex_dpr.models.templates import TemplateMode
from lex_dpr.utils.io import read_jsonl


def _parse_template(value: str) -> TemplateMode:
    value = (value or "").lower()
    try:
        return TemplateMode(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Unknown template mode: {value}") from err


def _load_texts(
    path: Path,
    *,
    id_field: str,
    text_field: str,
    limit: int | None = None,
) -> tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []

    for row in read_jsonl(path):
        ids.append(str(row[id_field]))
        texts.append(str(row[text_field]))
        if limit is not None and len(ids) >= limit:
            break

    if not ids:
        raise ValueError(f"No rows loaded from {path}")
    return ids, texts


def _save_embeddings(
    outdir: Path,
    prefix: str,
    ids: Iterable[str],
    embeddings: np.ndarray,
    *,
    fmt: str = "npz",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ids_array = np.asarray(list(ids), dtype=object)
    emb_array = embeddings.astype(np.float32, copy=False)

    if fmt in {"npz", "both"}:
        np.savez(outdir / f"{prefix}.npz", ids=ids_array, embeddings=emb_array)
    if fmt in {"npy", "both"}:
        np.save(outdir / f"{prefix}_ids.npy", ids_array)
        np.save(outdir / f"{prefix}_embeds.npy", emb_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed JSONL corpus with a bi-encoder checkpoint.")
    parser.add_argument("--model", required=True, help="Path or name of the SentenceTransformer checkpoint.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--outdir", required=True, help="Directory to write embeddings.")
    parser.add_argument("--prefix", default="corpus", help="Filename prefix for outputs.")
    parser.add_argument("--id-field", default="id", help="Field name for unique ids.")
    parser.add_argument("--text-field", default="text", help="Field name containing text to embed.")
    parser.add_argument(
        "--template",
        default="bge",
        choices=[mode.value for mode in TemplateMode],
        help="Template mode applied before encoding.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=0, help="Optional max sequence length.")
    parser.add_argument("--device", default=None, help="Force device (e.g. cuda, cpu).")
    parser.add_argument(
        "--encode-type",
        choices=["passage", "query"],
        default="passage",
        help="Apply passage or query template/encoding.",
    )
    parser.add_argument(
        "--output-format",
        default="npz",
        choices=["npz", "npy", "both"],
        help="How to save embeddings.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to encode.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    parser.add_argument("--peft-adapter", type=str, default=None, help="Path to PEFT adapter if not auto-detected.")
    args = parser.parse_args()

    template_mode = _parse_template(args.template)
    normalize = not args.no_normalize
    max_len = args.max_len if args.max_len > 0 else None

    encoder = BiEncoder(
        args.model,
        template=template_mode,
        normalize=normalize,
        max_seq_length=max_len,
        peft_adapter_path=args.peft_adapter,
    )
    if args.device:
        encoder.model.to(args.device)

    ids, texts = _load_texts(Path(args.input), id_field=args.id_field, text_field=args.text_field, limit=args.limit)
    if args.encode_type == "query":
        embeddings = encoder.encode_queries(texts, batch_size=args.batch_size)
    else:
        embeddings = encoder.encode_passages(texts, batch_size=args.batch_size)

    outdir = Path(args.outdir)
    _save_embeddings(outdir, args.prefix, ids, embeddings, fmt=args.output_format)
    print(f"[embed] saved {len(ids)} vectors → {outdir}")


if __name__ == "__main__":
    main()

