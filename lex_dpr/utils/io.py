# lex_dpr/utils/io.py
from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Iterable, Iterator, Dict, Any, Union, Sequence

PathLike = Union[str, Path]


def read_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".gz":
        raise ValueError("append_jsonl does not support .gz; use .jsonl")
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_or_append_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    if append:
        append_jsonl(path, rows)
    else:
        write_jsonl(path, rows)


def merge_jsonl(paths: Sequence[PathLike], out: PathLike) -> None:
    def gen():
        for pp in paths:
            yield from read_jsonl(pp)
    write_jsonl(out, gen())
