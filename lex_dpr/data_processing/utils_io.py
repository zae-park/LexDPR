from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Iterator, Union
import json, gzip

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
    # gzip append 지원 간단화: .gz면 통짜 다시 써야 하므로 여기선 일반 .jsonl만 append 권장
    if p.suffix == ".gz":
        raise ValueError("append_jsonl does not support .gz; use .jsonl")
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_or_append_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]], append: bool=False) -> None:
    if append:
        append_jsonl(path, rows)
    else:
        from .utils_io import write_jsonl  # reuse existing
        write_jsonl(path, rows)