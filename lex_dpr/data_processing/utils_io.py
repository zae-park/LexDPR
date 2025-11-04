from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Iterator, Union
import json, gzip

PathLike = Union[str, Path]

def read_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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
