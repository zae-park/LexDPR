# lex_dpr/utils/io.py
from pathlib import Path
import json, gzip
from typing import Iterator, Any, Dict, Union

PathLike = Union[str, Path]

def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: PathLike, rows: Iterator[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def merge_jsonl(paths: list[PathLike], out: PathLike) -> None:
    def gen():
        for pp in paths:
            yield from read_jsonl(pp)
    write_jsonl(out, gen())
