
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable
import json
import ujson
from pathlib import Path

@dataclass
class Passage:
    id: str
    text: str
    meta: Dict[str, Any]

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            yield ujson.loads(line)

def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(ujson.dumps(r, ensure_ascii=False)+"\n")
