
import argparse
from pathlib import Path
from typing import List, Dict
import re
from models.utils import write_jsonl

HEADER_PATTERNS = [
    r"^\s*질의요지\s*$",
    r"^\s*사실관계\s*$",
    r"^\s*관련\s*법령\s*$",
    r"^\s*검토\s*$",
    r"^\s*결론\s*$",
]

def split_into_passages(text: str, max_chars: int = 1200, stride: int = 200) -> List[str]:
    # naive sliding window splitter with respect to headers
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = end - stride
    return [c for c in chunks if c]

def process_file(path: Path) -> List[Dict]:
    # For a real system, parse HWP/PDF/HTML prior to this step.
    text = path.read_text(encoding="utf-8", errors="ignore")
    passages = split_into_passages(text)
    rows = []
    for i, p in enumerate(passages):
        rows.append({
            "id": f"{path.stem}::p{i:04d}",
            "text": p,
            "meta": {"source": str(path)}
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="directory with raw statute or letter txt files")
    ap.add_argument("--output", type=str, required=True, help="output jsonl path under data/processed/")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    rows = []
    for p in in_dir.rglob("*.txt"):
        rows.extend(process_file(p))
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} passages to {out_path}")

if __name__ == "__main__":
    main()
