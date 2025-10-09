
import argparse
from pathlib import Path
import ujson

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(ujson.dumps(r, ensure_ascii=False) + "\n")

def split_into_passages(text, max_chars=1200, stride=200):
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

def process_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    passages = split_into_passages(text)
    rows = []
    for i, p in enumerate(passages):
        rows.append({"id": f"{path.stem}::p{i:04d}", "text": p, "meta": {"source": str(path)}})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
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
