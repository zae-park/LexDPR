# lex_dpr/data_processing/preprocess_auto.py

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple
from .preprocess_law import law_iter_to_passages


def detect_type(js: Dict[str, Any]) -> str:
    if "법령" in js or "법령ID" in js:
        return "law"
    # AdmRulService 스키마 추정
    try:
        dumped = json.dumps(js, ensure_ascii=False)
    except Exception:
        dumped = ""
    if "AdmRulService" in js or "행정규칙기본정보" in dumped:
        return "admin"
    return "unknown"

def process_file(path: Path) -> Tuple[str, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    t = detect_type(js)

    if t == "law":
        rows = list(law_iter_to_passages(js))
        return t, rows

    if t == "admin":
        # ✅ 필요할 때만 임포트 + 에러 메시지 노출
        try:
            from .preprocess_admin_rule import admin_rule_iter_to_passages
        except Exception as e:
            raise RuntimeError(f"Failed to import preprocess_admin_rule: {e!r}")
        rows = admin_rule_iter_to_passages(js)
        return t, rows

    return "unknown", []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="입력 JSON 디렉토리 (재귀)")
    ap.add_argument("--out-law", required=False, default="data/processed/law_passages.jsonl")
    ap.add_argument("--out-admin", required=False, default="data/processed/admin_passages.jsonl")
    ap.add_argument("--append", action="store_true", help="출력 파일에 이어쓰기")
    ap.add_argument("--glob", default="**/*.json", help="파일 검색 패턴")
    args = ap.parse_args()

    from .utils_io import write_jsonl, append_jsonl  # 지연 임포트

    p = Path(args.src_dir)
    files = sorted(p.glob(args.glob))
    n_law = n_admin = n_unknown = 0
    buf_law, buf_admin = [], []

    for fp in files:
        try:
            t, rows = process_file(fp)
            if t == "law":
                n_law += 1; buf_law.extend(rows)
            elif t == "admin":
                n_admin += 1; buf_admin.extend(rows)
            else:
                n_unknown += 1
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")

    if buf_law:
        (append_jsonl if args.append else write_jsonl)(args.out_law, buf_law)
        print(f"[auto] law passages: +{len(buf_law)} → {args.out_law}")
    if buf_admin:
        (append_jsonl if args.append else write_jsonl)(args.out_admin, buf_admin)
        print(f"[auto] admin passages: +{len(buf_admin)} → {args.out_admin}")

    print(f"[auto] files: law={n_law}, admin={n_admin}, unknown={n_unknown}")

if __name__ == "__main__":
    main()
