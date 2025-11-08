# lex_dpr/data_processing/preprocess_auto.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple
from .preprocess_law import law_iter_to_passages


def _is_prec_dict(d: dict) -> bool:
    # 최상위 키에 직접 있거나
    if any(k in d for k in ["판례일련번호", "판결요지", "판시사항", "사건명", "법원명"]):
        return True
    # 래퍼 케이스
    if "PrecService" in d and isinstance(d["PrecService"], dict):
        ps = d["PrecService"]
        if any(k in ps for k in ["판례기본정보", "판결요지", "판시사항", "판결본문"]):
            return True
    # 목록/검색결과 케이스
    if any(k in d for k in ["판례목록", "precList", "prec_list"]):
        return True
    return False

# =========================
# 유형 감지
# =========================
def detect_type(js):
    # dict
    if isinstance(js, dict):
        if "법령" in js or "법령ID" in js:
            return "law"
        try:
            dumped = json.dumps(js, ensure_ascii=False)
        except Exception:
            dumped = ""
        if "AdmRulService" in js or "행정규칙기본정보" in dumped:
            return "admin"
        if _is_prec_dict(js):
            return "prec"
        return "unknown"

    # list → 첫 요소로 추정
    if isinstance(js, list) and js:
        first = js[0]
        if isinstance(first, dict):
            return detect_type(first)
        return "unknown"

    return "unknown"


# =========================
# 파일 처리기
# =========================
def process_file(path: Path) -> Tuple[str, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    t = detect_type(js)

    # 법령
    if t == "law":
        rows = list(law_iter_to_passages(js))
        return t, rows

    # 행정규칙
    if t == "admin":
        try:
            from .preprocess_admin_rule import admin_rule_iter_to_passages
        except Exception as e:
            raise RuntimeError(f"Failed to import preprocess_admin_rule: {e!r}")
        rows = admin_rule_iter_to_passages(js)
        return t, rows

    # 판례
    if t == "prec":
        try:
            from .preprocess_prec import prec_to_passages
        except Exception as e:
            raise RuntimeError(f"Failed to import preprocess_prec: {e!r}")
        rows = prec_to_passages(js)
        return t, rows

    return "unknown", []


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="입력 JSON 디렉토리 (재귀 탐색)")
    ap.add_argument("--out-law", default="data/processed/law_passages.jsonl")
    ap.add_argument("--out-admin", default="data/processed/admin_passages.jsonl")
    ap.add_argument("--out-prec", default="data/processed/prec_passages.jsonl")
    ap.add_argument("--append", action="store_true", help="출력 파일에 이어쓰기")
    ap.add_argument("--glob", default="**/*.json", help="파일 검색 패턴 (기본: **/*.json)")
    args = ap.parse_args()

    from .utils_io import write_jsonl, append_jsonl  # 지연 임포트

    p = Path(args.src_dir)
    files = sorted(p.glob(args.glob))
    n_law = n_admin = n_prec = n_unknown = 0
    n_law_empty = n_admin_empty = n_prec_empty = 0
    buf_law, buf_admin, buf_prec = [], [], []

    for fp in files:
        try:
            t, rows = process_file(fp)
            if t == "law":
                if rows:
                    n_law += 1
                    buf_law.extend(rows)
                else:
                    n_law_empty += 1
            elif t == "admin":
                if rows:
                    n_admin += 1
                    buf_admin.extend(rows)
                else:
                    n_admin_empty += 1
            elif t == "prec":
                if rows:
                    n_prec += 1
                    buf_prec.extend(rows)
                else:
                    n_prec_empty += 1
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

    if buf_prec:
        (append_jsonl if args.append else write_jsonl)(args.out_prec, buf_prec)
        print(f"[auto] prec passages: +{len(buf_prec)} → {args.out_prec}")

    print(f"[auto] files: law={n_law}(+{n_law_empty} empty), admin={n_admin}(+{n_admin_empty} empty), prec={n_prec}(+{n_prec_empty} empty), unknown={n_unknown}")

if __name__ == "__main__":
    main()
