# lex_dpr/data_processing/preprocess_prec.py
from __future__ import annotations
import argparse, re
from typing import Dict, Any, List, Iterable, Union
from .utils_io import read_json, write_jsonl
from .filters import normalize_whitespace

DEFAULT_SOURCE = "NLIC(국가법령정보)"
MIN_LEN_DEFAULT = 50
CHUNK_MAX_DEFAULT = 1200
CHUNK_OVERLAP_DEFAULT = 200

def _chunk(text: str, max_len: int, overlap: int) -> List[str]:
    """단순 길이 기반 슬라이딩 윈도우 청크"""
    if not text:
        return []
    s = re.sub(r"\s+", " ", text).strip()
    n = len(s)
    if n == 0:
        return []
    out: List[str] = []
    i = 0
    while i < n:
        j = min(n, i + max_len)
        seg = s[i:j]
        out.append(seg)
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def _dedup(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _short(s: str, n: int) -> str:
    s = (s or "").strip()
    return s[:n]

def _as_list(obj: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return obj if isinstance(obj, list) else [obj]

def prec_to_passages_one(
    js: Dict[str, Any],
    min_len: int,
    chunk_max: int,
    chunk_overlap: int,
    prefer_body_first: bool = True,
) -> List[Dict[str, Any]]:
    """
    한 건의 판례 JSON을 passage 리스트로 변환.
    예상 키:
      - 사건명/사건번호/법원명/선고일자/판시사항/판결요지/판결본문
      - 또는 title/case_number/court_name/judgment_date/headnote/summary/text
    """
    case_id = str(js.get("판례일련번호") or js.get("case_id") or js.get("id") or "000000").zfill(6)
    base_id = f"PREC_{case_id}"

    title = (js.get("사건명") or js.get("title") or "") or ""
    case_no = (js.get("사건번호") or js.get("case_number") or "") or ""
    court = (js.get("법원명") or js.get("court_name") or "") or ""
    jdate = (js.get("선고일자") or js.get("judgment_date") or "") or ""

    headnote = (js.get("판시사항") or js.get("headnote") or "") or ""
    summary  = (js.get("판결요지") or js.get("summary") or "") or ""
    body     = (js.get("판결본문") or js.get("text") or "") or ""

    source   = (js.get("source") or DEFAULT_SOURCE)

    # 본문 우선 청크, 없으면 판시사항+요지 묶어서 청크
    chunks: List[str] = []
    if prefer_body_first and body:
        chunks = _chunk(normalize_whitespace(body), chunk_max, chunk_overlap)
    if not chunks:
        alt = normalize_whitespace(f"{headnote}\n{summary}".strip())
        if len(alt) >= min_len:
            chunks = _chunk(alt, chunk_max, chunk_overlap)

    # title/headnote/summary도 passage로 포함하고 싶다면 아래 활성화(선택)
    # for seg in [title, headnote, summary]:
    #     t = normalize_whitespace(seg or "")
    #     if len(t) >= min_len:
    #         chunks.append(t)

    # 정규화/길이 필터/중복 제거
    normed = [normalize_whitespace(c) for c in chunks if len(normalize_whitespace(c)) >= min_len]
    uniq = _dedup(normed)

    out: List[Dict[str, Any]] = []
    for i, t in enumerate(uniq, 1):
        row = {
            "id": f"{base_id}_{i}",
            "parent_id": base_id,
            "source": source,
            "type": "판례",
            "title": title,
            "case_number": case_no,
            "court_name": court,
            "judgment_date": jdate,
            "text": t,
            # 메타 보존(최대 1,000자)
            "headnote": _short(headnote, 1000),
            "summary": _short(summary, 1000),
            "tags": ["판례"]
        }
        out.append(row)
    return out

def prec_to_passages(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    min_len: int = MIN_LEN_DEFAULT,
    chunk_max: int = CHUNK_MAX_DEFAULT,
    chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    prefer_body_first: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for js in _as_list(data):
        rows.extend(prec_to_passages_one(js, min_len, chunk_max, chunk_overlap, prefer_body_first))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="판례 원본 JSON 경로(단일 객체 또는 배열)")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로(prec_passages.jsonl)")
    ap.add_argument("--min_len", type=int, default=MIN_LEN_DEFAULT, help=f"passage 최소 길이 (기본 {MIN_LEN_DEFAULT})")
    ap.add_argument("--chunk_max", type=int, default=CHUNK_MAX_DEFAULT, help=f"청크 최대 길이 (기본 {CHUNK_MAX_DEFAULT})")
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP_DEFAULT, help=f"청크 오버랩 (기본 {CHUNK_OVERLAP_DEFAULT})")
    ap.add_argument("--no_prefer_body", action="store_true", help="본문 우선 분할 비활성화(요지/판시사항 우선)")
    args = ap.parse_args()

    data = read_json(args.src)
    rows = prec_to_passages(
        data=data,
        min_len=args.min_len,
        chunk_max=args.chunk_max,
        chunk_overlap=args.chunk_overlap,
        prefer_body_first=(not args.no_prefer_body),
    )
    write_jsonl(args.out, rows)
    print(f"[preprocess_prec] passages: {len(rows)} → {args.out}")

if __name__ == "__main__":
    main()
