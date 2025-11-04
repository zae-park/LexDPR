from __future__ import annotations
import argparse
from typing import Dict, Any, Iterator, List
from .utils_io import read_json, write_jsonl
from .filters import normalize_whitespace

MIN_LEN = 50

def prec_to_passages(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    판례 JSON을 간단히 분할:
    - title/headnote/summary/body류 필드를 추출하여 n개 passage 생성
    예상 키: 사건명/판시사항/판결요지/판결본문/사건번호/법원명/선고일자 등
    """
    case_id = str(js.get("판례일련번호") or js.get("case_id") or js.get("id") or "000000").zfill(6)
    case_no = js.get("사건번호") or js.get("case_number") or ""
    court = js.get("법원명") or js.get("court_name") or ""
    jdate = js.get("선고일자") or js.get("judgment_date") or ""
    title = js.get("사건명") or js.get("title") or ""
    headnote = js.get("판시사항") or js.get("headnote") or ""
    summary = js.get("판결요지") or js.get("summary") or ""
    body = js.get("판결본문") or js.get("text") or ""

    chunks = []
    for seg in [title, headnote, summary, body]:
        t = normalize_whitespace(seg or "")
        if len(t) >= MIN_LEN:
            chunks.append(t)
    # 중복 제거
    uniq, seen = [], set()
    for t in chunks:
        if t not in seen:
            uniq.append(t); seen.add(t)

    out = []
    for i, t in enumerate(uniq, 1):
        out.append({
            "id": f"PREC_{case_id}_{i}",
            "parent_id": f"PREC_{case_id}",
            "type": "판례",
            "case_number": case_no,
            "court_name": court,
            "judgment_date": jdate,
            "title": title,
            "text": t,
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="판례 JSON 파일")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    args = ap.parse_args()

    js = read_json(args.src)
    rows = prec_to_passages(js)
    write_jsonl(args.out, rows)
    print(f"[preprocess_prec] passages: {len(rows)} → {args.out}")

if __name__ == "__main__":
    main()
