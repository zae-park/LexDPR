#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Law.go.kr Open API ingestion for RAG
- Fetches "판례(prec)" and/or "법령(law)" by keyword (e.g., "저축")
- Builds passage-level JSONL suitable for retrievers
- Optionally creates naive query ↔ passage pairs from summaries/headnotes

Requirements:
  pip install requests lxml

How to run:
  python law_openapi_ingest.py --service-key <YOUR_KEY> --keyword 저축 --fetch prec law --max 200 --out-dir output_openapi

Docs:
- Guide list: https://open.law.go.kr/LSO/openApi/guideList.do
- 판례 목록:   http(s)://www.law.go.kr/DRF/lawSearch.do?target=prec&...
- 판례 본문:   http(s)://www.law.go.kr/DRF/lawService.do?target=prec&...
- 법령 목록:   http(s)://www.law.go.kr/DRF/lawSearch.do?target=law&...
- 법령 본문:   http(s)://www.law.go.kr/DRF/lawService.do?target=law&...
"""
import os, re, json, csv, time, argparse
from typing import List, Dict, Optional, Tuple
import requests
from lxml import etree

BASE = "https://www.law.go.kr/DRF"
SEARCH = BASE + "/lawSearch.do"
SERVICE = BASE + "/lawService.do"

def fetch_xml(url: str, params: Dict[str, str]) -> etree._Element:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return etree.fromstring(r.content)

def get_prec_list(key: str, keyword: str, page: int=1, display: int=100) -> etree._Element:
    params = {
        "OC": key,            # Open API Key (id in some docs; OC commonly used)
        "target": "prec",
        "type": "XML",
        "display": str(display),
        "page": str(page),
        "query": keyword
    }
    return fetch_xml(SEARCH, params)

def get_prec_detail(key: str, id_num: str) -> etree._Element:
    params = {
        "OC": key,
        "target": "prec",
        "type": "XML",
        "ID": id_num
    }
    return fetch_xml(SERVICE, params)

def get_law_list(key: str, keyword: str, page: int=1, display: int=100) -> etree._Element:
    params = {
        "OC": key,
        "target": "law",
        "type": "XML",
        "display": str(display),
        "page": str(page),
        "query": keyword
    }
    return fetch_xml(SEARCH, params)

def get_law_detail(key: str, law_id: str) -> etree._Element:
    params = {
        "OC": key,
        "target": "law",
        "type": "XML",
        "ID": law_id
    }
    return fetch_xml(SERVICE, params)

def xml_text(node, tag: str) -> Optional[str]:
    el = node.find(tag)
    return el.text.strip() if el is not None and el.text else None

def chunk(text: str, max_len: int=1200, overlap: int=200) -> List[str]:
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_len)
        seg = text[i:j]
        out.append(seg)
        if j == n:
            break
        i = max(0, j - overlap)
    # filter short
    return [s for s in out if len(s) >= 100]

def save_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def prec_to_passages(key: str, keyword: str, max_items: int, out_dir: str) -> str:
    # paginate search
    collected = []
    page = 1
    display = 100
    while len(collected) < max_items:
        root = get_prec_list(key, keyword, page=page, display=display)
        items = root.findall(".//prec")
        if not items:
            break
        for it in items:
            pid = xml_text(it, "판례일련번호") or xml_text(it, "판례상세링크")
            if not pid: 
                continue
            collected.append({
                "판례일련번호": pid,
                "사건명": xml_text(it, "사건명"),
                "사건번호": xml_text(it, "사건번호"),
                "선고일자": xml_text(it, "선고일자"),
                "법원명": xml_text(it, "법원명"),
            })
            if len(collected) >= max_items:
                break
        page += 1
    passages = []
    for meta in collected:
        try:
            detail = get_prec_detail(key, meta["판례일련번호"])
        except Exception as e:
            continue
        # Extract fields (tags per API schema)
        title = (xml_text(detail, "사건명") or "").strip()
        case_no = xml_text(detail, "사건번호")
        date = xml_text(detail, "선고일자")
        court = xml_text(detail, "선고법원")
        headnote = xml_text(detail, "판시사항") or ""
        summary = xml_text(detail, "판결요지") or ""
        body = xml_text(detail, "판결본문") or ""
        base_id = f"PREC_{meta['판례일련번호']}"
        # chunk body first; if empty, fallback to headnote/summary
        source_text = body if body else (headnote + "\n" + summary)
        chunks = chunk(source_text, max_len=1200, overlap=200)
        for idx, ch in enumerate(chunks, 1):
            passages.append({
                "id": f"{base_id}_{idx}",
                "parent_id": base_id,
                "source": "NLIC(국가법령정보)",
                "type": "판례",
                "title": title,
                "case_number": case_no,
                "court_name": court,
                "judgment_date": date,
                "text": ch,
                "headnote": headnote[:1000],
                "summary": summary[:1000],
                "tags": [keyword, "판례"]
            })
    out_path = os.path.join(out_dir, "prec_passages.jsonl")
    save_jsonl(out_path, passages)
    return out_path

def law_to_passages(key: str, keyword: str, max_items: int, out_dir: str) -> str:
    collected = []
    page = 1
    display = 100
    while len(collected) < max_items:
        root = get_law_list(key, keyword, page=page, display=display)
        items = root.findall(".//법령")
        if not items:
            break
        for it in items:
            law_id = xml_text(it, "법령ID")
            if not law_id:
                continue
            collected.append({
                "법령ID": law_id,
                "법령명": xml_text(it, "법령명한글"),
                "시행일자": xml_text(it, "시행일자"),
                "공포일자": xml_text(it, "공포일자"),
            })
            if len(collected) >= max_items:
                break
        page += 1

    passages = []
    for meta in collected:
        try:
            detail = get_law_detail(key, meta["법령ID"])
        except Exception:
            continue
        law_name = xml_text(detail, "법령명한글") or xml_text(detail, "법령명")
        eff = xml_text(detail, "시행일자")
        # Iterate 조문
        for jomun in detail.findall(".//조문"):
            art_no = xml_text(jomun, "조문번호")
            art_title = xml_text(jomun, "조문제목")
            art_body = (xml_text(jomun, "조문내용") or "").strip()
            if not art_body:
                continue
            chunks = chunk(art_body, max_len=800, overlap=120)
            base_id = f"LAW_{meta['법령ID']}_{art_no or 'NA'}"
            for idx, ch in enumerate(chunks, 1):
                passages.append({
                    "id": f"{base_id}_{idx}",
                    "parent_id": base_id,
                    "source": "NLIC(국가법령정보)",
                    "type": "법령",
                    "law_id": meta["법령ID"],
                    "law_name": law_name,
                    "article": art_no,
                    "title": art_title,
                    "effective_date": eff,
                    "text": ch,
                    "tags": [keyword, "법령"]
                })
    out_path = os.path.join(out_dir, "law_passages.jsonl")
    save_jsonl(out_path, passages)
    return out_path

def build_naive_pairs(prec_path: str, law_path: Optional[str], out_dir: str) -> str:
    # Very simple pair creation: make a query from headnote/summary and map to its own passage.
    pairs = []
    def iter_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    if prec_path and os.path.exists(prec_path):
        for p in iter_jsonl(prec_path):
            q = (p.get("headnote") or p.get("summary") or "")[:200]
            if len(q) < 20:
                continue
            pairs.append({
                "query_id": f"Q_{p['id']}",
                "query_text": q,
                "positive_passages": [p["id"]],
                "hard_negatives": []
            })
    # Optionally include law queries from article titles
    if law_path and os.path.exists(law_path):
        for p in iter_jsonl(law_path):
            title = p.get("title") or ""
            if title and len(title) >= 6:
                pairs.append({
                    "query_id": f"Q_{p['id']}",
                    "query_text": f"{p.get('law_name','')} {title}는 무엇을 규정하나요?",
                    "positive_passages": [p["id"]],
                    "hard_negatives": []
                })
    out_path = os.path.join(out_dir, "pairs_bootstrap.jsonl")
    save_jsonl(out_path, pairs)
    return out_path

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-key", required=True, help="국가법령정보 Open API 키(OC)")
    ap.add_argument("--keyword", required=True, help="검색 키워드 (예: 저축)")
    ap.add_argument("--fetch", nargs="+", default=["prec"], choices=["prec","law"], help="가져올 대상")
    ap.add_argument("--max", type=int, default=200, help="최대 문서 수 (목록 기준)")
    ap.add_argument("--out-dir", default="output_openapi")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prec_path = law_path = None

    if "prec" in args.fetch:
        prec_path = prec_to_passages(args.service_key, args.keyword, args.max, args.out_dir)
        print("Saved prec passages:", prec_path)
    if "law" in args.fetch:
        law_path = law_to_passages(args.service_key, args.keyword, args.max, args.out_dir)
        print("Saved law passages:", law_path)

    pairs_path = build_naive_pairs(prec_path, law_path, args.out_dir)
    print("Saved naive pairs:", pairs_path)

if __name__ == "__main__":
    main()
