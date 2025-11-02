#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supreme Court Portal (Scourt) crawler scaffold for RAG datasets.
- Searches for a keyword (e.g., "저축") in case law.
- Extracts search results (title, case number, court, judgment date, URL, summary).
- Follows detail pages to fetch full text when available.
- Saves raw cases and chunked passages (JSONL/CSV).

NOTE:
- The portal markup may change. Adjust CSS selectors in SELECTORS below.
- For headless/dynamic content, swap requests+BS4 to Selenium/Playwright.
- Respect robots.txt and site terms. Use small max-pages and polite delays.
"""

import os, re, csv, time, json, math, html, argparse, logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Tuple

import requests
from bs4 import BeautifulSoup

# ----------------------------- Config -----------------------------
DEFAULT_BASE = "https://portal.scourt.go.kr/pgp/index.on"
SEARCH_PATH = ""  # search is handled by the same endpoint with query params

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScourtRAGBot/1.0; +https://example.com/bot)",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}
# Polite crawl config
REQUEST_TIMEOUT = 15
RETRY = 3
SLEEP_BETWEEN = (0.8, 1.6)  # seconds (min, max)

# CSS/selectors. If the portal updates, adjust here.
SELECTORS = {
    # Search result list page
    "result_item": "ul.result > li, div.result_list > ul > li, .resultList > li",
    "title": "a",  # link text
    "url": "a",
    "summary": ".summary, .txt, .cont, p",
    # Attempt to find meta fields on result card if present
    "meta": ".meta, .info, .detail",
    # Detail page content
    "detail_body": ".contArea, .con, .content, #content, .txt",
    "detail_table": "table, .tbl, .tb",  # sometimes meta sits in tables
}

# ----------------------------- Data Types -----------------------------
@dataclass
class CaseDoc:
    id: str
    source: str
    title: str
    case_number: Optional[str]
    court_name: Optional[str]
    judgment_date: Optional[str]  # YYYY-MM-DD preferred
    url: str
    summary: Optional[str]
    body: Optional[str]
    tags: List[str]
    type: str = "판례"

# ----------------------------- Utils -----------------------------
def sleep_polite():
    import random
    time.sleep(random.uniform(*SLEEP_BETWEEN))

def request_url(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    last_exc = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            else:
                logging.warning("Non-200 status %s for %s", resp.status_code, resp.url)
        except Exception as e:
            last_exc = e
            logging.warning("Request error (attempt %d/%d): %s", attempt, RETRY, e)
        sleep_polite()
    if last_exc:
        logging.error("Failed to fetch %s after %d retries: %s", url, RETRY, last_exc)
    return None

def normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # Examples: "2015.12.03", "2015-12-03", "2015년 12월 3일"
    s = re.sub(r"[년./\- ]+", "-", s)
    s = s.strip("-")
    parts = s.split("-")
    nums = []
    for p in parts:
        p = re.sub(r"\D", "", p)
        if not p:
            continue
        nums.append(p.zfill(2))
    if not nums:
        return None
    if len(nums) == 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"
    if len(nums) == 2:
        return f"{nums[0]}-{nums[1]}-01"
    if len(nums) == 1:
        return f"{nums[0]}-01-01"
    return None

def extract_text(el) -> str:
    if not el:
        return ""
    txt = el.get_text(" ", strip=True)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def chunk_text(text: str, target_tokens: int = 600, overlap_tokens: int = 100) -> List[str]:
    """Heuristic chunker by sentence; token ≈ word pieces (rough)."""
    if not text:
        return []
    # Split by sentences (rough)
    sents = re.split(r"(?<=[.!?。？！]|다\.)\s+", text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        tok = max(1, len(s) // 4)  # rough tokenization heuristic
        if cur_len + tok > target_tokens and cur:
            chunks.append(" ".join(cur).strip())
            # overlap
            if overlap_tokens > 0:
                overlap_text = " ".join(cur)[-overlap_tokens*4:]
                cur = [overlap_text]
                cur_len = len(overlap_text) // 4
            else:
                cur = []
                cur_len = 0
        cur.append(s)
        cur_len += tok
    if cur:
        chunks.append(" ".join(cur).strip())
    # filter very short
    chunks = [c for c in chunks if len(c) > 50]
    return chunks

def guess_meta_from_text(text: str) -> Dict[str, Optional[str]]:
    # Try to pull case number, court, date from blocks of text
    cn = None
    court = None
    date = None
    # Case number patterns example: 2013다12345, 2013노123, 2013두1234
    m = re.search(r"(20\d{2}|19\d{2})\s*[가-힣A-Za-z]\s*\d{1,6}", text.replace(" ", ""))
    if m:
        cn = m.group(0)
    # Court name rough
    for kw in ["대법원", "서울고등법원", "서울중앙지방법원", "고등법원", "지방법원", "법원"]:
        if kw in text:
            court = kw
            break
    # Date
    dm = re.search(r"(20\d{2}|19\d{2})[.\-년 ]+\d{1,2}[.\-월 ]+\d{1,2}", text)
    if dm:
        date = normalize_date(dm.group(0))
    return {"case_number": cn, "court_name": court, "judgment_date": date}

# ----------------------------- Parsing -----------------------------
def parse_search_results(html_text: str) -> List[Dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    items = soup.select(SELECTORS["result_item"])
    results = []
    for it in items:
        a = it.select_one(SELECTORS["url"])
        title = extract_text(a) if a else ""
        href = a.get("href") if a and a.has_attr("href") else None
        url = None
        if href and not href.startswith("http"):
            url = DEFAULT_BASE + href if href.startswith("?") else DEFAULT_BASE + "?" + href
        else:
            url = href
        summary_el = it.select_one(SELECTORS["summary"])
        summary = extract_text(summary_el) if summary_el else ""
        meta_el = it.select_one(SELECTORS["meta"])
        meta_txt = extract_text(meta_el) if meta_el else ""
        results.append({
            "title": title,
            "url": url,
            "summary": summary,
            "meta_text": meta_txt,
        })
    return results

def parse_detail_page(html_text: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    body_el = soup.select_one(SELECTORS["detail_body"])
    # Fallback: collect paragraphs
    if not body_el:
        paragraphs = soup.select("p")
        body_text = "\n".join(extract_text(p) for p in paragraphs)
    else:
        body_text = extract_text(body_el)
    # Also scan tables for meta
    meta_blocks = []
    for tbl in soup.select(SELECTORS["detail_table"]):
        meta_blocks.append(extract_text(tbl))
    meta_text = " ".join(meta_blocks)
    return {"body": body_text, "meta_text": meta_text}

# ----------------------------- Crawler -----------------------------
def build_search_params(keyword: str, page: int) -> Dict[str, str]:
    """
    The portal uses query parameters. If these change, adjust here.
    Common params:
    - m=PGP1021M01 (case search)
    - l=N
    - c=900
    - page or pageIndex like params for pagination
    - srchWord or searchTxt for keyword
    """
    return {
        "m": "PGP1021M01",
        "l": "N",
        "c": "900",
        "pageIndex": str(page),
        "srchWord": keyword,
    }

def crawl(keyword: str, max_pages: int = 3, out_dir: str = "output") -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_scourt_cases.jsonl")
    passages_path = os.path.join(out_dir, "scourt_passages.jsonl")
    csv_path = os.path.join(out_dir, "scourt_cases.csv")

    raw_f = open(raw_path, "w", encoding="utf-8")
    passages_f = open(passages_path, "w", encoding="utf-8")
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=[
        "id","source","title","case_number","court_name","judgment_date","url","summary","type"
    ])
    csv_writer.writeheader()

    seen_urls = set()
    doc_count = 0

    for page in range(1, max_pages + 1):
        params = build_search_params(keyword, page)
        resp = request_url(DEFAULT_BASE, params=params)
        if not resp:
            continue
        results = parse_search_results(resp.text)
        if not results:
            # If no items found, break early
            if page == 1:
                logging.warning("No results parsed. Consider updating SELECTORS or params.")
            break

        for r in results:
            url = r.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # Fetch detail page
            sleep_polite()
            detail_resp = request_url(url)
            body = None
            detail_meta = ""
            if detail_resp:
                parsed = parse_detail_page(detail_resp.text)
                body = parsed.get("body")
                detail_meta = parsed.get("meta_text") or ""
            # Guess meta if missing
            meta_text = " ".join(filter(None, [r.get("meta_text",""), detail_meta]))
            meta_guess = guess_meta_from_text(" ".join([r.get("summary",""), meta_text, body or ""]))

            # Assemble document
            doc_id = f"SCOURT_{hash(url) & 0xffffffff:08x}"
            doc = {
                "id": doc_id,
                "source": "SupremeCourt",
                "title": r.get("title") or "",
                "case_number": meta_guess.get("case_number"),
                "court_name": meta_guess.get("court_name"),
                "judgment_date": meta_guess.get("judgment_date"),
                "url": url,
                "summary": r.get("summary") or "",
                "body": body,
                "tags": ["저축", "판례"],
                "type": "판례",
            }

            raw_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            csv_writer.writerow({
                "id": doc["id"],
                "source": doc["source"],
                "title": doc["title"],
                "case_number": doc["case_number"] or "",
                "court_name": doc["court_name"] or "",
                "judgment_date": doc["judgment_date"] or "",
                "url": doc["url"],
                "summary": doc["summary"],
                "type": doc["type"],
            })

            # Chunk passages for RAG
            text_for_chunk = (doc.get("body") or "").strip()
            if not text_for_chunk:
                # Fall back to summary if body missing
                text_for_chunk = doc.get("summary") or ""
            chunks = chunk_text(text_for_chunk, target_tokens=600, overlap_tokens=100)
            for idx, ch in enumerate(chunks, start=1):
                pid = f"{doc_id}_{idx}"
                passage = {
                    "id": pid,
                    "parent_id": doc_id,
                    "source": "SupremeCourt",
                    "title": doc["title"],
                    "case_number": doc["case_number"],
                    "court_name": doc["court_name"],
                    "judgment_date": doc["judgment_date"],
                    "url": doc["url"],
                    "text": ch,
                    "tags": ["저축", "판례"],
                    "type": "판례",
                }
                passages_f.write(json.dumps(passage, ensure_ascii=False) + "\n")

            doc_count += 1

        # Optional: stop early if few results on a page
        sleep_polite()

    raw_f.close()
    passages_f.close()
    csv_f.close()
    return raw_path, passages_path, csv_path

def main():
    parser = argparse.ArgumentParser(description="Scourt portal crawler for RAG")
    parser.add_argument("--keyword", type=str, required=True, help="검색 키워드 (e.g., 저축)")
    parser.add_argument("--max-pages", type=int, default=3, help="검색 결과 페이지 제한")
    parser.add_argument("--out-dir", type=str, default="output", help="출력 폴더")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    raw_path, passages_path, csv_path = crawl(args.keyword, args.max_pages, args.out_dir)
    print("Saved:")
    print(" -", raw_path)
    print(" -", passages_path)
    print(" -", csv_path)

if __name__ == "__main__":
    main()
