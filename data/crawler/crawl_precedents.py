"""
판례 크롤러: law.go.kr에서 판례 데이터 수집

판결유형이 "판결", "판결 : 확정", "결정"인 판례만 필터링하여 수집합니다.
JSON API를 사용하여 빠르고 효율적으로 데이터를 수집합니다.
"""

from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from tqdm import tqdm


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """JSON 파일을 저장합니다"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 판결유형 필터링 기준
VALID_JUDGMENT_TYPES = ["판결", "판결 : 확정", "결정"]

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "https://www.law.go.kr/DRF/lawService.do"

# 요청 간 지연 시간 (초)
REQUEST_DELAY = 1.0


def _fetch_detail_worker(case_id: str, delay: float = REQUEST_DELAY) -> Optional[Dict[str, Any]]:
    """판례 상세 정보를 가져오는 워커 함수 (멀티프로세싱용)"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    })
    
    params = {
        'OC': 'hanwhasbank01',
        'target': 'prec',
        'ID': case_id,
        'type': 'JSON',
    }
    
    time.sleep(delay)
    try:
        response = session.get(DETAIL_URL, params=params, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        data = response.json()
        
        # JSON 구조에 따라 파싱
        if "판례" in data:
            prec = data["판례"]
        elif "PrecService" in data:
            prec = data["PrecService"]
        elif isinstance(data, dict) and "판례일련번호" in data:
            prec = data
        else:
            prec = data.get("prec", data.get("data", data))
        
        if not isinstance(prec, dict):
            return None
        
        # 판결유형 확인 및 필터링
        judgment_type = str(prec.get("판결유형") or prec.get("판결종류") or "").strip()
        if not judgment_type or judgment_type not in VALID_JUDGMENT_TYPES:
            return None
        
        # 결과 구성
        result = {
            '판례일련번호': str(prec.get("판례일련번호") or prec.get("ID") or case_id),
            '사건명': str(prec.get("사건명") or prec.get("title") or "").strip(),
            '사건번호': str(prec.get("사건번호") or prec.get("사건번호") or "").strip(),
            '법원명': str(prec.get("법원명") or prec.get("court") or "").strip(),
            '사건유형': str(prec.get("사건유형") or prec.get("case_type") or "").strip(),
            '판결유형': judgment_type,
            '선고일자': str(prec.get("선고일자") or prec.get("date") or "").strip(),
            '판시사항': str(prec.get("판시사항") or prec.get("headnote") or "").strip(),
            '판결요지': str(prec.get("판결요지") or prec.get("summary") or "").strip(),
            '판결본문': str(prec.get("판결본문") or prec.get("text") or prec.get("body") or "").strip(),
            '참조조문': str(prec.get("참조조문") or prec.get("ref_law") or "").strip(),
            '참조판례': str(prec.get("참조판례") or prec.get("ref_prec") or "").strip(),
        }
        
        return result
    except Exception as e:
        print(f"  경고: 판례 {case_id} 크롤링 실패: {e}")
        return None


class PrecedentCrawler:
    """판례 크롤러 클래스 (JSON API 사용)"""
    
    def __init__(self, output_dir: str, delay: float = REQUEST_DELAY, max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        })
        self.collected_ids = set()
        
    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """JSON API를 호출하여 데이터를 가져옵니다"""
        time.sleep(self.delay)
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            # JSON 응답 파싱
            try:
                return response.json()
            except ValueError:
                # JSON이 아닌 경우 텍스트로 반환 (에러 메시지 등)
                print(f"Warning: Non-JSON response from {url}")
                return {}
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            raise
    
    def _parse_search_result(self, data: Dict[str, Any]) -> tuple[int, List[Dict[str, str]]]:
        """검색 결과 JSON에서 총 개수와 판례 목록 추출"""
        precedents = []
        total_count = 0
        
        # JSON 구조에 따라 파싱
        # 일반적인 구조: { "total": 165261, "prec": [...] } 또는 { "검색결과": {...} }
        if "total" in data:
            total_count = int(data.get("total", 0))
            prec_list = data.get("prec", [])
        elif "검색결과" in data:
            search_result = data["검색결과"]
            total_count = int(search_result.get("총건수", search_result.get("total", 0)))
            prec_list = search_result.get("판례", search_result.get("prec", []))
        elif "PrecSearch" in data:
            prec_search = data["PrecSearch"]
            total_count = int(prec_search.get("totalCnt", prec_search.get("total", 0)))
            prec_list = prec_search.get("prec", [])
        else:
            # 직접 리스트인 경우
            if isinstance(data, list):
                prec_list = data
            else:
                # 다른 구조 시도
                prec_list = data.get("list", data.get("items", []))
        
        # 판례 목록 파싱
        for prec in prec_list:
            if not isinstance(prec, dict):
                continue
            
            # 판례 ID 추출
            case_id = str(prec.get("판례일련번호") or prec.get("ID") or prec.get("id") or "").strip()
            if not case_id:
                continue
            
            # 판결유형 확인 (사전 필터링)
            judgment_type = str(prec.get("판결유형") or prec.get("판결종류") or "").strip()
            if judgment_type and judgment_type not in VALID_JUDGMENT_TYPES:
                continue
            
            precedents.append({
                'case_id': case_id,
                'title': str(prec.get("사건명") or prec.get("title") or "").strip(),
                'court': str(prec.get("법원명") or prec.get("court") or "").strip(),
                'case_type': str(prec.get("사건유형") or prec.get("case_type") or "").strip(),
                'judgment_type': judgment_type,
                'date': str(prec.get("선고일자") or prec.get("date") or "").strip(),
            })
        
        return total_count, precedents
    
    def _parse_detail_data(self, data: Dict[str, Any], case_id: str) -> Optional[Dict[str, Any]]:
        """판례 상세 JSON 데이터에서 정보 추출"""
        # JSON 구조에 따라 파싱
        if "판례" in data:
            prec = data["판례"]
        elif "PrecService" in data:
            prec = data["PrecService"]
        elif isinstance(data, dict) and "판례일련번호" in data:
            prec = data
        else:
            # 다른 구조 시도
            prec = data.get("prec", data.get("data", data))
        
        if not isinstance(prec, dict):
            return None
        
        # 판결유형 확인 및 필터링
        judgment_type = str(prec.get("판결유형") or prec.get("판결종류") or "").strip()
        if not judgment_type or judgment_type not in VALID_JUDGMENT_TYPES:
            return None
        
        # 결과 구성
        result = {
            '판례일련번호': str(prec.get("판례일련번호") or prec.get("ID") or case_id),
            '사건명': str(prec.get("사건명") or prec.get("title") or "").strip(),
            '사건번호': str(prec.get("사건번호") or prec.get("사건번호") or "").strip(),
            '법원명': str(prec.get("법원명") or prec.get("court") or "").strip(),
            '사건유형': str(prec.get("사건유형") or prec.get("case_type") or "").strip(),
            '판결유형': judgment_type,
            '선고일자': str(prec.get("선고일자") or prec.get("date") or "").strip(),
            '판시사항': str(prec.get("판시사항") or prec.get("headnote") or "").strip(),
            '판결요지': str(prec.get("판결요지") or prec.get("summary") or "").strip(),
            '판결본문': str(prec.get("판결본문") or prec.get("text") or prec.get("body") or "").strip(),
            '참조조문': str(prec.get("참조조문") or prec.get("ref_law") or "").strip(),
            '참조판례': str(prec.get("참조판례") or prec.get("ref_prec") or "").strip(),
        }
        
        return result
    
    def crawl_search_page(self, page: int = 1, display: int = 20) -> tuple[int, List[Dict[str, str]]]:
        """검색 결과 페이지 크롤링 (JSON API)"""
        params = {
            'query': '*',
            'target': 'prec',
            'OC': 'hanwhasbank01',
            'search': '1',
            'display': str(display),
            'nw': '3',
            'page': str(page),
            'refAdr': 'law.go.kr',
            'type': 'JSON',  # HTML 대신 JSON 사용
            'popYn': 'N',
        }
        
        data = self._get_json(SEARCH_URL, params=params)
        total_count, precedents = self._parse_search_result(data)
        return total_count, precedents
    
    def crawl_detail(self, case_id: str) -> Optional[Dict[str, Any]]:
        """판례 상세 페이지 크롤링 (JSON API)"""
        params = {
            'OC': 'hanwhasbank01',
            'target': 'prec',
            'ID': case_id,
            'type': 'JSON',  # HTML 대신 JSON 사용
        }
        
        data = self._get_json(DETAIL_URL, params=params)
        detail = self._parse_detail_data(data, case_id)
        return detail
    
    def crawl(self, max_pages: Optional[int] = None, start_page: int = 1):
        """전체 크롤링 프로세스 실행"""
        print(f"[크롤러 시작] 출력 디렉토리: {self.output_dir}")
        print("JSON API를 사용하여 빠르게 데이터를 수집합니다.\n")
        
        # 첫 페이지로 총 개수 확인
        print("검색 결과 확인 중...")
        total_count, first_page_precedents = self.crawl_search_page(page=1, display=20)
        
        if total_count == 0 and first_page_precedents:
            # total_count를 찾지 못한 경우, 첫 페이지 결과로 추정
            total_count = len(first_page_precedents) * 100  # 대략적인 추정
        
        print(f"총 판례 수: {total_count:,}건 (추정)")
        
        if max_pages:
            total_pages = min(max_pages, (total_count + 19) // 20) if total_count > 0 else max_pages
        else:
            total_pages = (total_count + 19) // 20 if total_count > 0 else 1000  # 기본값
        
        print(f"크롤링할 페이지 수: {total_pages} (시작 페이지: {start_page})")
        
        collected_count = 0
        filtered_count = 0
        
        # 페이지별로 크롤링
        for page in range(start_page, start_page + total_pages):
            print(f"\n[페이지 {page}/{start_page + total_pages - 1}] 검색 결과 수집 중...")
            _, precedents = self.crawl_search_page(page=page, display=20)
            
            if not precedents:
                print(f"페이지 {page}에서 판례를 찾을 수 없습니다.")
                # 연속으로 빈 페이지가 나오면 중단
                if page > start_page:
                    break
                continue
            
            print(f"  → {len(precedents)}건의 판례 발견 (이미 필터링됨)")
            
            # 수집할 판례 ID 목록 준비 (중복 제거)
            case_ids_to_fetch = [
                prec['case_id'] for prec in precedents
                if prec['case_id'] not in self.collected_ids
            ]
            
            if not case_ids_to_fetch:
                continue
            
            # 멀티프로세싱으로 각 판례의 상세 정보 크롤링
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 작업 제출
                future_to_case_id = {
                    executor.submit(_fetch_detail_worker, case_id, self.delay): case_id
                    for case_id in case_ids_to_fetch
                }
                
                # 진행 상황 표시
                with tqdm(total=len(future_to_case_id), desc=f"  상세 정보 수집") as pbar:
                    for future in as_completed(future_to_case_id):
                        case_id = future_to_case_id[future]
                        pbar.update(1)
                        
                        try:
                            detail = future.result()
                            
                            if detail:
                                # 판결유형 재확인 (안전장치)
                                if detail.get('판결유형') not in VALID_JUDGMENT_TYPES:
                                    filtered_count += 1
                                    continue
                                
                                # JSON 파일로 저장
                                output_file = self.output_dir / f"{case_id}.json"
                                write_json(output_file, detail)
                                
                                self.collected_ids.add(case_id)
                                collected_count += 1
                                
                                if collected_count % 10 == 0:
                                    print(f"    → 현재까지 수집: {collected_count}건 (필터링: {filtered_count}건)")
                            else:
                                filtered_count += 1
                        except Exception as e:
                            print(f"  경고: 판례 {case_id} 처리 중 오류: {e}")
                            filtered_count += 1
        
        print(f"\n[크롤링 완료]")
        print(f"  수집된 판례: {collected_count}건")
        print(f"  필터링된 판례: {filtered_count}건")
        print(f"  저장 위치: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="law.go.kr에서 판례 데이터 크롤링 (JSON API 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python crawl_precedents.py --output data/precedents
  python crawl_precedents.py --output data/precedents --max-pages 10
  python crawl_precedents.py --output data/precedents --start-page 5 --max-pages 20
        """
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/precedents',
        help='판례 JSON 파일 저장 디렉토리 (기본값: data/precedents)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='크롤링할 최대 페이지 수 (기본값: 전체)'
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=1,
        help='시작 페이지 번호 (기본값: 1)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=REQUEST_DELAY,
        help=f'요청 간 지연 시간(초) (기본값: {REQUEST_DELAY})'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='병렬 처리 워커 수 (기본값: 4)'
    )
    
    args = parser.parse_args()
    
    crawler = PrecedentCrawler(args.output, delay=args.delay, max_workers=args.max_workers)
    crawler.crawl(max_pages=args.max_pages, start_page=args.start_page)


if __name__ == "__main__":
    main()
