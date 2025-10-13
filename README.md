# 🏛️ LexDPR  
**구조화되고 계층적인 법령 및 규범 문서를 위한 Dense Passage Retrieval 모델**

LexDPR은 **법령, 규정, 비조치의견서 등과 같은 구조화된 문서**를 대상으로 하는 **Dense Passage Retrieval (DPR)** 모델입니다.  
조·항·호 단위의 계층적 구조를 가진 문서를 효율적으로 인덱싱하고, 의미적 일관성을 유지하며 검색 성능을 향상시키는 것을 목표로 합니다.

---

## 📘 프로젝트 개요

기존의 상용 임베딩 모델(OpenAI, Cohere, Sentence-Transformers 등)은 다음과 같은 문제를 가집니다:

- **계층 구조**가 깊은 문서(조/항/호 등)에 대한 표현 부족  
- **법령 문맥 의존성**이 높은 구문 처리의 불안정성  
- **의미적으로 연결된 문장 간 거리 문제**로 인한 검색 정확도 저하  

LexDPR은 **법령 문서 구조에 최적화된 Dense Passage Retrieval 파이프라인**으로, **RAG 시스템의 중간 검색기(retriever)** 역할을 수행합니다.

---

## ⚙️ 아키텍처 개요

LexDPR은 구조 인식형(Dense Passage Retrieval with structure-awareness) **듀얼 인코더(Dual Encoder)** 모델입니다.

```
Query Encoder (Sentence-BERT / Legal-BERT)
     │
     ▼
Query Vector
     │
     ├──> Passage Encoder (조문/항 단위)
     │         └─ 구조적 단위별 컨텍스트 윈도우 처리
     │
     ▼
Similarity Scoring (dot product / cosine)
     │
  Top-k 구조 단위 검색 결과 출력
```

LexDPR은 RAG 파이프라인의 생성기(generator)와 독립적으로 동작하며, **retriever 계층에만 집중**합니다.

---

## 🧩 프로젝트 구조

```
📁 LexDPR/
 ├── data/
 │    ├── statutes/             # 법령 문서 (조/항/호 단위)
 │    ├── no_action_letters/    # 비조치의견서 데이터
 │    ├── queries/              # 질의 데이터 (JSONL)
 │    └── processed/            # 전처리 후 corpus.jsonl
 │
 ├── scripts/
 │    ├── preprocess_acts.py    # 법령 문서 전처리 및 청크 생성
 │    ├── encode_passages.py    # 패시지 임베딩 생성
 │    ├── encode_queries.py     # 질의 임베딩 생성
 │    ├── build_index.py        # FAISS 인덱스 구축
 │    └── evaluate.py           # 평가 (Recall@k 등)
 │
 ├── configs/
 │    ├── base.yaml             # 기본 하이퍼파라미터 설정
 │    ├── model.yaml            # 인코더 아키텍처 설정
 │    └── data.yaml             # 데이터 경로 및 전처리 옵션
 │
 ├── run_demo_real.sh           # 전체 파이프라인 실행 스크립트
 ├── README.md
 └── requirements.txt
```

---

## 🔍 주요 기능

- **실제 Sentence-Transformers 기반 DPR 인코더 사용**  
  `sentence-transformers/all-MiniLM-L6-v2` 기반 인코딩 (필요 시 Legal-BERT 교체 가능)
- **구조 인식형 청크 분할**  
  조문·항 단위의 계층 구조를 분석해 문맥 단위별로 세분화
- **FAISS 기반 벡터 인덱싱**  
  대용량 법령 데이터의 효율적 검색
- **Recall@k 평가 자동화**  
  `data/queries/queries.jsonl`의 positive passage 기준으로 정확도 평가
- **간단한 실행**  
  `run_demo_real.sh` 하나로 전처리→임베딩→인덱스→평가 일괄 수행

---

## 🧠 활용 분야

- 법령 및 규제 문서 검색 시스템  
- 비조치의견서 / 행정해석 질의응답 검색  
- 규제 준수(Compliance) 자동화 도구  
- 계약서 / 정책 / 규정 기반 QA RAG 시스템  

---

## 🚀 사용 예시

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 전체 파이프라인 실행 (모델 가중치 자동 다운로드)
bash run_demo_real.sh
```

또는 수동 실행:

```bash
# 전처리
python scripts/preprocess_acts.py --input data/statutes --output data/processed/corpus.jsonl
python scripts/preprocess_acts.py --input data/no_action_letters --output data/processed/tmp.jsonl
cat data/processed/tmp.jsonl >> data/processed/corpus.jsonl && rm data/processed/tmp.jsonl

# 임베딩 생성
python scripts/encode_passages.py --input data/processed/corpus.jsonl --outdir checkpoint
python scripts/encode_queries.py --queries data/queries/queries.jsonl --outdir checkpoint

# 인덱스 빌드 및 평가
python scripts/build_index.py --input checkpoint --output index --factory Flat --metric dot
python scripts/evaluate.py --index_dir index --queries data/queries/queries.jsonl --top_k 10
```

---

## 📄 인용 정보

```
@misc{lexdpr2025,
  author = {박성재},
  title  = {LexDPR: 구조화된 법령 문서를 위한 Dense Passage Retrieval 모델},
  year   = {2025},
  url    = {https://github.com/zae-park/LexDPR}
}
```

---

## 🧾 라이선스

MIT License  
공공데이터(예: 비조치의견서, 법령 DB)는 각 출처의 오픈라이선스 정책을 반드시 준수해야 합니다.
