# LexDPR TODO List

이 문서는 LexDPR 프로젝트의 진행 상황을 추적하고, 단계별 작업을 세밀하게 관리하기 위한 TODO 리스트입니다.

## 진행 상태 표기

- ✅ **완료**: 구현 완료 및 검증됨
- 🚧 **진행 중**: 현재 작업 중
- ⏸️ **보류**: 일시 중단
- 📋 **계획**: 계획만 수립됨
- ❌ **취소**: 더 이상 필요 없음

---

## 1. 프로젝트 설정 및 인프라

### 1.1 프로젝트 구조
- ✅ 프로젝트 디렉토리 구조 설계 및 생성
- ✅ Poetry 기반 의존성 관리 설정 (`pyproject.toml`)
- ✅ 모듈 구조 설계 (`lex_dpr/` 패키지)
- ✅ 설정 파일 구조 (`configs/` 디렉토리)

### 1.2 의존성 관리
- ✅ 핵심 의존성 정의 (sentence-transformers, transformers, torch 등)
- ✅ 선택적 의존성 관리 (Poetry extras: wandb, mlflow)
- ✅ 개발 의존성 분리 (dev group)
- 📋 의존성 버전 고정 및 호환성 검증

### 1.3 환경 설정
- ✅ 설정 파일 시스템 (OmegaConf 기반)
- ✅ 설정 초기화 CLI (`lex-dpr config init`)
- ✅ 설정 확인 CLI (`lex-dpr config show`)
- ✅ 설정 오버라이드 지원 (명령줄 인자)
- 📋 환경 변수 기반 설정 지원

### 1.4 로깅 시스템
- ✅ 표준 logging 모듈 통합
- ✅ 로그 레벨 설정
- ✅ 웹 로깅 지원 (WandB, MLflow)
- ✅ 다중 웹 로깅 서비스 동시 사용
- 📋 로그 파일 저장 기능
- 📋 구조화된 로그 포맷 (JSON)

---

## 2. 데이터 처리 파이프라인

### 2.1 데이터 전처리
- ✅ 법령 데이터 전처리 (`preprocess_law.py`)
- ✅ 행정규칙 데이터 전처리 (`preprocess_admin_rule.py`)
- ✅ 판례 데이터 전처리 (`preprocess_prec.py`)
- ✅ 자동 전처리 파이프라인 (`preprocess_auto.py`)
- 📋 데이터 검증 및 품질 체크 강화
- 📋 전처리 통계 리포트 생성

### 2.2 Passage 생성
- ✅ 법령 Passage 생성 (`law_passages.jsonl`)
- ✅ 행정규칙 Passage 생성 (`admin_passages.jsonl`)
- ✅ 판례 Passage 생성 (`prec_passages.jsonl`)
- ✅ 통합 Corpus 생성 (`merged_corpus.jsonl`)
- ✅ Passage Corpus 분석 스크립트 (`analyze_passages.py`)
  - ✅ 중복 passage 탐지 및 통계
  - ✅ 길이 분포 분석 (문자 수, 토큰 수)
  - ✅ 소스별(법령/행정규칙/판례) 통계
  - ✅ CLI 명령어 통합 (`lex-dpr analyze-passages`)
- 📋 Passage 중복 제거 최적화 (자동 제거 도구) - 필요 시에만
- 📋 Passage 길이 분포 분석 (추가 분석 도구) - 필요 시에만

### 2.3 학습 쌍(Pairs) 생성
- ✅ 법령 기반 쌍 생성 (`build_pairs_from_law`)
- ✅ 행정규칙 기반 쌍 생성 (`build_pairs_from_admin`)
- ✅ 판례 기반 쌍 생성 (`build_pairs_from_prec`)
- ✅ Hard Negative 샘플링
- ✅ Cross-positive 쌍 생성 (판례 → 법령/행정규칙)
- ✅ Train/Valid/Test 자동 분할 (query_id 마지막 자리 기준)
- ✅ 데이터 품질 점검 스크립트 (`analyze_pairs.py`)
  - ✅ 쿼리 타입별 비율, positive/negative 개수 분포
  - ✅ 질의 및 passage 토큰 길이 분포 분석
  - ✅ Train/Valid/Test 데이터셋 통계 리포트
  - ✅ CLI 명령어 통합 (`lex-dpr analyze-pairs`)
- 📋 쌍 생성 통계 및 품질 리포트 (자동 생성)
- 📋 Negative 샘플링 전략 개선 (더 어려운 negative)
- 📋 전략별 모델 비교 실험 (법령만 / 행정규칙만 / 법령+행정규칙)

### 2.4 데이터 검증
- ✅ ID 정합성 검증 (`validate_dataset.py`)
- ✅ Passage-Pair 매칭 검증
- 📋 데이터 품질 메트릭 계산
- 📋 이상치 탐지 및 리포트

### 2.5 데이터 필터링
- ✅ 삭제 조문 필터링 (`filters.py`)
- ✅ 공백/빈 텍스트 필터링
- 📋 중복 Passage 자동 제거
- 📋 품질 점수 기반 필터링

---

## 3. 모델 구현

### 3.1 BiEncoder 아키텍처
- ✅ BiEncoder 클래스 구현 (`encoders.py`)
- ✅ 단일 모델 기반 인코딩 (질의/패시지 공유)
- ✅ 템플릿 시스템 (`templates.py`)
- ✅ BGE 템플릿 지원
- ✅ 모델 팩토리 (`factory.py`)
- 📋 모델 alias 확장
- 📋 커스텀 템플릿 지원

### 3.2 모델 로딩 및 저장
- ✅ HuggingFace 모델 로딩
- ✅ 로컬 체크포인트 로딩
- ✅ SentenceTransformer 호환 저장
- 📋 모델 버전 관리
- 📋 모델 메타데이터 저장

### 3.3 PEFT (LoRA) 지원
- ✅ LoRA 어댑터 부착 (`peft.py`)
- ✅ PEFT 체크포인트 로딩
- ✅ LoRA 파라미터만 학습 모드
- 📋 IA3, AdaLoRA 등 다른 PEFT 방법 지원
- 📋 PEFT 하이퍼파라미터 튜닝 가이드

### 3.4 모델 최적화
- ✅ Gradient Checkpointing 지원
- ✅ Mixed Precision Training (AMP) 지원
- 📋 모델 양자화 (INT8, INT4)
- 📋 ONNX 변환 지원
- 📋 TensorRT 최적화

---

## 4. 학습 파이프라인

### 4.1 Trainer 구현
- ✅ BiEncoderTrainer 클래스 (`base_trainer.py`)
- ✅ 학습 루프 통합
- ✅ 데이터 로딩 및 배치 처리
- ✅ Loss 함수 통합 (MultipleNegativesRankingLoss)
- ✅ Optimizer 설정 (`optim.py`)
- ✅ Scheduler 설정 (`scheduler.py`)
- 📋 학습 재개 (resume) 기능
- 📋 체크포인트 관리 개선

### 4.2 Loss 함수
- ✅ MultipleNegativesRankingLoss (`losses.py`)
- 📋 Triplet Loss 지원
- 📋 InfoNCE Loss 지원
- 📋 하드 네거티브 가중치 Loss
- 📋 커스텀 Loss 함수 지원

### 4.3 데이터셋 및 DataLoader
- ✅ PairDataset 구현 (`datasets.py`)
- ✅ InputExample 변환
- ✅ Collator 함수 (`collators.py`)
- ✅ 배치 크기 자동 조정
- 📋 데이터 증강 (Augmentation)
- 📋 동적 배치 샘플링

### 4.4 학습 모니터링
- ✅ 학습 Loss 로깅
- ✅ 웹 로깅 통합 (WandB, MLflow)
- ✅ 하이퍼파라미터 로깅
- ✅ 평가 메트릭 로깅
- 📋 학습 곡선 시각화
- 📋 모델 가중치 히스토그램

### 4.5 평가 통합
- ✅ InformationRetrievalEvaluator 통합
- ✅ 학습 중 평가 (evaluation_steps)
- ✅ MRR@k, NDCG@k, MAP@k 계산
- ✅ Recall@k, Precision@k 계산
- 📋 커스텀 평가 메트릭
- 📋 평가 결과 상세 리포트

### 4.6 특수 기능
- ✅ 테스트 실행 모드 (test_run)
- ✅ 에포크/스텝 제한
- ✅ 학습 조기 종료 (Early Stopping)
  - ✅ Validation 메트릭 기반 자동 중단
  - ✅ Best checkpoint 자동 저장
  - ✅ Patience 및 min_delta 설정 지원
- 📋 학습률 스케줄러 다양화
- ✅ 그라디언트 클리핑
  - ✅ 모델 backward hook을 통한 gradient clipping
  - ✅ 설정 파일 통합
  - ✅ 통계 로깅

### 4.7 하이퍼파라미터 튜닝
- ✅ 하이퍼파라미터 스윕 (Hyperparameter Sweep) 시스템
  - ✅ WandB Sweep 통합
  - ✅ Grid Search / Random Search / Bayesian Optimization 지원
  - ✅ 주요 튜닝 대상:
    - Learning Rate (lr)
    - Temperature (loss scaling)
    - Batch Size
    - Epochs
    - Warmup Steps
    - Gradient Accumulation Steps
  - ✅ 스윕 결과 비교 및 시각화 (WandB 대시보드)
  - ✅ SMOKE TEST 모드 지원 (빠른 검증)
- ✅ 하이퍼파라미터 스윕 CLI (`lex-dpr sweep`)
  - ✅ 스윕 설정 파일 (YAML) 생성 및 관리
  - ✅ 스윕 실행 및 모니터링
  - ✅ 여러 날짜/머신에서 나눠서 실행 지원
  - ✅ 시간 기반 실행 제어 (특정 시간대에만 실행)
  - ✅ 베이지안 탐색 수렴 조건 설정 (Early Termination)
  - ✅ 스윕 ID 자동 저장 및 관리
  - 📋 MLflow Experiments 통합
  - 📋 최적 파라미터 리포트 자동 생성

---

## 5. 평가 시스템

### 5.1 평가 스크립트
- ✅ 평가 함수 구현 (`eval.py`)
- ✅ InformationRetrievalEvaluator 설정
- ✅ 독립 평가 스크립트 (`scripts/evaluate.py`)
- ✅ 상세 평가 모듈 구현 (`eval_detailed.py`)

### 5.2 평가 메트릭
- ✅ MRR@k
- ✅ NDCG@k
- ✅ MAP@k
- ✅ Accuracy@k
- ✅ Precision@k
- ✅ Recall@k
- 📋 커스텀 법률 도메인 메트릭

### 5.3 평가 리포트
- ✅ 상세 분석 리포트 생성 (`eval_detailed.py`)
  - ✅ 쿼리별 성능 분석
  - ✅ 소스별 성능 분석
  - ✅ 실패 케이스 분석
  - ✅ 쿼리 길이별 성능 분석
- ✅ 리포트 포맷 개선 (텍스트, JSON)
- ✅ 여러 모델 비교 분석 기능 (`compare_models` 함수)
- 📋 선택적 시각화 (그래프, 차트)

### 5.4 평가 데이터셋
- 📋 표준 평가 데이터셋 구축
- 📋 법률 도메인 특화 평가셋
- 📋 다양한 난이도 평가셋

---

## 6. 임베딩 추출 및 인덱싱

### 6.1 임베딩 추출
- ✅ CLI 명령어 (`lex-dpr embed`)
- ✅ Passage 임베딩 추출
- ✅ Query 임베딩 추출
- ✅ 배치 처리 지원
- ✅ NPZ/NPY 포맷 저장
- 📋 HDF5 포맷 지원
- 📋 임베딩 압축 옵션

### 6.2 벡터 인덱싱
- ✅ FAISS 인덱스 구축 (`scripts/build_index.py`) - 평가/분석용
- ✅ 임베딩 생성 및 저장 (NPY/NPZ 포맷)
- 📋 평가용 FAISS 인덱스 개선 (필요 시)
- 📋 벡터 검색 성능 벤치마크 (평가용)
- ⏸️ PostgreSQL + pgvector 통합 - 

### 6.3 검색 기능
- ✅ 평가용 검색 구현 (InformationRetrievalEvaluator 사용)
- ⏸️ 프로덕션 검색 API - 
- ⏸️ PostgreSQL + pgvector 기반 검색 - 

---

## 7. API 서버

### 7.1 기본 API
- ✅ FastAPI 기반 서버 (`api.py`) - 모델 테스트용
- ✅ Health Check 엔드포인트
- ✅ Passage 임베딩 엔드포인트 (`POST /embed/passage`)
- ✅ Query 임베딩 엔드포인트 (`POST /embed/query`)
- ✅ 배치 임베딩 지원 (여러 텍스트 동시 처리)
- ✅ API 문서 자동 생성 (Swagger/OpenAPI) - FastAPI 기본 제공
- ⏸️ 검색 엔드포인트 (Top-k) - 

### 7.2 API 기능
- 📋 인증 및 권한 관리
- 📋 Rate Limiting
- 📋 요청 로깅
- 📋 에러 핸들링 개선
- 📋 API 문서 자동 생성 (Swagger/OpenAPI)

### 7.3 성능 최적화
- 📋 비동기 처리 (AsyncIO)
- 📋 모델 캐싱
- 📋 배치 처리 최적화
- 📋 응답 압축

### 7.4 모니터링
- 📋 API 메트릭 수집
- 📋 응답 시간 모니터링
- 📋 에러율 추적

---

## 8. CLI 인터페이스

### 8.1 메인 CLI
- ✅ 통합 CLI 래퍼 (`lex-dpr`)
- ✅ Typer 기반 명령어 구조
- ✅ 하위 명령어 통합 (train, embed, api, config)

### 8.2 학습 CLI
- ✅ 학습 명령어 (`lex-dpr train`)
- ✅ 설정 오버라이드 지원
- ✅ 로깅 통합
- 📋 학습 진행 상황 상세 표시

### 8.3 임베딩 CLI
- ✅ 임베딩 추출 명령어 (`lex-dpr embed`) - 파일 기반 배치 처리
- ✅ 타입별 처리 (query/passage)
- 📋 단일 텍스트 임베딩 테스트 명령어 (`lex-dpr embed-text`)
  - 단일 질의/Passage 텍스트 입력
  - 간단한 JSON/텍스트 출력 (시범 테스트용)
  - 배치 처리는 API 사용 권장
- 📋 배치 크기 자동 조정

### 8.4 API CLI
- ✅ API 서버 실행 명령어 (`lex-dpr api`)
- 📋 서버 설정 옵션 확장

### 8.5 설정 CLI
- ✅ 설정 초기화 (`lex-dpr config init`)
- ✅ 설정 표시 (`lex-dpr config show`)
- ✅ 설정 검증
- 📋 설정 템플릿 생성

### 8.6 데이터 분석 CLI
- ✅ 데이터 품질 분석 명령어 (`lex-dpr analyze-pairs`)
- ✅ Train/Valid/Test 통계 분석
- ✅ 토큰 길이 분포 분석
- 📋 시각화 기능 (히스토그램, 박스플롯 등)

---

## 9. 문서화

### 9.1 사용자 문서
- ✅ README.md (기본 사용법)
- ✅ GUIDE.md (학습 가이드)
- 📋 API 문서
- 📋 설정 가이드
- 📋 배포 가이드

### 9.2 개발자 문서
- ✅ 모델 구조 문서 (`lex_dpr/models/README.md`)
- ✅ 데이터 처리 문서 (`lex_dpr/data_processing/README.md`)
- 📋 아키텍처 다이어그램
- 📋 개발 가이드라인
- 📋 기여 가이드

### 9.3 예제 및 튜토리얼
- 📋 빠른 시작 튜토리얼
- 📋 엔드투엔드 예제
- 📋 고급 사용 예제
- 📋 문제 해결 가이드

---

## 10. 테스트 및 검증

### 10.1 단위 테스트
- 📋 모델 테스트
- 📋 데이터 처리 테스트
- 📋 유틸리티 함수 테스트
- 📋 CLI 명령어 테스트

### 10.2 통합 테스트
- 📋 학습 파이프라인 테스트
- 📋 평가 파이프라인 테스트
- 📋 API 서버 테스트
- 📋 엔드투엔드 테스트

### 10.3 검증
- 📋 데이터 검증 스크립트
- 📋 모델 출력 검증
- 📋 성능 회귀 테스트

---

## 11. 배포 및 운영

### 11.1 패키징
- ✅ Poetry 패키지 설정
- 📋 Docker 이미지 빌드
- 📋 배포 스크립트
- 📋 버전 관리 전략

### 11.2 CI/CD
- 📋 GitHub Actions 워크플로우
- 📋 자동 테스트
- 📋 자동 배포
- 📋 릴리스 자동화

### 11.3 모니터링 및 로깅
- ✅ 웹 로깅 통합 (WandB, MLflow)
- 📋 프로덕션 로깅 시스템
- 📋 모델 성능 모니터링
- 📋 알림 시스템

---

## 12. 성능 최적화

### 12.1 학습 최적화
- ✅ Mixed Precision Training
- ✅ Gradient Checkpointing
- 📋 데이터 로딩 최적화 (멀티프로세싱)
- 📋 학습 속도 벤치마크

### 12.2 추론 최적화
- 📋 모델 양자화
- 📋 배치 추론 최적화
- 📋 ONNX 변환
- 📋 TensorRT 최적화

### 12.3 메모리 최적화
- 📋 메모리 사용량 프로파일링
- 📋 메모리 효율적인 데이터 로딩
- 📋 모델 압축

---

## 13. 확장 기능

### 13.1 고급 학습 기능
- 📋 지식 증류 (Knowledge Distillation)
- 📋 하드 네거티브 마이닝
- 📋 데이터 증강 전략
- 📋 앙상블 학습

### 13.2 검색 기능 확장
- 📋 하이브리드 검색 (Dense + Sparse)
- 📋 리랭킹 (Reranking) 통합
- 📋 필터링 및 페이싱
- 📋 검색 결과 설명 (Explainability)

### 13.3 도메인 확장
- 📋 다른 법률 도메인 지원
- 📋 다국어 지원
- 📋 크로스 도메인 전이 학습

### 13.4 실험 관리
- ✅ 웹 로깅 (WandB, MLflow)
- 📋 실험 추적 시스템
- 📋 하이퍼파라미터 튜닝 자동화
- 📋 A/B 테스트 프레임워크

---

## 14. 버그 수정 및 개선

### 14.1 알려진 이슈
- ✅ Neptune 의존성 제거 및 패칭
- ✅ Trainer report_to 파라미터 호환성 처리
- 📋 기타 알려진 버그 수정

### 14.2 코드 품질
- 📋 코드 리팩토링
- 📋 타입 힌트 완성
- 📋 문서 문자열 보완
- 📋 린터 및 포매터 통합

### 14.3 성능 개선
- 📋 병목 지점 최적화
- 📋 불필요한 연산 제거
- 📋 캐싱 전략 개선

---

## 진행 상황 요약

### 완료된 주요 기능 (✅)
1. 프로젝트 기본 구조 및 설정
2. 데이터 처리 파이프라인 (법령, 행정규칙, 판례)
3. 학습 쌍 생성 및 검증 (Train/Valid/Test 자동 분할)
4. BiEncoder 모델 구현
5. 학습 파이프라인 (Trainer, Loss, Optimizer)
6. 웹 로깅 통합 (WandB, MLflow)
7. CLI 인터페이스 (train, embed, api, config, eval, gen-data, analyze-pairs, analyze-passages, smoke-train, crawl-precedents)
8. 평가 시스템 기본 구현
9. 임베딩 API 서버 기본 구현 (모델 테스트용)
10. 문서화 (README, GUIDE)
11. 데이터 품질 분석 도구
    - analyze_pairs.py (질의-passage 쌍 분석)
    - analyze_passages.py (passage corpus 분석)
12. WandB Sweep 하이퍼파라미터 튜닝 시스템
    - 스윕 설정 파일 생성 및 관리
    - 여러 날짜/머신에서 나눠서 실행 지원
    - 시간 기반 실행 제어 (특정 시간대에만 실행)
    - 베이지안 탐색 수렴 조건 설정 (Early Termination)
13. 벡터 인덱싱 시스템
    - FAISS 인덱스 구축 (평가/분석용)

### 진행 중인 작업 (🚧)
- 없음 (현재)

### 우선순위 높은 다음 작업 (📋)
1. 평가 시스템 강화
   - 독립 평가 스크립트 개선
   - 상세 분석 리포트 생성
2. ✅ 학습 조기 종료 (Early Stopping) - 완료
   - Validation 메트릭 기반 자동 중단
   - Best checkpoint 자동 저장
3. 단일 텍스트 임베딩 테스트 CLI (`embed-text`)
   - `lex-dpr embed-text` 명령어 구현
   - 대화형 모드 (`lex-dpr embed-interactive`)
4. 테스트 코드 작성
   - 단위 테스트
   - 통합 테스트
5. 모델 배포 준비
   - Docker 이미지 빌드 (모델 서빙용)
   - 모델 체크포인트 관리
   - ⏸️ 프로덕션 API 서버 - 
   - ⏸️ CI/CD 파이프라인 - 

---

## 참고사항

- 이 TODO 리스트는 정기적으로 업데이트됩니다
- 각 작업의 우선순위는 프로젝트 요구사항에 따라 변경될 수 있습니다
- 완료된 작업은 ✅로 표시하고, 다음 단계로 진행합니다
- 새로운 요구사항이 발생하면 적절한 섹션에 추가합니다


