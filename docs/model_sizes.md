# 모델 크기 및 메모리 사용량 가이드

## 현재 사용 가능한 모델

### 1. bge-m3-ko (dragonkue/BGE-m3-ko)
- **파라미터 수**: ~560M (BAAI/bge-m3 기반)
- **모델 크기**: ~1.2GB (가중치만)
- **메모리 사용량**: 배치 256, max_len 512 기준 ~78-100GB (AMP + Gradient Checkpointing 사용 시)
- **특징**: 
  - 한국어 최적화
  - 최대 8192 토큰 지원
  - Multi-vector retrieval 지원
- **OOM 위험**: 높음 (배치 크기와 시퀀스 길이에 따라)

### 2. ko-simcse (jhgan/ko-sroberta-multitask)
- **파라미터 수**: ~110M
- **모델 크기**: ~440MB
- **메모리 사용량**: 배치 256, max_len 128 기준 ~5-10GB
- **특징**:
  - 한국어 전용
  - 작은 모델로 빠른 학습
  - max_len 128로 제한됨
- **OOM 위험**: 낮음

### 3. multilingual-minilm (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **파라미터 수**: ~117M
- **모델 크기**: ~470MB
- **메모리 사용량**: 배치 256, max_len 512 기준 ~10-15GB
- **특징**:
  - 다국어 지원
  - 경량 모델
- **OOM 위험**: 낮음

### 4. multilingual-e5-small (intfloat/multilingual-e5-small)
- **파라미터 수**: ~118M
- **모델 크기**: ~470MB
- **메모리 사용량**: 배치 256, max_len 512 기준 ~10-15GB
- **특징**:
  - 다국어 지원
  - E5 시리즈
- **OOM 위험**: 낮음

## BGE-m3-ko의 다양한 사이즈

**참고**: BGE-m3-ko는 일반적으로 단일 사이즈만 제공됩니다 (560M 파라미터).
- BAAI/bge-m3: 560M 파라미터 (기본)
- dragonkue/BGE-m3-ko: 동일한 아키텍처, 한국어 최적화

**더 작은 BGE 모델 옵션**:
- BGE-small: ~33M 파라미터 (한국어 지원 여부 확인 필요)
- BGE-base: ~110M 파라미터 (한국어 지원 여부 확인 필요)

## 메모리 사용량 비교 (배치 256, max_len 512 기준)

| 모델 | 파라미터 | 가중치 크기 | 예상 메모리 (AMP+GC) | OOM 위험 |
|------|---------|------------|---------------------|----------|
| bge-m3-ko | 560M | ~1.2GB | ~78-100GB | 높음 |
| ko-simcse | 110M | ~440MB | ~5-10GB (max_len 128) | 낮음 |
| multilingual-minilm | 117M | ~470MB | ~10-15GB | 낮음 |
| multilingual-e5-small | 118M | ~470MB | ~10-15GB | 낮음 |

## 권장 사항

### bge-m3-ko 사용 시 (OOM 방지)
1. **배치 크기 줄이기**: 256 → 128 또는 64
2. **시퀀스 길이 줄이기**: 512 → 256 또는 384
3. **Gradient Checkpointing 활성화**: 필수
4. **AMP 활성화**: 필수

### 대안 모델 사용 시
- **ko-simcse**: 한국어 전용, 빠른 학습, 작은 메모리
- **multilingual-minilm**: 다국어 지원, 경량
- **multilingual-e5-small**: 다국어 지원, E5 시리즈

