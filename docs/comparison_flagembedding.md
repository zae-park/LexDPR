# FlagEmbedding Fine-tuning 예제 vs 현재 코드 비교

[FlagEmbedding fine-tuning 예제](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)와 현재 LexDPR 코드의 주요 차이점을 분석합니다.

## 1. 학습 프레임워크

### FlagEmbedding
- **직접 구현된 학습 루프**: PyTorch로 직접 학습 루프를 구현
- **Multi-GPU 지원**: `torchrun`을 사용한 분산 학습
- **커스텀 학습 스크립트**: `train.py`에서 직접 optimizer, scheduler, loss 관리

### 현재 코드 (LexDPR)
- **Sentence-Transformers 기반**: `sentence-transformers` 라이브러리의 `fit()` 메서드 사용
- **단일 GPU 우선**: 현재는 단일 GPU 학습에 최적화
- **고수준 API**: 학습 루프를 직접 구현하지 않고 라이브러리에 위임

**차이점**:
```python
# FlagEmbedding 방식
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

# 현재 코드 방식
model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    optimizer_params=optimizer_params,
    use_amp=True
)
```

## 2. Loss 함수

### FlagEmbedding
- **InfoNCE Loss**: 직접 구현된 contrastive learning loss
- **Temperature scaling**: 하이퍼파라미터로 조절 가능
- **Hard negative mining**: 별도 스크립트(`hn_mine.py`)로 하드 네거티브 추출

### 현재 코드
- **MultipleNegativesRankingLoss**: Sentence-Transformers의 표준 loss
- **In-batch negatives**: 배치 내 다른 샘플을 자동으로 negative로 사용
- **Hard negatives**: 데이터셋에 미리 포함된 hard negatives 사용

**차이점**:
- FlagEmbedding은 학습 중 동적으로 hard negative를 마이닝할 수 있음
- 현재 코드는 사전에 생성된 hard negative를 사용

## 3. 데이터 형식

### FlagEmbedding
```json
{"query": "질의 텍스트", "pos": ["긍정 패시지1", "긍정 패시지2"], "neg": ["부정 패시지1", "부정 패시지2"]}
```
- 각 줄이 하나의 샘플
- `pos`와 `neg`가 리스트로 여러 개 포함 가능

### 현재 코드
```json
{"query_text": "질의 텍스트", "positive_passages": ["pid1", "pid2"], "hard_negatives": ["pid3", "pid4"]}
```
- Passage ID를 참조하는 방식
- 실제 passage 텍스트는 별도 corpus 파일에서 로드

**차이점**:
- FlagEmbedding: 텍스트가 직접 포함 (중복 가능)
- 현재 코드: ID 참조 방식 (메모리 효율적, 중복 제거)

## 4. 하드 네거티브 샘플링

### FlagEmbedding
- **학습 중 동적 마이닝**: `hn_mine.py`로 학습 중 hard negative 추출
- **별도 스크립트**: 학습 전에 한 번 실행하여 hard negative 생성
- **BM25 기반**: 초기에는 BM25로 hard negative 생성

### 현재 코드
- **사전 생성**: `make_pairs.py`로 학습 전에 hard negative 생성
- **법률 도메인 특화**: 판례의 참조조문을 기반으로 positive/hard negative 구성
- **동일 법령 내 샘플링**: 같은 법령의 다른 조문을 hard negative로 사용

**차이점**:
- FlagEmbedding: 일반적인 정보 검색 방식
- 현재 코드: 법률 도메인 특화 (참조조문 기반)

## 5. 모델 설정

### FlagEmbedding
- **전체 모델 fine-tuning**: 기본적으로 전체 모델을 학습
- **LoRA 지원**: 옵션으로 LoRA 사용 가능
- **Query/Passage 길이 분리**: `query_max_len`, `passage_max_len` 별도 설정

### 현재 코드
- **LoRA 기본 사용**: 기본적으로 LoRA만 학습 (메모리 효율적)
- **통합 길이 설정**: `max_len` 또는 `query_max_len`/`passage_max_len` 분리 설정
- **PEFT 통합**: `peft` 라이브러리로 LoRA 관리

**차이점**:
- FlagEmbedding: 전체 fine-tuning 우선, LoRA는 옵션
- 현재 코드: LoRA 우선, 메모리 효율성 중시

## 6. 평가 방식

### FlagEmbedding
- **Retrieval 평가**: Query-Passage retrieval 성능 평가
- **Recall@K**: Top-K recall 메트릭
- **평가 스크립트**: 별도 평가 스크립트 제공

### 현재 코드
- **IR 평가 통합**: `build_ir_evaluator`로 평가 통합
- **다양한 메트릭**: NDCG@K, Recall@K, MRR 등
- **학습 중 평가**: `evaluation_steps`로 주기적 평가

**차이점**:
- FlagEmbedding: 학습과 평가 분리
- 현재 코드: 학습 중 통합 평가

## 7. 메모리 최적화

### FlagEmbedding
- **Gradient Checkpointing**: 옵션으로 제공
- **Mixed Precision**: AMP 지원
- **배치 크기 조절**: 메모리에 따라 배치 크기 조절

### 현재 코드
- **Gradient Checkpointing**: ✅ 추가됨 (sweep 설정에 포함)
- **Mixed Precision**: ✅ `use_amp: true` 기본 활성화
- **LoRA 기본 사용**: 메모리 효율성 중시

**최근 추가**:
```yaml
trainer.gradient_checkpointing: true  # 메모리 절약 (약 10GB 절감)
trainer.use_amp: true  # Mixed precision training
```

## 8. 학습 하이퍼파라미터

### FlagEmbedding
- **Learning Rate**: 일반적으로 1e-5 ~ 5e-5
- **Batch Size**: 32 ~ 128
- **Epochs**: 3 ~ 10
- **Warmup**: 학습률 warmup 지원

### 현재 코드
- **Learning Rate**: Sweep에서 5e-7 ~ 1e-4 탐색
- **Batch Size**: Sweep에서 32 ~ 256 탐색
- **Epochs**: 200 (Early stopping으로 실제로는 조기 종료)
- **Warmup**: Cosine annealing with warmup

**차이점**:
- FlagEmbedding: 고정된 하이퍼파라미터
- 현재 코드: WandB sweep으로 하이퍼파라미터 탐색

## 9. 로깅 및 모니터링

### FlagEmbedding
- **기본 로깅**: 학습 loss 출력
- **모델 저장**: 체크포인트 저장

### 현재 코드
- **WandB 통합**: 실험 추적 및 시각화
- **Early Stopping**: Validation 메트릭 기반 조기 종료
- **상세 로깅**: 학습/평가 메트릭 상세 로깅

**차이점**:
- FlagEmbedding: 기본적인 로깅
- 현재 코드: 실험 관리 및 모니터링 강화

## 10. 주요 개선 사항 (현재 코드)

### 장점
1. **도메인 특화**: 법률 도메인에 특화된 데이터 처리
2. **메모리 효율성**: LoRA 기본 사용, Gradient Checkpointing 지원
3. **실험 관리**: WandB 통합으로 하이퍼파라미터 탐색 용이
4. **Early Stopping**: Overfitting 방지
5. **유연한 평가**: 다양한 IR 메트릭 지원

### FlagEmbedding에서 참고할 점
1. **동적 Hard Negative Mining**: 학습 중 hard negative 업데이트
2. **Multi-GPU 지원**: 분산 학습 지원
3. **더 세밀한 제어**: 학습 루프 직접 구현으로 더 세밀한 제어 가능

## 결론

현재 코드는 **법률 도메인 특화**와 **메모리 효율성**에 중점을 두고 있으며, FlagEmbedding은 **일반적인 정보 검색**과 **전체 모델 fine-tuning**에 중점을 둡니다.

**현재 코드의 특징**:
- ✅ LoRA 기본 사용 (메모리 효율)
- ✅ Gradient Checkpointing 지원 (메모리 절약)
- ✅ 법률 도메인 특화 데이터 처리
- ✅ WandB 통합 실험 관리
- ✅ Early Stopping으로 overfitting 방지

**추가 고려 사항**:
- Multi-GPU 지원 추가 고려
- 동적 hard negative mining 고려 (선택사항)

