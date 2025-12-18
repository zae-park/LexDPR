# 학습 문제 해결 가이드

## 현재 코드의 학습 문제 진단

### 1. 학습이 잘 안되는 증상

#### 가능한 증상들:
- Train loss는 감소하지만 valid loss가 증가 (Overfitting)
- Train loss와 valid loss가 모두 감소하지 않음 (Underfitting)
- Loss가 불안정하게 변동
- 성능 메트릭(Recall@K, NDCG@K)이 개선되지 않음

### 2. 현재 코드의 장점

#### 이미 구현된 기능:
- ✅ **LoRA 기본 사용**: 메모리 효율적 학습
- ✅ **Gradient Checkpointing**: 메모리 절약 (약 10GB)
- ✅ **AMP (Mixed Precision)**: 메모리 절약 및 학습 속도 향상
- ✅ **Early Stopping**: Overfitting 방지
- ✅ **WandB 통합**: 실험 추적 및 모니터링
- ✅ **법률 도메인 특화**: 참조조문 기반 데이터 처리

### 3. FlagEmbedding vs 현재 코드

#### FlagEmbedding의 장점:
- 검증된 학습 방법 (BGE 모델 개발팀의 공식 방법)
- Multi-GPU 지원 (분산 학습)
- 동적 Hard Negative Mining (학습 중 업데이트)
- 더 세밀한 학습 루프 제어

#### FlagEmbedding의 단점:
- LoRA 지원 여부 불명확 (확인 필요)
- 법률 도메인 특화 기능 없음
- 실험 관리 도구 통합 필요

### 4. 학습 문제 해결 방법

#### A. 현재 코드로 해결 시도

**Overfitting 문제:**
```yaml
# 이미 적용된 설정
trainer.early_stopping.enabled: true
trainer.early_stopping.patience: 8
trainer.weight_decay: 0.01 ~ 0.5  # 정규화 강화
model.peft.dropout: 0.05 ~ 0.1  # Dropout 증가
```

**추가 조치:**
1. 데이터 증폭 줄이기: `data.multiply: [0, 1]` (2 제거)
2. Learning rate 낮추기: `trainer.lr: 5e-7 ~ 1e-5` (더 보수적)
3. Batch size 조절: 더 작은 배치로 안정화

**Underfitting 문제:**
1. Learning rate 높이기: `trainer.lr: 1e-5 ~ 1e-4`
2. Epochs 증가: Early stopping patience 증가
3. 데이터 증폭: `data.multiply: [1, 2]`

**Loss 불안정:**
1. Learning rate 낮추기
2. Gradient clipping 활성화: `trainer.gradient_clip_norm: 1.0`
3. Warmup steps 증가

#### B. FlagEmbedding으로 전환 고려

**전환을 고려해야 하는 경우:**
- 현재 코드로 여러 하이퍼파라미터를 시도했지만 성능이 개선되지 않음
- Multi-GPU 학습이 필요함
- 동적 Hard Negative Mining이 필요함

**전환 시 주의사항:**
1. **LoRA 지원 확인 필수**: FlagEmbedding이 LoRA를 지원하는지 확인 필요
   - LoRA 미지원 시 메모리 사용량 급증 (전체 모델 학습)
   - H100 100GB에서도 배치 크기 256, max_len 512는 어려울 수 있음

2. **법률 도메인 특화 기능 재구현 필요**:
   - 참조조문 파싱 로직
   - 판례 데이터 처리
   - 법령/행정규칙 passage 생성

3. **실험 관리 도구 통합 필요**:
   - WandB 통합
   - Early Stopping 구현
   - 평가 메트릭 통합

### 5. 권장 접근 방법

#### 단계 1: 현재 코드 최적화 (우선 시도)

1. **하이퍼파라미터 Sweep 실행**:
   ```bash
   poetry run lex-dpr sweep preset
   ```
   - 이미 설정된 범위로 다양한 조합 시도
   - WandB에서 결과 분석

2. **학습 곡선 분석**:
   - Train/Valid loss 추이 확인
   - Overfitting 패턴 확인
   - Early stopping이 적절히 작동하는지 확인

3. **데이터 품질 확인**:
   - Hard negative 품질 확인
   - Positive passage 매칭 정확도 확인
   - 데이터 증폭 효과 확인

#### 단계 2: FlagEmbedding 검토 (필요 시)

1. **LoRA 지원 확인**:
   - FlagEmbedding GitHub 이슈/문서 확인
   - 예제 코드에서 LoRA 사용 예시 확인

2. **Pilot 실험**:
   - 작은 데이터셋으로 FlagEmbedding 학습 시도
   - 메모리 사용량 및 성능 비교

3. **전환 결정**:
   - FlagEmbedding이 LoRA를 지원하고 성능이 더 좋으면 전환 고려
   - 그렇지 않으면 현재 코드 최적화 계속

### 6. 현재 코드 개선 제안

#### 즉시 시도할 수 있는 개선:

1. **Gradient Clipping 추가**:
   ```yaml
   trainer.gradient_clip_norm: 1.0
   ```

2. **Learning Rate Schedule 조정**:
   - 현재: Warmup + Cosine Annealing
   - 대안: Linear decay, Step decay 등 시도

3. **Temperature 조정**:
   ```yaml
   trainer.temperature: [0.01, 0.05, 0.1]  # Sweep에서 탐색
   ```

4. **Hard Negative 비율 조정**:
   - 현재: `use_hard_negatives: true`
   - 대안: Hard negative 비율을 조절하여 in-batch와의 균형 조정

### 7. 결론

**현재 코드로 충분한 경우:**
- LoRA 기본 사용으로 메모리 효율적
- Gradient Checkpointing + AMP로 메모리 최적화
- 법률 도메인 특화 기능
- WandB 통합 실험 관리
- Early Stopping으로 Overfitting 방지

**FlagEmbedding 전환을 고려하는 경우:**
- LoRA 지원 확인 후
- 현재 코드로 여러 시도 후에도 성능이 개선되지 않을 때
- Multi-GPU 학습이 필수일 때

**권장 사항:**
1. 먼저 현재 코드로 하이퍼파라미터 sweep 실행
2. 학습 곡선 및 성능 메트릭 분석
3. 문제가 지속되면 FlagEmbedding LoRA 지원 확인
4. LoRA 지원 시 Pilot 실험으로 비교

