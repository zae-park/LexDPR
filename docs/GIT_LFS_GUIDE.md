# Git LFS 사용 가이드

## 📋 개요

이 프로젝트는 모델 체크포인트와 대용량 바이너리 파일을 관리하기 위해 **Git LFS (Large File Storage)**를 사용합니다.

## 🗓️ LFS 도입 이력

- **도입일**: 2025-11-14
- **커밋**: `919e0cf4` - "Add checkpoint files with Git LFS and update .gitignore"
- **배경**: 모델 학습 과정에서 생성되는 체크포인트 파일들이 Git 저장소에 포함되면서 저장소 크기가 증가하고, GitHub의 파일 크기 제한에 걸리는 문제 발생

## 📁 LFS로 추적되는 파일

### 1. 체크포인트 디렉토리
```
checkpoint/**  # 모든 체크포인트 파일
```

**생성 위치**: `checkpoint/lexdpr/bi_encoder/` (기본값, `configs/base.yaml`에서 설정)

**포함 파일**:
- `adapter_model.safetensors` - 모델 가중치 (가장 큰 파일)
- `tokenizer.json`, `tokenizer_config.json` - 토크나이저 설정
- `config_sentence_transformers.json` - Sentence-Transformers 설정
- `sentence_bert_config.json` - Sentence-BERT 설정
- `modules.json` - 모듈 구조
- `special_tokens_map.json` - 특수 토큰 매핑
- `vocab.txt` - 어휘 사전
- `1_Pooling/config.json` - 풀링 레이어 설정
- `adapter_config.json` - PEFT 어댑터 설정 (LoRA 사용 시)
- `README.md` - 모델 정보

### 2. 모델 가중치 파일
```
*.safetensors  # SafeTensors 형식 모델 파일
*.bin          # PyTorch 바이너리 모델 파일
```

## 🔧 파일 생성 과정

### 학습 명령어

체크포인트 파일은 다음 명령어로 생성됩니다:

```bash
# Hydra를 사용한 학습 (권장)
poetry run python entrypoint_train.py

# 또는 설정 파일 직접 지정
poetry run python -c "from lex_dpr.trainer.base_trainer import BiEncoderTrainer; from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/base.yaml'); trainer = BiEncoderTrainer(cfg); trainer.train()"
```

### 코드에서의 생성 위치

**`lex_dpr/trainer/base_trainer.py`**:
```python
def train(self) -> None:
    # ... 학습 과정 ...
    
    # 체크포인트 저장 (line 224-225)
    os.makedirs(self.cfg.out_dir, exist_ok=True)
    save_path = os.path.join(self.cfg.out_dir, "bi_encoder")
    self.model.save(save_path)  # sentence-transformers의 save() 메서드
    print(f"[BiEncoderTrainer] saved model to {save_path}")
```

**설정 파일**: `configs/base.yaml`
```yaml
out_dir: checkpoint/lexdpr  # 체크포인트 저장 경로
```

### 생성되는 파일 크기

- `adapter_model.safetensors`: 약 수십 MB ~ 수백 MB (모델 크기에 따라)
- 기타 설정 파일들: 각각 수 KB ~ 수십 KB

## ⚙️ LFS 설정

### `.gitattributes` 파일

```gitattributes
# Git LFS 파일 추적 설정
checkpoint/** filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
```

### LFS 초기화 (처음 설정 시)

```bash
# Git LFS 설치 확인
git lfs version

# LFS 초기화 (이미 완료됨)
git lfs install

# 특정 패턴 추적 시작
git lfs track "checkpoint/**"
git lfs track "*.safetensors"
git lfs track "*.bin"
```

## 🔄 org-mirror와 origin 동기화

### 저장소 분리 관리 전략

이 프로젝트는 **두 개의 저장소를 분리해서 관리**합니다:

- **org-mirror** (비공개 조직, 폐쇄망 환경)
  - 작업 환경
  - checkpoint 파일 포함 (LFS 사용)
  - 실제 모델 학습 및 개발
  
- **origin** (공개 저장소)
  - 코드와 문서만 공유
  - checkpoint 파일 제외 (`.gitignore`에 포함)
  - 공개적으로 공유 가능한 내용만

### 동기화 방법

### org-mirror → origin 동기화

**중요**: org-mirror는 폐쇄망 환경이므로, origin과 직접 동기화할 수 없습니다.

**권장 워크플로우**:
1. org-mirror에서 작업 (checkpoint 포함)
2. 코드 변경사항만 커밋
3. checkpoint 파일은 `.gitignore`에 의해 자동 제외됨
4. origin에 push (코드와 문서만)

### 문제 상황 (참고)

만약 checkpoint 파일이 포함된 상태로 origin에 push하려고 하면 다음 에러가 발생합니다:

```
remote: error: GH008: Your push referenced at least N unknown Git LFS objects
```

**원인**: LFS 포인터 파일은 있지만 실제 LFS 객체가 LFS 서버에 업로드되지 않았을 때 발생

### 해결 방법

#### 방법 1: LFS 스킵하고 pull (권장)

```bash
# org-mirror에서 pull할 때
GIT_LFS_SKIP_SMUDGE=1 git pull --rebase org-mirror main

# 또는 영구 설정
git config filter.lfs.smudge "git-lfs smudge --skip %f"
git config filter.lfs.process "git-lfs filter-process --skip"
```

#### 방법 2: LFS 객체와 함께 push

```bash
# LFS 객체를 포함하여 push
git lfs push origin main --all

# 또는 일반 push (LFS 객체 자동 업로드)
git push origin main
```

#### 방법 3: checkpoint 파일 제외 (origin용, 권장)

**origin 저장소에서는 checkpoint를 제외하는 것이 권장됩니다**:

```bash
# .gitignore에 이미 포함되어 있음
# checkpoint/  # org-mirror에서는 LFS로 추적, origin에서는 제외

# 이미 추적 중인 파일 제거 (필요시)
git rm --cached -r checkpoint/
git commit -m "Remove checkpoint from Git tracking"
```

**주의**: org-mirror에서는 `.gitattributes`의 LFS 설정을 유지하고, origin으로 push할 때는 checkpoint가 자동으로 제외됩니다.

## ⚠️ 주의사항

### 1. LFS 포인터 파일

LFS로 추적되는 파일은 실제로는 **포인터 파일**만 Git에 저장됩니다:
```
version https://git-lfs.github.com/spec/v1
oid sha256:1c8bc7bd750c5c20d8707f1c5c578f5d69bb3d1d5ebcf4b2fde5128de154ec1c
size 296
```

실제 파일을 사용하려면:
```bash
# LFS 파일 다운로드
git lfs pull

# 특정 파일만 다운로드
git lfs pull --include="checkpoint/**"
```

### 2. 저장소 크기 관리

- 체크포인트 파일은 일반적으로 Git에 포함하지 않는 것이 좋습니다
- `.gitignore`에 `checkpoint/`를 추가하여 로컬에서만 관리하는 것을 권장합니다
- 공유가 필요한 경우에만 LFS 사용을 고려하세요

### 3. org-mirror와 origin 동기화 시

- org-mirror에서 LFS 파일을 추가한 경우, origin에 push하기 전에 LFS 객체가 업로드되었는지 확인
- LFS 객체가 없으면 `GIT_LFS_SKIP_SMUDGE=1`로 pull하여 포인터만 받고, 필요시 수동으로 다운로드

## 📝 체크리스트

### 새 체크포인트 생성 시

- [ ] `.gitignore`에 `checkpoint/`가 포함되어 있는지 확인
- [ ] LFS로 추적하려는 경우 `.gitattributes`에 패턴 추가
- [ ] `git lfs track` 명령어로 추적 시작
- [ ] 커밋 전에 `git lfs ls-files`로 추적 상태 확인

### org-mirror → origin 동기화 시

- [ ] `GIT_LFS_SKIP_SMUDGE=1`로 pull하여 LFS 에러 방지
- [ ] 필요시 `git lfs pull`로 실제 파일 다운로드
- [ ] origin에 push할 때 LFS 객체 업로드 확인

## 🔗 참고 자료

- [Git LFS 공식 문서](https://git-lfs.github.com/)
- [GitHub LFS 가이드](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- 커밋 히스토리: `git log --all --oneline --grep="LFS"`

## 📌 요약

1. **체크포인트 생성**: `entrypoint_train.py` 실행 → `checkpoint/lexdpr/bi_encoder/`에 저장
2. **LFS 추적**: `.gitattributes`에 `checkpoint/**` 패턴으로 자동 추적
3. **동기화**: org-mirror에서 pull 시 `GIT_LFS_SKIP_SMUDGE=1` 사용 권장
4. **권장사항**: 체크포인트는 `.gitignore`에 추가하여 Git에서 제외하는 것이 일반적

