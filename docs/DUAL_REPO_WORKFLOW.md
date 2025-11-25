# 이중 저장소 워크플로우 가이드

## 📋 개요

이 프로젝트는 **org-mirror** (비공개 조직, 폐쇄망)와 **origin** (공개 저장소) 두 개의 저장소를 분리해서 관리합니다.

## 🔄 Push 가능 여부

### ✅ 일반적인 코드 변경사항

**두 저장소 모두에 자유롭게 push 가능합니다:**

```bash
# 코드 변경 후
git add .
git commit -m "코드 변경사항"

# 양쪽 모두 push
git push origin main
git push org-mirror main
```

**이유**: checkpoint 파일은 `.gitignore`에 의해 자동으로 제외되므로, 코드 변경사항만 push됩니다.

### ⚠️ Checkpoint 파일 관련 작업

#### org-mirror에 checkpoint 추가하려는 경우

```bash
# 1. checkpoint 파일 생성 (학습 등)
poetry run python entrypoint_train.py

# 2. .gitignore에서 checkpoint 제외 (임시)
# .gitignore에서 checkpoint/ 라인을 주석 처리하거나 제거

# 3. LFS로 추적 시작
git lfs track "checkpoint/**"

# 4. 커밋 및 push
git add checkpoint/
git commit -m "Add checkpoint files"
git push org-mirror main  # org-mirror에만 push
```

**주의**: 
- origin에는 push하지 마세요 (LFS 객체가 없어서 실패합니다)
- 작업 후 `.gitignore`를 다시 복원하세요

#### origin에 push할 때

```bash
# checkpoint는 .gitignore에 의해 자동 제외됨
git push origin main  # 문제없음
```

## 🎯 권장 워크플로우

### 시나리오 1: 일반 코드 변경

```bash
# 1. 코드 수정
vim some_file.py

# 2. 커밋
git add .
git commit -m "Fix bug in some_file.py"

# 3. 양쪽 모두 push
git push origin main
git push org-mirror main
```

### 시나리오 2: Checkpoint 생성 후 org-mirror에만 push

```bash
# 1. 모델 학습 (checkpoint 생성)
poetry run python entrypoint_train.py

# 2. .gitignore 임시 수정 (checkpoint 제외 해제)
# .gitignore에서 checkpoint/ 라인 주석 처리

# 3. LFS 추적 확인
git lfs track "checkpoint/**"  # 이미 설정되어 있으면 생략

# 4. org-mirror에만 push
git add checkpoint/
git commit -m "Add new checkpoint"
git push org-mirror main

# 5. .gitignore 복원
# checkpoint/ 라인 다시 활성화

# 6. origin에 push (checkpoint 제외됨)
git commit --amend --no-edit  # checkpoint 제외하고 커밋 수정
git push origin main
```

### 시나리오 3: org-mirror에서 최신 변경사항 가져오기

```bash
# LFS 스킵하고 pull (포인터만 받음)
GIT_LFS_SKIP_SMUDGE=1 git pull --rebase org-mirror main

# checkpoint는 .gitignore에 의해 로컬에 생성되지 않음
# 필요시 수동으로 다운로드: git lfs pull --include="checkpoint/**"
```

## ⚠️ 주의사항

### 1. Checkpoint 파일은 org-mirror에만

- **org-mirror**: checkpoint 포함 가능 (LFS 사용)
- **origin**: checkpoint 제외 (`.gitignore`에 의해 자동 처리)

### 2. .gitignore 우선순위

- `.gitignore`에 `checkpoint/`가 있으면, `.gitattributes`의 LFS 설정보다 우선합니다
- 따라서 checkpoint를 추가하려면 `.gitignore`에서 임시로 제외해야 합니다

### 3. LFS 객체 없이 push 시도

```bash
# ❌ 이렇게 하면 실패합니다
git push origin main  # checkpoint가 포함되어 있으면 LFS 에러

# ✅ 올바른 방법
# checkpoint를 .gitignore에 추가하고
git push origin main  # checkpoint 자동 제외
```

## 🔧 유용한 명령어

### 현재 상태 확인

```bash
# 어떤 파일이 LFS로 추적되는지 확인
git lfs ls-files

# 두 저장소 상태 비교
git log --oneline --graph --all --decorate -10

# checkpoint 파일이 있는지 확인
Test-Path checkpoint/lexdpr/bi_encoder
```

### Checkpoint 제외 확인

```bash
# .gitignore에 checkpoint가 있는지 확인
grep checkpoint .gitignore

# 실제로 제외되는지 테스트
git status --ignored | grep checkpoint
```

## 📝 요약

| 작업 | origin push | org-mirror push | 비고 |
|------|------------|----------------|------|
| 일반 코드 변경 | ✅ 가능 | ✅ 가능 | checkpoint 자동 제외 |
| Checkpoint 추가 | ❌ 불가 | ✅ 가능 | LFS 객체 필요 |
| 문서 업데이트 | ✅ 가능 | ✅ 가능 | 문제없음 |
| 설정 파일 변경 | ✅ 가능 | ✅ 가능 | 문제없음 |

**핵심**: checkpoint를 제외한 모든 변경사항은 두 저장소 모두에 push 가능합니다.

