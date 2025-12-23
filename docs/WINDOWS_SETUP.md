# Windows 환경 설정 가이드

## WandB API 키 설정

### 방법 1: 명령 프롬프트(CMD)에서 임시 설정

```cmd
# 현재 세션에서만 유효
set WANDB_API_KEY=your_api_key_here

# 확인
echo %WANDB_API_KEY%
```

### 방법 2: PowerShell에서 임시 설정

```powershell
# 현재 세션에서만 유효
$env:WANDB_API_KEY="your_api_key_here"

# 확인
echo $env:WANDB_API_KEY
```

### 방법 3: 시스템 환경 변수로 영구 설정 (권장)

1. **시스템 속성 열기**
   - `Win + R` → `sysdm.cpl` 입력 → Enter
   - 또는 제어판 → 시스템 → 고급 시스템 설정

2. **환경 변수 설정**
   - "환경 변수" 버튼 클릭
   - "사용자 변수" 또는 "시스템 변수"에서 "새로 만들기" 클릭
   - 변수 이름: `WANDB_API_KEY`
   - 변수 값: `your_api_key_here`
   - 확인 클릭

3. **새 터미널 열기**
   - 환경 변수는 새로 열린 터미널에서 적용됩니다.

### 방법 4: .env 파일 사용 (Poetry 환경)

```bash
# 프로젝트 루트에 .env 파일 생성
echo WANDB_API_KEY=your_api_key_here > .env
```

Python에서 자동으로 로드하려면 `python-dotenv` 패키지가 필요합니다.

## 모델 다운로드

### CMD에서 실행

```cmd
# 1. API 키 설정
set WANDB_API_KEY=your_api_key_here

# 2. 모델 다운로드
python scripts/download_artifact_for_package.py --artifact artifacts/model/model_trim-sweep-12 --output lex_dpr/models/default_model --project lexdpr --entity zae-park
```

### PowerShell에서 실행

```powershell
# 1. API 키 설정
$env:WANDB_API_KEY="your_api_key_here"

# 2. 모델 다운로드
python scripts/download_artifact_for_package.py --artifact artifacts/model/model_trim-sweep-12 --output lex_dpr/models/default_model --project lexdpr --entity zae-park
```

### Poetry 환경에서 실행

```bash
# 1. API 키 설정 (CMD)
set WANDB_API_KEY=your_api_key_here

# 또는 PowerShell
$env:WANDB_API_KEY="your_api_key_here"

# 2. Poetry 환경에서 실행
poetry run python scripts/download_artifact_for_package.py --artifact artifacts/model/model_trim-sweep-12 --output lex_dpr/models/default_model --project lexdpr --entity zae-park
```

## WandB API 키 확인

WandB 웹사이트에서 API 키를 확인할 수 있습니다:
1. https://wandb.ai/settings 접속
2. "API keys" 섹션에서 키 확인 또는 생성

## 문제 해결

### API 키가 인식되지 않는 경우

1. **터미널 재시작**: 환경 변수 설정 후 새 터미널 열기
2. **대소문자 확인**: `WANDB_API_KEY` (대문자)
3. **공백 확인**: 키 앞뒤에 공백이 없어야 함
4. **따옴표 확인**: PowerShell에서는 따옴표 필요, CMD에서는 불필요

### 모델 다운로드 실패 시

```bash
# WandB 로그인 확인
poetry run python -c "import wandb; wandb.login()"
```

또는 직접 로그인:

```bash
poetry run wandb login
```

