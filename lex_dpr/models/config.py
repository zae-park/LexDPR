# lex_dpr/models/config.py
"""
패키지 배포 시 사용할 기본 모델 설정

패키지 배포 시 이 파일의 설정을 업데이트하여
특정 학습된 모델을 기본으로 사용하도록 설정할 수 있습니다.

방법 1: 패키지에 모델 포함 (권장)
  - DEFAULT_MODEL_PATH: 패키지에 포함된 모델 경로 (상대 경로)
  - DEFAULT_MAX_LEN: 학습 시 사용된 max_seq_length
  - 패키지에 PEFT 어댑터만 포함 (작은 크기)

방법 2: WandB에서 자동 다운로드
  - DEFAULT_RUN_ID: WandB Run ID
  - DEFAULT_MAX_LEN: 학습 시 사용된 max_seq_length
  - 첫 실행 시 자동 다운로드, 이후 캐시 재사용
"""

# 방법 1: 패키지에 포함된 모델 사용 (권장)
# 패키지에 모델을 포함시킨 경우 이 경로를 설정하세요.
# 예: "models/default_model" (패키지 내부 상대 경로)
# 주의: "lex_dpr/" 접두사는 필요 없습니다. 패키지 루트 기준 상대 경로입니다.
DEFAULT_MODEL_PATH: str | None = "models/default_model"  # 패키지에 포함된 모델 경로

# 학습 시 사용된 max_seq_length (패키지에 포함된 모델 또는 WandB 다운로드 모델 모두에 적용)
# 이 값이 설정되면 BiEncoder가 자동으로 max_seq_length로 사용합니다.
# sweep.yaml에서 model.max_len: 128로 설정되어 있음
DEFAULT_MAX_LEN: int | None = 128  # 학습 시 사용된 max_seq_length

# 방법 2: WandB에서 자동 다운로드 (DEFAULT_MODEL_PATH가 None인 경우 사용)
# 기본 WandB Run ID (패키지 배포 시 업데이트)
# 사용자가 run ID를 몰라도 자동으로 이 모델을 다운로드하여 사용합니다.
DEFAULT_RUN_ID: str | None = None  # 예: "abc123xyz"

# 기본 WandB 프로젝트 및 엔티티
DEFAULT_WANDB_PROJECT: str = "lexdpr"
DEFAULT_WANDB_ENTITY: str = "zae-park"

# 기본 다운로드 경로 (사용자 홈 디렉토리 기준)
DEFAULT_MODEL_CACHE_DIR: str = "~/.lexdpr/models"

# 기본 메트릭 (최고 성능 모델 선택 시 사용)
DEFAULT_METRIC: str = "eval/recall_at_10"
DEFAULT_GOAL: str = "maximize"

