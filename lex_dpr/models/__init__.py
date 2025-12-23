# lex_dpr/models/__init__.py
"""
LexDPR 모델 모듈

주요 클래스:
- BiEncoder: 질의/패시지 임베딩 생성
- TemplateMode: 템플릿 모드 (BGE/NONE)
"""

from .encoders import BiEncoder
from .templates import TemplateMode
from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MAX_LEN,
    DEFAULT_RUN_ID,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_MODEL_CACHE_DIR,
)

__all__ = [
    "BiEncoder",
    "TemplateMode",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MAX_LEN",
    "DEFAULT_RUN_ID",
    "DEFAULT_WANDB_PROJECT",
    "DEFAULT_WANDB_ENTITY",
    "DEFAULT_MODEL_CACHE_DIR",
]

