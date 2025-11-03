# lex_dpr/models/types.py
from typing import Literal, List, Optional
from pydantic import BaseModel

class BiModelCfg(BaseModel):
    name: str = "BAAI/bge-m3"
    template: Literal["bge","none"] = "bge"
    normalize: bool = True
    max_len: int = 512

class TrainCfg(BaseModel):
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 64
    temperature: float = 0.05
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    eval_steps: int = 300
