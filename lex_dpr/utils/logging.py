# lex_dpr/utils/logging.py
import logging, time, json, os
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, logdir: Optional[str] = None, jsonl_path: Optional[str] = None):
        self.logger = logging.getLogger("lexdpr")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)
        self.tb = SummaryWriter(log_dir=logdir) if logdir else None
        self.jsonl_path = jsonl_path
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    def info(self, msg: str): self.logger.info(msg)

    def scalars(self, step: int, **metrics: float):
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)
        if self.jsonl_path:
            rec: Dict[str, Any] = {"time": int(time.time()), "step": step, **metrics}
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
