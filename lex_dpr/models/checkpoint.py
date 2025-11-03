# lex_dpr/models/checkpoint.py
import os, math

class BestScoreSaver:
    def __init__(self, out_dir: str, metric_key: str = "cosine_accuracy_at_10"):
        self.best = -math.inf
        self.key = metric_key
        self.path = os.path.join(out_dir, "bi_encoder_best")

    def __call__(self, metrics: dict, model):
        val = float(metrics.get(self.key, float("nan")))
        if val != val:  # NaN
            return False
        if val > self.best:
            self.best = val
            model.save(self.path)
            return True
        return False
