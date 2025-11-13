# scripts/train_cfg.py

import sys
import warnings

# FutureWarning 억제 (선택사항)
warnings.filterwarnings("ignore", category=FutureWarning)

from omegaconf import OmegaConf

from lex_dpr.trainer.base_trainer import BiEncoderTrainer


def main():
    base = OmegaConf.load("configs/base.yaml")
    data = OmegaConf.load("configs/data.yaml")
    model = OmegaConf.load("configs/model.yaml")
    cfg = OmegaConf.merge(base, {"data": data, "model": model})

    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))
    trainer = BiEncoderTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
