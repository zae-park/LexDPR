# scripts/train_cfg.py

import sys

from omegaconf import OmegaConf

from lex_dpr.training.bi_encoder import train_bi


def main():
    base = OmegaConf.load("configs/base.yaml")
    data = OmegaConf.load("configs/data.yaml")
    model = OmegaConf.load("configs/model.yaml")
    cfg = OmegaConf.merge(base, {"data": data, "model": model})

    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))
    train_bi(cfg)


if __name__ == "__main__":
    main()
