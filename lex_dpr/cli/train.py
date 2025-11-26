"""
학습 엔트리포인트

사용 예시:
  poetry run train
  poetry run train trainer.epochs=5 trainer.lr=3e-5
"""

import sys
import warnings

# FutureWarning 억제 (선택사항)
warnings.filterwarnings("ignore", category=FutureWarning)

from omegaconf import OmegaConf

from lex_dpr.trainer.base_trainer import BiEncoderTrainer


def main():
    """학습 메인 함수"""
    base = OmegaConf.load("configs/base.yaml")
    data = OmegaConf.load("configs/data.yaml")
    model = OmegaConf.load("configs/model.yaml")
    cfg = OmegaConf.merge(base, {"data": data, "model": model})

    # 커맨드라인 인자로 오버라이드 (예: trainer.epochs=5)
    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))
    trainer = BiEncoderTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

