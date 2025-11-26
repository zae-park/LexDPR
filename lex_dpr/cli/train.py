"""
학습 엔트리포인트

사용 예시:
  poetry run train
  poetry run train trainer.epochs=5 trainer.lr=3e-5
"""

import sys
import warnings
from pathlib import Path

# FutureWarning 억제 (선택사항)
warnings.filterwarnings("ignore", category=FutureWarning)

from omegaconf import OmegaConf

from lex_dpr.trainer.base_trainer import BiEncoderTrainer


def _get_config_path(filename: str) -> Path:
    """설정 파일 경로 반환 (사용자 configs 우선, 없으면 패키지 기본값)"""
    user_configs_dir = Path.cwd() / "configs"
    user_path = user_configs_dir / filename
    
    if user_path.exists():
        return user_path
    
    # 패키지 내부 기본값 사용
    import lex_dpr.configs
    package_configs_dir = Path(lex_dpr.configs.__file__).parent
    return package_configs_dir / filename


def main():
    """학습 메인 함수"""
    # 설정 파일 로드 (사용자 configs 우선, 없으면 패키지 기본값)
    base_path = _get_config_path("base.yaml")
    data_path = _get_config_path("data.yaml")
    model_path = _get_config_path("model.yaml")
    
    base = OmegaConf.load(base_path)
    
    # data.yaml 로드 및 병합 (원래 로직과 동일하게)
    if data_path.exists():
        data = OmegaConf.load(data_path)
        base = OmegaConf.merge(base, {"data": data})
    
    # model.yaml 로드 및 병합 (원래 로직과 동일하게)
    if model_path.exists():
        model = OmegaConf.load(model_path)
        base = OmegaConf.merge(base, {"model": model})
    
    cfg = base

    # 커맨드라인 인자로 오버라이드 (예: trainer.epochs=5)
    overrides = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, overrides)

    print(OmegaConf.to_yaml(cfg))
    trainer = BiEncoderTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

