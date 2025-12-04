"""
WandB Sweep을 위한 Trainer 래퍼

WandB Sweep의 wandb.config를 읽어서 OmegaConf 설정과 병합한 후
BiEncoderTrainer를 실행합니다.
"""

import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from .base_trainer import BiEncoderTrainer

logger = logging.getLogger("lex_dpr.trainer.sweep")


class SweepTrainer:
    """
    WandB Sweep과 통합된 Trainer 래퍼
    
    wandb.config에서 파라미터를 읽어서 기본 설정과 병합한 후
    BiEncoderTrainer를 실행합니다.
    """

    def __init__(self, base_cfg: DictConfig, wandb_config: Optional[dict] = None):
        """
        Args:
            base_cfg: 기본 OmegaConf 설정 (configs/base.yaml 등)
            wandb_config: WandB Sweep에서 제공하는 파라미터 딕셔너리
                         None이면 wandb.config에서 자동으로 읽음
        """
        self.base_cfg = base_cfg
        
        # WandB config 읽기
        if wandb_config is None:
            wandb_config = self._read_wandb_config()
        
        # WandB config를 OmegaConf 형식으로 변환
        if wandb_config:
            sweep_overrides = self._convert_wandb_config(wandb_config)
            logger.info("=" * 80)
            logger.info("🔍 WandB Sweep 파라미터 적용:")
            logger.info(f"   적용된 파라미터 수: {len(sweep_overrides.keys())}")
            logger.info(f"   파라미터 목록: {list(sweep_overrides.keys())}")
            
            # 주요 파라미터 값 로깅
            for key in ['trainer.lr', 'trainer.temperature', 'trainer.weight_decay', 
                       'trainer.warmup_ratio', 'model.peft.r', 'model.peft.alpha']:
                if key in wandb_config:
                    logger.info(f"   {key} = {wandb_config[key]}")
            logger.info("=" * 80)
            logger.info("")
            
            # 기본 설정과 병합
            self.cfg = OmegaConf.merge(base_cfg, sweep_overrides)
        else:
            logger.warning("=" * 80)
            logger.warning("⚠️  WandB config를 찾을 수 없습니다!")
            logger.warning("   기본 설정을 사용합니다.")
            logger.warning("=" * 80)
            logger.warning("")
            self.cfg = base_cfg
        
        # BiEncoderTrainer 생성
        self.trainer = BiEncoderTrainer(self.cfg)
    
    def _read_wandb_config(self) -> Optional[dict]:
        """wandb.config에서 파라미터 읽기"""
        try:
            import wandb
            logger.info("WandB config 읽기 시도 중...")
            
            # wandb.run이 None일 수 있으므로 직접 wandb.config 접근 시도
            if wandb.run is not None:
                logger.info(f"wandb.run 존재: True, sweep_id: {getattr(wandb.run, 'sweep_id', None)}")
                config_dict = dict(wandb.config)
                logger.info(f"wandb.config에서 읽은 파라미터 수: {len(config_dict)}")
                logger.info(f"wandb.config 키 목록: {list(config_dict.keys())[:10]}...")  # 처음 10개만
                return config_dict
            else:
                # wandb.run이 None이지만 wandb.config는 접근 가능할 수 있음
                try:
                    config_dict = dict(wandb.config)
                    if config_dict:
                        logger.info(f"wandb.run이 None이지만 wandb.config에서 읽음: {len(config_dict)}개 파라미터")
                        return config_dict
                except:
                    pass
                
                logger.warning("wandb.run이 None입니다. WandB Sweep 모드가 아닐 수 있습니다.")
        except ImportError:
            logger.warning("wandb가 설치되지 않았습니다.")
        except Exception as e:
            logger.error(f"wandb.config 읽기 실패: {e}", exc_info=True)
        return None
    
    def _convert_wandb_config(self, wandb_config: dict) -> DictConfig:
        """
        WandB config 딕셔너리를 OmegaConf 형식으로 변환
        
        WandB는 점(.)으로 구분된 키를 사용하지만, 중첩 구조로 변환해야 함.
        예: 'trainer.lr' -> {'trainer': {'lr': value}}
        """
        result = {}
        
        for key, value in wandb_config.items():
            # 점(.)으로 구분된 키를 중첩 딕셔너리로 변환
            parts = key.split('.')
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 마지막 키에 값 할당
            current[parts[-1]] = value
        
        return OmegaConf.create(result)
    
    def train(self):
        """학습 실행"""
        return self.trainer.train()


def train_with_sweep(base_cfg: DictConfig, wandb_config: Optional[dict] = None) -> None:
    """
    편의 함수: SweepTrainer를 사용하여 학습 실행
    
    Args:
        base_cfg: 기본 OmegaConf 설정
        wandb_config: WandB Sweep 파라미터 (None이면 wandb.config에서 자동 읽기)
    """
    trainer = SweepTrainer(base_cfg, wandb_config)
    trainer.train()

