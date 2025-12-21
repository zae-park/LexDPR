"""
웹 로깅 서비스 통합 (WandB, MLflow)

사용 예시:
  # config에서 설정
  web_logging:
    service: wandb  # wandb, mlflow
    token: ${WANDB_API_KEY}  # 환경 변수 또는 직접 입력
    project: lexdpr
    name: experiment-001
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("lex_dpr.web_logging")


class WebLogger:
    """웹 로깅 서비스 통합 래퍼"""
    
    def __init__(self, service: str, token: Optional[str] = None, **kwargs):
        """
        Args:
            service: 'wandb', 'mlflow'
            token: API 토큰 (환경 변수에서 읽거나 직접 입력)
            **kwargs: 각 서비스별 추가 설정
        """
        self.service = service.lower()
        self.token = token
        self.logger_impl = None
        self.is_active = False
        
        if not token:
            logger.warning(f"{service} 토큰이 제공되지 않았습니다. 웹 로깅을 건너뜁니다.")
            return
        
        try:
            if self.service == "wandb":
                self._init_wandb(token, **kwargs)
            elif self.service == "mlflow":
                self._init_mlflow(token, **kwargs)
            else:
                logger.warning(f"지원하지 않는 웹 로깅 서비스: {service}. 지원 서비스: wandb, mlflow")
                return
            
            self.is_active = True
            logger.info(f"✅ {service.upper()} 웹 로깅 초기화 완료")
        except Exception as e:
            logger.warning(f"{service} 초기화 실패: {e}. 웹 로깅을 건너뜁니다.")
            self.is_active = False
    
    def _init_wandb(self, token: str, project: str, name: Optional[str] = None, **kwargs):
        """WandB 초기화"""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
            raise
        
        # WandB 로그인
        wandb.login(key=token)
        
        # Sweep 모드에서는 wandb.agent()가 이미 wandb.init()을 호출했을 수 있음
        # 이미 run이 존재하는 경우 재초기화하지 않음
        if wandb.run is not None:
            logger.info("WandB run이 이미 존재합니다. 기존 run을 사용합니다.")
            logger.info(f"  기존 run ID: {wandb.run.id}")
            logger.info(f"  기존 run name: {wandb.run.name}")
            self.logger_impl = wandb.run
            return
        
        # run_name 충돌 방지를 위해 reinit=True 옵션 추가
        init_kwargs = kwargs.copy()
        init_kwargs["name"] = name
        # reinit=True로 설정하여 기존 run과 충돌 시 재초기화 허용
        init_kwargs["reinit"] = True
        
        run = wandb.init(
            project=project,
            **init_kwargs
        )
        self.logger_impl = run
        logger.info(f"WandB 프로젝트: {project}, 실행 이름: {run.name}")
    
    def _init_mlflow(self, token: str, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None, run_name: Optional[str] = None, **kwargs):
        """MLflow 초기화"""
        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow가 설치되지 않았습니다. 'poetry install --extras mlflow'로 설치하세요.")
            raise
        
        # MLflow는 토큰을 직접 사용하지 않고, tracking_uri에 포함하거나 환경 변수로 설정
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except Exception:
                # 이미 존재하는 경우
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else None
            
            if experiment_id:
                mlflow.set_experiment(experiment_name)
        
        run = mlflow.start_run(run_name=run_name, **kwargs)
        self.logger_impl = run
        logger.info(f"MLflow 추적 URI: {mlflow.get_tracking_uri()}, 실험: {experiment_name}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """하이퍼파라미터 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "wandb":
                self.logger_impl.config.update(params)
            elif self.service == "mlflow":
                try:
                    import mlflow
                    mlflow.log_params(params)
                except ImportError:
                    logger.warning("mlflow가 설치되지 않았습니다. 파라미터 로깅을 건너뜁니다.")
        except Exception as e:
            logger.warning(f"파라미터 로깅 실패: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """메트릭 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "wandb":
                if step is not None:
                    self.logger_impl.log(metrics, step=step)
                else:
                    self.logger_impl.log(metrics)
            elif self.service == "mlflow":
                try:
                    import mlflow
                    if step is not None:
                        mlflow.log_metrics(metrics, step=step)
                    else:
                        mlflow.log_metrics(metrics)
                except ImportError:
                    logger.warning("mlflow가 설치되지 않았습니다. 메트릭 로깅을 건너뜁니다.")
        except Exception as e:
            logger.warning(f"메트릭 로깅 실패: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """아티팩트(파일 또는 디렉토리) 로깅"""
        if not self.is_active:
            return
        
        try:
            if self.service == "wandb":
                import wandb
                import os
                from pathlib import Path
                
                path = Path(local_path)
                base_artifact_name = artifact_path or "model"
                
                # WandB run 이름을 포함하여 아티팩트 이름을 고유하게 만들기
                # 같은 run 내에서도 여러 번 업로드할 수 있도록 타임스탬프 추가
                try:
                    run_name = wandb.run.name if wandb.run else None
                    if run_name:
                        # run 이름을 안전한 형식으로 변환 (특수 문자 제거)
                        safe_run_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in run_name)
                        artifact_name = f"{base_artifact_name}_{safe_run_name}"
                    else:
                        # run 이름이 없으면 타임스탬프 사용
                        import time
                        timestamp = int(time.time())
                        artifact_name = f"{base_artifact_name}_{timestamp}"
                except Exception:
                    # run 정보를 가져올 수 없으면 기본 이름 사용
                    artifact_name = base_artifact_name
                
                # 디렉토리인 경우 Artifact 객체 사용
                if path.is_dir():
                    artifact = wandb.Artifact(name=artifact_name, type="model")
                    artifact.add_dir(str(path))
                    self.logger_impl.log_artifact(artifact)
                    logger.info(f"WandB에 디렉토리 artifact 업로드: {local_path} -> {artifact_name}")
                # 파일인 경우
                elif path.is_file():
                    artifact = wandb.Artifact(name=artifact_name, type="model")
                    artifact.add_file(str(path))
                    self.logger_impl.log_artifact(artifact)
                    logger.info(f"WandB에 파일 artifact 업로드: {local_path} -> {artifact_name}")
                else:
                    logger.warning(f"경로를 찾을 수 없습니다: {local_path}")
            elif self.service == "mlflow":
                try:
                    import mlflow
                    mlflow.log_artifact(local_path, artifact_path)
                except ImportError:
                    logger.warning("mlflow가 설치되지 않았습니다. 아티팩트 로깅을 건너뜁니다.")
        except Exception as e:
            logger.warning(f"아티팩트 로깅 실패: {e}")
    
    def finish(self) -> None:
        """로깅 종료"""
        if not self.is_active:
            return
        
        try:
            if self.service == "wandb":
                self.logger_impl.finish()
            elif self.service == "mlflow":
                try:
                    import mlflow
                    mlflow.end_run()
                except ImportError:
                    logger.warning("mlflow가 설치되지 않았습니다. 로깅 종료를 건너뜁니다.")
                    return
            logger.info(f"{self.service.upper()} 로깅 종료")
        except Exception as e:
            logger.warning(f"로깅 종료 실패: {e}")


def _create_single_web_logger(web_logging_cfg) -> Optional[WebLogger]:
    """단일 웹 로깅 서비스 설정에서 WebLogger 생성"""
    service = getattr(web_logging_cfg, "service", None)
    if not service:
        return None
    
    # 서비스 이름 정규화
    service = str(service).lower().strip()
    
    # 지원하는 서비스인지 확인
    if service not in ["wandb", "mlflow"]:
        logger.warning(f"지원하지 않는 웹 로깅 서비스: {service}. 지원 서비스: wandb, mlflow")
        return None
    
    # 토큰 가져오기 (환경 변수 또는 직접 입력)
    token = getattr(web_logging_cfg, "token", None)
    if token and isinstance(token, str) and token.startswith("${") and token.endswith("}"):
        # 환경 변수 참조 (예: ${WANDB_API_KEY})
        env_var = token[2:-1]
        token = os.getenv(env_var)
        if not token:
            logger.warning(f"환경 변수 {env_var}를 찾을 수 없습니다.")
            return None
    
    if not token:
        logger.warning(f"{service} 토큰이 제공되지 않았습니다.")
        return None
    
    # 서비스별 설정 추출 (지정된 서비스에 대해서만)
    kwargs = {}
    if service == "wandb":
        kwargs["project"] = getattr(web_logging_cfg, "project", "lexdpr")
        kwargs["name"] = getattr(web_logging_cfg, "name", None)
        kwargs["entity"] = getattr(web_logging_cfg, "entity", None)
    elif service == "mlflow":
        kwargs["tracking_uri"] = getattr(web_logging_cfg, "tracking_uri", None)
        kwargs["experiment_name"] = getattr(web_logging_cfg, "experiment_name", "lexdpr")
        kwargs["run_name"] = getattr(web_logging_cfg, "run_name", None)
    
    # 지정된 서비스만 초기화
    try:
        return WebLogger(service=service, token=token, **kwargs)
    except Exception as e:
        logger.warning(f"{service} 웹 로거 생성 실패: {e}")
        return None


class MultiWebLogger:
    """여러 웹 로깅 서비스를 동시에 관리하는 래퍼"""
    
    def __init__(self, loggers: list[WebLogger]):
        self.loggers = [lg for lg in loggers if lg and lg.is_active]
        self.is_active = len(self.loggers) > 0
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """모든 활성 웹 로거에 하이퍼파라미터 로깅"""
        for logger in self.loggers:
            logger.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """모든 활성 웹 로거에 메트릭 로깅"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """모든 활성 웹 로거에 아티팩트 로깅"""
        for logger in self.loggers:
            logger.log_artifact(local_path, artifact_path)
    
    def finish(self) -> None:
        """모든 활성 웹 로거 종료"""
        for logger in self.loggers:
            logger.finish()


def create_web_logger(cfg) -> Optional[MultiWebLogger]:
    """
    설정에서 웹 로거 생성 (단일 또는 다중 서비스 지원)
    
    Args:
        cfg: OmegaConf 설정 객체 (web_logging 섹션 포함)
    
    Returns:
        MultiWebLogger 인스턴스 (여러 서비스 래핑) 또는 None
    """
    web_logging_cfg = getattr(cfg, "web_logging", None)
    if not web_logging_cfg:
        return None
    
    # 리스트 형태인지 확인 (여러 서비스)
    if isinstance(web_logging_cfg, (list, tuple)):
        loggers = []
        for item in web_logging_cfg:
            logger_impl = _create_single_web_logger(item)
            if logger_impl:
                loggers.append(logger_impl)
        
        if not loggers:
            return None
        
        return MultiWebLogger(loggers)
    
    # 단일 서비스 (기존 방식)
    logger_impl = _create_single_web_logger(web_logging_cfg)
    if not logger_impl:
        return None
    
    return MultiWebLogger([logger_impl])


def upload_artifact_to_existing_run(
    run_id: str,
    local_path: str,
    artifact_path: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """
    기존 WandB run에 artifact를 업로드합니다.
    
    Args:
        run_id: WandB run ID (예: "abc123")
        local_path: 업로드할 로컬 파일/디렉토리 경로
        artifact_path: 아티팩트 이름 (기본값: "model")
        project: WandB 프로젝트 이름 (기본값: run에서 자동 감지)
        entity: WandB entity 이름 (기본값: run에서 자동 감지)
        token: WandB API 토큰 (기본값: 환경 변수에서 읽기)
    
    Returns:
        업로드 성공 여부
    """
    try:
        import wandb
        from pathlib import Path
    except ImportError:
        logger.warning("wandb가 설치되지 않았습니다. 'poetry install --extras wandb'로 설치하세요.")
        return False
    
    # WandB 로그인
    if token:
        wandb.login(key=token)
    elif os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        logger.warning("WandB API 토큰이 제공되지 않았습니다.")
        return False
    
    try:
        # 방법 1: wandb.init(resume="allow")를 사용하여 run을 resume하고 artifact 업로드
        # 이 방법이 종료된 run에도 확실하게 작동합니다
        
        # run 경로 구성 (init에 필요한 정보)
        init_kwargs = {
            "id": run_id,
            "resume": "allow",  # 종료된 run도 resume 가능
        }
        
        if project:
            init_kwargs["project"] = project
        if entity:
            init_kwargs["entity"] = entity
        
        # Run을 resume하여 artifact 업로드
        with wandb.init(**init_kwargs) as resumed_run:
            logger.info(f"기존 run을 resume했습니다: {resumed_run.name} (ID: {resumed_run.id})")
            logger.info(f"Run 상태: {resumed_run.state}")
            
            # 아티팩트 이름 설정
            base_artifact_name = artifact_path or "model"
            safe_run_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in resumed_run.name)
            artifact_name = f"{base_artifact_name}_{safe_run_name}"
            
            # 아티팩트 생성 및 업로드
            path = Path(local_path)
            if not path.exists():
                logger.warning(f"경로를 찾을 수 없습니다: {local_path}")
                return False
            
            artifact = wandb.Artifact(name=artifact_name, type="model")
            
            if path.is_dir():
                artifact.add_dir(str(path))
                logger.info(f"디렉토리를 artifact에 추가: {local_path}")
            elif path.is_file():
                artifact.add_file(str(path))
                logger.info(f"파일을 artifact에 추가: {local_path}")
            else:
                logger.warning(f"경로가 파일도 디렉토리도 아닙니다: {local_path}")
                return False
            
            # Resume된 run에 artifact 업로드
            resumed_run.log_artifact(artifact)
            logger.info(f"✅ 기존 run에 artifact 업로드 완료: {artifact_name}")
            logger.info(f"   Run: {resumed_run.name} ({resumed_run.id})")
            logger.info(f"   Artifact: {artifact_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"기존 run에 artifact 업로드 실패: {e}")
        import traceback
        logger.debug(f"상세 에러: {traceback.format_exc()}")
        return False

