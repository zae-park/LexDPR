#!/usr/bin/env python3
"""
WandB artifact를 패키지에 포함하기 위해 다운로드하는 스크립트

사용법:
    python scripts/download_artifact_for_package.py \
        --artifact artifacts/model/model_trim-sweep-12 \
        --output lex_dpr/models/default_model \
        --project lexdpr \
        --entity zae-park
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import wandb
    from wandb import Api
except ImportError:
    print("wandb가 설치되지 않았습니다. 'pip install wandb' 또는 'poetry install --extras wandb'로 설치하세요.")
    sys.exit(1)


def download_artifact(
    artifact_path: str,
    output_dir: Path,
    project: str = "lexdpr",
    entity: str = "zae-park",
) -> Optional[Path]:
    """WandB artifact를 다운로드합니다."""
    print(f"\n📥 Artifact 다운로드 중...")
    print(f"   Artifact: {artifact_path}")
    print(f"   Project: {project}, Entity: {entity}")
    
    # WandB 로그인 확인
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("⚠️  WANDB_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\nWindows 환경에서 설정 방법:")
        print("  CMD:     set WANDB_API_KEY=your_api_key")
        print("  PowerShell: $env:WANDB_API_KEY=\"your_api_key\"")
        print("\n또는 WandB 로그인:")
        print("  poetry run wandb login")
        print("  또는: python -c \"import wandb; wandb.login()\"")
        return None
    
    try:
        api = Api()
        
        # Artifact 경로 파싱
        # WandB artifact 경로 형식: entity/project/artifact_name:version
        # 입력 형식: artifacts/model/model_trim-sweep-12 또는 model_trim-sweep-12
        
        # artifacts/ 접두사 제거
        if artifact_path.startswith("artifacts/"):
            artifact_path = artifact_path[len("artifacts/"):]
        
        # model/ 접두사가 있으면 제거 (artifact 타입은 별도로 지정)
        artifact_name = artifact_path
        artifact_type = "model"
        if "/" in artifact_path:
            parts = artifact_path.split("/")
            if len(parts) >= 2:
                artifact_type = parts[0]  # 예: "model"
                artifact_name = "/".join(parts[1:])  # 예: "model_trim-sweep-12"
        
        # WandB API 형식: entity/project/artifact_name:version
        # artifact_name에 버전이 포함되어 있지 않으면 최신 버전 사용
        full_artifact_path = f"{entity}/{project}/{artifact_name}"
        
        # 버전이 명시되지 않았으면 최신 버전 사용
        if ":" not in artifact_name:
            full_artifact_path = f"{full_artifact_path}:latest"
        
        print(f"   Artifact 경로: {full_artifact_path}")
        print(f"   Artifact 타입: {artifact_type}")
        
        # Artifact 다운로드
        try:
            artifact = api.artifact(full_artifact_path)
        except Exception as e:
            # latest가 실패하면 버전 없이 시도
            if ":latest" in full_artifact_path:
                print(f"   ⚠️  :latest 버전 실패, 버전 없이 시도...")
                full_artifact_path = f"{entity}/{project}/{artifact_name}"
                artifact = api.artifact(full_artifact_path)
            else:
                raise
        
        print(f"✅ Artifact 발견: {artifact.name}")
        print(f"   타입: {artifact.type}")
        print(f"   버전: {artifact.version}")
        
        # Artifact를 생성한 run 정보 가져오기
        run = None
        try:
            # Artifact의 사용된 run 찾기
            if hasattr(artifact, 'used_by'):
                used_by = artifact.used_by()
                if used_by and len(used_by) > 0:
                    # 사용된 run이 있으면 첫 번째 run 사용
                    run_id = used_by[0].id if hasattr(used_by[0], 'id') else str(used_by[0])
                    run = api.run(f"{entity}/{project}/{run_id}")
                    print(f"   Run 정보 발견: {run.name} (ID: {run.id})")
            elif hasattr(artifact, 'logged_by'):
                # 로그한 run 사용
                logged_by = artifact.logged_by()
                if logged_by:
                    run_id = logged_by.id if hasattr(logged_by, 'id') else str(logged_by)
                    run = api.run(f"{entity}/{project}/{run_id}")
                    print(f"   Run 정보 발견: {run.name} (ID: {run.id})")
        except Exception as e:
            print(f"   ⚠️  Run 정보 가져오기 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 다운로드
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = artifact.download(root=str(output_dir))
        artifact_path_obj = Path(artifact_dir)
        
        print(f"✅ 다운로드 완료: {artifact_path_obj}")
        
        # artifact 내부에 bi_encoder 디렉토리가 있는지 확인
        if (artifact_path_obj / "bi_encoder").exists():
            model_dir = artifact_path_obj / "bi_encoder"
            print(f"   모델 디렉토리: {model_dir}")
        elif (artifact_path_obj / "adapter_config.json").exists():
            model_dir = artifact_path_obj
            print(f"   모델 디렉토리: {model_dir} (adapter_config.json 발견)")
        else:
            # 하위 디렉토리 확인
            model_dir = None
            for subdir in artifact_path_obj.iterdir():
                if subdir.is_dir() and (subdir / "adapter_config.json").exists():
                    model_dir = subdir
                    print(f"   모델 디렉토리: {model_dir} (하위 디렉토리에서 발견)")
                    break
            
            if model_dir is None:
                model_dir = artifact_path_obj
                print(f"   모델 디렉토리: {model_dir} (기본 경로 사용)")
        
        # training_config.json이 있는지 확인하고, 없으면 WandB run에서 생성
        training_config_path = model_dir / "training_config.json"
        if not training_config_path.exists() and run:
            try:
                # Run 정보에서 config 가져오기
                config = run.config
                
                # 학습 시 사용된 max_len 찾기
                max_len = None
                if "model" in config and isinstance(config["model"], dict):
                    max_len = config["model"].get("max_len")
                elif "max_len" in config:
                    max_len = config["max_len"]
                
                # template 정보 찾기
                use_bge_template = True
                if "model" in config and isinstance(config["model"], dict):
                    use_bge_template = config["model"].get("use_bge_template", True)
                elif "use_bge_template" in config:
                    use_bge_template = config["use_bge_template"]
                
                # training_config.json 생성
                import json
                training_config = {
                    "max_len": max_len,
                    "use_bge_template": use_bge_template,
                    "run_id": run.id,
                    "run_name": run.name,
                    "project": run.project,
                    "entity": run.entity,
                }
                
                with open(training_config_path, "w", encoding="utf-8") as f:
                    json.dump(training_config, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ training_config.json 생성 완료: {training_config_path}")
                if max_len:
                    print(f"   학습 시 사용된 max_len: {max_len}")
            except Exception as e:
                print(f"   ⚠️  training_config.json 생성 실패: {e}")
                print(f"   수동으로 config.py의 DEFAULT_MAX_LEN을 설정하세요.")
        elif not training_config_path.exists():
            print(f"   ⚠️  training_config.json이 없고 run 정보도 없습니다.")
            print(f"   수동으로 config.py의 DEFAULT_MAX_LEN을 설정하세요.")
        
        return model_dir
        
    except Exception as e:
        print(f"❌ Artifact 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="WandB artifact를 패키지에 포함하기 위해 다운로드"
    )
    parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Artifact 경로 (예: artifacts/model/model_trim-sweep-12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 디렉토리 (예: lex_dpr/models/default_model)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="lexdpr",
        help="WandB 프로젝트 이름 (기본값: lexdpr)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="zae-park",
        help="WandB entity 이름 (기본값: zae-park)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # 다운로드
    model_dir = download_artifact(
        artifact_path=args.artifact,
        output_dir=output_dir,
        project=args.project,
        entity=args.entity,
    )
    
    if model_dir:
        print(f"\n✅ 모델 다운로드 완료!")
        print(f"   경로: {model_dir}")
        print(f"\n다음 단계:")
        print(f"1. config.py 업데이트:")
        print(f"   DEFAULT_MODEL_PATH = \"models/default_model\"")
        print(f"   DEFAULT_MAX_LEN = <학습 시 사용된 max_len>")
        print(f"2. pyproject.toml에 모델 파일 포함 설정 확인")
        print(f"3. 패키지 빌드: poetry build")


if __name__ == "__main__":
    main()

