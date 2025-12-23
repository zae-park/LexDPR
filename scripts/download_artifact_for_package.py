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
    if not os.getenv("WANDB_API_KEY"):
        print("⚠️  WANDB_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   export WANDB_API_KEY=your_api_key")
        return None
    
    try:
        api = Api()
        
        # Artifact 경로 파싱
        # 형식: artifacts/model/model_trim-sweep-12 또는 entity/project/artifacts/model/model_trim-sweep-12
        if "/" in artifact_path and not artifact_path.startswith("artifacts/"):
            # entity/project/artifacts/... 형식
            parts = artifact_path.split("/")
            if len(parts) >= 4 and parts[2] == "artifacts":
                entity = parts[0]
                project = parts[1]
                artifact_path = "/".join(parts[2:])
        
        # Artifact 다운로드
        print(f"   전체 경로: {entity}/{project}/{artifact_path}")
        artifact = api.artifact(f"{entity}/{project}/{artifact_path}")
        
        print(f"✅ Artifact 발견: {artifact.name}")
        print(f"   타입: {artifact.type}")
        print(f"   버전: {artifact.version}")
        
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
        
        # training_config.json이 있는지 확인하고, 없으면 생성
        training_config_path = model_dir / "training_config.json"
        if not training_config_path.exists():
            print(f"   ⚠️  training_config.json이 없습니다. 수동으로 설정해야 합니다.")
            print(f"   config.py의 DEFAULT_MAX_LEN을 설정하세요.")
        
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

