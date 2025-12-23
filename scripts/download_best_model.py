#!/usr/bin/env python3
"""
WandB Sweep에서 최고 성능 run을 찾아 모델 artifact를 다운로드하는 스크립트

사용법:
    python scripts/download_best_model.py --sweep-id <sweep-id> --metric eval/recall_at_10
    python scripts/download_best_model.py --sweep-id <sweep-id> --output-dir checkpoint/best_model
    python scripts/download_best_model.py --project lexdpr --entity zae-park --metric eval/recall_at_10
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


def find_best_run(
    sweep_id: Optional[str] = None,
    project: str = "lexdpr",
    entity: str = "zae-park",
    metric: str = "eval/recall_at_10",
    goal: str = "maximize",
) -> Optional[wandb.apis.public.Run]:
    """WandB sweep에서 최고 성능 run 찾기"""
    api = Api()
    
    if sweep_id:
        # Sweep ID로 직접 접근
        try:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            runs = list(sweep.runs)
        except Exception as e:
            print(f"⚠️  Sweep을 찾을 수 없습니다: {e}")
            print(f"   Sweep ID: {sweep_id}")
            print(f"   Project: {project}, Entity: {entity}")
            return None
    else:
        # Project에서 모든 run 검색
        runs = api.runs(f"{entity}/{project}")
    
    if not runs:
        print("⚠️  실행된 run이 없습니다.")
        return None
    
    print(f"📊 총 {len(runs)}개의 run을 검색 중...")
    
    # 성공한 run만 필터링 (실패한 run 제외)
    successful_runs = []
    for run in runs:
        if run.state == "finished" and run.summary:
            # 메트릭이 있는지 확인
            metric_key = metric.replace("@", "_at_")  # WandB는 @를 _at_로 변환
            if metric_key in run.summary:
                successful_runs.append(run)
    
    if not successful_runs:
        print(f"⚠️  성공한 run을 찾을 수 없습니다.")
        print(f"   찾는 메트릭: {metric}")
        return None
    
    print(f"✅ 성공한 run: {len(successful_runs)}개")
    
    # 최고 성능 run 찾기
    best_run = None
    best_score = float('-inf') if goal == "maximize" else float('inf')
    metric_key = metric.replace("@", "_at_")
    
    for run in successful_runs:
        score = run.summary.get(metric_key)
        if score is None:
            continue
        
        is_better = False
        if goal == "maximize":
            is_better = score > best_score
        else:
            is_better = score < best_score
        
        if is_better:
            best_score = score
            best_run = run
    
    if best_run:
        print(f"\n🏆 최고 성능 run 발견!")
        print(f"   Run ID: {best_run.id}")
        print(f"   Run Name: {best_run.name}")
        print(f"   {metric}: {best_score:.4f}")
        print(f"   URL: {best_run.url}")
        return best_run
    else:
        print(f"⚠️  최고 성능 run을 찾을 수 없습니다.")
        return None


def download_model_artifact(
    run: wandb.apis.public.Run,
    output_dir: Path,
    artifact_name: str = "model",
) -> Optional[Path]:
    """WandB run에서 모델 artifact 다운로드 및 학습 설정 저장"""
    print(f"\n📥 Artifact 다운로드 중...")
    print(f"   Run: {run.name}")
    print(f"   Artifact: {artifact_name}")
    
    try:
        # Run의 artifact 목록 확인
        artifacts = run.logged_artifacts()
        if not artifacts:
            print("⚠️  이 run에 artifact가 없습니다.")
            # Checkpoint 경로 확인 (로컬에 저장된 경우)
            print("   로컬 checkpoint 경로를 확인하세요.")
            return None
        
        # 모델 artifact 찾기
        model_artifact = None
        for artifact in artifacts:
            if artifact_name in artifact.name.lower() or "model" in artifact.name.lower():
                model_artifact = artifact
                break
        
        if not model_artifact:
            print(f"⚠️  '{artifact_name}' artifact를 찾을 수 없습니다.")
            print(f"   사용 가능한 artifact:")
            for artifact in artifacts:
                print(f"     - {artifact.name}")
            return None
        
        print(f"✅ Artifact 발견: {model_artifact.name}")
        
        # Artifact 다운로드
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = model_artifact.download(root=str(output_dir))
        artifact_path = Path(artifact_dir)
        
        # 학습 설정 정보 저장 (run config에서 max_len 등 정보 가져오기)
        try:
            import json
            config = run.config
            
            # 학습 시 사용된 max_len 찾기
            max_len = None
            if "model" in config and isinstance(config["model"], dict):
                max_len = config["model"].get("max_len")
            elif "max_len" in config:
                max_len = config["max_len"]
            
            # template 정보 찾기
            use_bge_template = True  # 기본값
            if "model" in config and isinstance(config["model"], dict):
                use_bge_template = config["model"].get("use_bge_template", True)
            elif "use_bge_template" in config:
                use_bge_template = config["use_bge_template"]
            
            # 학습 설정 정보 저장
            training_config = {
                "max_len": max_len,
                "use_bge_template": use_bge_template,
                "run_id": run.id,
                "run_name": run.name,
                "project": run.project,
                "entity": run.entity,
            }
            
            # training_config.json 파일로 저장
            training_config_path = artifact_path / "training_config.json"
            with open(training_config_path, "w", encoding="utf-8") as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 학습 설정 정보 저장 완료: {training_config_path}")
            if max_len:
                print(f"   학습 시 사용된 max_len: {max_len}")
            print(f"   BGE 템플릿 사용: {use_bge_template}")
            
        except Exception as e:
            print(f"⚠️  학습 설정 정보 저장 실패 (무시하고 계속): {e}")
        
        print(f"✅ 다운로드 완료: {artifact_dir}")
        return artifact_path
        
    except Exception as e:
        print(f"❌ Artifact 다운로드 실패: {e}")
        return None


def get_checkpoint_path_from_run(run: wandb.apis.public.Run) -> Optional[str]:
    """Run의 config에서 checkpoint 경로 확인"""
    try:
        # WandB config에서 checkpoint 경로 확인
        config = run.config
        if "out_dir" in config:
            out_dir = config["out_dir"]
            checkpoint_path = f"{out_dir}/bi_encoder"
            return checkpoint_path
        elif "checkpoint" in config:
            return config["checkpoint"]
    except Exception:
        pass
    
    # 기본 경로
    return "checkpoint/lexdpr/bi_encoder"


def main():
    parser = argparse.ArgumentParser(
        description="WandB Sweep에서 최고 성능 run의 모델을 다운로드"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="WandB Sweep ID (없으면 project의 모든 run 검색)",
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
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/recall_at_10",
        help="최적화할 메트릭 (기본값: eval/recall_at_10)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        choices=["maximize", "minimize"],
        default="maximize",
        help="메트릭 최적화 목표 (기본값: maximize)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint/best_model",
        help="모델 다운로드 경로 (기본값: checkpoint/best_model)",
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default="model",
        help="다운로드할 artifact 이름 (기본값: model)",
    )
    parser.add_argument(
        "--use-local-checkpoint",
        action="store_true",
        help="WandB artifact 대신 로컬 checkpoint 경로 사용 (run의 config에서 확인)",
    )
    
    args = parser.parse_args()
    
    # WandB 로그인 확인
    if not os.getenv("WANDB_API_KEY"):
        print("⚠️  WANDB_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   export WANDB_API_KEY=your_api_key")
        return
    
    # 최고 성능 run 찾기
    best_run = find_best_run(
        sweep_id=args.sweep_id,
        project=args.project,
        entity=args.entity,
        metric=args.metric,
        goal=args.goal,
    )
    
    if not best_run:
        print("\n❌ 최고 성능 run을 찾을 수 없습니다.")
        return
    
    # 모델 다운로드
    output_dir = Path(args.output_dir)
    
    if args.use_local_checkpoint:
        # 로컬 checkpoint 경로 확인
        checkpoint_path = get_checkpoint_path_from_run(best_run)
        print(f"\n📁 로컬 checkpoint 경로: {checkpoint_path}")
        if Path(checkpoint_path).exists():
            print(f"✅ 로컬 checkpoint 발견: {checkpoint_path}")
            # 심볼릭 링크 또는 복사
            output_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            target_path = output_dir / "bi_encoder"
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(checkpoint_path, target_path)
            
            # 학습 설정 정보 저장 (로컬 checkpoint인 경우에도)
            try:
                import json
                config = best_run.config
                max_len = None
                if "model" in config and isinstance(config["model"], dict):
                    max_len = config["model"].get("max_len")
                elif "max_len" in config:
                    max_len = config["max_len"]
                
                use_bge_template = True
                if "model" in config and isinstance(config["model"], dict):
                    use_bge_template = config["model"].get("use_bge_template", True)
                elif "use_bge_template" in config:
                    use_bge_template = config["use_bge_template"]
                
                training_config = {
                    "max_len": max_len,
                    "use_bge_template": use_bge_template,
                    "run_id": best_run.id,
                    "run_name": best_run.name,
                    "project": best_run.project,
                    "entity": best_run.entity,
                }
                
                training_config_path = target_path / "training_config.json"
                with open(training_config_path, "w", encoding="utf-8") as f:
                    json.dump(training_config, f, indent=2, ensure_ascii=False)
                print(f"✅ 학습 설정 정보 저장 완료: {training_config_path}")
                if max_len:
                    print(f"   학습 시 사용된 max_len: {max_len}")
            except Exception as e:
                print(f"⚠️  학습 설정 정보 저장 실패 (무시하고 계속): {e}")
            
            print(f"✅ 모델 복사 완료: {target_path}")
        else:
            print(f"⚠️  로컬 checkpoint를 찾을 수 없습니다: {checkpoint_path}")
            print("   WandB artifact를 다운로드합니다...")
            download_model_artifact(best_run, output_dir, args.artifact_name)
    else:
        # WandB artifact 다운로드
        artifact_path = download_model_artifact(best_run, output_dir, args.artifact_name)
        
        if artifact_path:
            print(f"\n✅ 모델 다운로드 완료!")
            print(f"   경로: {artifact_path}")
            print(f"\n다음 명령으로 임베딩을 생성할 수 있습니다:")
            print(f"   python entrypoint_embed.py \\")
            print(f"     --model {artifact_path} \\")
            print(f"     --input data/processed/merged_corpus.jsonl \\")
            print(f"     --outdir embeds \\")
            print(f"     --prefix passages \\")
            print(f"     --type passage")


if __name__ == "__main__":
    main()

