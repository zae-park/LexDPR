# lex_dpr/utils/gpu_utils.py
"""
GPU 프로세스 관리 유틸리티

사용 예시:
  python -m lex_dpr.utils.gpu_utils list
  python -m lex_dpr.utils.gpu_utils kill <pid>
  python -m lex_dpr.utils.gpu_utils kill-all
"""

import subprocess
import sys
from typing import List, Dict, Optional


def get_gpu_processes() -> List[Dict[str, str]]:
    """GPU를 사용하는 프로세스 목록 반환"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,user", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                processes.append({
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_memory_mb": parts[2],
                    "user": parts[3]
                })
        
        return processes
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다. NVIDIA GPU가 설치되어 있는지 확인하세요.")
        return []


def kill_process(pid: int, force: bool = False) -> bool:
    """프로세스 종료"""
    try:
        signal = "SIGKILL" if force else "SIGTERM"
        subprocess.run(["kill", f"-{9 if force else 15}", str(pid)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"❌ 프로세스 종료 실패: {e}")
        return False


def format_memory(mb_str: str) -> str:
    """메모리 크기를 읽기 쉬운 형식으로 변환"""
    try:
        mb = int(mb_str)
        if mb >= 1024:
            return f"{mb / 1024:.2f} GB"
        else:
            return f"{mb} MB"
    except (ValueError, TypeError):
        return mb_str


def list_processes():
    """GPU 프로세스 목록 출력"""
    processes = get_gpu_processes()
    
    if not processes:
        print("✅ GPU를 사용하는 프로세스가 없습니다.")
        return
    
    print("=" * 80)
    print("GPU 프로세스 목록:")
    print("=" * 80)
    print(f"{'PID':<10} {'프로세스명':<30} {'메모리':<15} {'사용자':<15}")
    print("-" * 80)
    
    total_memory = 0
    for proc in processes:
        pid = proc["pid"]
        name = proc["process_name"][:28]  # 이름 길이 제한
        memory = format_memory(proc["used_memory_mb"])
        user = proc["user"][:13]  # 사용자 이름 길이 제한
        
        try:
            total_memory += int(proc["used_memory_mb"])
        except (ValueError, TypeError):
            pass
        
        print(f"{pid:<10} {name:<30} {memory:<15} {user:<15}")
    
    print("-" * 80)
    print(f"총 사용 메모리: {format_memory(str(total_memory))}")
    print("=" * 80)
    print("")
    print("💡 사용 방법:")
    print("  python -m lex_dpr.utils.gpu_utils kill <PID>        # 프로세스 종료")
    print("  python -m lex_dpr.utils.gpu_utils kill-all          # 모든 프로세스 종료")
    print("  python -m lex_dpr.utils.gpu_utils kill <PID> --force # 강제 종료")


def kill_process_by_pid(pid: int, force: bool = False):
    """PID로 프로세스 종료"""
    processes = get_gpu_processes()
    pid_str = str(pid)
    
    # 해당 PID가 GPU 프로세스인지 확인
    found = False
    for proc in processes:
        if proc["pid"] == pid_str:
            found = True
            name = proc["process_name"]
            memory = format_memory(proc["used_memory_mb"])
            print(f"프로세스 발견: PID={pid}, 이름={name}, 메모리={memory}")
            break
    
    if not found:
        print(f"⚠️  PID {pid}는 GPU를 사용하는 프로세스가 아닙니다.")
        response = input("그래도 종료하시겠습니까? (y/N): ")
        if response.lower() != "y":
            print("취소되었습니다.")
            return
    
    if kill_process(pid, force=force):
        print(f"✅ 프로세스 {pid}를 종료했습니다.")
    else:
        print(f"❌ 프로세스 {pid} 종료에 실패했습니다.")
        if not force:
            response = input("강제 종료를 시도하시겠습니까? (y/N): ")
            if response.lower() == "y":
                if kill_process(pid, force=True):
                    print(f"✅ 프로세스 {pid}를 강제 종료했습니다.")
                else:
                    print(f"❌ 강제 종료에도 실패했습니다.")


def kill_all_processes(force: bool = False):
    """모든 GPU 프로세스 종료"""
    processes = get_gpu_processes()
    
    if not processes:
        print("✅ 종료할 GPU 프로세스가 없습니다.")
        return
    
    print(f"⚠️  {len(processes)}개의 GPU 프로세스를 종료하려고 합니다:")
    for proc in processes:
        pid = proc["pid"]
        name = proc["process_name"]
        memory = format_memory(proc["used_memory_mb"])
        print(f"  - PID {pid}: {name} (메모리: {memory})")
    
    print("")
    response = input("정말로 모든 프로세스를 종료하시겠습니까? (y/N): ")
    if response.lower() != "y":
        print("취소되었습니다.")
        return
    
    success_count = 0
    for proc in processes:
        pid = int(proc["pid"])
        if kill_process(pid, force=force):
            success_count += 1
            print(f"✅ PID {pid} 종료 완료")
        else:
            print(f"❌ PID {pid} 종료 실패")
    
    print(f"\n✅ {success_count}/{len(processes)}개 프로세스 종료 완료")


def main():
    """CLI 진입점"""
    if len(sys.argv) < 2:
        list_processes()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_processes()
    elif command == "kill":
        if len(sys.argv) < 3:
            print("❌ 사용법: python -m lex_dpr.utils.gpu_utils kill <PID> [--force]")
            sys.exit(1)
        
        try:
            pid = int(sys.argv[2])
            force = "--force" in sys.argv or "-f" in sys.argv
            kill_process_by_pid(pid, force=force)
        except ValueError:
            print("❌ PID는 숫자여야 합니다.")
            sys.exit(1)
    elif command == "kill-all":
        force = "--force" in sys.argv or "-f" in sys.argv
        kill_all_processes(force=force)
    else:
        print("❌ 알 수 없는 명령어입니다.")
        print("사용법:")
        print("  python -m lex_dpr.utils.gpu_utils list")
        print("  python -m lex_dpr.utils.gpu_utils kill <PID> [--force]")
        print("  python -m lex_dpr.utils.gpu_utils kill-all [--force]")
        sys.exit(1)


if __name__ == "__main__":
    main()

