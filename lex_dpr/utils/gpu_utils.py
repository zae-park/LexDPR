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
    """GPU를 사용하는 프로세스 목록 반환 (compute apps + 일반 프로세스)"""
    processes = []
    seen_pids = set()
    
    # 방법 1: Compute apps (CUDA compute API 사용 프로세스)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,user", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                pid = parts[0]
                processes.append({
                    "pid": pid,
                    "process_name": parts[1],
                    "used_memory_mb": parts[2],
                    "user": parts[3],
                    "type": "compute"
                })
                seen_pids.add(pid)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # 방법 2: 일반 nvidia-smi 출력에서 프로세스 정보 추출 (VLLM 등)
    try:
        result_full = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result_full.returncode == 0:
            import re
            lines = result_full.stdout.split("\n")
            in_processes = False
            
            for line in lines:
                # 프로세스 섹션 시작 확인
                if "Processes:" in line or ("GPU" in line and "PID" in line and "Type" in line):
                    in_processes = True
                    continue
                
                if in_processes:
                    # 프로세스 라인 파싱: "|    0  12345    C   python ...  1234MiB |"
                    if "|" in line and ("MiB" in line or "GiB" in line):
                        # PID 추출 (일반적으로 5자리 이상 숫자)
                        pid_matches = re.findall(r'\b(\d{4,})\b', line)
                        # 메모리 추출
                        memory_match = re.search(r'(\d+(?:\.\d+)?)\s*(MiB|GiB)', line)
                        
                        if pid_matches and memory_match:
                            pid = pid_matches[0]  # 첫 번째 PID 사용
                            if pid not in seen_pids:
                                memory_value = float(memory_match.group(1))
                                memory_unit = memory_match.group(2)
                                
                                # GiB를 MiB로 변환
                                if memory_unit == "GiB":
                                    memory_mb = int(memory_value * 1024)
                                else:
                                    memory_mb = int(memory_value)
                                
                                # 프로세스명 추출 (PID 다음 부분에서 찾기)
                                proc_name = "unknown"
                                parts = [p.strip() for p in line.split("|") if p.strip()]
                                for part in parts:
                                    # PID 다음에 오는 부분에서 프로세스명 찾기
                                    if pid in part:
                                        words = part.split()
                                        pid_idx = -1
                                        for i, word in enumerate(words):
                                            if word == pid:
                                                pid_idx = i
                                                break
                                        if pid_idx >= 0 and pid_idx + 1 < len(words):
                                            # PID 다음 단어가 프로세스명일 가능성
                                            next_word = words[pid_idx + 1]
                                            if next_word not in ["C", "G", "M"]:  # Type이 아닌 경우
                                                proc_name = next_word
                                                break
                                
                                # VLLM 관련 프로세스명 확인
                                if "vllm" in line.lower() or "vllm" in proc_name.lower():
                                    proc_name = "vllm"
                                
                                processes.append({
                                    "pid": pid,
                                    "process_name": proc_name,
                                    "used_memory_mb": str(memory_mb),
                                    "user": "unknown",
                                    "type": "general"
                                })
                                seen_pids.add(pid)
                    elif line.strip().startswith("+") or (line.strip() and not "|" in line and not "MiB" in line and not "GiB" in line):
                        # 테이블 끝 또는 다른 섹션 시작
                        if not any(c in line for c in ["|", "MiB", "GiB", "Processes"]):
                            break
    except Exception as e:
        # 파싱 실패해도 계속 진행
        pass
    
    return processes


def kill_process(pid: int, force: bool = False, use_sudo: bool = False) -> bool:
    """프로세스 종료"""
    try:
        signal = 9 if force else 15
        cmd = ["kill", f"-{signal}", str(pid)]
        
        if use_sudo:
            cmd = ["sudo"] + cmd
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            return True
        elif "Operation not permitted" in result.stderr or "Permission denied" in result.stderr:
            if not use_sudo:
                # 권한 문제인 경우 sudo 시도
                print("⚠️  권한이 없습니다. sudo를 사용하여 시도합니다...")
                return kill_process(pid, force=force, use_sudo=True)
            else:
                print(f"❌ sudo를 사용해도 권한이 없습니다. root 권한이 필요할 수 있습니다.")
                return False
        else:
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
        print("⚠️  compute apps로 등록된 GPU 프로세스가 없습니다.")
        print("")
        print("💡 nvidia-smi에서 프로세스가 보이지만 여기서 안 보이는 경우:")
        print("   1. nvidia-smi를 직접 실행하여 PID 확인:")
        print("      nvidia-smi")
        print("   2. 확인한 PID로 직접 종료:")
        print("      poetry run lex-dpr gpu kill <PID>")
        print("")
        print("또는 전체 GPU 메모리 사용량 확인:")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            print("GPU 메모리 사용량:")
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 3:
                        gpu_id = parts[0].strip()
                        used = int(parts[1].strip())
                        total = int(parts[2].strip())
                        usage_pct = (used / total * 100) if total > 0 else 0
                        print(f"  GPU {gpu_id}: {used}MB / {total}MB ({usage_pct:.1f}%)")
        except Exception:
            pass
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


def kill_process_by_pid(pid: int, force: bool = False, use_sudo: bool = False):
    """PID로 프로세스 종료"""
    processes = get_gpu_processes()
    pid_str = str(pid)
    
    # 해당 PID가 GPU 프로세스인지 확인
    found = False
    proc_info = None
    for proc in processes:
        if proc["pid"] == pid_str:
            found = True
            proc_info = proc
            name = proc["process_name"]
            memory = format_memory(proc["used_memory_mb"])
            user = proc.get("user", "unknown")
            print(f"프로세스 발견: PID={pid}, 이름={name}, 메모리={memory}, 사용자={user}")
            break
    
    if not found:
        # 프로세스 정보 확인 시도
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "user,comm,pid"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"⚠️  PID {pid}는 GPU 프로세스 목록에 없지만 시스템 프로세스입니다.")
                print(f"프로세스 정보:\n{result.stdout}")
            else:
                print(f"⚠️  PID {pid}를 찾을 수 없습니다.")
        except Exception:
            print(f"⚠️  PID {pid} 정보를 확인할 수 없습니다.")
        
        response = input("그래도 종료하시겠습니까? (y/N): ")
        if response.lower() != "y":
            print("취소되었습니다.")
            return
    
    # 프로세스 소유자 확인
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "user="],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            owner = result.stdout.strip()
            import os
            current_user = os.getenv("USER") or os.getenv("USERNAME", "unknown")
            if owner != current_user:
                print(f"⚠️  이 프로세스는 '{owner}' 사용자의 것입니다. (현재 사용자: {current_user})")
                if not use_sudo:
                    response = input("sudo를 사용하여 종료하시겠습니까? (y/N): ")
                    if response.lower() == "y":
                        use_sudo = True
                    else:
                        print("취소되었습니다.")
                        return
    except Exception:
        pass
    
    if kill_process(pid, force=force, use_sudo=use_sudo):
        print(f"✅ 프로세스 {pid}를 종료했습니다.")
    else:
        print(f"❌ 프로세스 {pid} 종료에 실패했습니다.")
        if not force:
            response = input("강제 종료를 시도하시겠습니까? (y/N): ")
            if response.lower() == "y":
                if kill_process(pid, force=True, use_sudo=use_sudo):
                    print(f"✅ 프로세스 {pid}를 강제 종료했습니다.")
                else:
                    print(f"❌ 강제 종료에도 실패했습니다.")
                    if not use_sudo:
                        print("💡 sudo 권한이 필요할 수 있습니다:")
                        print(f"   sudo kill -9 {pid}")


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

