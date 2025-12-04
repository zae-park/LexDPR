#!/bin/bash
# 여러 WandB Sweep Agent를 동시에 실행하는 스크립트

# 설정
NUM_AGENTS=${1:-3}  # 기본값: 3개
CONFIG_FILE=${2:-"configs/sweep.yaml"}
LOG_DIR="logs/agents"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "WandB Sweep Agent 다중 실행"
echo "=========================================="
echo "Agent 수: $NUM_AGENTS"
echo "설정 파일: $CONFIG_FILE"
echo "로그 디렉토리: $LOG_DIR"
echo "=========================================="
echo ""

# 각 agent 실행
for i in $(seq 1 $NUM_AGENTS); do
    echo "Agent $i 시작 중..."
    nohup poetry run lex-dpr sweep agent --config "$CONFIG_FILE" > "$LOG_DIR/agent_${i}.log" 2>&1 &
    echo "  PID: $!"
    echo "  로그: $LOG_DIR/agent_${i}.log"
    sleep 2  # 각 agent 시작 사이에 약간의 지연
done

echo ""
echo "✅ $NUM_AGENTS개의 Agent가 백그라운드에서 실행 중입니다."
echo ""
echo "실행 중인 Agent 확인:"
ps aux | grep "lex-dpr sweep agent" | grep -v grep
echo ""
echo "로그 확인:"
echo "  tail -f $LOG_DIR/agent_1.log"
echo "  tail -f $LOG_DIR/agent_2.log"
echo "  ..."
echo ""
echo "Agent 종료:"
echo "  pkill -f 'lex-dpr sweep agent'"

# ./scripts/run_multiple_agents.sh 10
# ./scripts/run_multiple_agents.sh 5 configs/my_sweep.yaml
# ps aux | grep "lex-dpr sweep agent" | grep -v grep
