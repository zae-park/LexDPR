#!/bin/bash
# 긴 텍스트 테스트 스크립트 (512 토큰을 살짝 넘는 경우)

echo "=== 긴 텍스트 생성 (512 토큰을 살짝 넘는 경우) ==="

# Python으로 정확한 길이의 텍스트 생성
python3 << 'PYEOF'
import json

# 기본 텍스트
base_text = "법령 조문 내용 법률 제1조 목적 이 법은 법령의 해석 및 적용에 관한 사항을 규정함을 목적으로 한다. 제2조 정의 이 법에서 사용하는 용어의 뜻은 다음과 같다. 제3조 적용 범위 이 법은 모든 법령에 적용된다. 제4조 해석 원칙 법령은 그 목적과 취지에 맞게 해석하여야 한다. 제5조 적용 순서 특별법이 일반법에 우선한다. 제6조 시행일 이 법은 공포한 날부터 시행한다. "

# 약 550-600 토큰 정도 (512를 살짝 넘는 정도)
long_text = base_text * 6

# JSON 파일 생성
with open("/tmp/long_text.json", "w", encoding="utf-8") as f:
    json.dump({"texts": [long_text]}, f, ensure_ascii=False, indent=2)

print(f"텍스트 길이: {len(long_text)} 문자")
print(f"추정 토큰 수: 약 {len(long_text) // 2.5:.0f} 토큰 (512를 살짝 넘음)")
print("JSON 파일 생성 완료: /tmp/long_text.json")
PYEOF

echo ""

echo "1. Passage 임베딩 (auto_chunk=True, 긴 텍스트 자동 chunking):"
curl -X POST http://localhost:8000/embed/passage \
  -H "Content-Type: application/json" \
  -d @/tmp/long_text.json

echo ""
echo ""
echo "2. Query 임베딩 (auto_chunk=False, truncation만):"
curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d @/tmp/long_text.json

echo ""
echo ""
echo "테스트 완료!"

