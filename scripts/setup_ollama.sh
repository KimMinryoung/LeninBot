#!/bin/bash
# Ollama + Qwen3.5:4b 설치 스크립트 (Hetzner VPS)
# 사용법: bash scripts/setup_ollama.sh

set -eE

echo "=== Ollama + Qwen3.5:4b 설치 시작 ==="

# 1. Ollama 설치
if command -v ollama &> /dev/null; then
    echo "Ollama 이미 설치됨: $(ollama --version)"
else
    echo "Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama 설치 완료"
fi

# 2. Ollama 서비스 시작 (systemd)
if systemctl is-active --quiet ollama; then
    echo "Ollama 서비스 실행 중"
else
    echo "Ollama 서비스 시작..."
    sudo systemctl enable ollama
    sudo systemctl start ollama
    sleep 3
fi

# 3. 모델 다운로드
echo "Qwen3.5:4b 다운로드 중... (약 2.5GB)"
ollama pull qwen3.5:4b

# 4. nothink 커스텀 모델 생성 (thinking 비활성화 — CPU 속도 최적화)
echo "nothink 모델 생성 중..."
cat > /tmp/Modelfile << 'MODELEOF'
FROM qwen3.5:4b
PARAMETER num_predict 512
PARAMETER temperature 0.3
TEMPLATE "{{- if .System }}{{ .System }}{{ end }}{{ .Prompt }}"
MODELEOF

ollama create qwen3.5-nothink -f /tmp/Modelfile
rm -f /tmp/Modelfile
echo "qwen3.5-nothink 생성 완료"

# 5. 동작 확인
echo ""
echo "=== 설치 검증 ==="
ollama list
echo ""
echo "테스트 실행 (nothink)..."
RESPONSE=$(ollama run qwen3.5-nothink "Say 'Hello' in Korean. Reply in one word only." 2>/dev/null | head -1)
echo "모델 응답: $RESPONSE"

# 6. API 접근 확인
echo ""
echo "API 확인 (localhost:11434)..."
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  {m['name']} ({m['size'] / 1e9:.1f} GB)\")
" 2>/dev/null || echo "  API 응답 실패 — systemctl status ollama 확인"

echo ""
echo "=== 설치 완료 ==="
echo "사용법: ollama run qwen3.5-nothink"
echo "API: http://localhost:11434"
echo "telegram_bot.py 기본 모델: qwen3.5-nothink"
