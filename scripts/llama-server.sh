#!/usr/bin/env bash
# llama-server.sh — leninbot 로컬 LLM 서버 실행 스크립트
#
# 사용법:
#   ./scripts/llama-server.sh          # 기본 실행
#   ./scripts/llama-server.sh --fg     # 포그라운드 (디버깅용)
#
# systemd 서비스(leninbot-llama.service)의 ExecStart에서도 이 스크립트를 호출.
# 설정 변경은 여기서만 하면 됨.

set -euo pipefail

# ── 설정 ──────────────────────────────────────────────────────────────
LLAMA_DIR="/home/grass/llama-cpp/llama-b8502"
MODEL="/home/grass/llama-cpp/models/qwen3.5-4b-q4_k_m.gguf"
HOST="127.0.0.1"
PORT=11435
CTX_SIZE=8192
THREADS=4

# qwen3.5 Small 시리즈(4B, 9B)는 thinking이 기본 비활성화이지만,
# llama-server에 명시적으로 전달해야 확실히 꺼짐.
# ref: https://unsloth.ai/docs/models/qwen3.5
CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'

# ── 실행 ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="${LLAMA_DIR}"

exec "${LLAMA_DIR}/llama-server" \
    -m "${MODEL}" \
    --host "${HOST}" --port "${PORT}" \
    -c "${CTX_SIZE}" -t "${THREADS}" \
    --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
