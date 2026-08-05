#!/usr/bin/env bash
# remove_llm_provider_keys.sh — LLM 게이트웨이 enforcement 최종 단계 (root 필요).
#
# anthropic/deepseek/moonshot/openai/gemini 및 scoped provider 키 credential을 각 서비스에서
# 주석 처리한다. 이후 실키는 leninbot-llm-proxy.service에만 남고, 클라이언트는
# placeholder + 프록시 경유로만 프로바이더에 닿을 수 있다 (물리적 우회 차단).
#
# - 주석 처리 방식 + 원본 .bak-llmkeys 백업 → 롤백은 백업 복원 + daemon-reload.
# - gemini 포함 (2026-08-05 2차): graphiti 추출·임베딩과 browser-use vision
#   폴백이 프록시 경유 클라이언트로 전환된 뒤부터. 이미 주석된 라인은 매치되지
#   않으므로 재실행(idempotent) 안전.
# - 제외: leninbot-llm-proxy(키 보관소), research-document-translation
#   (직접 클라이언트 일회성 스크립트 — 문서화된 예외).
# - 사전 조건 (이미 적용됨): .env의 OPENAI_API_KEY=via-llm-proxy +
#   OPENAI_BASE_URL → graphiti 내부 reranker가 프록시 경유.
# - skills/kg-maintenance 등 ad-hoc 스크립트는 이후 GEMINI_API_KEY를
#   /run/credentials/leninbot-llm-proxy.service/gemini_api_key 에서 읽어 export.
#
# 실행: sudo bash scripts/remove_llm_provider_keys.sh
# 이후 서비스 재시작은 별도로 수행한다.

set -euo pipefail

PATTERN='^(LoadCredentialEncrypted=(anthropic|writer_anthropic|deepseek|moonshot|openai|gemini|kg_gemini)_api_key.*)$'
MARK='# key moved to leninbot-llm-proxy (2026-08-05)'

DROPIN_SERVICES=(
  leninbot-a2a-api
  leninbot-api
  leninbot-autonomous
  leninbot-browser
  leninbot-experience
  leninbot-razvedchik
  leninbot-roleplay
  leninbot-telegram
  novel-writer-api
)
MAIN_UNIT_SERVICES=(
  leninbot-commulingo-enrich
  leninbot-commulingo-maintainer
  leninbot-commulingo-new
  leninbot-commulingo-terms
  leninbot-kg-integrity
  leninbot-event-backfill
)

edit_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "skip (missing): $f"
    return
  fi
  local before
  before=$(grep -cE "$PATTERN" "$f" || true)
  if [ "$before" -eq 0 ]; then
    echo "skip (already clean): $f"
    return
  fi
  sed -i.bak-llmkeys -E "s@$PATTERN@# \\1  $MARK@" "$f"
  echo "edited: $f ($before key line(s) commented)"
}

for s in "${DROPIN_SERVICES[@]}"; do
  edit_file "/etc/systemd/system/${s}.service.d/credentials.conf"
done
for s in "${MAIN_UNIT_SERVICES[@]}"; do
  edit_file "/etc/systemd/system/${s}.service"
done

systemctl daemon-reload
echo
echo "완료. 남은 LLM 키 credential 현황:"
grep -rlE '^LoadCredentialEncrypted=(anthropic|writer_anthropic|deepseek|moonshot|openai|gemini|kg_gemini)_api_key' \
  /etc/systemd/system/ 2>/dev/null | sed 's/^/  keeps keys: /' || true
echo
echo "다음: 상주 서비스 재시작 (Claude가 수행) → KG 초기화·프록시 경유 스모크."
