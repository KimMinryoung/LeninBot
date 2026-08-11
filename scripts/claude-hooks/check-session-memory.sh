#!/usr/bin/env bash
#
#
# [저장소 사본] 실제 설치 위치는 ~/.claude/hooks/check-session-memory.sh 이고,
# ~/.claude/settings.json 의 hooks.PreToolUse (matcher: "Agent|Workflow")에 등록돼 있다.
# 여기 사본은 버전 관리·재구축용. 수정 시 두 곳을 같이 갱신할 것.
# PreToolUse 훅 (Agent|Workflow): 서브에이전트/워크플로를 띄우기 전에
# 이 세션이 속한 user-*.slice의 메모리 수위를 재고, 임계를 넘으면 생성을 막는다.
#
# 배경: 2026-08-10 04:23 UTC — Claude Code가 8.74 GiB까지 부풀어 호스트가 5분
# 멈췄다. cgroup 상한(high 6G / max 8G)은 커널 층의 최후 방어선이고, 이 훅은
# 그 전에 에이전트가 스스로 물러나게 하는 응용 층 방어선이다.
#
# 임계 기본값 5 GiB (MemoryHigh 6G보다 낮게). SUBAGENT_MEM_LIMIT_GIB로 조정.

set -u

LIMIT_GIB="${SUBAGENT_MEM_LIMIT_GIB:-5}"
LIMIT_BYTES=$((LIMIT_GIB * 1073741824))

# 이 프로세스가 속한 user-N.slice를 찾는다. 못 찾으면 조용히 허용(fail-open).
rel=$(awk -F: '{print $3; exit}' /proc/self/cgroup 2>/dev/null)
slice=$(printf '%s' "$rel" | grep -o '^/user\.slice/user-[0-9]*\.slice')
[ -n "$slice" ] || exit 0

cur=$(cat "/sys/fs/cgroup${slice}/memory.current" 2>/dev/null)
case "$cur" in ''|*[!0-9]*) exit 0;; esac

if [ "$cur" -gt "$LIMIT_BYTES" ]; then
  cur_gib=$(awk -v n="$cur" 'BEGIN{printf "%.1f", n/1073741824}')
  cat <<EOF
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"세션 메모리가 ${cur_gib} GiB로 임계(${LIMIT_GIB} GiB)를 넘어 서브에이전트/워크플로 생성을 차단했습니다. 6 GiB부터 커널 스로틀, 8 GiB에서 OOM 킬이 일어납니다. 이 작업은 서브에이전트 없이 인라인으로 처리하고, 사용자에게 /compact 또는 새 세션을 권하세요."},"systemMessage":"세션 메모리 ${cur_gib} GiB > ${LIMIT_GIB} GiB — 서브에이전트 생성을 차단했습니다."}
EOF
fi
exit 0
