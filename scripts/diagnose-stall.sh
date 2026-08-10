#!/usr/bin/env bash
#
# 호스트 정지 구간의 커널·systemd 로그를 모아 보여 준다.
#
# 2026-08-10 04:17~04:23 UTC 정지 사건 분석용으로 만들었지만, 창을 바꿔
# 재사용할 수 있다. 알고 싶은 것은 딱 셋이다:
#   1. OOM 킬러가 돌았나, 돌았다면 무엇을 죽였나
#   2. 커널이 hung task(120초 이상 블록)를 신고했나
#   3. 그 창에서 어떤 systemd 유닛이 시작했나
#
# 사용법:  bash scripts/diagnose-stall.sh                    # 기본 창(2026-08-10 04:10~04:30 UTC)
#          bash scripts/diagnose-stall.sh "04:10" "04:30"    # 오늘 날짜의 다른 창
#          bash scripts/diagnose-stall.sh "2026-08-09 22:00" "2026-08-09 22:30"
#
# 관련: scripts/setup-session-limits.sh (재발 방지), dev_docs/standby_operations.md

set -uo pipefail

SINCE="${1:-2026-08-10 04:10:00}"
UNTIL="${2:-2026-08-10 04:30:00}"

# 시:분만 준 경우 오늘 날짜를 붙인다
if [[ "$SINCE" =~ ^[0-9]{2}:[0-9]{2} ]]; then SINCE="$(date -u +%Y-%m-%d) $SINCE"; fi
if [[ "$UNTIL" =~ ^[0-9]{2}:[0-9]{2} ]]; then UNTIL="$(date -u +%Y-%m-%d) $UNTIL"; fi

if [ "$(id -u)" -ne 0 ]; then
  echo "root 권한이 필요합니다. sudo로 재실행합니다."
  exec sudo -- bash "$0" "$SINCE" "$UNTIL"
fi

echo "════ 조사 창: ${SINCE} ~ ${UNTIL} (UTC 기준 로그) ════"
echo

echo "──[1] OOM 킬러 ──────────────────────────────"
# 창을 넓게 잡아 놓친 게 없는지도 같이 본다
OOM=$(journalctl -k --since "$SINCE" --until "$UNTIL" --no-pager 2>/dev/null \
      | grep -iE "out of memory|oom-kill|oom_reaper|killed process" || true)
if [ -n "$OOM" ]; then
  echo "$OOM" | head -40
else
  echo "  (이 창에서는 없음)"
  echo "  ── 최근 7일 전체에서 OOM 흔적 ──"
  journalctl -k --since "-7 days" --no-pager 2>/dev/null \
    | grep -iE "out of memory|oom-kill|killed process" | tail -15 \
    || echo "  (없음)"
fi
echo

echo "──[2] hung task / 스톨 경고 ─────────────────"
journalctl -k --since "$SINCE" --until "$UNTIL" --no-pager 2>/dev/null \
  | grep -iE "blocked for more than|hung_task|soft lockup|rcu_sched|watchdog|call trace" \
  | head -30 || true
echo "  (위가 비었으면 해당 없음)"
echo

echo "──[3] 커널 로그 전체 (창 내부) ──────────────"
journalctl -k --since "$SINCE" --until "$UNTIL" --no-pager 2>/dev/null | head -60
echo "  (위가 비었으면 커널이 아무것도 남기지 않은 것 — 순수 메모리 압박일 수 있다)"
echo

echo "──[4] 이 창에서 시작한 systemd 유닛 ─────────"
journalctl --since "$SINCE" --until "$UNTIL" --no-pager 2>/dev/null \
  | grep -iE "Starting |Started |Stopping |Stopped |Failed |succeeded" \
  | head -50 || true
echo

echo "──[5] 경고 이상 로그 (전 유닛) ──────────────"
journalctl --since "$SINCE" --until "$UNTIL" -p warning --no-pager 2>/dev/null | head -50
echo

echo "──[6] 참고: 그때의 자원 지표 (sar) ──────────"
SA_DAY=$(date -u -d "$SINCE" +%d 2>/dev/null || echo "")
if [ -n "$SA_DAY" ] && [ -f "/var/log/sysstat/sa${SA_DAY}" ]; then
  echo "  [load]"
  LC_ALL=C sar -q -f "/var/log/sysstat/sa${SA_DAY}" 2>/dev/null | awk 'NR<=3 || ($1>="04:00:00" && $1<="04:40:00")' | head -12
  echo "  [memory]"
  LC_ALL=C sar -r -f "/var/log/sysstat/sa${SA_DAY}" 2>/dev/null | awk 'NR<=3 || ($1>="04:00:00" && $1<="04:40:00")' | head -12
else
  echo "  (sa${SA_DAY} 파일 없음)"
fi

echo
echo "════ 끝 ════"
