#!/usr/bin/env bash
#
# 대화형 세션(user-1000.slice) 메모리 관찰기.
#
# 2026-08-10 04:23 UTC 사건 이후 만들었다. Claude Code가 anon-rss 8.74 GiB까지
# 부풀어 전역 OOM 킬러에게 죽었고, 그 사이 시스템 파일 캐시가 전멸해 서버가
# 5분간 멈췄다. cgroup 상한(scripts/setup-session-limits.sh)으로 막아 두었지만,
# 상한이 적절한지·문맥 상한 환경변수가 실제로 듣는지는 관찰해야 안다.
#
# 사용법:
#   bash scripts/watch-session-memory.sh              # 한 번 재고 로그에 append (cron용)
#   bash scripts/watch-session-memory.sh --watch      # 5초 간격 실시간
#   bash scripts/watch-session-memory.sh --watch 30   # 30초 간격
#   bash scripts/watch-session-memory.sh --report     # 쌓인 로그 요약
#   bash scripts/watch-session-memory.sh --reset-peak # 최고점 기록 초기화(sudo)
#
# cron으로 5분마다 쌓으려면:
#   */5 * * * * /usr/bin/bash /home/grass/leninbot/scripts/watch-session-memory.sh >/dev/null 2>&1

set -uo pipefail

CG="/sys/fs/cgroup/user.slice/user-1000.slice"
LOG="/home/grass/leninbot/logs/session-memory.tsv"

if [ ! -d "$CG" ]; then
  echo "오류: $CG 가 없습니다. cgroup v2가 아니거나 세션이 없습니다." >&2
  exit 1
fi

gib() { awk -v n="${1:-0}" 'BEGIN{ if (n=="max"||n=="") print "-"; else printf "%.2f", n/1073741824 }'; }
rd()  { [ -r "$1" ] && cat "$1" 2>/dev/null || echo ""; }

# cgroup 안의 프로세스를 RSS 내림차순으로. "이름:GiB" 형태로 상위 N개.
top_procs() {
  local n="${1:-3}"
  {
    while IFS= read -r pidfile; do
      while IFS= read -r pid; do
        [ -r "/proc/$pid/status" ] || continue
        local rss comm
        rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
        comm=$(tr -d '\0' < "/proc/$pid/comm" 2>/dev/null)
        [ -n "${rss:-}" ] && echo "$rss $comm"
      done < "$pidfile"
    done < <(find "$CG" -name cgroup.procs 2>/dev/null)
  } | sort -rn | head -"$n" | awk '{printf "%s:%.2fGiB ", $2, $1/1048576}'
}

sample() {
  local ts cur peak high max oomk highev
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cur=$(rd "$CG/memory.current")
  peak=$(rd "$CG/memory.peak")
  high=$(rd "$CG/memory.high")
  max=$(rd "$CG/memory.max")
  oomk=$(awk '/^oom_kill /{print $2}' "$CG/memory.events" 2>/dev/null)
  highev=$(awk '/^high /{print $2}' "$CG/memory.events" 2>/dev/null)

  CUR_G=$(gib "$cur"); PEAK_G=$(gib "$peak")
  HIGH_G=$(gib "$high"); MAX_G=$(gib "$max")
  OOMK="${oomk:-0}"; HIGHEV="${highev:-0}"
  TOP=$(top_procs 3)
  TS="$ts"
}

append_log() {
  mkdir -p "$(dirname "$LOG")"
  if [ ! -f "$LOG" ]; then
    printf 'ts\tcurrent_gib\tpeak_gib\thigh_gib\tmax_gib\toom_kill\thigh_events\ttop_procs\n' > "$LOG"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TS" "$CUR_G" "$PEAK_G" "$HIGH_G" "$MAX_G" "$OOMK" "$HIGHEV" "$TOP" >> "$LOG"
}

print_status() {
  echo "[$TS]  현재 ${CUR_G} GiB   최고점 ${PEAK_G} GiB   (상한: high ${HIGH_G} / max ${MAX_G} GiB)"
  echo "  memory.events: oom_kill=${OOMK}  high(스로틀 발동)=${HIGHEV}"
  echo "  상위 프로세스: ${TOP:-(없음)}"
}

case "${1:-}" in
  --watch)
    INT="${2:-5}"
    echo "${INT}초 간격 관찰. Ctrl-C로 종료."
    while true; do sample; print_status; echo; sleep "$INT"; done
    ;;

  --reset-peak)
    if [ ! -w "$CG/memory.peak" ] && [ "$(id -u)" -ne 0 ]; then
      echo "root 권한이 필요합니다. sudo로 재실행합니다."
      exec sudo -- bash "$0" --reset-peak
    fi
    echo 0 > "$CG/memory.peak" 2>/dev/null \
      && echo "최고점 기록을 초기화했습니다." \
      || echo "초기화 실패 (커널이 쓰기를 지원하지 않을 수 있습니다)."
    sample; print_status
    ;;

  --report)
    if [ ! -f "$LOG" ]; then
      echo "아직 쌓인 기록이 없습니다: $LOG"
      echo "먼저 한 번 실행하거나 cron에 걸어 두세요."
      sample; echo; print_status
      exit 0
    fi
    echo "════ 세션 메모리 요약 ($LOG) ════"
    awk -F'\t' 'NR>1{
      n++; s+=$2; if($2>mx){mx=$2; mxts=$1; mxtop=$8}
      if($6>oom) oom=$6; if($7>hi) hi=$7
      last=$1; lastcur=$2; lastpeak=$3
    } END{
      if(n==0){print "  (기록 없음)"; exit}
      printf "  표본 %d개\n", n
      printf "  평균 현재 사용량   %.2f GiB\n", s/n
      printf "  기록된 최대치      %.2f GiB   (%s)\n", mx, mxts
      printf "    그때 상위 프로세스: %s\n", mxtop
      printf "  cgroup 최고점      %.2f GiB\n", lastpeak
      printf "  누적 oom_kill      %d\n", oom
      printf "  누적 high 스로틀   %d\n", hi
      printf "  마지막 표본        %s (%.2f GiB)\n", last, lastcur
    }' "$LOG"
    echo
    echo "── 최근 10개 ──"
    { head -1 "$LOG"; tail -n +2 "$LOG" | tail -10; } | column -t -s $'\t' 2>/dev/null || tail -10 "$LOG"
    echo
    echo "해석: high 스로틀이 계속 오르면 MemoryHigh가 너무 낮은 것이고,"
    echo "      oom_kill이 오르면 세션이 상한에 닿아 죽고 있다는 뜻입니다."
    echo "      둘 다 0이고 최대치가 상한보다 한참 낮으면 여유가 있는 것입니다."
    ;;

  -h|--help)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  *)
    sample
    append_log
    print_status
    echo "  → $LOG 에 기록했습니다."
    ;;
esac
