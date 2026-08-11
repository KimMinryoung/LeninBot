#!/usr/bin/env bash
#
# 대화형 세션(user.slice 아래 모든 user-*.slice) 메모리 관찰기.
#
# 2026-08-10 04:23 UTC 사건 이후 만들었다. Claude Code가 anon-rss 8.74 GiB까지
# 부풀어 전역 OOM 킬러에게 죽었고, 그 사이 시스템 파일 캐시가 전멸해 서버가
# 5분간 멈췄다. cgroup 상한(scripts/setup-session-limits.sh)으로 막아 두었지만,
# 상한이 적절한지·문맥 상한 환경변수가 실제로 듣는지는 관찰해야 안다.
#
# 2026-08-11: 상한이 user-1000 전용에서 user-.slice.d 템플릿(전 사용자)으로
# 옮겨졌다. PC에서 root로 SSH 접속하면 세션이 user-0.slice에 실리므로,
# 관찰도 특정 uid 고정이 아니라 존재하는 user-*.slice 전체를 합산한다.
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

LOG="/home/grass/leninbot/logs/session-memory.tsv"

slices() {
  find /sys/fs/cgroup/user.slice -maxdepth 1 -type d -name 'user-*.slice' 2>/dev/null | sort
}

if [ -z "$(slices)" ]; then
  echo "오류: /sys/fs/cgroup/user.slice 아래에 user-*.slice 가 없습니다. cgroup v2가 아니거나 세션이 없습니다." >&2
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
    done < <(slices | while IFS= read -r cg; do find "$cg" -name cgroup.procs 2>/dev/null; done)
  } | sort -rn | head -"$n" | awk '{printf "%s:%.2fGiB ", $2, $1/1048576}'
}

# 슬라이스별 값을 합산한다. cur·oom_kill·high 이벤트는 합, peak는 최대,
# high/max 상한은 템플릿이라 모든 슬라이스가 같으므로 첫 값을 쓴다.
sample() {
  local ts cur peak high max oomk highev cg v
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cur=0; peak=0; oomk=0; highev=0; high=""; max=""
  while IFS= read -r cg; do
    v=$(rd "$cg/memory.current"); [ -n "$v" ] && cur=$((cur + v))
    v=$(rd "$cg/memory.peak");    [ -n "$v" ] && [ "$v" -gt "$peak" ] 2>/dev/null && peak=$v
    [ -z "$high" ] && high=$(rd "$cg/memory.high")
    [ -z "$max" ]  && max=$(rd "$cg/memory.max")
    v=$(awk '/^oom_kill /{print $2}' "$cg/memory.events" 2>/dev/null); [ -n "$v" ] && oomk=$((oomk + v))
    v=$(awk '/^high /{print $2}' "$cg/memory.events" 2>/dev/null);     [ -n "$v" ] && highev=$((highev + v))
  done < <(slices)

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
    if [ "$(id -u)" -ne 0 ]; then
      echo "root 권한이 필요합니다. sudo로 재실행합니다."
      exec sudo -- bash "$0" --reset-peak
    fi
    while IFS= read -r cg; do
      echo 0 > "$cg/memory.peak" 2>/dev/null \
        && echo "최고점 초기화: $cg" \
        || echo "초기화 실패: $cg (커널이 쓰기를 지원하지 않을 수 있습니다)."
    done < <(slices)
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
