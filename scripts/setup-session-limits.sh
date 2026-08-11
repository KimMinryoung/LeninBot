#!/usr/bin/env bash
#
# 대화형 세션(SSH 셸, Claude Code)에 자원 상한을 건다.
#
# 배경: 2026-08-10 04:17~04:23 UTC, 호스트가 5분간 정지했다.
#   load 120, %system 44%, 디스크 읽기 1 GB/s, 가용 메모리 9.0 GB → 1.2 GB,
#   페이지 캐시 6.5 GB → 0.94 GB. sysstat의 04:20 샘플이 통째로 누락되고
#   04:23:10에 밀려 찍혔다. nginx·sshd가 스케줄을 못 받아 사이트는 522,
#   SSH는 무응답이 됐고 Postgres 복제도 이때 끊겼다.
#
#   원인은 메모리 폭주로 인한 전역 페이지 캐시 축출과 뒤이은 스래싱이다.
#   스왑도 systemd-oomd도 earlyoom도 없고 user 슬라이스에 상한이 없어서
#   프로세스 하나가 서버 전체를 멈춰 세울 수 있는 상태였다.
#
# 적용 범위: user-.slice.d 템플릿 — 모든 사용자 슬라이스(root 포함)에 걸린다.
#   2026-08-11에 user-1000 전용에서 옮겼다. PC에서 root로 SSH 접속하면 세션이
#   user-0.slice에 실려 uid 1000 전용 상한이 실세션을 못 덮었기 때문.
#   leninbot-* 서비스는 User=grass여도 system.slice 소속이라 영향이 없다.
#   예산은 uid별로 따로다(user-0과 user-1000이 각각 6G/8G).
#
# 사용법:  bash scripts/setup-session-limits.sh          # 적용 (필요하면 sudo로 자동 재실행)
#          bash scripts/setup-session-limits.sh --show   # 현재 상태만 확인
#          bash scripts/setup-session-limits.sh --undo   # 되돌리기
#
# 관련 문서: dev_docs/standby_operations.md (이 사건으로 복제가 끊겼다)

set -euo pipefail

DROPIN_DIR="/etc/systemd/system/user-.slice.d"
DROPIN="${DROPIN_DIR}/limits.conf"

MEM_HIGH="6G"    # 소프트 상한 — 죽이지 않고 이 cgroup 안에서만 회수 시작
MEM_MAX="8G"     # 하드 상한 — 닿으면 폭주한 세션만 죽고 서버는 산다
CPU_WEIGHT="30"  # 경합 시 시스템 서비스에 양보 (기본 100)
# MEM_MAX 근거: 2026-08-10 사건에서 Claude Code(2.1.226)가 anon-rss 8.74 GiB까지
#   부풀었다. 평소 이 슬라이스는 1.4 GiB를 쓴다. 8G면 정상 사용의 5배라 안 닿고,
#   폭주는 관측된 정점(8.74 GiB) 아래에서 잘라 낸다.
# 스왑이 없고 MemorySwapMax=0이라 MemoryHigh는 익명 메모리를 회수하지 못한다.
#   즉 6~8 GiB 구간은 "느려지는 구간"이고 실제 차단은 MemoryMax가 한다.

show() {
  local found=0 s slice
  for s in /sys/fs/cgroup/user.slice/user-*.slice; do
    [ -d "$s" ] || continue
    found=1
    slice=$(basename "$s")
    echo "── ${slice} — 커널이 실제로 물고 있는 값 ──"
    for f in memory.high memory.max memory.swap.max cpu.weight; do
      [ -r "$s/$f" ] && echo "  ${f} = $(cat "$s/$f")" || echo "  ${f} = (없음)"
    done
    [ -r "$s/memory.current" ] \
      && awk '{printf "  memory.current = %.2f GiB\n", $1/1073741824}' "$s/memory.current"
  done
  [ "$found" -eq 1 ] || echo "  (활성 user-*.slice 없음)"
  echo "── 드롭인 파일 ──"
  if [ -f "$DROPIN" ]; then echo "  $DROPIN (있음)"; else echo "  $DROPIN (없음)"; fi
}

if [ "${1:-}" = "--show" ]; then show; exit 0; fi

# root가 아니면 sudo로 재실행
if [ "$(id -u)" -ne 0 ]; then
  echo "root 권한이 필요합니다. sudo로 재실행합니다."
  exec sudo -- bash "$0" "$@"
fi

if [ "${1:-}" = "--undo" ]; then
  if [ -f "$DROPIN" ]; then
    rm -f "$DROPIN"
    rmdir "$DROPIN_DIR" 2>/dev/null || true
    systemctl daemon-reload
    echo "제거했습니다. 상한이 풀렸습니다."
  else
    echo "드롭인이 없습니다. 할 일 없음."
  fi
  echo
  show
  exit 0
fi

# 사전 점검 — cgroup v2가 아니면 이 설정은 의미가 없다
if [ "$(stat -fc %T /sys/fs/cgroup)" != "cgroup2fs" ]; then
  echo "오류: cgroup v2가 아닙니다. 이 설정은 적용되지 않습니다." >&2
  exit 1
fi

# 기존 파일이 있으면 백업
if [ -f "$DROPIN" ]; then
  BAK="${DROPIN}.$(date +%Y%m%d_%H%M%S).bak"
  cp -a "$DROPIN" "$BAK"
  echo "기존 파일을 백업했습니다: $BAK"
fi

install -d -m 755 "$DROPIN_DIR"
cat > "$DROPIN" <<EOF
# 2026-08-10 04:17~04:23 UTC 호스트 정지 사건 대응.
# 모든 사용자 세션(root SSH 포함, SSH 셸·Claude Code)을 제한한다.
# 2026-08-11: user-1000.slice.d에서 user-.slice.d(전 사용자 템플릿)로 이동.
#   PC에서 root로 접속하므로 uid 1000 전용 제한은 실세션을 못 덮었다.
# leninbot-* 서비스는 system.slice 소속이라 영향 없음.

[Slice]

# 소프트 상한. 넘어서면 죽이지 않고 "이 cgroup 안에서만" 회수를 시작한다.
# 시스템 전체 페이지 캐시를 날리는 대신 자기 캐시를 먼저 게워내므로
# 전역 스래싱이 생기지 않는다. 세션이 느려질 뿐이다.
MemoryHigh=${MEM_HIGH}

# 하드 상한(최후 방어선). 여기 닿으면 이 cgroup 안에서만 OOM 킬이 일어난다.
# 서버 전체가 아니라 폭주한 세션만 죽는다.
MemoryMax=${MEM_MAX}

# 경합 시 시스템 서비스에 양보한다.
CPUWeight=${CPU_WEIGHT}

# 스왑은 현재 없지만, 생기더라도 대화형 세션이 쓰지 않게 한다.
MemorySwapMax=0

# IOWeight는 넣지 않았다. sda의 I/O 스케줄러가 [none]이라 io.weight가
# 동작하지 않는다(bfq 또는 io.cost 설정 필요). 문제의 1 GB/s 읽기는
# 캐시 축출의 결과였으므로 메모리 상한만으로 해소된다.
EOF
chmod 644 "$DROPIN"

systemctl daemon-reload
echo "적용했습니다: $DROPIN"
echo

show

echo
echo "값을 바꾸려면 스크립트 상단의 MEM_HIGH / MEM_MAX 를 고치고 다시 실행하세요."
echo "되돌리려면: bash scripts/setup-session-limits.sh --undo"
