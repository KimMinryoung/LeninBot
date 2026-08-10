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
# 적용 범위: user-1000.slice(= grass의 로그인 세션)에만 걸린다.
#   leninbot-* 서비스는 User=grass여도 system.slice 소속이라 영향이 없다.
#
# 사용법:  bash scripts/setup-session-limits.sh          # 적용 (필요하면 sudo로 자동 재실행)
#          bash scripts/setup-session-limits.sh --show   # 현재 상태만 확인
#          bash scripts/setup-session-limits.sh --undo   # 되돌리기
#
# 관련 문서: dev_docs/standby_operations.md (이 사건으로 복제가 끊겼다)

set -euo pipefail

UID_TARGET=1000
DROPIN_DIR="/etc/systemd/system/user-${UID_TARGET}.slice.d"
DROPIN="${DROPIN_DIR}/limits.conf"
SLICE="user-${UID_TARGET}.slice"

MEM_HIGH="6G"    # 소프트 상한 — 죽이지 않고 이 cgroup 안에서만 회수 시작
MEM_MAX="10G"    # 하드 상한 — 닿으면 폭주한 세션만 죽고 서버는 산다
CPU_WEIGHT="30"  # 경합 시 시스템 서비스에 양보 (기본 100)

show() {
  echo "── 현재 설정 ──"
  systemctl show "$SLICE" -p MemoryHigh -p MemoryMax -p MemorySwapMax -p CPUWeight 2>/dev/null || true
  echo "── 커널이 실제로 물고 있는 값 ──"
  for f in memory.high memory.max memory.swap.max cpu.weight; do
    p="/sys/fs/cgroup/user.slice/${SLICE}/${f}"
    [ -r "$p" ] && echo "  ${f} = $(cat "$p")" || echo "  ${f} = (없음)"
  done
  echo "── 이 슬라이스가 지금 쓰는 메모리 ──"
  p="/sys/fs/cgroup/user.slice/${SLICE}/memory.current"
  [ -r "$p" ] && awk '{printf "  memory.current = %.2f GiB\n", $1/1073741824}' "$p" || echo "  (없음)"
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
# 대화형 세션(SSH 셸, Claude Code)만 제한한다.
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
