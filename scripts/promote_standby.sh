#!/usr/bin/env bash
# Promote leninbot-standby and repoint this VM's services at it.
#
# SCOPE: this handles "the database is gone but this VM is fine" — a wedged or
# corrupted leninbot-pg, a bad volume, a botched migration. It does NOT handle
# "the whole VM is gone": everything that serves traffic (nginx, frontend, the
# APIs, neo4j, the embedding server) lives here, and the standby has 3.8 GB of
# RAM and none of it installed. Losing this host means provisioning a new one;
# the standby's job there is that the data is already hot, not that the site
# stays up. See dev_docs/standby_operations.md.
#
# Promotion is one-way. The old primary cannot simply resume replicating from
# the promoted node — timelines diverge and you need pg_rewind or a fresh base
# backup. Do not run this as a drill.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDBY_HOST=100.124.58.85
STANDBY_PORT=5434
STANDBY_SSH="root@${STANDBY_HOST}"
STANDBY_CTR=leninbot-pg-standby
PRIMARY_CTR=leninbot-pg
MAIN_ENV="$ROOT/.env"
FRONTEND_ENV=/home/grass/frontend/.env
SERVICES=(leninbot-api leninbot-telegram leninbot-a2a-api leninbot-email-api leninbot-roleplay)

DRY_RUN=0
CONFIRM=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --confirm=*) CONFIRM="${arg#--confirm=}" ;;
    *) echo "usage: $0 [--dry-run] [--confirm=PROMOTE_STANDBY]" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then printf '   [dry-run] %s\n' "$*"; else eval "$@"; fi; }
die()  { printf 'ABORT: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
step "사전 점검"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$STANDBY_SSH" true \
  || die "스탠바이에 SSH가 안 된다. tailnet과 호스트를 먼저 확인할 것."
say "   스탠바이 SSH 도달 OK"

in_recovery="$(ssh -o BatchMode=yes "$STANDBY_SSH" \
  "docker exec $STANDBY_CTR psql -U postgres -tAc 'SELECT pg_is_in_recovery();'" 2>/dev/null || true)"
[[ "$in_recovery" == "t" ]] \
  || die "스탠바이가 복구 모드가 아니다 (pg_is_in_recovery=${in_recovery:-?}). 이미 승격됐거나 다른 상태다."
say "   스탠바이 복구 모드 확인 OK"

# Split brain is the one unrecoverable mistake here: two nodes both taking
# writes means two divergent datasets and no automatic way to merge them.
if docker exec "$PRIMARY_CTR" psql -U postgres -tAc 'SELECT 1;' >/dev/null 2>&1; then
  primary_recovery="$(docker exec "$PRIMARY_CTR" psql -U postgres -tAc 'SELECT pg_is_in_recovery();' 2>/dev/null || echo '?')"
  if [[ "$primary_recovery" == "f" ]]; then
    # A dry run changes nothing, so it must still be usable while the primary is
    # healthy — that is the only time anyone will read the plan calmly.
    msg="primary($PRIMARY_CTR)가 아직 살아서 쓰기를 받고 있다.
     승격하면 양쪽이 각자 쓰기를 받아 데이터가 갈라지고, 되돌릴 방법이 없다.
     정말 넘길 것이면 primary를 먼저 확실히 정지시켜라:
       docker stop $PRIMARY_CTR"
    if [[ $DRY_RUN -eq 1 ]]; then
      say "   !! 실제 실행이었다면 여기서 중단됐다: $msg"
    else
      die "$msg"
    fi
  else
    say "   primary가 쓰기를 받지 않는 상태 확인 OK"
  fi
else
  say "   primary가 응답하지 않음 — 승격 대상 상황이 맞다"
fi

last_lsn="$(ssh -o BatchMode=yes "$STANDBY_SSH" \
  "docker exec $STANDBY_CTR psql -U postgres -tAc 'SELECT pg_last_wal_replay_lsn();'" 2>/dev/null || echo '?')"
say "   스탠바이 마지막 재생 LSN: $last_lsn"

if [[ $DRY_RUN -eq 0 && "$CONFIRM" != "PROMOTE_STANDBY" ]]; then
  die "이 작업은 되돌릴 수 없다. 확인하려면 --confirm=PROMOTE_STANDBY 를 붙일 것.
     먼저 --dry-run 으로 무엇이 바뀌는지 볼 것을 권한다."
fi

# ------------------------------------------------------------------ backup
step "설정 백업"
stamp="$(date +%Y%m%d-%H%M%S)"
run "cp -a '$MAIN_ENV' '$MAIN_ENV.bak-promote-$stamp'"
run "cp -a '$FRONTEND_ENV' '$FRONTEND_ENV.bak-promote-$stamp'"
say "   .bak-promote-$stamp 로 보관"

# ----------------------------------------------------------------- promote
step "스탠바이 승격"
run "ssh -o BatchMode=yes '$STANDBY_SSH' \"docker exec $STANDBY_CTR psql -U postgres -c 'SELECT pg_promote();'\""
if [[ $DRY_RUN -eq 0 ]]; then
  for _ in $(seq 1 30); do
    r="$(ssh -o BatchMode=yes "$STANDBY_SSH" \
      "docker exec $STANDBY_CTR psql -U postgres -tAc 'SELECT pg_is_in_recovery();'" 2>/dev/null || echo '?')"
    [[ "$r" == "f" ]] && break
    sleep 2
  done
  [[ "$r" == "f" ]] || die "승격이 30초 안에 끝나지 않았다 (pg_is_in_recovery=$r). 스탠바이 로그를 볼 것."
  say "   승격 완료 — 쓰기 가능 상태"
fi

# ------------------------------------------------------------------ repoint
step "메인 서비스를 새 주소로"
run "sed -i 's|^DB_HOST=.*|DB_HOST=$STANDBY_HOST|; s|^DB_PORT=.*|DB_PORT=$STANDBY_PORT|; s|^WRITER_DB_HOST=.*|WRITER_DB_HOST=$STANDBY_HOST|; s|^WRITER_DB_PORT=.*|WRITER_DB_PORT=$STANDBY_PORT|' '$MAIN_ENV'"

# The frontend keeps its own .env and reaches Postgres by container name over
# the docker network. That container is gone, so it needs the tailnet address —
# and env is baked in at `docker run`, so a restart is not enough.
step "frontend 재생성 (env는 기동 시점에 박히므로 restart로는 안 바뀐다)"
run "sed -i 's|^DB_HOST=.*|DB_HOST=$STANDBY_HOST|; s|^DB_PORT=.*|DB_PORT=$STANDBY_PORT|' '$FRONTEND_ENV'"
run "docker rm -f leninbot-frontend"
run "docker run -d --name leninbot-frontend --restart unless-stopped \
  --env-file '$FRONTEND_ENV' -p 127.0.0.1:3000:3000 \
  -v /home/grass/frontend/data:/app/data leninbot-frontend"
run "docker network connect leninbot_default leninbot-frontend"

step "서비스 재시작"
for svc in "${SERVICES[@]}"; do
  run "sudo systemctl restart $svc"
done
run "sudo systemctl restart novel-writer-api.service"

# --------------------------------------------------------------- aftermath
step "확인"
if [[ $DRY_RUN -eq 0 ]]; then
  sleep 5
  say "   서비스 상태:"
  for svc in "${SERVICES[@]}"; do printf '     %-24s %s\n' "$svc" "$(systemctl is-active "$svc")"; done
  curl -s -o /dev/null -w "     cyber-lenin.com HTTP %{http_code}\n" --max-time 25 "https://cyber-lenin.com/?cb=$RANDOM" || true
fi

cat <<'AFTER'

════════════════════════════════════════════════════════════════════
승격 후 남은 일 — 자동으로 되지 않는다

1. 이제 스탠바이가 없다. 복제·백업 구성이 전부 옛 primary를 가리킨다:
   - 슬롯 standby_hel1, max_slot_wal_keep_size
   - 일일 백업 3종 (leninbot-pg 컨테이너명 하드코딩)
   - leninbot-replication-health.timer → 계속 실패 알림이 온다.
     새 구성을 세우기 전까지는 멈춰 둘 것:
       sudo systemctl stop leninbot-replication-health.timer

2. 새 스탠바이를 세우거나, 옛 primary를 고쳐 역방향 복제를 구성한다.
   옛 primary는 그냥 재기동해서는 복제에 합류하지 못한다 (타임라인 분기).
   pg_rewind 또는 새 base backup이 필요하다.

3. 워치독의 데드맨 스위치는 백업 잡이 안 돌면 곧 알림을 보낸다. 정상이다.

4. 롤백하려면 .bak-promote-* 파일을 되돌리고 서비스를 재시작한다.
   단 승격 후 스탠바이에 들어간 쓰기는 옛 primary에 없다.
════════════════════════════════════════════════════════════════════
AFTER
