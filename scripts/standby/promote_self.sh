#!/usr/bin/env bash
# Promote this standby to primary. Runs ON the standby host, standalone.
#
# WHY THIS EXISTS SEPARATELY FROM scripts/promote_standby.sh:
#   That script lives on the main VM and also repoints the app tier at the new
#   primary. It is the right tool when the main VM is alive and only the
#   database is broken. But if the main VM is gone, so is that script — the one
#   situation where you most need to promote is the one where you cannot reach
#   it. This copy lives on the standby and depends on nothing else.
#
# SCOPE: it promotes, and stops. Pointing the application at the new primary is
# a separate job, because at that point the application may not have a host to
# run on yet. The next steps are printed at the end.
#
# Deployed to /root/pgstandby/promote.sh — keep this repo copy as the source of
# truth and redeploy after edits.
set -euo pipefail

CTR=leninbot-pg-standby
PRIMARY_HOST=100.122.248.77
PRIMARY_PORT=5434
PASSFILE=/root/pgstandby/pgpass
IMAGE="$(docker inspect "$CTR" --format '{{.Config.Image}}' 2>/dev/null || echo '')"

DRY_RUN=0
CONFIRM=""
for arg in "${@:-}"; do
  case "$arg" in
    ""|--help|-h)
      # 셰뱅 다음의 주석 블록을 그대로 설명문으로 쓴다. 줄 번호로 자르면
      # 파일을 고칠 때마다 조용히 어긋난다.
      awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
      echo
      echo "usage: $0 --dry-run"
      echo "       $0 --confirm=PROMOTE_STANDBY"
      exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --confirm=*) CONFIRM="${arg#--confirm=}" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
die()  { printf '\nABORT: %s\n' "$*" >&2; exit 1; }
sql()  { docker exec "$CTR" psql -U postgres -tAc "$1" 2>/dev/null || true; }

step "1/4  이 노드 상태"
[[ -n "$IMAGE" ]] || die "컨테이너 $CTR 를 찾을 수 없다. 여기가 스탠바이 호스트가 맞나?"
in_rec="$(sql 'SELECT pg_is_in_recovery();')"
case "$in_rec" in
  t) say "     복구 모드 — 승격 대상이 맞다" ;;
  f) die "이미 primary다 (pg_is_in_recovery=f). 승격할 것이 없다." ;;
  *) die "상태를 읽지 못했다 (응답: '${in_rec:-없음}'). 컨테이너가 살아 있나?
     docker logs --tail 50 $CTR" ;;
esac
say "     마지막 수신 LSN: $(sql 'SELECT pg_last_wal_receive_lsn();')"
say "     마지막 재생 LSN: $(sql 'SELECT pg_last_wal_replay_lsn();')"

step "2/4  옛 primary가 살아 있나 (스플릿브레인 검사)"
# 양쪽이 동시에 쓰기를 받으면 데이터가 갈라지고 합칠 방법이 없다.
# 이것이 이 스크립트에서 유일하게 되돌릴 수 없는 실수다.
primary_state="$(docker run --rm -v "$PASSFILE":/pp:ro -e PGPASSFILE=/pp "$IMAGE" \
  psql "host=$PRIMARY_HOST port=$PRIMARY_PORT user=replicator dbname=postgres connect_timeout=5" \
  -tAc 'SELECT pg_is_in_recovery();' 2>/dev/null || echo 'UNREACHABLE')"

case "$primary_state" in
  UNREACHABLE)
    say "     primary에 도달할 수 없다 — 승격해도 되는 상황이다" ;;
  f)
    # dry-run은 아무것도 바꾸지 않으므로 여기서 멈추면 안 된다. primary가
    # 건강할 때가 이 계획을 차분히 읽을 유일한 시점이다.
    split_brain_msg="옛 primary($PRIMARY_HOST)가 아직 살아서 쓰기를 받고 있다.

     지금 승격하면 두 노드가 각자 쓰기를 받아 데이터가 두 갈래로
     갈라지고, 되돌리거나 합칠 방법이 없다.

     먼저 확인할 것:
       - 정말 메인 VM이 죽었나? 살아 있다면 거기서
         scripts/promote_standby.sh 를 쓰는 편이 낫다 (앱 재배선까지 해준다)
       - DB만 이상한 것이라면 docker restart leninbot-pg 부터 시도할 것

     그래도 넘길 것이면 옛 primary를 확실히 정지시킨 뒤 다시 실행:
       ssh grass@$PRIMARY_HOST 'docker stop leninbot-pg'"
    if [[ $DRY_RUN -eq 1 ]]; then
      say "     !! 실제 실행이었다면 여기서 중단됐다:"
      printf '%s\n' "$split_brain_msg" | sed 's/^/     /'
    else
      die "$split_brain_msg"
    fi ;;
  t)
    say "     primary가 복구 모드다 (이미 강등됐거나 다른 스탠바이다) — 진행 가능" ;;
  *)
    say "     판정 불가 (응답: '$primary_state') — 직접 확인할 것" ;;
esac

step "3/4  승격"
if [[ $DRY_RUN -eq 1 ]]; then
  say "     [dry-run] SELECT pg_promote();  — 실행하지 않음"
  say "     [dry-run] 이후 pg_is_in_recovery() 가 f 가 될 때까지 대기"
else
  [[ "$CONFIRM" == "PROMOTE_STANDBY" ]] || die "되돌릴 수 없는 작업이다. 확인하려면:
       $0 --confirm=PROMOTE_STANDBY
     먼저 --dry-run 으로 계획을 볼 것을 권한다."
  docker exec "$CTR" psql -U postgres -c 'SELECT pg_promote();' >/dev/null
  for _ in $(seq 1 30); do
    [[ "$(sql 'SELECT pg_is_in_recovery();')" == "f" ]] && break
    sleep 2
  done
  [[ "$(sql 'SELECT pg_is_in_recovery();')" == "f" ]] \
    || die "30초 안에 승격이 끝나지 않았다. docker logs --tail 50 $CTR 를 볼 것."
  say "     승격 완료 — 이 노드가 이제 primary이고 쓰기를 받는다"
fi

step "4/4  확인"
if [[ $DRY_RUN -eq 0 ]]; then
  say "     in_recovery : $(sql 'SELECT pg_is_in_recovery();')  (f 여야 정상)"
  say "     현재 WAL LSN : $(sql 'SELECT pg_current_wal_lsn();')"
  say "     lenin_corpus : $(docker exec "$CTR" psql -U postgres -d leninbot -tAc 'SELECT count(*) FROM lenin_corpus;' 2>/dev/null || echo '?') 행"
fi

cat <<EOF

════════════════════════════════════════════════════════════════════
다음: 앱을 이 노드로 보내야 한다. 승격만으로는 사이트가 살아나지 않는다.

이 호스트에는 nginx도 frontend도 API도 없다. 앱이 도는 호스트에서
DB 주소를 여기로 바꿔야 한다.

  접속 주소 : $PRIMARY_HOST 자리에 → 100.124.58.85 : 5434
  DB        : leninbot / writer / legacy_game

메인 VM이 살아 있다면 (DB만 죽었던 경우):
  /home/grass/leninbot/.env       DB_HOST, DB_PORT, WRITER_DB_HOST, WRITER_DB_PORT
  /home/grass/frontend/.env       DB_HOST, DB_PORT
    → frontend는 env가 docker run 시점에 박히므로 재생성해야 한다
  sudo systemctl restart leninbot-api leninbot-telegram leninbot-a2a-api \\
       leninbot-email-api leninbot-roleplay novel-writer-api
  sudo systemctl stop leninbot-replication-health.timer   # 스탠바이가 없어졌다

메인 VM이 죽었다면:
  새 호스트를 띄우고 저장소를 클론한 뒤 위 값들을 이 주소로 설정한다.
  Cloudflare DNS를 새 origin IP로, 방화벽에 80/443(Cloudflare 대역) 규칙.
  데이터는 여기 이미 뜨겁게 살아 있으니 R2 복원을 기다릴 필요는 없다.

옛 primary는 그냥 켠다고 복제에 합류하지 않는다 — 타임라인이 갈라져서
pg_rewind 또는 새 base backup이 필요하다.

자세한 것: dev_docs/standby_operations.md
════════════════════════════════════════════════════════════════════
EOF
