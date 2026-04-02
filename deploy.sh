#!/bin/bash
# ~/leninbot/deploy.sh
# 차분 배포 — 변경분만 pull, 의존성 변경 시에만 pip install, 서비스 재시작
# 텔레그램 /deploy 명령에서도 호출됨 (로그: /tmp/leninbot-deploy.log)
#
# 에러 발생 시에도 텔레그램으로 실패 알림 전송

DEPLOY_DIR="/home/grass/leninbot"
BRANCH="main"
VENV="$DEPLOY_DIR/venv"
LOG="/tmp/leninbot-deploy.log"
DEPLOY_META="/tmp/leninbot-deploy-meta.json"
LOCK_FILE="/tmp/leninbot-deploy.lock"
FORCE_RESTART=0
SERVICE="all"  # all | api | telegram
FRONTEND_DIR="/home/grass/frontend"

for arg in "$@"; do
    case "$arg" in
        --restart|-r)
            FORCE_RESTART=1
            ;;
        --telegram)
            SERVICE="telegram"
            ;;
        --api)
            SERVICE="api"
            ;;
        --frontend)
            SERVICE="frontend"
            ;;
        --all)
            SERVICE="all"
            ;;
        --help|-h)
            echo "Usage: bash deploy.sh [--restart] [--telegram|--api|--frontend|--all]"
            echo "  --restart, -r   Restart services even when no new commit is found."
            echo "  --telegram      Restart leninbot-telegram only."
            echo "  --api           Restart leninbot-api only."
            echo "  --frontend      Rebuild and restart frontend container only."
            echo "  --all           Restart all services (default)."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage."
            exit 2
            ;;
    esac
done

exec > >(tee "$LOG") 2>&1

# ── 텔레그램 알림 함수 ──────────────────────────────────────────
_notify_telegram() {
    local msg="$1"
    source "$DEPLOY_DIR/.env" 2>/dev/null
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$ALLOWED_USER_IDS" ]; then
        IFS=',' read -ra UIDS <<< "$ALLOWED_USER_IDS"
        for uid in "${UIDS[@]}"; do
            uid=$(echo "$uid" | tr -d ' ')
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d chat_id="$uid" -d text="$msg" -d parse_mode="Markdown" > /dev/null 2>&1 || true
        done
    fi
}

# ── 동시 실행 방지 락 ───────────────────────────────────────────
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "이미 다른 deploy가 실행 중입니다. 이번 요청은 스킵합니다."
    _notify_telegram "⚠️ *Deploy 스킵*: 이미 다른 deploy가 진행 중입니다."
    exit 0
fi

# ── 에러 트랩 — 실패 시 알림 전송 후 종료 ──────────────────────
_on_error() {
    local exit_code=$?
    local failed_line=$1
    local error_context
    error_context=$(tail -5 "$LOG" 2>/dev/null | head -4)
    echo "=== Deploy 실패 (line $failed_line, exit $exit_code) ==="

    _notify_telegram "❌ *Deploy 실패* (exit $exit_code)
단계: line $failed_line
\`\`\`
$error_context
\`\`\`"

    # 실패 메타도 저장 (새 봇 인스턴스가 읽을 수 있도록)
    cat > "$DEPLOY_META" <<METAEOF
{
  "timestamp": "$(date -Iseconds)",
  "status": "failed",
  "exit_code": $exit_code,
  "failed_line": $failed_line,
  "error": $(echo "$error_context" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))' 2>/dev/null || echo '""')
}
METAEOF

    exit $exit_code
}

trap '_on_error $LINENO' ERR
set -eE  # -E ensures trap propagates to functions

cd "$DEPLOY_DIR"
echo "=== $(date) Deploy 시작 ==="

# git fetch helper: handles transient ref races/retries.
_git_fetch_with_retry() {
    local max_attempts=4
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
        if git fetch --prune origin; then
            return 0
        fi
        echo "git fetch 실패 (시도 $attempt/$max_attempts) — 정리 후 재시도"
        # Clean potentially stale ref state and retry.
        git remote prune origin || true
        git gc --prune=now --quiet || true
        sleep $((attempt * 2))
        attempt=$((attempt + 1))
    done
    return 1
}

# Run systemctl with least-friction privilege path:
# - root: direct call
# - non-root: sudo with absolute path (matches sudoers NOPASSWD entry)
_run_systemctl() {
    if [ "$(id -u)" -eq 0 ]; then
        /usr/bin/systemctl "$@"
    else
        sudo /usr/bin/systemctl "$@"
    fi
}

# 프론트엔드 전용 모드 — 백엔드 단계 건너뛰기
if [ "$SERVICE" = "frontend" ]; then
    CHANGES="(프론트엔드 전용 배포)"
    SUMMARY="frontend-only"
    DID_PULL=0
else

# 1. 코드 업데이트 (변경분만)
_git_fetch_with_retry
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")
DID_PULL=0

if [ "$LOCAL" = "$REMOTE" ]; then
    if [ "$FORCE_RESTART" -eq 0 ]; then
        echo "이미 최신. 변경 없음."
        exit 0
    fi
    echo "이미 최신이지만 --restart 옵션으로 서비스 재시작을 진행합니다."
    CHANGES="(코드 변경 없음: 서비스 재시작만 수행)"
else
    CHANGES=$(git log "$LOCAL..$REMOTE" --oneline)
    echo "변경 감지: $LOCAL → $REMOTE"
    echo "$CHANGES"
    git pull origin "$BRANCH"
    DID_PULL=1
fi

fi  # end: SERVICE != frontend

# 2. 의존성 — requirements.txt 변경 시에만 설치
if [ "$DID_PULL" -eq 1 ] && git diff "$LOCAL" "$REMOTE" --name-only | grep -q "requirements"; then
    echo "의존성 변경 감지 → pip install"
    source "$VENV/bin/activate"
    pip install -r requirements.txt --quiet
    # Playwright 브라우저 설치 (chromium만, 이미 있으면 스킵)
    if pip show playwright > /dev/null 2>&1; then
        echo "Playwright chromium 설치 확인..."
        playwright install --with-deps chromium 2>/dev/null || true
    fi
else
    echo "의존성 변경 없음 → 스킵"
fi

# 3. Deploy 메타데이터 저장 (새 봇 인스턴스가 읽음)
cat > "$DEPLOY_META" <<METAEOF
{
  "timestamp": "$(date -Iseconds)",
  "status": "success",
  "prev_commit": "$LOCAL",
  "new_commit": "$REMOTE",
  "changes": $(echo "$CHANGES" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'),
  "deps_updated": $(git diff "$LOCAL" "$REMOTE" --name-only | grep -q "requirements" && echo "true" || echo "false")
}
METAEOF

# 4. systemd unit 파일 변경 시 daemon-reload
if [ "$DID_PULL" -eq 1 ] && git diff "$LOCAL" "$REMOTE" --name-only | grep -q "systemd/"; then
    echo "systemd unit 파일 변경 감지 → daemon-reload"
    _run_systemctl daemon-reload
fi

# 5. 서비스 재시작 (SERVICE 플래그에 따라 선택적)
SUMMARY=$(git log --oneline -1)
RESTARTED=""

if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "api" ]; then
    echo "leninbot-api 재시작..."
    _run_systemctl restart leninbot-api
    _run_systemctl is-active --quiet leninbot-api
    RESTARTED="api"
fi

# 6. 성공 알림 (텔레그램 재시작 전에 전송)
_notify_telegram "✅ *Deploy 완료* [$SERVICE]
\`$SUMMARY\`
변경 커밋:
\`\`\`
$CHANGES
\`\`\`"

# 7. Browser worker 재시작 (코드 변경 시 lazy import 캐시 무효화 필요)
if [ "$SERVICE" = "all" ] && _run_systemctl is-active --quiet leninbot-browser 2>/dev/null; then
    echo "leninbot-browser 재시작..."
    _run_systemctl restart leninbot-browser
    RESTARTED="${RESTARTED:+$RESTARTED+}browser"
fi

# 8. 텔레그램 재시작은 마지막 단계
# 주의: systemd 환경에서는 이 스크립트가 같은 cgroup에 있다면
# telegram 서비스 재시작 시 즉시 종료될 수 있으므로, 이후 후처리는 두지 않는다.
if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "telegram" ]; then
    echo "leninbot-telegram 재시작..."
    _run_systemctl restart leninbot-telegram
    RESTARTED="${RESTARTED:+$RESTARTED+}telegram"
fi

# 9. 프론트엔드 배포 (Docker)
_deploy_frontend() {
    cd "$FRONTEND_DIR"

    git fetch origin
    local LOCAL_FE
    LOCAL_FE=$(git rev-parse HEAD)
    local REMOTE_FE
    REMOTE_FE=$(git rev-parse origin/master)

    if [ "$LOCAL_FE" = "$REMOTE_FE" ] && [ "$FORCE_RESTART" -eq 0 ]; then
        echo "프론트엔드: 변경 없음 → 스킵"
        return 0
    fi

    git pull origin master

    docker build -t leninbot-frontend .

    docker stop leninbot-frontend 2>/dev/null || true
    docker rm leninbot-frontend 2>/dev/null || true

    docker run -d \
        --name leninbot-frontend \
        --add-host=host.docker.internal:host-gateway \
        -p 127.0.0.1:3000:3000 \
        --restart unless-stopped \
        --env-file "$FRONTEND_DIR/.env" \
        -v "$FRONTEND_DIR/data:/app/data" \
        leninbot-frontend

    docker image prune -f

    echo "프론트엔드 배포 완료"
    cd "$DEPLOY_DIR"
}

if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "frontend" ]; then
    if [ -d "$FRONTEND_DIR" ]; then
        echo "프론트엔드 배포 시작..."
        _deploy_frontend
        RESTARTED="${RESTARTED:+$RESTARTED+}frontend"
    fi
fi

echo "=== Deploy 완료 [$RESTARTED]: $SUMMARY ==="
