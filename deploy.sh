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
# - non-root: non-interactive sudo (fails fast if not permitted)
_run_systemctl() {
    if [ "$(id -u)" -eq 0 ]; then
        systemctl "$@"
    else
        sudo -n systemctl "$@"
    fi
}

# 1. 코드 업데이트 (변경분만)
_git_fetch_with_retry
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "이미 최신. 변경 없음."
    exit 0
fi

CHANGES=$(git log "$LOCAL..$REMOTE" --oneline)
echo "변경 감지: $LOCAL → $REMOTE"
echo "$CHANGES"
git pull origin "$BRANCH"

# 2. 의존성 — requirements.txt 변경 시에만 설치
if git diff "$LOCAL" "$REMOTE" --name-only | grep -q "requirements"; then
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

# 4. 서비스 재시작 (API 먼저)
echo "서비스 재시작..."
_run_systemctl restart leninbot-api
_run_systemctl is-active --quiet leninbot-api

# 5. 성공 알림 (텔레그램 재시작 전에 전송)
SUMMARY=$(git log --oneline -1)
_notify_telegram "✅ *Deploy 완료*
\`$SUMMARY\`
변경 커밋:
\`\`\`
$CHANGES
\`\`\`"

# 6. 텔레그램 재시작은 마지막 단계
# 주의: systemd 환경에서는 이 스크립트가 같은 cgroup에 있다면
# telegram 서비스 재시작 시 즉시 종료될 수 있으므로, 이후 후처리는 두지 않는다.
_run_systemctl restart leninbot-telegram

echo "=== Deploy 완료: $SUMMARY ==="
