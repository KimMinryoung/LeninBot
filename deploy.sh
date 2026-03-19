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

# 1. 코드 업데이트 (변경분만)
git fetch origin
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

# 4. 서비스 재시작 (API 먼저, 텔레그램은 마지막)
echo "서비스 재시작..."
sudo systemctl restart leninbot-api
sudo systemctl restart leninbot-telegram

# 5. 성공 알림 (봇이 죽은 후이므로 curl로 직접 전송)
SUMMARY=$(git log --oneline -1)
_notify_telegram "✅ *Deploy 완료*
\`$SUMMARY\`
변경 커밋:
\`\`\`
$CHANGES
\`\`\`"

echo "=== Deploy 완료: $SUMMARY ==="
