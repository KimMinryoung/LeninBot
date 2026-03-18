#!/bin/bash
# ~/leninbot/deploy.sh
# 차분 배포 — 변경분만 pull, 의존성 변경 시에만 pip install, 서비스 재시작
# 텔레그램 /deploy 명령에서도 호출됨 (로그: /tmp/leninbot-deploy.log)

set -e

DEPLOY_DIR="/home/grass/leninbot"
BRANCH="main"
VENV="$DEPLOY_DIR/venv"
LOG="/tmp/leninbot-deploy.log"

exec > >(tee "$LOG") 2>&1

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
else
    echo "의존성 변경 없음 → 스킵"
fi

# 3. 서비스 재시작 (API 먼저, 텔레그램은 마지막)
echo "서비스 재시작..."
sudo systemctl restart leninbot-api
sudo systemctl restart leninbot-telegram
echo "=== Deploy 완료: $(git log --oneline -1) ==="
