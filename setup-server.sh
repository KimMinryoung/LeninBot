#!/bin/bash
# Hetzner 서버 초기 세팅 (root로 1회 실행)
# 사용법: ssh root@YOUR_IP 'bash -s' < setup-server.sh

set -e

DEPLOY_USER="grass"
REPO_URL="https://github.com/KimMinryoung/LeninBot.git"
DEPLOY_DIR="/home/$DEPLOY_USER/leninbot"

echo "=== 1. 시스템 패키지 설치 ==="
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip git

echo "=== 2. 유저 생성 (이미 있으면 스킵) ==="
id "$DEPLOY_USER" &>/dev/null || useradd -m -s /bin/bash "$DEPLOY_USER"

echo "=== 3. 코드 클론 ==="
sudo -u "$DEPLOY_USER" git clone "$REPO_URL" "$DEPLOY_DIR" || {
    echo "이미 클론됨, pull 실행"
    cd "$DEPLOY_DIR" && sudo -u "$DEPLOY_USER" git pull origin main
}

echo "=== 4. venv 생성 + 의존성 설치 ==="
cd "$DEPLOY_DIR"
sudo -u "$DEPLOY_USER" python3 -m venv venv
sudo -u "$DEPLOY_USER" venv/bin/pip install -r requirements.txt --quiet

echo "=== 4.5. Playwright 브라우저 설치 ==="
sudo -u "$DEPLOY_USER" venv/bin/playwright install --with-deps chromium

echo "=== 5. .env 파일 ==="
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    echo "⚠️  .env 파일을 직접 생성하세요: $DEPLOY_DIR/.env"
    echo "필요한 키: TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE 등"
fi

echo "=== 6. systemd 서비스 및 타이머 등록 ==="
cp "$DEPLOY_DIR/systemd/leninbot-api.service" /etc/systemd/system/
cp "$DEPLOY_DIR/systemd/leninbot-telegram.service" /etc/systemd/system/
cp "$DEPLOY_DIR/systemd/leninbot-neo4j.service" /etc/systemd/system/
cp "$DEPLOY_DIR/systemd/leninbot-experience.service" /etc/systemd/system/
cp "$DEPLOY_DIR/systemd/leninbot-experience.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable leninbot-api leninbot-telegram leninbot-neo4j \
    leninbot-experience.timer

echo "=== 7. deploy.sh 실행 권한 ==="
chmod +x "$DEPLOY_DIR/deploy.sh"

echo ""
echo "=== 초기 세팅 완료 ==="
echo "다음 단계:"
echo "  1. .env 파일 작성: nano $DEPLOY_DIR/.env"
echo "  2. 서비스 시작: systemctl start leninbot-api leninbot-telegram"
echo "  3. 로그 확인: journalctl -u leninbot-api -f"
echo "  4. 이후 배포: sudo -u $DEPLOY_USER $DEPLOY_DIR/deploy.sh"
