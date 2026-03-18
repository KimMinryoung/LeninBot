# Hetzner 배포 가이드

## 파일 구성

| 파일 | 용도 |
|------|------|
| `setup-server.sh` | Hetzner 서버 초기 프로비저닝 (1회) |
| `deploy.sh` | 차분 배포 스크립트 (변경분만 pull + 재시작) |
| `systemd/leninbot-api.service` | FastAPI 서비스 (port 8000) |
| `systemd/leninbot-telegram.service` | Telegram 봇 서비스 |

## 서버 디렉토리 구조

```
/home/grass/leninbot/
  ├── .env                  # 환경변수 (수동 생성)
  ├── venv/                 # Python 가상환경
  ├── deploy.sh             # 배포 스크립트
  ├── api.py                # FastAPI 서버
  ├── telegram_bot.py       # Telegram 봇
  ├── systemd/              # systemd 서비스 파일
  └── ...                   # 나머지 프로젝트 파일
```

---

## 1단계: 초기 서버 세팅 (1회)

Hetzner 서버에 root로 접속하여 실행:

```bash
ssh root@YOUR_HETZNER_IP 'bash -s' < setup-server.sh
```

이 스크립트가 수행하는 작업:
1. 시스템 패키지 설치 (python3, git 등)
2. `grass` 유저 생성
3. GitHub 저장소 클론 → `/home/grass/leninbot/`
4. Python venv 생성 + `requirements.txt` 설치
5. systemd 서비스 파일 복사 및 등록 (`enable`)
6. `deploy.sh` 실행 권한 부여

## 2단계: 환경변수 설정

```bash
ssh root@YOUR_HETZNER_IP
nano /home/grass/leninbot/.env
```

필요한 환경변수:

```env
# Telegram
TELEGRAM_BOT_TOKEN=...
ALLOWED_USER_IDS=123456789,987654321

# LLM
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...

# Supabase (Vector DB)
SUPABASE_URL=...
SUPABASE_KEY=...

# PostgreSQL (태스크 큐, 채팅 히스토리)
DB_HOST=...
DB_PORT=5432
DB_NAME=postgres
DB_USER=...
DB_PASSWORD=...

# Neo4j (Knowledge Graph)
NEO4J_URI=...
NEO4J_USER=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=...
```

## 3단계: 서비스 시작

```bash
systemctl start leninbot-api leninbot-telegram
```

확인:

```bash
systemctl status leninbot-api
systemctl status leninbot-telegram
journalctl -u leninbot-telegram -f  # 실시간 로그
```

## 4단계: sudoers 설정

`deploy.sh`가 `grass` 유저로 `systemctl restart`를 실행하므로, 비밀번호 없이 허용:

```bash
visudo
```

추가:

```
grass ALL=(ALL) NOPASSWD: /bin/systemctl restart leninbot-api, /bin/systemctl restart leninbot-telegram
```

---

## 이후 배포 방법

### 방법 A: 텔레그램 `/deploy` 명령 (권장)

텔레그램에서 봇에게:

```
/deploy
```

### 방법 B: SSH 직접 실행

```bash
ssh grass@YOUR_HETZNER_IP '~/leninbot/deploy.sh'
```

---

## CI/CD 흐름 상세

```
개발자 PC                    GitHub                     Hetzner 서버
─────────                    ──────                     ─────────────
코드 수정
    │
git commit + push ──────→ origin/main 업데이트

텔레그램 /deploy ──────────────────────────────────────→ deploy.sh 실행
    (또는 SSH)                                              │
                                                            ├─ 1. git fetch
                                                            │   LOCAL vs REMOTE 비교
                                                            │   동일하면 exit 0
                                                            │
                                                            ├─ 2. git pull origin main
                                                            │   (변경분만 다운로드)
                                                            │
                                                            ├─ 3. requirements.txt 변경 확인
                                                            │   변경 시 → pip install
                                                            │   미변경 → 스킵
                                                            │
                                                            ├─ 4. deploy-meta.json 저장
                                                            │   (커밋 정보, 변경 내역)
                                                            │
                                                            ├─ 5. systemctl restart
                                                            │   leninbot-api (먼저)
                                                            │   leninbot-telegram (마지막)
                                                            │
                                                            └─ 6. curl → Telegram API
                                                                "Deploy 완료" 메시지 전송

                                                        새 봇 프로세스 기동
                                                            │
                                                            ├─ deploy-meta.json 읽기
                                                            ├─ system_alert에 주입
                                                            └─ 파일 삭제 (재트리거 방지)

텔레그램에서 확인:
  - curl 알림: "Deploy 완료" + 커밋 목록
  - 봇 대화: deploy 사실을 인식하고 맥락에 반영
```

### 차분 업데이트 조건 분기

| 조건 | 동작 |
|------|------|
| `LOCAL == REMOTE` | "이미 최신" → 즉시 종료 |
| 코드만 변경 | git pull → 서비스 재시작 |
| `requirements.txt` 변경 | git pull → pip install → 서비스 재시작 |

### Deploy 알림 이중 경로

| 경로 | 방식 | 목적 |
|------|------|------|
| deploy.sh → curl | Telegram Bot API 직접 호출 | 사용자에게 완료 통보 (봇 죽은 상태에서도 동작) |
| 새 봇 → system_alert | deploy-meta.json → `_system_alerts` 주입 | 봇이 자신의 업데이트를 인식, 대화 맥락에 반영 |

---

## 로그 확인

```bash
# systemd 서비스 로그
journalctl -u leninbot-api -f
journalctl -u leninbot-telegram -f

# deploy 실행 로그
cat /tmp/leninbot-deploy.log
```

## 트러블슈팅

| 증상 | 확인 |
|------|------|
| `/deploy` 응답 없음 | `deploy.sh` 존재 + 실행 권한 확인 |
| "이미 최신" | `git push`를 먼저 했는지 확인 |
| pip install 실패 | `journalctl -u leninbot-telegram`에서 에러 확인 |
| 봇 기동 안 됨 | `.env` 파일의 `TELEGRAM_BOT_TOKEN` 확인 |
| curl 알림 안 옴 | `.env`에 `ALLOWED_USER_IDS` 설정 확인 |
| sudoers 권한 오류 | `visudo`로 `grass` 유저의 systemctl 권한 확인 |
