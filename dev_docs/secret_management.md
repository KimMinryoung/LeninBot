# Secret Management

## 1. Purpose

프로덕션 서비스의 시크릿(API 키·패스워드·토큰)을 `.env` 평문 파일이 아닌 **systemd-creds 로 암호화된 서비스별 tmpfs 마운트** 에서 읽도록 한 계층. 개발 CLI 는 여전히 `.env` fallback 경로로 동작하되, Tier A 시크릿은 `.env` 에 더 이상 존재하지 않는다.

**설계 원칙**:
- **Single source of truth**: 프로덕션은 `/etc/credstore.encrypted/`, dev 는 `.env` (Tier B 한정)
- **Least privilege**: 서비스는 자기가 실제로 쓰는 키만 mount — kg-backup 은 21개가 아니라 2개
- **Zero-config clone**: 코드 repo 를 클론한 사람은 PROJECT_ROOT 같은 경로 환경변수를 건드리지 않는다 — 코드가 `__file__` 로 스스로 위치를 안다
- **Graceful rotation**: `rotate --restart` 한 번으로 값 교체 → active 서비스만 재시작 → 검증 → 백업 정리까지 원자적

## 2. Three-Tier Classification

`.env` / systemd credential / 코드 유도 3곳 중 어디에 두는지 결정하는 기준.

| Tier | 성격 | 저장 위치 | 예시 |
|------|------|-----------|------|
| **A. 진짜 비밀** | 노출 시 즉시 피해 | `/etc/credstore.encrypted/<name>.cred` (systemd-creds 암호화) | `*_API_KEY`, `*_TOKEN`, `*_PASSWORD`, `TELEGRAM_BOT_TOKEN`, 지갑 프라이빗키 |
| **B. 비-비밀 설정** | 공개돼도 무해, 머신마다 다름 | `.env` (계속 사용) | `DB_HOST`, `NEO4J_URI`, `REDIS_URL`, `MY_DOMAIN`, 이메일 서버 주소/포트/사용자명, 지갑 공개주소 |
| **C. 런타임 상태** | env 가 아닌 것 | 코드에서 유도 | `PROJECT_ROOT` (← `Path(__file__)`), 배포 lock/log 경로 (하드코드), 현재 git 브랜치 (← `git`) |

**핵심 기준**: "운영자가 반드시 설정해줘야만 하는 입력"만 env. 코드가 스스로 알 수 있는 건 코드에.

## 3. Architecture

```
                       sudo manage_secrets add/rotate
                                │
                                v
                   systemd-creds encrypt --name=<lower>
                                │
                                v
               /etc/credstore.encrypted/<lower>.cred  (host-bound, 0400 root)
                                │
                   systemd unit drop-in declares:
                 LoadCredentialEncrypted=<lower>:<path>
                                │
                           service start
                                │
                                v
      /run/credentials/<svc>.service/<lower>   (tmpfs, 0400, service-private)
                                │
                                v
                    secrets_loader.get_secret(NAME)
                    ├─ CREDENTIALS_DIRECTORY/<lower> 우선
                    └─ os.environ[NAME] fallback (.env 경로, dev 전용)
                                │
                                v
                  application code (bot_config, shared, db, ...)
```

**host-bound 의미**: `systemd-creds encrypt` 는 `/var/lib/systemd/credential.secret`(root-only) 로 파생한 키로 암호화 → `.cred` 파일을 다른 머신에 복사해도 복호화 불가. LUKS/TPM 이 없는 이 환경에선 TPM-bind 보다 한 단계 약하지만 "디스크 이미지 탈취" 시나리오는 막아낸다.

## 4. Per-Service Credential Scopes

`scripts/migrate_secrets_to_credstore.py` 의 `SERVICE_CREDS` 가 단일 소스. `scripts/dropins/<svc>.conf` (gitignored, 생성됨) 로 rendering 되어 `/etc/systemd/system/<svc>.service.d/credentials.conf` 로 설치.

| 서비스 | 마운트되는 cred 수 | 근거 |
|---|---|---|
| leninbot-api | 22 (전체 Tier A) | agent host, tool surface 광범위 |
| leninbot-telegram | 22 (전체 Tier A) | agent host, tool surface 광범위 |
| leninbot-browser | 3 (anthropic, openai, deepseek) | browser-use 는 LLM 호출만 |
| leninbot-autonomous | 10 | planning LLMs + KG/DB + telegram/razvedchik |
| leninbot-razvedchik | 2 | Moltbook patrol writes + optional Telegram notification |
| leninbot-experience | 6 | Gemini(writing) + KG/DB + bot_config imports |
| leninbot-kg-backup | 2 (r2_cf_api_token, neo4j_password) | R2 업로드 + Neo4j 덤프만 |
| leninbot-embedding | 0 | BGE-M3 로컬 모델, 외부 API 없음 |
| leninbot-neo4j | 0 | Docker wrapper, compose env 별도 |

**합산**: 67 credential mount (만약 전 서비스에 전부 attach 했으면 154 → -56%). 한 서비스가 뚫려도 다른 서비스용 시크릿은 노출 안 됨.

## 5. Runtime Resolution (`secrets_loader.py`)

```python
from secrets_loader import get_secret, require_secret

api_key = get_secret("GEMINI_API_KEY")        # None 가능
api_key = get_secret("GEMINI_API_KEY", "")    # 기본값 허용
api_key = require_secret("GEMINI_API_KEY")    # 없으면 RuntimeError
```

**우선순위**:
1. `$CREDENTIALS_DIRECTORY/<name_lower>` (systemd mount, 프로덕션)
2. `os.environ[<NAME>]` (`.env` → `load_dotenv()` 로 주입됨, dev CLI)

**네이밍 규칙**: env var 이름을 lowercase 하여 cred 파일명으로 사용. `GEMINI_API_KEY` ↔ `gemini_api_key.cred` ↔ `LoadCredentialEncrypted=gemini_api_key:...` 전부 통일.

**캐싱**: `@lru_cache` — 프로세스 수명 동안 첫 호출 결과를 캐싱. rotate 후에는 반드시 서비스 재시작해야 새 값 인지 (이것이 `rotate --restart` 가 존재하는 이유).

## 6. Operational CLI (`scripts/manage_secrets.py`)

전부 root 권한 필요. 값은 절대 stdout 으로 출력 안 됨. 입력은 `getpass` hidden 프롬프트 또는 stdin 파이프.

```bash
# 조회 (크기 · 수정일 · age · 마운트된 서비스)
sudo manage_secrets.py list

# 노후 키 필터 (90일 이상)
sudo manage_secrets.py list --stale 90

# 노후 키 있으면 Telegram 알림까지
sudo manage_secrets.py list --stale 90 --notify

# 신규 cred 등록 (아직 드롭인에 없음)
sudo manage_secrets.py add NEW_API_KEY

# 기존 cred 값 교체 + 사용 중인 active 서비스 재시작까지
sudo manage_secrets.py rotate GEMINI_API_KEY --restart

# stdin 파이프로 값 공급 (자동화)
echo "new-value" | sudo manage_secrets.py rotate GEMINI_API_KEY --restart

# 삭제 (드롭인에서 먼저 빠져야 경고 없이 됨)
sudo manage_secrets.py delete OLD_KEY -f
```

## 7. Key Rotation Flow (`rotate --restart`)

```
1. 현재 .cred 를 .cred.bak 로 이름변경 (원자적 백업)
2. 새 값을 systemd-creds encrypt → .cred.tmp → .cred (atomic rename)
   실패 시: .cred.bak → .cred 복구하고 종료
3. _services_mounting(name) 으로 드롭인에서 이 cred 를 mount 하는 서비스 나열
4. _filter_active 로 현재 active 상태인 것만 추림 (timer-driven 은 제외)
5. systemctl restart <active_services>
6. 2초 대기 후 is-active 재확인
   실패 시: 롤백 명령 출력하고 종료 (.cred.bak 유지)
7. 성공 시: .cred.bak 삭제, "rotation verified" 출력
```

**timer-driven 서비스** (autonomous, experience, kg-backup): 재시작 대상 아님. 다음 timer firing 시 새 cred 자동 로드 — 재시작은 무의미.

## 8. Staleness Monitoring

주 1회 월요일 10:00 UTC 에 `leninbot-stale-secrets.timer` 발동.

```
/etc/systemd/system/leninbot-stale-secrets.{service,timer}
    │
    └─> sudo manage_secrets.py list --stale 90 --notify
            │
            ├─ /etc/credstore.encrypted/*.cred 의 mtime 조회
            ├─ 90일 이상 된 것 필터
            └─ 결과 있으면 Telegram sendMessage API POST
```

**설치**: `sudo scripts/install_stale_secrets_timer.sh` (idempotent)

**임계값 변경**: `scripts/systemd/leninbot-stale-secrets.service` 의 `ExecStart` 에서 `--stale 90` 수정 후 `daemon-reload`

**Telegram 경로**: 오직 `TELEGRAM_BOT_TOKEN`(credstore) + `TELEGRAM_CHAT_ID`(.env) 만 사용. 의존성은 stdlib `urllib` — 새 패키지 추가 없음.

## 9. Security Properties

**얻는 것**:
- 디스크 평문 Tier A 제거 — `.env` 에 더 이상 API 키 없음
- 서비스 격리 — 한 서비스가 뚫려도 다른 서비스용 시크릿은 접근 불가 (`/run/credentials/<svc>.service/` 는 `User=grass` 전용)
- Host-bound 암호화 — `.cred` 파일 복사해 갔더라도 다른 머신에선 복호화 불가
- Root-only at rest — `/etc/credstore.encrypted/` 는 0700 root
- tmpfs only at runtime — 디스크에 복호화된 값이 landing 안 함

**얻지 못하는 것**:
- TPM/HSM 레벨 하드웨어 바인딩 (현재 호스트는 TPM 미장착, `/var/lib/systemd/credential.secret` 파일 기반)
- 메모리 덤프 방지 — process 가 읽은 순간 heap 에 있음
- Insider / root 접근 방지 — 이건 근본적으로 가능하지 않음
- 자동 교체 — 키 로테이션은 운영자 trigger

**위협 모델 위치**: 이 layer 는 "디스크 탈취 + git 유출" 까지만 방어. 프로세스 RCE 이후는 범위 밖 (일반적인 container/process 격리 layer 몫).

## 10. Cookbook

### 신규 시크릿 추가

```
1. scripts/migrate_secrets_to_credstore.py:
   - TIER_A 리스트에 NEW_API_KEY 추가
   - SERVICE_CREDS 에서 이 키를 필요로 하는 서비스 집합에 포함
2. /home/grass/leninbot/.env 에 NEW_API_KEY=... 임시 추가 (마이그레이션 입력용)
3. sudo scripts/migrate_secrets_to_credstore.py  → /etc/credstore.encrypted/new_api_key.cred 생성 + 드롭인 regen
4. sudo scripts/apply_credentials_dropin.sh      → 드롭인 설치 + daemon-reload + 재시작
5. .env 에서 NEW_API_KEY 라인 삭제 (이제 credstore 가 SoT)
6. 코드에서: get_secret("NEW_API_KEY") 로 읽기
```

### 시크릿 교체

```
sudo manage_secrets.py rotate NEW_API_KEY --restart
# 프롬프트에 새 값 2회 입력 → active 서비스 자동 재시작 → is-active 검증 → 백업 정리
```

### 시크릿 은퇴 (서비스 자체 안 쓰게 됨)

```
1. scripts/migrate_secrets_to_credstore.py:
   - TIER_A 에서 제거
   - SERVICE_CREDS 의 어떤 집합에도 이 키가 없는지 확인
2. sudo scripts/apply_credentials_dropin.sh
   (이제 어느 서비스도 LoadCredentialEncrypted=<name>:... 선언 안 함)
3. sudo manage_secrets.py delete NAME -f
4. 코드에서 get_secret("NAME") 호출부와 관련 dead code 제거
5. 커밋
```

### 신규 서비스가 기존 cred 를 쓰게 됨

```
1. SERVICE_CREDS["leninbot-newsvc"] = {"NEEDED_KEY", ...}
2. sudo scripts/apply_credentials_dropin.sh
3. (신규 서비스 유닛 파일은 별도 작업)
```

## 11. Migration History

| 날짜 | 상태 |
|------|------|
| 2026-04-06 | `eth.privkey`, `sol.keypair` — 지갑 프라이빗키 최초 credstore 이전 (crypto_wallet 모듈이 선례) |
| 2026-04-23 | Phase A: `secrets_loader.py` + 21 Tier A 전체 credstore 이전 + 17개 파일 `os.getenv` → `get_secret` |
| 2026-04-23 | Phase B: `.env` 에서 Tier A 제거 (credstore 단일 소스) |
| 2026-04-23 | Per-service least-privilege scope 적용 (126 → 60 mount) |
| 2026-04-23 | `RENDER_API_KEY` retire — 사용 종료된 서비스 은퇴 첫 사례 |
| 2026-04-23 | `rotate --restart` + 주 1회 staleness 알림 타이머 |

## 12. Known Limitations

- **TPM 부재**: 현재 credential.secret 은 일반 파일 시스템에 존재. TPM 장착 시 `systemd-creds encrypt --with-key=tpm2` 로 단계 상승 가능
- **Dev CLI UX 저하**: `.env` 에 Tier A 없어서 로컬에서 `python some_script.py` 실행 시 API 키 없음. 필요 시 한시적으로 `export GEMINI_API_KEY=$(systemd-creds decrypt --name=gemini_api_key /etc/credstore.encrypted/gemini_api_key.cred)` 하거나 임시 `.env.dev` 사용
- **TELEGRAM_BOT_TOKEN 교체의 교착 가능성**: staleness 알림 자체가 이 토큰 쓰므로 토큰이 잘못 교체되면 알림 자체가 실패 — 다만 manage_secrets rotate 플로우의 롤백 경로로 복구 가능
- **credstore 백업 없음**: `/etc/credstore.encrypted/` 자체는 호스트 손실 시 재생성 필요 (원본 API 키를 provider 콘솔에서 재발급하거나 별도 안전한 곳에 평문 복사본 유지). KG 백업(R2)은 있지만 credstore 는 설계상 host-bound 라 백업이 의미 없음 — 재생성이 정답.
