# leninbot DB 계정 전환: postgres → leninbot_app

2026-10-05 준비(frontend `dev_docs/commulingo-admin-mcp.md` 5단계, `commulingo-agent-pipeline.md` W6). leninbot 서비스는 지금 `postgres` 슈퍼유저로 접속한다. 이 상태에서는 테이블 권한이 아무 의미가 없으므로, leninbot 전용 로그인 `leninbot_app`으로 바꾸고 CommuLingo(frontend 소유) 테이블에는 권한을 주지 않는다.

## 설계

| 대상 | 처리 |
|---|---|
| leninbot 테이블 37개 + 출처 캐시 3개(`commulingo_pipeline_{sources,fetch_cache,job_sources}`) | 소유자를 `leninbot_app`으로 |
| leninbot이 실행 중 DDL(`ensure_*`)을 거는 공용 발행 테이블 `research_documents`, `chat_logs`, `hub_curations`, `static_pages` | 소유자를 `leninbot_app`으로(frontend 권한은 그대로 남음) |
| leninbot 스키마 코드가 정의하는 함수 `prevent_telegram_task_tool_log_loss`, `prevent_tool_audit_log_mutation` | 소유자를 `leninbot_app`으로 |
| frontend가 소유하고 leninbot도 읽고 쓰는 `posts`, `ai_diary`, `users`, `user_passkeys`, `user_fingerprints` | DML 권한만(소유자 postgres 유지) |
| CommuLingo 테이블과 frontend 전용 테이블(`site_menu_views` 등) | 권한 없음 |
| 스키마 | `public` USAGE·CREATE(앞으로 leninbot이 만드는 테이블은 `leninbot_app` 소유), `extensions` USAGE |
| 기본 권한 | `leninbot_app`이 만드는 테이블·시퀀스에 frontend가 지금처럼 접근(postgres 기본 권한과 같게) |

- 소유자가 바뀌어도 기존 권한(`frontend`, `leninbot_audit`, `leninbot_ro`)은 남는다. frontend의 `scripts/apply-migration`과 백업(`pg_dump`), 복구 스크립트는 컨테이너 안 `postgres`로 돌기 때문에 영향이 없다.
- 슈퍼유저가 필요한 관리 작업은 `docker exec leninbot-pg psql -U postgres`로 한다. 예: `schema_migrations.py --only audit-sink-role`(이제 기본 실행에서 빠진다), `scripts/setup_readonly_db_role.sh`, `scripts/install_audit_sink_role.sh`, role 생성. 전환 뒤 서비스 credential `db_password`는 `leninbot_app` 비밀번호다.
- 2026-10-05 운영 스키마·role을 복원한 격리 DB에서 확인한 것:
  - SQL 두 번 적용(멱등). 결과는 `leninbot_app` 42개, `postgres` 55개.
  - `check_app_role.py` 통과.
  - `leninbot_app`으로 `schema_migrations.py` 전체 블록 성공(`writer-tables`는 별도 DB라 제외).
  - frontend·감사 role 권한 유지.

## 순서

`(root)` 표시가 있는 단계만 root 비밀번호가 필요하다. 나머지는 grass로 실행한다.

### 1. 비밀번호와 role (서비스 영향 없음)

서비스는 아직 postgres로 접속하고, 슈퍼유저는 소유자와 상관없이 동작한다. 테이블마다 짧은 잠금을 잡으며(`lock_timeout 5s`), 실패하면 다시 실행한다.

```bash
cd /home/grass/leninbot
umask 077; openssl rand -hex 24 > ~/.config/leninbot/app_db_password
{ printf "\\set app_password '%s'\n" "$(cat ~/.config/leninbot/app_db_password)"; cat scripts/db/leninbot_app_role.sql; } \
  | docker exec -i leninbot-pg psql -U postgres -d leninbot -v ON_ERROR_STOP=1
```

비밀번호는 명령행이 아니라 stdin으로 들어간다. 마지막 출력은 `leninbot_app 42`, `postgres 55`여야 한다.

### 2. 새 계정으로 사전 확인 (쓰기 없음)

```bash
APP_DB_PASSWORD="$(cat ~/.config/leninbot/app_db_password)" venv/bin/python scripts/check_app_role.py
```

`FAIL` 줄이 없어야 한다.

### 3. 전환 (root)

```bash
cd /home/grass/leninbot
sudo cp -p /etc/credstore.encrypted/db_password.cred /etc/credstore.encrypted/db_password.cred.pre-app-role
sed -i 's/^DB_USER=postgres$/DB_USER=leninbot_app/' .env && grep '^DB_USER=' .env
sudo venv/bin/python scripts/manage_secrets.py rotate DB_PASSWORD < ~/.config/leninbot/app_db_password
sudo systemctl restart leninbot-api leninbot-telegram leninbot-worker leninbot-a2a-api leninbot-roleplay \
  leninbot-browser leninbot-email-api leninbot-llm-proxy novel-writer-api
```

- `rotate -r`는 drop-in(`*.service.d/*.conf`)으로 credential을 마운트하는 서비스만 찾는다. 그래서 `-r` 대신, `db_password`를 쓰는 상시 서비스 9개를 직접 재시작한다. 본 unit 파일에 credential을 둔 `leninbot-worker`, `leninbot-llm-proxy`도 여기에 포함된다.
- oneshot 서비스(kg-sync, kg-integrity, kg-report, email-poller, experience, autonomous, research-document-translation)는 다음 실행부터 새 계정을 쓴다.
- `.env` 변경과 재시작 사이에 시작된 oneshot 실행은 한 번 실패할 수 있으니, 명령을 이어서 실행한다.

### 4. 확인

```bash
systemctl is-active leninbot-api leninbot-telegram leninbot-worker leninbot-a2a-api leninbot-roleplay leninbot-browser leninbot-email-api leninbot-llm-proxy novel-writer-api
journalctl --since "-15 min" -u 'leninbot-*' -u novel-writer-api | grep -iE "permission denied|InsufficientPrivilege|password authentication failed"
docker logs leninbot-pg --since 15m 2>&1 | grep -iE "permission denied|authentication failed"
```

그다음 대표 경로를 하나씩 확인한다:
- 웹 채팅 한 번
- 텔레그램 `/status`
- 일꾼 작업 하나
- `sudo systemctl start leninbot-kg-sync` 대신 `venv/bin/python -m jobs.kg_sync --source commulingo --dry-run`(쓰기 없음; systemd 밖에서는 읽기 전용 계정을 쓰므로, 확인은 서비스 로그로 한다)

문제가 없으면 비밀번호 파일을 지운다:

```bash
shred -u ~/.config/leninbot/app_db_password
```

### 되돌리기 (root)

```bash
sed -i 's/^DB_USER=leninbot_app$/DB_USER=postgres/' /home/grass/leninbot/.env
sudo cp -p /etc/credstore.encrypted/db_password.cred.pre-app-role /etc/credstore.encrypted/db_password.cred
sudo systemctl restart leninbot-api leninbot-telegram leninbot-worker leninbot-a2a-api leninbot-roleplay \
  leninbot-browser leninbot-email-api leninbot-llm-proxy novel-writer-api
```

소유권 변경은 되돌릴 필요가 없다. postgres로 접속하는 동안에는 소유자가 의미 없다.

## 이후

- leninbot 새 테이블은 `leninbot_app` 소유로 생긴다. frontend가 읽어야 하면 기본 권한으로 이미 열려 있다.
- frontend가 새로 만드는 테이블은 leninbot에 보이지 않는다. leninbot이 공용으로 써야 하는 테이블이면 그 frontend 마이그레이션에 명시적으로 `GRANT … TO leninbot_app`을 넣는다.
- 반대 방향의 분리(frontend가 leninbot 테이블 전체에 쓰기 권한을 가진 기본 권한)는 별도 과제다.
