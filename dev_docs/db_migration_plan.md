# DB 마이그레이션 계획 — Supabase → 자체 호스팅 Postgres

작성: 2026-07-28. 상태: **Phase 2 컷오버·Phase 3 writer 통합 완료, Supabase pause 완료 (2026-07-28). Phase 4 백업 VM 스트리밍 스탠바이 가동 (2026-08-01).** 다음: Phase 5 해지(~2026-08-11), pgBackRest PITR.

## 목표

메인 DB(Supabase 호스팅 Postgres)를 이 VM의 Docker Postgres로 이전하고, writer DB(`leninbot-writer-pg`)와 단일 인스턴스로 통합한다. 백업용 VM 1대를 추가해 스트리밍 스탠바이 + PITR 백업 체계를 구축한다.

## 조사 결과 요약 (2026-07-28 기준)

| 항목 | 값 |
|---|---|
| Supabase 사용 방식 | 순수 Postgres (psycopg2 직결). SDK/Auth/Storage/Realtime/RLS 미사용 |
| DB 크기 | 8,343 MB — 단 `lenin_corpus`가 8,112 MB |
| lenin_corpus 실체 | 17,046행. 힙 504MB + TOAST 2.4GB + **인덱스 5.2GB** (ivfflat 레거시 + HNSW 중복, 112만 쓰기 churn 비대화). 재빌드 시 전체 DB 약 2~3GB 예상 |
| 기타 테이블 | 전부 합쳐 ~250MB (tool_audit_log 64MB가 최대) |
| 필요 확장 | `vector`, `pgcrypto`, `uuid-ossp`, `pg_stat_statements`. `supabase_vault`/`index_advisor`/`hypopg`는 Supabase 부속 — 버림 |
| 접속 경로 | `db.py` 풀 (모든 서비스 공용), `scripts/psql-supabase`, `scripts/translate_db_content.py`(일회성) |
| 암호 관리 | systemd LoadCredentialEncrypted (`db_password.cred`) + `get_secret("DB_PASSWORD")`. **로컬 인스턴스도 동일 암호 재사용 → credstore 변경 불필요** |
| 함정 | `db.py`가 `sslmode="require"` 하드코딩 (`.env`의 `DB_SSL`은 읽지 않음) |
| VM 자원 | RAM 15GB(가용 10GB), 디스크 241GB 여유. 기존 컨테이너: neo4j, redis, writer-pg(5433), frontend |
| 롤 정리 | Supabase 롤(anon/authenticated/service_role 등) grants는 `--no-owner --no-privileges`로 제거하고 `postgres` 단일 롤 사용 |

## 목표 아키텍처

```
메인 VM                              백업 VM (신규, 2vCPU/4GB/50GB)
┌─────────────────────────┐          ┌──────────────────────────┐
│ leninbot-pg (Docker)    │─WAL/복제→│ 스트리밍 스탠바이 (warm)  │
│ pgvector/pgvector:pg17  │          │ pgBackRest 리포지토리     │
│ 127.0.0.1:5434          │          │ (주1 full / 일1 incr)     │
│ DB: leninbot + writer   │          └──────────────────────────┘
└─────────────────────────┘                     │
        │ (일일 dump, 기존 패턴)                 │ (선택: 리포를 R2에)
        ▼                                       ▼
   Cloudflare R2 (오프사이트 3차)
```

- 챗봇 쿼리가 localhost로 붙어 왕복 지연 제거. Supavisor 풀러 불필요 (`DB_POOL_MODE=direct` 유지).
- 자동 failover 없음 — 스탠바이 승격은 수동 (이 규모에서 자동화는 오버엔지니어링).

## 단계별 계획

### Phase 0 — 준비 ✅ 완료 (2026-07-28)

1. **`db.py` sslmode 수정** ✅: 하드코딩된 `"require"` 대신 `DB_SSL` env를 읽도록 변경 (`.env`는 `require`라 현행 무변경). 컷오버 시 `prefer`로 바꾼다.
2. **컨테이너** ✅: `leninbot-pg` (`pgvector/pgvector:pg17`, Debian), `127.0.0.1:5434`, 볼륨 `leninbot_pg_data`, `shared_buffers=2GB`, locale `C.UTF-8`. **`docker-compose.pg.yml`이 아니라 `docker-compose.neo4j.yml`에 병합** — 같은 compose 프로젝트(leninbot)라 orphan 경고를 피하고 `leninbot-neo4j.service` 유닛이 함께 관리. 주의: 그 유닛을 stop하면 이제 메인 DB도 내려간다. `POSTGRES_PASSWORD`는 initdb 때만 credstore 값으로 주입했고 compose 파일에는 없음 (데이터 볼륨이 보유).
3. **초기화** ✅: `leninbot` DB 생성, 확장 설치 — Supabase 레이아웃과 동일하게 `vector`는 `public`, `pgcrypto`/`uuid-ossp`/`pg_stat_statements`는 `extensions` 스키마. `writer` DB는 Phase 3에서 생성.
4. **스크립트 rename** ✅: `psql-supabase` → `psql-main` (+구명 심링크). 10곳의 기존 참조는 심링크로 호환 — 점진 이행.
5. 함정 기록: Docker 기본 브리지에 IPv6가 없어 컨테이너 안에서 Supabase 직결 불가 → 덤프는 `docker run --network host pgvector/pgvector:pg17 pg_dump ...`로 수행 (호스트 pg_dump는 v16이라 v17 서버 덤프 불가).

### Phase 1 — 코퍼스 프리로드 ✅ 완료 (2026-07-28)

1. 스키마 복원 ✅: `pg_dump --schema-only --schema=public --no-owner --no-privileges`. Supabase 잔재 없음. `CREATE SCHEMA public` 라인만 주석 처리 필요했음. 테이블 73/73, 인덱스 162/163 (차이 = 의도적으로 뺀 ivfflat).
2. `lenin_corpus` 데이터 복원 ✅: 17,046행 스트리밍 (파이프, 수 분).
3. 레거시 ivfflat 제외 ✅ — 코드 참조 없음 확인. HNSW만 유지.
4. 검증 ✅: 행수·content 바이트합(42,049,446)·임베딩 수·id 수 완전 일치. 동일 쿼리 벡터 top-5 결과 양쪽 동일.
5. **결과: 로컬 DB 277MB** (Supabase 8,343MB — 96%가 인덱스/TOAST 비대화였음). 참고: db.py 주석 기준 Supabase RTT ~560ms → localhost 전환의 지연 이득 매우 큼.

### Phase 2 — 컷오버 ✅ 완료 (2026-07-28, 다운타임 약 4분)

실행 기록:
- 중지 대상: DB 사용 서비스 6개(telegram, api, a2a-api, email-api, browser, roleplay) + 활성 타이머 8개. `leninbot-neo4j`(pg 컨테이너 포함 Docker 인프라)와 `leninbot-embedding`(DB 미사용)은 유지.
- 서비스 중지 전에 암호를 임시 파일(0600)로 확보 필요 — 중지하면 `/run/credentials/*/db_password` 마운트가 사라짐. 사용 후 즉시 삭제함.
- truncate 시 append-only 가드 존재: `leninbot.audit_log_mutation_approved` + `leninbot.task_tool_log_mutation_approved`를 `SET LOCAL`로 켠 트랜잭션 필요.
- 재적재는 `pg_dump --data-only --schema=public --disable-triggers` 파이프, 74초. 시퀀스 46개 setval 포함.
- 검증: 73개 테이블 행수 diff 완전 일치. db.py 풀 경유 로컬 접속·벡터 검색 top-3 일치. API 4개 health 200. 전 서비스 로그 무경고. 가드 트리거 재활성 확인.
- `.env` 백업: `.env.bak-supabase-cutover` (0600) — 롤백 시 이 파일로 복원 + 서비스 재시작. Supabase 해지 후 삭제할 것.

원래 절차 (참고):

1. leninbot 서비스·타이머 전체 중지.
2. **전 테이블 데이터 재동기화**: 실데이터가 ~300MB뿐이므로 로컬 전 테이블 truncate 후 `pg_dump --data-only --schema=public` 전체를 신선하게 재적재 (프리로드에서 리허설 완료, ~10분). corpus 스테일 걱정 없음 — corpus는 churn 테이블임 (7.5개월간 ins 543k/del 460k).
3. 시퀀스 값 확인 (pg_dump가 포함하지만 명시 검증).
4. `.env` 변경: `DB_HOST=127.0.0.1`, `DB_PORT=5434`, `DB_NAME=leninbot`, `DB_SSL=prefer`.
5. 서비스 기동 → 스모크: 텔레그램 1턴, 웹챗, RAG 검색, tool_audit_log 기록, commulingo 타이머 1회.
6. Supabase 프로젝트 **pause** (데이터 보존, 롤백 대비 2주 유지).

**롤백**: `.env` 원복 + 서비스 재시작. 컷오버 후 로컬에 쌓인 쓰기는 컷오버~롤백 사이 분량만 손실 (짧게 유지).

### Phase 3 — writer DB 통합 ✅ 완료 (2026-07-28, 다운타임 ~1분)

1. leninbot-pg에 `writer` 롤·DB 생성 → writer-pg에서 `pg_dump | psql` 이전. 7테이블 행수 완전 일치 검증.
2. `.env` `WRITER_DB_PORT=5434`로 변경, 서비스 재기동, db.py 경유 접속 확인 (구 컨테이너 중지 상태에서 재검증해 확정).
3. `leninbot-writer-pg` 컨테이너 제거. **볼륨 `leninbot_writer_pg_data`는 2주 보험으로 유지** — Supabase 해지 시점에 함께 삭제할 것. 포트는 5434 유지.
4. `backup_writer_db_to_r2.py`는 env가 아니라 **컨테이너명 하드코딩**이었음 → `leninbot-pg`로 수정, 1회 실행으로 R2 업로드 확인 (73.8MB).

### Phase 4 — 백업 체계 (1단계 ✅ 완료 2026-07-28, VM 단계 대기)

1. **메인 DB + legacy_game 일일 dump→R2** ✅: `scripts/backup_main_db_to_r2.py` + `leninbot-main-backup.timer` (03:40 KST, kg 03:00·writer 03:20와 시차). 2026-07-29 운영 테이블 정리 후 main 덤프는 162.2MB, `legacy_game`은 0.1MB이며 각각 로컬 3일·R2 15일 롤링 보관한다 (2026-08-01 7일→15일 연장, KG는 2일→14일).
   - **restore drill 통과** (임시 DB 복원→17,046행+HNSW 인덱스 확인→드롭). 드릴이 결함 발견: Docker 기본 `/dev/shm` 64MB로는 병렬 HNSW 인덱스 빌드 실패 → compose에 `shm_size: 1g` 추가로 해결. 프리로드 때 안 걸린 이유: 빈 테이블에 인덱스 생성 후 COPY라 대량 빌드가 없었음.
   - **2026-07-29 DRI 재검증 통과**: R2에서 당일 main/writer 객체를 실제 재다운로드해 로컬 사본과 SHA-256·바이트 동일성을 확인한 뒤, 격리된 `pgvector/pgvector:pg17` 컨테이너에 둘 다 복원. main 73테이블·200,692행·162 인덱스·47 시퀀스, writer 7테이블·1,225행·16 인덱스·5 시퀀스를 검증했고 invalid/unready 인덱스와 뒤처진 시퀀스는 0건. `lenin_corpus` 17,046행과 HNSW 벡터 질의, writer 본문 8개·441,491자도 확인. 측정 복원 시간은 main 8.3초, writer 9.3초였으며 운영 Postgres restart 0·관련 API health 200으로 무영향 확인.
   - **2026-07-29 운영 정리 후 3-DB DRI 통과**: 정리 직후 생성·R2 업로드한 최신 백업을 일회용 Postgres 17에 복원했다. main 59테이블·199,422행·37 시퀀스와 corpus 17,046행/HNSW, `legacy_game` 1테이블·415행·1시퀀스·원본 체크섬, writer 7테이블·1,225행·5시퀀스와 본문 441,491자를 모두 검증했다. 복원 시간은 main 8.3초, legacy 0.2초, writer 9.6초였고 전체 `DRILL PASS`.
   - 유닛 파일은 `systemd/`에 추적 (sudoers가 `cp systemd/* /etc/systemd/system/`만 NOPASSWD 허용).
   - 참고: R2에 `main-db-backup-2026-07-20.dump`라는 과거 객체가 있었음 (구 백업 규칙 잔재로 추정) — 롤링 삭제에 걸려 제거됨.
   - **2026-08-01 롤링 삭제 방식 교체**: 세 잡 모두 만료일 키 1개만 지우는 방식이라, 실행이 실패·누락된 날의 객체는 아무도 다시 지우지 않고 영구히 남았다. 실제로 2일 보관인 KG 버킷에 `kg-backup-2026-04-22/04-23/06-30`(349MB)이 살아 있었다. `scripts/r2_retention.py`의 `prune_r2_prefix()`로 prefix 전체를 커트오프 대비 sweep하도록 바꿔 누락분이 다음 성공 실행에서 자동 회수되게 했다. 안전장치 3종: (1) `<prefix>-YYYY-MM-DD<suffix>` 정확 매칭만 대상 — 날짜 없는 키나 `frontend-archives/` 같은 타 prefix는 인식조차 안 됨, (2) 목록 조회 실패 시 아무것도 삭제하지 않고 경고만, (3) `min_keep=2` — 남는 객체가 2개 미만이 되는 sweep은 커트오프 버그로 보고 전면 중단.
2. **백업 VM 셋업** ✅ 완료 (2026-08-01): Hetzner `hel1` 2vCPU/3.8GB/38GB, 호스트명 `ubuntu-4gb-hel1-2`, tailnet `leninbot-standby` = `100.124.58.85`.
   - **같은 로케이션(hel1)이다.** fsn1은 5배 비싸 리전 분리를 포기했다. 리전 분리가 막는 것은 DC 단위 장애라는 꼬리 위험이고 그건 R2 오프사이트가 이미 담당한다. 실제로 잦은 장애(디스크 풀·OOM·커널 패닉·Docker 엉킴·논리적 실수)는 같은 DC의 별도 VM으로도 전부 커버된다. 대신 외부 워치독은 **VM이 아닌 외부 무료 서비스**로 둬야 지리적 독립성이 생긴다 (미구현).
   - **경로는 tailnet 전용.** Tailscale 직결(IPv6, RTT 1~2ms). standby의 Hetzner 방화벽은 **인바운드 규칙 0개**가 정답이다 — Hetzner는 방화벽 미적용이 곧 전면 허용이라, 차단하려면 빈 방화벽을 붙여야 한다.
   - **함정 (2026-08-01 장애)**: `firewall-1`이 메인 VM과 신규 VM에 **공유 적용**돼 있어, 신규 VM 기준으로 규칙을 지우자 메인 VM의 80/443까지 닫혀 cyber-lenin.com이 내려갔다(HTTP 522). Cloudflare 캐시 때문에 몇 분간 정상으로 보였다. 서버별로 방화벽을 분리할 것.
   - **primary 노출**: `leninbot-pg`는 `127.0.0.1:5434`만 열려 있어 스트리밍이 불가능했다. compose `ports`에 `100.122.248.77:5434`(tailnet)를 추가하고 pg 서비스만 재생성했다(다운타임 6.5초). 특정 IP 바인딩은 부팅 시 tailscaled가 늦으면 컨테이너가 아예 못 뜨므로, `leninbot-neo4j.service`에 `After=tailscaled.service`와 주소 등장까지 최대 90초 대기하는 `ExecStartPre`를 넣었다.
   - **인증**: `replicator` 롤(REPLICATION, non-superuser) + `pg_hba.conf`에 `host replication replicator 100.124.58.85/32 scram-sha-256`. Docker 퍼블리싱에도 출발지 IP가 보존돼 `/32`로 좁힐 수 있었다. 비밀번호는 standby의 `/root/pgstandby/pgpass`(0600, uid 999)에 두고 `primary_conninfo`가 `passfile=`로 참조한다 — `postgresql.auto.conf`에 평문을 남기지 않기 위함이다. passfile이 데이터 디렉터리 **밖에** 있는 이유는 `pg_basebackup -D`가 빈 디렉터리를 요구하기 때문이다.
   - **standby 구성**: `/root/pgstandby/docker-compose.yml`, 컨테이너 `leninbot-pg-standby`, 볼륨 `leninbot_standby_pg_data`, primary와 **동일 이미지 digest**(`pgvector/pgvector@sha256:d2ef61f4…`, 17.10). 포트는 공개하지 않는다(부팅 경합 회피). `shared_buffers=1GB`(호스트 3.8GB 기준).
   - **슬롯 폭주 방지**: 슬롯은 스탠바이가 죽어 있는 동안 primary의 WAL을 무한 보관시킨다. `max_slot_wal_keep_size = 8GB`를 `ALTER SYSTEM` + reload로 걸었다. 초과 시 슬롯이 무효화되고 스탠바이는 재시드가 필요하지만 primary는 살아남는다.
   - **남은 것**: 복제 지연 모니터링, `REPLICATION_PASSWORD` credstore 등록, 방화벽 서버별 분리.
   - ⚠️ **스탠바이는 덤프의 대체가 아니다.** 물리 복제는 `DROP TABLE` 같은 논리적 실수를 즉시 따라 복제한다. 그 방어는 일일 덤프와 PITR의 몫이며 3층을 모두 유지해야 한다.
3. **pgBackRest**: 백업 VM을 리포로, WAL 아카이빙 + 주1 full/일1 incr → **PITR** 확보. (대안: 리포를 R2로 직접 — VM 디스크 절약, 복원 속도는 느림.)
4. **Postgres 복구/드릴 스크립트** ✅ (2026-07-29): `scripts/restore_db.py`.
   - 기본 안전 경로: `venv/bin/python scripts/restore_db.py drill` — 최신 로컬 main/legacy/writer 백업을 선택해 `shm_size=1g`인 일회용 Postgres 17 컨테이너를 만들고, TOC 확인→복원→전체 테이블 정확 행수·인덱스·시퀀스·HNSW·legacy 체크섬·writer 소유권/본문 검증 후 컨테이너를 제거한다. `--scope {all,main,legacy,writer}`, `--main-backup`, `--legacy-backup`, `--writer-backup`, `--keep-container` 지원.
   - 실제 복구: 실행 중인 별도 대상 컨테이너에 `restore --target-container <name> --confirm RECREATE_DATABASES`. 선택한 DB를 drop/create하므로 모든 DB client를 먼저 중지해야 하며, 활성 연결이 남아 있으면 거부한다. writer 역할 암호는 기존 `secrets_loader`의 `WRITER_DB_PASSWORD`를 사용하고, 복구 호스트에 `.env`/credential이 없으면 권한 0600인 `--writer-password-file`을 명시한다. main 복원은 `frontend` 로그인 역할과 DB CONNECT·schema USAGE·전체 table/sequence·default privileges도 재구성한다. 새 컨테이너에는 `FRONTEND_DB_PASSWORD` 또는 frontend `.env`의 기존 암호만 담은 0600 `--frontend-password-file`이 필요하다.
   - 운영 컨테이너 `leninbot-pg`는 `--force-production --confirm RECREATE_LENINBOT_PRODUCTION`을 동시에 주지 않으면 거부한다. 스크립트는 복구 전에 archive TOC, Postgres major version(17), 컨테이너 `/dev/shm` 1GiB 이상을 먼저 확인한다.
5. 안정화 후 1의 일일 dump는 유지(3차 방어) 또는 주기 완화.

### 운영 테이블 정리 (2026-07-29)

현재 LeninBot·frontend 런타임 코드, FK, DB 뷰 참조를 교차검증하고 사용자에게 테이블별 승인을 받은 뒤 main DB에서 14개 테이블을 `CASCADE` 없이 한 트랜잭션으로 제거했다.

- 삭제: `aicommunism_saves`, `aicommunism_sessions`, `user_sessions`.
- writer 이전 보험 사본 삭제: `writer_documents_migrated_20260707`, `writer_manuscript_chunks_migrated_20260707`, `writer_manuscript_revisions_migrated_20260707`, `writer_manuscripts_migrated_20260707`, `writer_messages_migrated_20260707`, `writer_projects_migrated_20260707`, `writer_settings_migrated_20260707`. 활성 writer 데이터는 별도 `writer` DB에 있고 백업/복구 검증을 통과했다.
- 미사용 구독 추적 삭제: `subscriptions`, `receipts`, `usage_snapshots`.
- `story_scenes`는 새 `legacy_game` DB로 테이블·415행·시퀀스·인덱스 4개를 이전했다. 이전 전후 내용 체크섬 `1b3bdc9d7dac48fafc1e216ffd1066f0`, ID 범위 1–831, 시퀀스 831이 일치한 뒤 main 원본만 삭제하고, `default_transaction_read_only=on`을 강제해 보관 DB의 쓰기를 차단했다.
- `game_saves`는 승인 범위가 아니므로 main DB에 유지했다. `leninbot_test`·`writer_test`도 변경하지 않았고 운영 백업/DRI scope에는 포함하지 않는다.

### 컷오버 후 장애 기록 (2026-07-28): frontend 전 콘텐츠 미표시

Supabase pause 직후 cyber-lenin.com의 모든 글(일기·포스트·리포트)이 사라짐. 원인 두 겹:

1. **frontend 컨테이너는 자체 DB 커넥션을 가짐** — `/home/grass/frontend/.env`의 `DB_*`가 Supabase 직결이었고, 메인 `.env` 컷오버 범위 밖이라 누락. env는 `docker run` 시점에 박히므로 파일 수정 후 컨테이너 재생성 필요.
2. **RLS 함정**: Supabase 스키마 덤프가 73개 테이블 전부의 `ENABLE ROW LEVEL SECURITY`를 갖고 옴 (정책은 0개 = superuser 외 전부 차단). Supabase에선 postgres 롤이 BYPASSRLS라 증상이 없었고, 로컬에서도 메인 서비스는 postgres(superuser)라 통과 — **비-superuser 롤을 만들자마자 발현**. 전 테이블 `DISABLE ROW LEVEL SECURITY` 적용 (서버사이드 신뢰 모델이라 RLS 무의미).

조치: 전용 `frontend` 롤 생성 (public 스키마 전권한 + default privileges, 암호는 frontend `.env` 기존 값 재사용 — 메인 superuser 암호를 frontend에 복사하지 않음), frontend `.env`를 `DB_HOST=leninbot-pg`/`DB_NAME=leninbot`/`DB_USER=frontend`/`DB_SSL=false`로 전환(백업 `.env.bak-supabase`), 컨테이너 재생성, Redis의 빈 리스트 캐시(`diary:index:*`, `post:index:*`, `report:*list*`) 수동 삭제. 일기·포스트·리포트 퍼블릭 복구 확인.

교훈: 다음 DB 이전 때는 **`pg_stat_activity`의 application_name 전수조사**로 접속 주체를 먼저 확인할 것 — `leninbot-frontend`가 보였다면 사전에 잡았다.

### Phase 5 — 해지

Supabase pause 완료: 2026-07-28 (사용자 실행). pause 후 ~6시간 시점 헬스 스윕 통과 — pg_stat_activity 전원 로컬, 원격 5432/6543 아웃바운드 없음, 서비스 7개 active·재시작 0, 사이트 일기 20/포스트 20/리포트 171 링크 정상 렌더.

2주 병행 관찰 후 (~2026-08-11): Supabase 최종 스냅샷을 R2에 보관 → 프로젝트 삭제. 그때 함께: `.env.bak-supabase-cutover` 삭제, `leninbot_writer_pg_data` 볼륨 삭제, `dev_docs/project_state.md`·`secret_management.md` 갱신.

## 쓰기 가드 + 테스트 DB (2026-07-29)

ad-hoc 스크립트(테스트, 일회성 CLI)가 프로덕션 DB에 실수로 쓰는 것을 막는 가드레일. 배경: 2026-07-28 tool_trace 테스트 스크립트가 프로덕션 `chat_logs`에 테스트 행 3개를 삽입·삭제(id 2660~2662 공백).

- **메커니즘**: `db.py`가 풀 생성 시 서비스 컨텍스트가 아니면 커넥션을 `default_transaction_read_only=on`으로 내림. 헬퍼 우회(`get_conn()` 직접 사용) 경로까지 Postgres 레벨에서 차단됨. 차단 시 `RuntimeError`로 안내 메시지 표출.
- **쓰기 허용 조건** (`_writes_allowed`): ① systemd 서비스(`INVOCATION_ID` 자동 주입, 자식 프로세스 상속) ② `LENINBOT_SERVICE=1` (비-systemd 서비스 컨텍스트 명시용) ③ `LENINBOT_ALLOW_WRITE=1` (승인된 프로덕션 쓰기 opt-in) ④ DB 이름이 `*_test`.
- **테스트 DB**: `leninbot_test`·`writer_test` (같은 leninbot-pg, 스키마 전용 클론). 테스트는 `DB_NAME=leninbot_test` / `WRITER_DB_NAME=writer_test`만 지정하면 됨. 갱신: `scripts/refresh_test_db.sh` (drop→create→schema-only dump 재적재).
- **한계**: 사고 방지용이며 보안 경계 아님 — `docker exec … psql -U postgres` 직접 경로는 막지 않음. `.env`에 플래그를 넣으면 안 됨(`secrets_loader`의 `load_dotenv()`로 ad-hoc도 물려받아 가드 무력화).
- 서비스는 재시작 시점부터 신규 코드 적용 — `INVOCATION_ID` 조건으로 유닛 파일 수정 없이 자동 통과. cron `metrics_collector.py`는 DB 미사용이라 무관.

## 미결 사항 (사용자 결정 필요)

1. **백업 VM**: 어느 프로바이더/스펙으로 발급할지 (권장 2vCPU/4GB/50GB). Phase 4 전까지만 결정하면 됨 — Phase 0~3은 블로킹 없음.
2. **컷오버 시점**: 트래픽 적은 시간대 선호.
3. pgBackRest 리포 위치: 백업 VM 디스크 vs R2 직접.

## 변경 이력

- 2026-07-29: 미사용 운영 테이블 13개 삭제, `story_scenes`를 `legacy_game` DB로 분리, main/legacy/writer 3-DB 백업·복구 드릴 통과.
- 2026-07-28: 최초 작성 (조사 + 계획).
- 2026-07-28 (심링크 이행 완료): `psql-supabase` 호환 심링크 제거 — 참조 스크립트 10곳 전부 `psql-main`으로 전환. `db.py`/`psql-main`의 sslmode 기본값 `require`→`prefer` (로컬 기준; `.env`가 명시 설정하므로 동작 무변경). 코드·툴 설명·봇 자기소개(shared.py)의 Supabase 잔재 문구 정리, `SUPABASE_KEY` env 레퍼런스 제거.
