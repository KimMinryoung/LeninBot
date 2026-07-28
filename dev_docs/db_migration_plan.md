# DB 마이그레이션 계획 — Supabase → 자체 호스팅 Postgres

작성: 2026-07-28. 상태: **Phase 2 컷오버 완료 (2026-07-28) — 프로덕션이 로컬 leninbot-pg에서 가동 중.** 다음: Supabase pause(사용자), Phase 3 writer 통합, Phase 4 백업 VM.

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

### Phase 4 — 백업 체계 (백업 VM 발급 후)

1. **즉시 (VM 발급 전)**: 메인 DB 일일 dump→R2 타이머 추가 (기존 kg-backup/writer-backup 패턴 복제). 가변 테이블 일 1회 + `lenin_corpus` 월 1회 분리로 용량 절약.
2. **백업 VM 셋업**: WireGuard(또는 Tailscale)로 사설 터널 → 스트리밍 레플리카 구성 (physical replication slot, `wal_keep_size` 설정).
3. **pgBackRest**: 백업 VM을 리포로, WAL 아카이빙 + 주1 full/일1 incr → **PITR** 확보. (대안: 리포를 R2로 직접 — VM 디스크 절약, 복원 속도는 느림.)
4. restore drill 스크립트 작성 (restore_kg.py 패턴).
5. 안정화 후 1의 일일 dump는 유지(3차 방어) 또는 주기 완화.

### Phase 5 — 해지

2주 병행 관찰 후: Supabase 최종 스냅샷을 R2에 보관 → 프로젝트 삭제. `dev_docs/project_state.md`·`secret_management.md` 갱신.

## 미결 사항 (사용자 결정 필요)

1. **백업 VM**: 어느 프로바이더/스펙으로 발급할지 (권장 2vCPU/4GB/50GB). Phase 4 전까지만 결정하면 됨 — Phase 0~3은 블로킹 없음.
2. **컷오버 시점**: 트래픽 적은 시간대 선호.
3. pgBackRest 리포 위치: 백업 VM 디스크 vs R2 직접.

## 변경 이력

- 2026-07-28: 최초 작성 (조사 + 계획).
