# PostgreSQL 운영과 복구

2026-09-07 저장소 코드·unit 정의 기준으로 정리했다. 설치된 서비스의 상태나 외부 계정 해지 여부를 새로 확인한 기록은 아니다. 기존 참조를 위해 파일명은 유지한다.

## 구성과 접속

- `docker-compose.neo4j.yml`의 `pg` 서비스가 `leninbot-pg` 컨테이너와 `leninbot_pg_data` 볼륨을 관리한다. 이미지 `pgvector/pgvector:pg17`, `shm_size: 1g`, `shared_buffers=2GB`다.
- `leninbot-neo4j.service`가 PostgreSQL·Neo4j·Redis를 함께 관리하므로 이 unit을 중지하면 메인 DB도 내려간다.
- 호스트 앱은 `127.0.0.1:5434`, 복제는 tailnet `100.122.248.77:5434`를 사용한다. unit은 Tailscale 주소가 나타날 때까지 기다린다.
- 활성 DB는 `leninbot`과 `writer`, 보관 DB는 읽기 전용 `legacy_game`이다. 테스트 DB `leninbot_test`·`writer_test`는 운영 백업 범위에 포함하지 않는다.
- 메인 풀은 `db.py`의 `DB_*` (`DB_SSL` 기본 `prefer`), writer 풀은 `WRITER_DB_*` (`WRITER_DB_SSLMODE` 기본 `disable`)로 분리된다. 시크릿 로딩은 [secret_management.md](secret_management.md)를 따른다.
- 로컬 SQL 진입점은 `scripts/psql-main`이다. 개발 진단은 기본 MCP `inspect`, SQL이 필요하면 `operator`의 `readonly_query_db` 또는 `scripts/query-db`를 사용한다. 구 `psql-supabase` 심링크는 없다.
- frontend는 별도 저장소 `/home/grass/frontend`와 자체 DB 설정을 사용한다. 컨테이너에서는 `leninbot-pg:5432`, DB `leninbot`, 전용 `frontend` 역할을 쓴다. DB 접속 변경 시 backend 설정만 고쳐서는 부족하다.

## 백업과 복구

| 대상 | timer / 스크립트 | 보관 |
|---|---|---|
| main + legacy | `leninbot-main-backup.timer`, 매일 03:40 KST → `scripts/backup_main_db_to_r2.py` | 로컬 3일, R2 15일 |
| writer | `leninbot-writer-backup.timer`, 매일 03:20 KST → `scripts/backup_writer_db_to_r2.py` | 로컬 3일, R2 15일 |

백업은 Postgres custom dump와 archive TOC 검증을 사용한다. `scripts/r2_retention.py`는 날짜 형식이 일치하는 prefix만 정리하고, 목록 조회 실패 또는 최소 보관 수 미달이면 삭제하지 않는다. KG 백업은 [knowledge_graph_design.md](knowledge_graph_design.md)를 따른다.

복구 검증은 프로젝트 venv로 실행한다:

```bash
venv/bin/python scripts/restore_db.py drill
```

기본 드릴은 최신 로컬 main/legacy/writer 백업을 일회용 Postgres 17 컨테이너에 복원하고 테이블 행수·인덱스·시퀀스·HNSW·legacy 체크섬·writer 소유권/본문을 검증한다. `--scope {all,main,legacy,writer}`, 개별 `--*-backup`, `--keep-container`를 지원한다. 로컬 사본 드릴만으로 R2 다운로드 경로까지 검증했다고 간주하지 않는다.

실제 복구는 `restore --target-container <name> --confirm RECREATE_DATABASES`로 선택 DB를 drop/create한다. 활성 DB client가 남아 있으면 거부한다. 운영 컨테이너에는 추가로 `--force-production --confirm RECREATE_LENINBOT_PRODUCTION`이 필요하다. `/dev/shm` 1GiB와 Postgres major version 17을 검사한다.

writer/frontend 역할 복구에는 기존 credential 또는 암호만 담은 0600 `--writer-password-file`/`--frontend-password-file`이 필요하다. frontend의 CONNECT·schema USAGE·table/sequence·default privileges도 복구 대상이다. 서비스를 중지하면 `/run/credentials/<unit>/` 마운트가 사라지므로 복구에 필요한 credential의 가용성을 먼저 확인한다.

## 스트리밍 스탠바이

기존 운영 기록상 `leninbot-standby` (`100.124.58.85`, Hetzner hel1)는 `/root/pgstandby/`의 `leninbot-pg-standby`로 물리 복제한다. 자동 failover는 없다. 승격·재시드·복제 진단은 [standby_operations.md](standby_operations.md), 알림은 [monitoring.md](monitoring.md)가 소유한다.

- primary/standby는 동일 Postgres 이미지 digest를 사용한다. 별도 VM이지만 같은 DC이므로 R2 오프사이트 백업을 유지한다.
- `max_slot_wal_keep_size=8GB`로 primary의 WAL 무한 증가를 제한한다. 슬롯 무효화 시 재시드가 필요하다.
- 복제 인증은 `replicator` 역할, standby IP `/32`, 데이터 디렉터리 밖의 0600 passfile을 사용한다. Tailscale SNAT로 출발지 IP가 바뀌면 인증이 실패할 수 있다.
- 기존 `pg_hba` 운영 설정은 Docker `172.16.0.0/12`, tailnet `100.64.0.0/10`, replication `100.124.58.85/32`다. 호스트 접속은 Docker 브리지 주소로 보일 수 있다. 규칙 변경은 신규 연결로 검증한다.
- 방화벽은 서버별로 분리한다. 공유 방화벽 수정은 primary에도 적용된다. standby의 공용 인바운드를 차단하려면 빈 규칙의 방화벽을 붙이는 구성을 유지한다.
- 물리 복제는 논리적 삭제도 전달하므로 일일 dump의 대체가 아니다. pgBackRest/WAL 아카이빙 기반 PITR는 아직 도입 완료를 확인할 근거가 없다.

## 쓰기 가드와 테스트 DB

`db.py`는 ad-hoc 연결을 `default_transaction_read_only=on`으로 연다. `_writes_allowed`는 `INVOCATION_ID`, `LENINBOT_SERVICE=1`, 명시적 `LENINBOT_ALLOW_WRITE=1`, 또는 DB 이름 `*_test`에서 쓰기를 허용한다. 이 플래그는 사고 방지용이며 직접 psql을 막는 보안 경계가 아니다. `.env`에 쓰기 opt-in을 상시 넣지 않는다.

테스트는 `DB_NAME=leninbot_test` / `WRITER_DB_NAME=writer_test`를 사용한다. `scripts/refresh_test_db.sh`는 테스트 DB를 drop/create하고 스키마만 다시 적재한다. 서비스 startup DDL은 없으며 스키마 변경은 `scripts/schema_migrations.py`로 적용한다.

## 이전 완료 사항과 미확인 항목

Supabase → 로컬 main/writer 통합과 Supabase pause는 2026-07-28 완료 기록이 있다. `story_scenes`는 `legacy_game`으로 분리했고 구 writer 복제 테이블은 제거했다. 과거 행수·크기·성능 측정은 현재 상태로 사용하지 않는다.

Supabase 최종 해지, 최종 스냅샷 R2 보관, `.env.bak-supabase-cutover`와 구 `leninbot_writer_pg_data` 보험 볼륨 삭제 여부는 이번 코드 점검으로 확인할 수 없다. 과거 8월 11일 예정일은 현재 일정이 아니다. 외부 계정·백업·볼륨을 확인한 뒤 별도 작업으로 처리한다.

다음 이전에서는 `pg_stat_activity.application_name`으로 frontend를 포함한 접속 주체를 확인한다. 외부 덤프의 RLS·역할·권한을 비-superuser로 검증하고, 컨테이너 환경 변경 후에는 재생성이 필요한지 확인한다.
