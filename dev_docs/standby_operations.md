# 스탠바이 운영

`leninbot-standby`(Hetzner hel1, tailnet `100.124.58.85`)에서 도는 물리 스트리밍 레플리카를 **어떻게 쓰는가**에 대한 문서. 어떻게 구축했는지와 그 과정의 함정은 `db_migration_plan.md` Phase 4-2에 있다.

## 구성

| 항목 | 값 |
|---|---|
| 호스트 | `ubuntu-4gb-hel1-2`, 2 vCPU / 3.8 GB / 38 GB |
| 접속 | `ssh root@100.124.58.85` (tailnet 전용, 공인 인바운드 0) |
| 컨테이너 | `leninbot-pg-standby` (primary와 **동일 이미지 digest**, PG 17.10) |
| compose | `/root/pgstandby/docker-compose.yml` |
| 볼륨 | `leninbot_standby_pg_data` |
| passfile | `/root/pgstandby/pgpass` (0600, uid 999) |
| 슬롯 | primary의 `standby_hel1` |
| 퍼블리시 포트 | **없음** — `docker exec`로만 접근 |

```bash
# 조회는 항상 이 형태
ssh root@100.124.58.85 'docker exec leninbot-pg-standby psql -U postgres -d leninbot -c "..."'
```

## 할 수 있는 일

### 1. 하드웨어·호스트 장애 시 서비스 복구 (주 목적)

RPO는 사실상 0이다. 절차는 아래 "승격 런북".

### 2. 무거운 읽기 쿼리 오프로드

`hot_standby=on`이라 읽기 전용 질의가 된다. 통계 집계, 대량 export, 조사성 스캔을 프로덕션 I/O를 건드리지 않고 돌릴 수 있다.

```bash
ssh root@100.124.58.85 'docker exec leninbot-pg-standby psql -U postgres -d leninbot -tAc "
  SELECT count(*) FROM tool_audit_log WHERE created_at > now() - interval \"30 days\";"'
```

**단, 오래 걸리는 쿼리는 복제와 충돌한다.** WAL 재생이 그 쿼리가 읽는 행을 지우려 하면 Postgres가 둘 중 하나를 선택한다 — 기본값(`max_standby_streaming_delay=30s`)에서는 30초까지 재생을 미루고, 그래도 안 되면 **쿼리를 취소한다**. 몇 분짜리 분석을 돌리려면 그 값을 늘려야 하고, 늘린 만큼 복제 지연이 커진다. 2 vCPU / 3.8 GB라 애초에 무거운 분석에는 맞지 않는다.

### 3. 백업을 primary 대신 여기서 뜨기

현재 일일 덤프 3종은 전부 primary에서 뜬다. 스탠바이에서 뜨면 primary의 I/O와 잠금을 아예 건드리지 않는다. 지금 규모(main 덤프 8초)에서는 이득이 작아 옮기지 않았다.

### 4. 조용한 손상 탐지

primary와 스탠바이의 행수·체크섬을 비교하면 논리적 이상을 잡을 수 있다. 물리 복제는 바이트 단위 사본이므로 **정상이라면 반드시 일치해야 한다.**

```bash
# 양쪽에서 같은 질의를 돌려 비교
docker exec leninbot-pg psql -U postgres -d leninbot -tAc "SELECT count(*) FROM lenin_corpus;"
ssh root@100.124.58.85 'docker exec leninbot-pg-standby psql -U postgres -d leninbot -tAc "SELECT count(*) FROM lenin_corpus;"'
```

### 5. 복구 드릴 호스트

`scripts/restore_db.py drill`을 여기서 돌리면 R2 백업 검증을 프로덕션 자원 없이 매일 할 수 있다. 다만 드릴은 HNSW 인덱스를 재빌드하고 `/dev/shm` 1 GiB를 요구해서 **3.8 GB RAM으로는 빠듯하다.** 아직 primary에 남겨 두었다.

## 할 수 없는 일 / 함정

- **쓰기 불가.** 읽기 전용이다. 스테이징으로 쓰려면 스탠바이 자체가 아니라 별도 사본을 떠야 한다.
- **논리적 실수를 막지 못한다.** `DROP TABLE`은 1초 뒤 스탠바이에도 반영된다. 그 방어는 일일 덤프와 (도입한다면) PITR의 몫이다.
- **같은 DC(hel1)다.** DC 단위 장애에는 primary와 함께 죽는다. 그 층은 R2 오프사이트가 담당한다.
- **승격은 되돌리기 어렵다.** 승격하면 타임라인이 갈라져 원래 primary로 복제를 되돌리려면 `pg_rewind`나 재시드가 필요하다. 훈련 삼아 승격하지 말 것.

## 승격 런북 (primary 소실 시)

> 자동 failover는 없다. 이 규모에서 자동화는 스플릿브레인 위험만 늘린다.

**0. 정말 primary가 죽었는지 확인.** 살아 있는 primary와 승격된 스탠바이가 동시에 쓰기를 받으면 데이터가 갈라진다. 확실하지 않으면 primary를 먼저 확실히 정지시킨다.

**1. 스탠바이에 포트를 연다.** 현재 퍼블리시 포트가 없어 앱이 붙을 수 없다. `/root/pgstandby/docker-compose.yml`에 추가:

```yaml
    ports:
      - "100.124.58.85:5434:5432"
```
```bash
cd /root/pgstandby && docker compose up -d
```

**2. 승격.**
```bash
docker exec leninbot-pg-standby psql -U postgres -c "SELECT pg_promote();"
docker exec leninbot-pg-standby psql -U postgres -tAc "SELECT pg_is_in_recovery();"  # f 여야 한다
```

**3. 앱을 새 주소로.** 메인 VM이 살아 있다면 `/home/grass/leninbot/.env`의 `DB_HOST=100.124.58.85`, `WRITER_DB_HOST`도 같이. frontend는 **자체 `.env`를 갖는다**(`/home/grass/frontend/.env`) — 2026-07-28에 이걸 놓쳐 사이트 콘텐츠가 전부 사라진 적이 있다. 컨테이너는 재생성해야 env가 반영된다.

**4. 서비스 재시작.**
```bash
sudo systemctl restart leninbot-api leninbot-telegram leninbot-a2a-api leninbot-email-api leninbot-roleplay novel-writer-api
```

**5. 승격 후 정리.** 이제 스탠바이가 없는 상태다. `max_slot_wal_keep_size`, 슬롯 `standby_hel1`, 백업 타이머, 워치독 핑이 모두 옛 primary를 가리키고 있으니 새 구성에 맞게 다시 세운다.

## 재시드 (슬롯이 무효화됐을 때)

`check_replication_health.py`가 `wal_status=lost`를 보고하면 스탠바이가 `max_slot_wal_keep_size`(8 GB)를 초과해 뒤처진 것이다. primary는 무사하고 스탠바이만 다시 만들면 된다.

```bash
ssh root@100.124.58.85
docker compose -f /root/pgstandby/docker-compose.yml down
docker volume rm leninbot_standby_pg_data && docker volume create leninbot_standby_pg_data

IMG=pgvector/pgvector@sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0
docker run --rm \
  -v leninbot_standby_pg_data:/var/lib/postgresql/data \
  -v /root/pgstandby/pgpass:/var/lib/postgresql/pgpass:ro \
  "$IMG" bash -c '
    chown postgres:postgres /var/lib/postgresql/data && chmod 700 /var/lib/postgresql/data
    su postgres -c "PGPASSFILE=/var/lib/postgresql/pgpass pg_basebackup \
      -h 100.122.248.77 -p 5434 -U replicator \
      -D /var/lib/postgresql/data -Fp -Xs -P -R -S standby_hel1"'

docker compose -f /root/pgstandby/docker-compose.yml up -d
```

슬롯이 이미 있으므로 primary 쪽에서 새로 만들 것은 없다. 완료 후 `venv/bin/python scripts/check_replication_health.py`로 확인한다.
