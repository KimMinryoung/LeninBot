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
| 퍼블리시 포트 | `100.124.58.85:5434` (tailnet 전용) |

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
- **승격은 되돌리기 어렵다.** 자세한 것은 아래 런북.
- **메인 VM이 통째로 죽으면 승격만으로 사이트가 살아나지 않는다.** 앱 계층이 전부 거기 있다. 아래 런북의 시나리오 B 참고.

## 승격 런북

> 자동 failover는 없다. 이 규모에서 자동화는 스플릿브레인 위험만 늘린다.
>
> **승격은 최후 수단이다.** 재시작은 되돌릴 수 있고 승격은 되돌릴 수 없다. 아래 분류를 건너뛰고 바로 승격하지 말 것.

### 0. 어떤 알림이 왔나

| 텔레그램 | 의미 |
|---|---|
| 🔴 `cyber-lenin.com 응답 이상` (5분 내) | 사이트가 안 보인다. 원인은 아직 모름 |
| 🔴 `... 콘텐츠 이상 — DB 유래 링크 0개` | 응답은 오는데 DB에서 글이 안 나온다 |
| 🔴 `복제 상태 점검` 실패 (15분 주기) | DB 또는 스탠바이 문제 |
| 🔴 `복제 상태 점검 무소식 N분` (1시간 내) | **메인 VM이 죽었다** — 데드맨 스위치 |

### 1. 분류: VM이 살아 있나

```bash
ssh grass@100.122.248.77 'uptime'      # tailnet 경유
```

붙으면 **A**, 안 붙으면 **B**.

### 2. (A) 정말 DB 문제인지 먼저 확인

```bash
docker exec leninbot-pg psql -U postgres -c 'SELECT 1;'
```

**응답이 오면 DB 문제가 아니다. 승격하지 말 것.** 2026-08-01 장애가 그랬듯 방화벽·nginx·frontend 쪽일 가능성이 크다.

```bash
docker ps
systemctl status nginx
curl -sI -k -H "Host: cyber-lenin.com" https://127.0.0.1/    # origin 자체는 사나
```

origin이 200인데 밖에서 안 보이면 **Hetzner 방화벽**을 본다 (인바운드 80/443, Cloudflare 대역).

### 3. (A) DB가 응답 없으면 — 재시작이 먼저

```bash
docker logs --tail 50 leninbot-pg
docker restart leninbot-pg
```

컨테이너가 엉킨 정도면 여기서 끝난다. 되돌릴 수 있는 조치를 먼저 소진한다.

### 4. (A) 그래도 안 살아나면 — 승격 vs 복원

여기서 갈린다. **둘 다 정답일 수 있다.**

| | 승격 | 백업 복원 (`restore_db.py`) |
|---|---|---|
| 데이터 손실 | **0** | 최대 24시간 (03:40 KST 이후분) |
| 소요 | 몇 분 | 10분 내외 |
| 이후 상태 | **스탠바이가 없어진다.** 복제 구성 재구축 필요 | 구성 그대로 유지 |

오늘치 쓰기가 아까우면 승격, 어제 상태로 충분하면 복원이다. 참고로 2026-08-01 기준 하루 쓰기는 대화 0건·원고 0건이고 대부분 `tool_audit_log`와 재생성 가능한 큐레이션이었다 — **그런 날에는 복원이 덜 번거롭다.** 승격은 그 대가로 복제 구성을 통째로 다시 세우게 만든다.

### 5. (A) 승격을 택했다면

```bash
scripts/promote_standby.sh --dry-run                      # 계획 확인
docker stop leninbot-pg                                   # 스플릿브레인 방지
scripts/promote_standby.sh --confirm=PROMOTE_STANDBY
```

두 번째 줄을 건너뛰면 스크립트가 거부한다. primary가 반쯤 살아 쓰기를 받는 상태로 승격하면 데이터가 두 갈래로 갈라지고 합칠 방법이 없다.

승격 직후 반드시:

```bash
sudo systemctl stop leninbot-replication-health.timer      # 스탠바이가 없으니 계속 알림이 온다
```

### 6. (B) 메인 VM이 통째로 죽었다

**승격만으로는 사이트가 살아나지 않는다.** nginx·frontend·API·neo4j·임베딩 서버가 전부 그 VM에 있고, 스탠바이는 RAM 3.8 GB에 그중 아무것도 깔려 있지 않다.

1. Hetzner Console에서 VM 상태 확인 — 재부팅으로 살아나면 그게 제일 빠르다
2. 안 되면 새 VM 발급 → Docker·Tailscale 설치 → 저장소 클론
3. 데이터는 스탠바이에 뜨겁게 살아 있다. 승격해 새 호스트가 붙게 하거나, 스탠바이에서 덤프를 떠 옮긴다. **R2 복원을 기다릴 필요가 없다는 것이 스탠바이의 실제 값어치다**
4. Cloudflare DNS를 새 origin IP로, 방화벽에 80/443(Cloudflare 대역) 규칙

몇 시간짜리 작업이다. **스탠바이가 줄이는 것은 복구 시간이 아니라 데이터 손실이다.**

> ⚠️ **사전에 해둘 것**: B에서 SSH가 안 되면 Hetzner Console이 유일한 진입로인데, 메인 VM은 SSH 키로 생성해 **root 비밀번호가 없어 콘솔 로그인이 안 된다**(22번도 방화벽으로 닫혀 있다). Console → Rescue → Reset root password를 **장애 전에** 해둘 것. 정작 필요할 때는 이 조작조차 못 할 수 있다.

### A 시나리오 상세: `scripts/promote_standby.sh`

```bash
scripts/promote_standby.sh --dry-run                      # 무엇이 바뀌는지 먼저 본다
scripts/promote_standby.sh --confirm=PROMOTE_STANDBY      # 실행
```

스크립트가 하는 일: 사전 점검 → `.env` 2개 백업 → `pg_promote()` → 메인 `.env`와 frontend `.env`를 스탠바이 주소로 → **frontend 컨테이너 재생성** → 서비스 재시작 → 사후 체크리스트 출력.

안전장치:

- **스플릿브레인 차단** — primary가 아직 쓰기를 받고 있으면 거부한다. 양쪽이 각자 쓰기를 받으면 데이터가 갈라지고 합칠 방법이 없다. 넘길 것이면 `docker stop leninbot-pg`로 먼저 확실히 정지시킨다.
- **`--confirm=PROMOTE_STANDBY`** 없이는 실행되지 않는다 (`restore_db.py`와 같은 패턴).
- `--dry-run`은 primary가 살아 있어도 계획 전체를 보여준다. 차분할 때 읽어두는 용도다.
- 스탠바이가 복구 모드가 아니면 (이미 승격됐거나 이상 상태) 거부한다.

**frontend를 재생성하는 이유**: 자체 `.env`를 갖고 있고 컨테이너명(`leninbot-pg`)으로 DB에 붙는다. env는 `docker run` 시점에 박히므로 **restart로는 안 바뀐다.** 2026-07-28에 정확히 이걸 놓쳐 사이트 콘텐츠가 전부 사라졌다. 재생성 명령은 그림자 컨테이너로 검증해 둔 것이다(본문 바이트수까지 프로덕션과 일치 확인).

### 승격은 되돌리기 어렵다

승격하면 타임라인이 갈라진다. 옛 primary는 재기동만으로는 복제에 합류하지 못하고 `pg_rewind`나 새 base backup이 필요하다. **훈련 삼아 승격하지 말 것.**

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
