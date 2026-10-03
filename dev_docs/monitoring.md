# 감시와 알림

## 원칙

모든 알림은 결국 사람에게 도달해야 하고, **감시자는 감시 대상 밖에 있어야 한다.**

2026-08-01 이전까지 모든 알림 잡이 메인 VM에서 돌며 텔레그램으로 보고했다. 그래서 알릴 수 없는 단 하나의 사건이 **VM 자신의 죽음**이었고, 실제로 그날 방화벽 변경으로 cyber-lenin.com이 몇 분간 HTTP 522를 뱉는 동안 아무 알림도 없었다. Cloudflare 캐시 때문에 브라우저에서는 정상으로 보여 발견이 더 늦었다. 그 사각지대를 닫으려고 워치독을 VM 밖(Cloudflare Workers)에 두었다.

## 알림 채널

전부 같은 곳으로 간다 — 텔레그램 봇 **`@LeninBichonBot`**, 개인 대화방 `chat_id=5804296818`.

- VM 안의 잡: `TELEGRAM_BOT_TOKEN`(credstore) + `TELEGRAM_CHAT_ID`(`.env`)
- 워치독 Worker: 같은 값을 `wrangler secret`으로 보관

## 계층

### 1. 외부 워치독 — Cloudflare Worker (VM 밖)

`watchdog/` · `https://leninbot-watchdog.minryoung93.workers.dev` · cron 5분

**사이트 감시.** `cyber-lenin.com`을 캐시 우회(`?watchdog=<ts>` + `cache: no-store`)로 fetch한다. 캐시된 사본을 보면 2026-08-01의 사각지대를 그대로 재현하게 되므로 이 우회는 필수다.

- 2xx가 아니면 이상 (522·502·404는 물론 **3xx도 이상** — 이 사이트는 200을 직접 주므로 리다이렉트 자체가 사람이 볼 변화다)
- 요청 실패(20초 타임아웃, DNS·연결 오류)도 이상
- **200이어도 콘텐츠를 검사한다.** 2026-07-28에 frontend가 옛 DB를 보면서 200을 반환하는데 글은 전부 사라진 적이 있다. 상태코드만 보는 감시는 그것을 정상으로 판정한다.

  | 단언 | 기준 | 현재 |
  |---|---|---|
  | 본문 길이 | ≥ 5,000 bytes | ~33,700 |
  | 타이틀 | `<title>`에 `Cyber-Lenin` | 있음 |
  | DB 유래 링크 | `/reports`·`/commulingo` ≥ 3개 | 5개 |

  기준은 느슨하게 잡아 홈 개편으로 오탐이 나지 않게 했고, 실패 시 어느 단언이 몇 개에서 깨졌는지 수치까지 메시지에 넣어 **사이트가 깨진 것인지 검사가 낡은 것인지** 즉시 구분되게 했다.

**데드맨 스위치.** systemd 잡이 성공하면 `${WATCHDOG_PING_BASE}/<job>`로 핑을 보내고(유닛의 `ExecStartPost`), cron이 밀린 핑을 알린다. **VM이 통째로 죽으면 핑이 끊겨 이쪽이 잡는다.**

| job | 기대 주기 | 유예 | 비고 |
|---|---|---|---|
| `replication-health` | 15분 | 45분 | 사실상 VM 하트비트 |
| `main-backup` | 24시간 | 3시간 | 2026-09-23부터 standby에서 핑 |
| `writer-backup` | 24시간 | 3시간 | standby에서 핑 |
| `kg-backup` | 24시간 | 3시간 | main VM (Neo4j가 main에만 있음) |
| `restore-drill` | 7일 | 6시간 | standby 주간 R2 복원 드릴 |

설계상 알아둘 것:

- **상태 전이 시에만 알린다** (정상→이상 1회, 이상→정상 1회). 도배를 막고 KV 무료 쓰기 한도에도 여유가 크다.
- **핑 이력이 없는 잡은 밀린 것으로 보지 않는다.** 그렇지 않으면 유닛 배선 전에 배포하는 순간 전부 울린다.
- `ExecStartPost`는 `-` 접두사를 붙였다. **워치독이 죽어도 백업이 실패로 표시되지 않는다.**
- `scheduled`는 `ctx.waitUntil`이 아니라 직접 `await`한다. 예외가 cron 실패로 대시보드에 잡히게 하기 위함이다.

상태 조회:
```bash
curl -s "https://leninbot-watchdog.minryoung93.workers.dev/status/$(cat .watchdog_ping_token)" | venv/bin/python -m json.tool
```

배포는 `scripts/deploy_watchdog.sh` (자세한 것은 `watchdog/README.md`). 배포용 Cloudflare API 토큰은 상시 보관하지 않는다 — 필요할 때 발급하고 끝나면 폐기한다.

### 2. 복제 상태 — `leninbot-replication-health.timer` (15분)

`scripts/check_replication_health.py`. 모든 값을 primary에서만 읽는다(`pg_stat_replication`이 스탠바이가 보고한 LSN을 담고 있어 한쪽 시점으로 충분하고, 스탠바이용 자격증명이 필요 없다).

네 가지를 본다 — 바이트 지연, 시간 지연, walreceiver 연결, **슬롯 `wal_status`**. 마지막이 가장 중요하다: `lost`/`unreserved`면 `max_slot_wal_keep_size`(8 GB)를 초과해 슬롯이 무효화된 것이고, 스탠바이는 재시드가 필요하다(`standby_operations.md`). primary는 무사하며, **이것이 디스크가 차는 대신 일어나도록 설계한 실패다.**

15분 주기인 이유: 스탠바이가 떨어져 나가는 순간부터 슬롯이 primary의 WAL을 붙잡는다. 일일 잡으로는 8 GB 예산을 다 쓴 뒤에 알게 된다.

### 3. 서비스 생존 — `leninbot-service-health.timer` (5분)

`scripts/check_service_health.py`. enabled 상태인 `leninbot-*.service` 전부의 `is-active`와 API `/health`(`API_HEALTH_URL`, 기본 `http://172.17.0.1:8000/health`)를 본다. 문제는 **연속 두 번** 보여야 알리고(배포 재시작은 조용하다), 풀리면 한 번 복구를 알린다. 상태는 `data/service_health_state.json`에 둔다.

2026-09-23 PG 장애 때 `leninbot-neo4j`(Docker DB 스택)의 첫 시작이 실패하자, 이것을 `Requires=`로 건 `leninbot-api`·`leninbot-a2a-api`의 시작 작업이 의존성 실패로 취소됐다. `Restart=always`는 프로세스 종료에만 반응하므로 스택이 12:21에 복구된 뒤에도 두 서비스는 2026-09-25 수동 시작 때까지 죽어 있었고 웹 채팅(`/api/proxy/chat`)이 그동안 실패했다. 외부 워치독은 200을 주던 frontend만 봐서 잡지 못했다. 재발 방지로 두 유닛은 `Wants=`+`After=`로 바꿨고(DB 접근은 지연 연결이라 스택 복구 후 요청이 다시 성공한다), 이 점검을 추가했다. 2026-09-25에 systemd 경유 복구 알림 전송까지 확인했다.

### 4. VM 안의 기존 알림 잡

| 유닛 | 주기 | 내용 |
|---|---|---|
| `leninbot-kg-integrity.timer` | 매시 | KG 무결성 + 검색 스모크 |
| `leninbot-commulingo-health.timer` | 매일 | 큐레이션 레인 헬스 |
| `leninbot-variant-scan.timer` | 매주 | 이름 변형 후보 |
| `leninbot-stale-secrets.timer` | 매주 | 오래된 credential |

이 계층은 **메인 VM이 살아 있어야만 작동한다.** 그래서 1번이 필요하다.

## 아직 못 잡는 것

- **워치독 자체의 죽음.** 상태 전이 시에만 알리므로 "조용함 = 정상"인데, 워치독이 죽어도 조용하다. Cloudflare 대시보드의 cron 실행 이력에서만 보인다. 이걸 닫으려면 워치독을 감시하는 무언가가 또 필요해 수확이 급격히 준다 — 대신 가끔 `/status`를 확인한다.
- **Cloudflare 장애.** 워치독이 Cloudflare 위에 있어 함께 영향을 받는다. origin 감시 용도로는 문제가 아니다(Cloudflare가 죽으면 사이트도 안 보인다).
- **콘텐츠 검사의 프로덕션 실증.** 사이트 다운/복구는 2026-08-01에 실제 알림까지 확인했지만, "200인데 콘텐츠 이상"은 로컬 테스트(27/27)로만 검증했다. 프로덕션 재현은 DB를 실제로 끊어야 해서 하지 않았다.
- **`archive_mode` 관련 감시 없음.** pgBackRest PITR을 도입하면 `archive_command` 실패로 WAL이 쌓여 디스크가 찰 수 있다. 그때 `check_replication_health.py`에 `pg_stat_archiver` 검사를 추가해야 한다.

## 테스트 방법

알림 체계는 **실제로 울려봐야** 검증된 것이다. 정상일 때 조용한 것은 아무것도 증명하지 않는다.

- **데드맨 스위치**: KV에 낡은 핑을 주입 → cron이 감지 → 알림 → 키 삭제 → 복구 알림.
  ```bash
  cd watchdog && npx --yes wrangler@3 kv:key put --namespace-id=<id> "ping:main-backup" "<30시간 전 epoch ms>"
  ```
- **사이트 감시**: `SITE_URL`을 404가 나는 경로로 임시 변경 후 배포 → cron 대기 → 알림 확인 → 원복 배포. 실제 사이트는 건드리지 않는다.
- **복제 점검**: 스탠바이 컨테이너를 잠깐 중지하면 `exit 1`과 함께 슬롯 inactive·walreceiver 부재를 보고한다.

셋 다 2026-08-01에 프로덕션에서 통과했다.

## 번역 배치 실패 상태

저장소의 `research-document-translation.service`는 `scripts/run_translation_batch.py`로 두 DB 번역 작업을 실행하고 하나라도 실패하면 exit 1을 반환한다. 보류된 검증 실패 행도 실패 상태에 포함하므로 호출이 없다는 이유로 정상처럼 보이지 않는다. 48시간 보류 정보는 `output/translation_failures/`에, 실행 결과는 journal에 있다. provider·통신 오류는 다음 정기 실행에서 다시 시도하며 무기한 보류는 없다. 2026-09-07 새 유닛 설치·daemon-reload 후 `ignore_errors=no`를 확인했다. 실제 번역 배치는 수동 실행하지 않았으며, 외부 워치독/텔레그램 알림 연결은 추가하지 않았다.


## 서버 자원 진단

2026-10-03 04:48–04:52 UTC 운영 서버 관찰: 8 vCPU, 메모리 15.24 GiB,
사용 약 6.6 GiB·available 약 8.6 GiB, swap 약 752 MiB, 루트 디스크 301 GB 중
138 GB 사용(48%). `sar`의 10월 2일 CPU 평균 사용은 약 4.0%, 3일 04:50까지는
약 5.6%였다. 짧은 `vmstat` 관찰에서 swap-in/out은 0이었다. 낮은 free만으로
메모리 부족을 판단하지 말고 available·swap 입출력·CPU iowait를 함께 본다.
이 수치는 당시 관찰이며 현재 용량/부하 보장이 아니다.

| 프로세스/컨테이너 | 관찰 메모리 | 비고 |
|---|---:|---|
| embedding | PSS 약 2,347 MiB | CPU BGE-M3, reranker 미로딩 |
| API | PSS 약 601 MiB | Chromium/Playwright 자식 포함 |
| Telegram | PSS 약 576 MiB | Chromium/Playwright 자식 포함 |
| roleplay | PSS 약 170 MiB | 별도 봇 |
| LLM proxy / web gateway | PSS 약 46 / 41 MiB | 각각 별도 서비스 |
| A2A / email / writer / browser worker | PSS 약 14 / 5 / 4 / 1 MiB | 각각 swap 약 17 / 31 / 30 / 11 MiB |
| Neo4j | Docker working set 약 1,842 MiB | heap 최대 1 GiB, page cache 512 MiB |
| PostgreSQL | Docker working set 약 1,115 MiB | shared_buffers 2 GiB 설정, DB cache hit 높음 |
| frontend / Redis | Docker working set 약 354 / 15 MiB | frontend는 별도 저장소 |

PSS는 shared page를 비례 배분한 실제 상주량이다. Docker stats는 inactive file cache를
뺀 cgroup working set이므로 위 두 지표를 정밀 합산하지 않는다. 서비스의 `MemoryCurrent`는
페이지 회계/스왑 때문에 프로세스 RSS와 다를 수 있다. 15초간 개별 Python 서비스 CPU는
한 코어 기준 0–0.21%였고, Docker 순간 관찰은 PostgreSQL 8.7%, Neo4j 0.87%, Redis 0.68%였다.
짧은 표본은 정기 작업 최대 부하를 대표하지 않는다.

확인 명령:

```bash
uptime
free -h
vmstat 1 6
df -h /
docker stats --no-stream
systemctl show 'leninbot-*.service' novel-writer-api.service -p MainPID -p ControlGroup -p MemoryCurrent -p CPUUsageNSec
sar -u -r -S -W
journalctl --disk-usage
```

root에서 `/sys/fs/cgroup<ControlGroup>/cgroup.procs`(하위 cgroup 포함)의 PID별
`/proc/<pid>/smaps_rollup` PSS/Swap을 합산하면 브라우저 자식까지 포함한다.
CPUUsageNSec 두 표본의 차이를 관찰 초와 1e9로 나누고 100을 곱하면 한 코어 기준 CPU%다.
DB 조회는 `scripts/query-db`의 read-only guard를 사용한다.

최적화: 본문 수집 Chromium은 5분 유휴 뒤 자동 해제한다(`web_research.md`).
임베딩 8→4 CPU thread 실험은 짧은 한·영·러 검색어의 CPU 계산량을 약 45% 줄였지만,
문서 8개(반복 한·영 문단) 묶음은 중앙값 4.99초→11.87초, CPU 시간 39.08초→39.21초였다.
단일 모델에서 각 조건 3회 측정했으며 max vector difference는 검색어 2.61e-7,
문서 0이었다. 문서 처리 저하 때문에 thread 설정은 변경하지 않았다.
journal 약 3 GiB와 모델 cache 약 6.4 GiB는 디스크 여유가 충분하고 로그/재시작에 필요해
삭제하지 않았다. Docker 로그에는 현재 rotation cap이 없지만 총 관찰 로그가 수 MiB였으므로
DB 컨테이너를 재생성하는 변경은 수행하지 않았다.


2026-10-03 04:56 UTC 검증·적용: 전체 unittest 1,361개(23 skip), pytest 66 pass
(7 skip), Python name 검사 573개 파일 통과. 유휴 pool 회귀 테스트 6개와 실제 Chromium의
본문 수집→유휴 종료→재생성 2회도 통과했다(각 종료 후 드라이버 자식 프로세스 0개,
임시 파일에 쿠키 저장 확인). 실제 smoke는 독립 프로세스의 idle을 1초로 줄여 실행했다.
운영 API/Telegram은 기본 300초다. DB/Redis restart guard가 진행 중 작업 없음으로 통과한 뒤
두 서비스를 재시작했고 새 MainPID, API `/health` 정상, Telegram polling 재개를 확인했다.
두 서비스 PSS는 API 600.9→184.7 MiB, Telegram 576.0→295.3 MiB였다.
합계 감소 약 697 MiB에는 재시작으로 해제된 Python 캐시도 포함된다. 변경으로 회수 가능한
기존 Chromium/드라이버 자식만의 PSS는 API 198.4 + Telegram 199.3 = 약 398 MiB였다.
따라서 유휴 자동 종료의 지속 절감량과 재시작 직후의 전체 감소량을 구분한다.
