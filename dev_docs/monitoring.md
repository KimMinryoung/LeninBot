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
| `main-backup` | 24시간 | 3시간 | |
| `writer-backup` | 24시간 | 3시간 | |
| `kg-backup` | 24시간 | 3시간 | |

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

### 3. VM 안의 기존 알림 잡

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
