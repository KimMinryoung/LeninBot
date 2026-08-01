# leninbot-watchdog

VM 바깥(Cloudflare Workers)에서 도는 감시자.

지금 모든 알림은 메인 VM에서 텔레그램으로 나간다. 그래서 **알릴 수 없는 단 하나의 사건이 VM 자신의 죽음**이다. 이 Worker는 그 사각지대를 덮는다.

- **cron (5분)** — `cyber-lenin.com`을 캐시 우회로 fetch. origin이 죽으면 알림.
- **`/ping/<token>/<job>`** — systemd 타이머용 데드맨 스위치. 잡이 성공하면 핑을 보내고, cron이 "핑이 늦었다"를 감지해 알린다. **VM이 통째로 죽으면 핑이 끊겨서** 이쪽이 잡는다.

알림은 상태가 바뀔 때만 나간다(정상→이상 1회, 이상→정상 1회). 5분마다 도배하지 않고, KV 무료 쓰기 한도에도 여유가 크다.

## 감시 대상

| job | 기대 주기 | 유예 | 비고 |
|---|---|---|---|
| `replication-health` | 15분 | 45분 | 사실상 VM 하트비트 |
| `main-backup` | 24시간 | 3시간 | |
| `writer-backup` | 24시간 | 3시간 | |
| `kg-backup` | 24시간 | 3시간 | |

## 배포

```bash
cd watchdog
../scripts/deploy_watchdog.sh
```

스크립트가 KV 네임스페이스 생성 → `wrangler.toml`의 id 치환 → 시크릿 3종 입력 → 배포까지 한다. 필요한 것:

- `CLOUDFLARE_API_TOKEN` — "Edit Cloudflare Workers" 템플릿으로 발급
- 텔레그램 봇 토큰·chat id — 메인 VM credstore와 같은 값
- `PING_TOKEN` — 스크립트가 생성

배포가 끝나면 스크립트가 `.env`에 넣을 `WATCHDOG_PING_BASE` 줄을 출력한다. 그 값을 넣어야 systemd 유닛의 `ExecStartPost` 핑이 동작한다.

## 확인

```bash
curl https://leninbot-watchdog.<subdomain>.workers.dev/status/<PING_TOKEN>
```

## 주의

핑 토큰이 새면 **알림을 억제**당할 수 있다(가짜 핑으로 "살아있다"를 위조). 유출 시 `wrangler secret put PING_TOKEN`으로 교체하고 `.env`의 `WATCHDOG_PING_BASE`도 함께 갱신할 것.

Worker 자체는 Cloudflare 장애와 운명을 같이한다. origin 감시 용도로는 문제가 없지만(어차피 Cloudflare가 죽으면 사이트도 안 보인다), 완전한 독립을 원하면 제3의 SaaS를 병행해야 한다.
