# Web 검색·본문 추출 게이트웨이

## 소유권과 실행 경로

`leninbot-web-gateway.service` (`web_gateway/app.py`)가 `127.0.0.1:8111`에서
검색·유료 본문 추출을 전담한다. LLM 게이트웨이와 별도 프로세스/예산이다.

```text
runtime_tools/web_search.py → web_gateway/client.py → POST /search
content_fetch/urls.py: Playwright → 무료 HTTP → client.extract → POST /extract
                                                  ↓
                      web_gateway: 검증 → 비용 예약 → Tavily/Brave → 정산
```

- `web_gateway/search.py`: 공급자 순서, 요청 검증, Tavily/Brave 호출, 캐시·동시 요청 합치기와 circuit.
- `web_gateway/budget.py`: 게이트웨이 내부에서만 예산 예약·정산·사용량 집계.
- `web_gateway/credentials.py`: 게이트웨이의 `CREDENTIALS_DIRECTORY`에서만 검색 키 로딩.
- `web_gateway/client.py`: 키 없는 로컬 HTTP 클라이언트. 공급자 직접 호출·재시도 fallback 없음.
- `runtime_tools/registry.py`: 기존 `web_search` 도구 이름·인자·노출 경계 유지.
- `content_fetch/urls.py`: 로컬 URL/DNS/redirect 검증과 무료 본문 추출 유지.
  유료 Extract는 게이트웨이가 URL을 다시 검증하고 single-URL basic으로만 실행한다.

서버는 임의 upstream/path/API 인자를 받는 프록시가 아니다. 허용된 요청 모델만 받으며
추가 필드·잘못된 depth·결과 수를 거부한다. POST 본문은 32 KiB 이하로 제한한다.

## 키 격리와 API 경계

서비스는 `DynamicUser=yes`, `SupplementaryGroups=grass`, `ProtectSystem=strict`,
`ProtectHome=read-only`, `StateDirectory=leninbot-web-gateway`로 실행한다. 실제 키
`tavily_api_key`/`brave_search_api_key`는 이 서비스의 systemd credential에만 마운트한다.
다른 소비 서비스와 `.env`에는 검색 키를 두지 않는다. 환경변수 키는 게이트웨이에서도
인증 대체값으로 쓰지 않는다. root는 호스트 관리자이므로 이 격리의 대상이 아니다.

| 메서드 | 경로 | 계약 |
|---|---|---|
| GET | `/health` | 키 존재·예산 설정·장부 접근 준비 확인. 유료 요청 없음 |
| POST | `/search` | `{arguments: <기존 web_search 인자>, caller?: {...}}` → `{result: string, error: bool}` |
| POST | `/extract` | `{url: string, caller?: {...}}` → `{results: [{raw_content}], error: false}` 또는 `{error: true, message}` |
| GET | `/usage?days=7&by=service` | 1~366일 서비스별 집계. `by=task`는 작업/요청별 |

루프백 연결만 허용하며 proxy header 신뢰를 끈다. 별도 로컬 인증 토큰은 없고 단일
사용자 호스트 내부 서비스에 한정한다(LLM 프록시와 같은 로컬 신뢰 모델). Nginx/공개 API에
이 경로를 노출하지 않는다. caller의 서비스·작업 라벨은 로컬 클라이언트의 자기 신고이며
권한이나 예산 증액 근거가 아니다. 라벨을 바꿔도 모든 요청에 하나의 공용 예산을 적용한다.
키·검색어·URL·본문은 사용량 장부에 저장하지 않는다. 공급자 실패 원문은 클라이언트로
반환하지 않는다. 검색 결과의 `<external>` 출처 표시와 `ToolFailure` 오류 구분은 유지한다.

## 일일 예산·장부

`config/web_research.json`은 유료 요청 직전에 다시 읽는다. 현재 한도는 **$10/UTC일**이며
0은 유료 요청 중지다. `tavily_credit_usd=0.008`, `brave_search_usd=0.005`는 공개 종량제
추정 단가다. 무료 크레딧·계약 할인·세금은 제외하며 LLM 비용과 별도로 집계한다.
음수·누락·잘못된 JSON·유효하지 않은 공급자 단가는 유료 호출을 차단한다.

운영 장부는 **`/var/lib/leninbot-web-gateway/usage.sqlite3`**다. systemd가 서비스 전용
0700 StateDirectory를 제공한다. `WEB_RESEARCH_USAGE_DB`는 이 서비스의 저장 경로 설정이며
클라이언트는 이 파일을 직접 읽거나 쓰지 않는다. SQLite `BEGIN IMMEDIATE`에서 해당 일자의
사용/예약액을 합산하고 새 예약을 기록한 뒤에만 공급자 요청을 시작한다.

검색 basic/fast/ultra-fast는 1크레딧, advanced는 2크레딧, 단일 URL basic Extract는
보수적으로 1크레딧을 예약한다. Tavily의 실제 `usage.credits`(0 포함)로 정산한다.
사용량이 없거나 비정상이면 예약 추정액을 유지한다. 실패·취소는 `outcome_unknown`,
강제 종료는 `reserved`로 남아 예산에 계속 포함된다. 정산 기록 실패도 예약을 유지한다.
Brave는 요청당 단가를 사용한다. 공급자 규칙이 바뀌어 예약보다 큰 사용량이 보고되면
그대로 기록하므로 단가·과금 규칙 변경은 운영자가 반영해야 한다.

예산 부족은 공급자 장애가 아니다. 검색은 오류로 끝나며 Brave 등 다른 공급자로
우회하거나 circuit을 열지 않는다. `use_cache=false`도 예산을 우회할 수 없다.
게이트웨이/장부가 내려가면 유료 경로는 실패하고, 기존 무료 추출과 저장된 근거는 사용할 수 있다.
다른 서버나 외부 프로그램이 별도 보유한 같은 키로 쓰는 비용은 이 장부 범위 밖이다.

게이트웨이 시작 시 기존 `data/web_research_usage.sqlite3`가 있으면 행 ID 기준으로
중복 없이 가져온다. 과거 미확정 예약도 유지하므로 전환 당일 예산이 초기화되지 않는다.
초기 전환은 기존 소비자를 정리한 뒤 게이트웨이를 다시 시작해 마지막 기록까지 이관한다.

요금 근거(2026-09-12 확인): [Tavily](https://docs.tavily.com/documentation/api-credits),
[Brave](https://brave.com/search/api/). 실제 청구서가 아닌 운영 추정치다.

## 캐시와 조사 중단

기존 `WEB_SEARCH_PROVIDERS=tavily,brave`, `WEB_SEARCH_PROVIDER_COOLDOWN_SECONDS=300`,
`WEB_SEARCH_CACHE_TTL_SECONDS=300`은 게이트웨이가 읽는다. 검색 캐시는 256개 LRU이며
일반 300초, news/finance/day 60초, 빈 결과 30초 상한을 유지한다. 게이트웨이 프로세스의
캐시이므로 이제 서로 다른 소비 서비스도 동일 요청을 재사용하고 동시 요청을 합친다.
재시작하면 캐시는 사라지지만 예산 장부는 유지된다. 키에는 검색 제약과 공급자 순서를
포함하며 caller는 제외한다. 공유 유료 요청 비용은 최초 요청자에게 귀속한다.

도구와 writer/autonomous/CommuLingo 지침은 필요한 주장에 충분한 근거가 확보되면
검색을 멈추고 작성·저장하도록 한다. 누락 근거·상충 정보·변경 가능성 검증은 허용한다.
문장·번역·저장 형식 오류는 기존 근거로 고친다. CommuLingo의 기존 영속 조사 기억과
형식 오류 후 신규 조사 차단은 유지한다. 의미 유사도에 의한 기계적 검색 차단은 없다.

## 조회·배포·검증

```bash
venv/bin/python scripts/web_research_usage.py --days 7
venv/bin/python scripts/web_research_usage.py --days 7 --by task
```

조회 CLI는 `/usage`를 호출한다. 일자·공급자·search/extract·depth별 요청 수,
`accounted_usd`, 보고 크레딧, 미확정/추정 요청 수와 서비스/작업 귀속을 반환한다.

`systemd/leninbot-web-gateway.service`가 정식 unit 원본이다.
`scripts/prepare_web_gateway.py`는 grass로 실행해 현재 설치된 unit/drop-in의 검색 credential
마운트만 제거한 파일과 게이트웨이 credential/dependency drop-in을
`systemd/.web-gateway-stage/`에 준비한다. 실제 키는 읽거나 복사하지 않는다. 이 파일들을
`/etc/systemd/system/`에 설치하고 daemon-reload 후 게이트웨이와 소비 서비스를 재시작한다.
기존 소비자의 credential 마운트는 프로세스 재시작 후에야 사라진다. 예약 작업은 다음 실행부터
새 credential 구성을 사용한다. 최종 게이트웨이 재시작은 기존 사용량을 다시 확인·이관한다.
`scripts/migrate_secrets_to_credstore.py`의 공통 credential 생성기도 검색 키를 게이트웨이에만
배정하므로 이후 credential 갱신이 옛 마운트를 되살리지 않는다.

```bash
venv/bin/python -m unittest discover -s tests -p 'test_web_gateway.py'
venv/bin/python -m unittest discover -s tests -p 'test_paid_web_budget.py'
venv/bin/python -m unittest discover -s tests -p 'test_free_fetch_first.py'
venv/bin/python -m unittest discover -s tests -p 'test_web_search*.py'
venv/bin/python scripts/smoke_web_search_providers.py
venv/bin/python scripts/smoke_url_security.py
```

위 테스트는 모의 공급자를 사용한다. 운영 검증도 `/health`, `/usage`, 잘못된 요청의
거절과 파일 접근 거절로 진행할 수 있으며 실제 유료 검색을 할 필요가 없다.
