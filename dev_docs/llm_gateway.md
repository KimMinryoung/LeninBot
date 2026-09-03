# LLM 게이트웨이 (llm/gateway.py + llm_proxy/)

최종 확인: 2026-08-29 (툴 루프 SDK 이중 감사 제거, GPT-5.6 가격·캐시 쓰기·장문 티어 갱신).

모든 LLM API 호출이 지나는 단일 seam. 툴 보안 게이트웨이(`security_gateway/`)의
LLM 버전으로, 같은 패턴을 따른다: 단일 관문 + 이중 싱크 감사 + shadow→enforce 롤아웃.
두 절반으로 구성된다:

- **관찰/정책 절반** — `llm/gateway.py` (in-process): 호출 전 정책 체크, 호출 후
  토큰/비용 감사. LiteLLM proxy의 spend-log/예산 데이터 모델 차용.
- **강제 절반** — `llm_proxy/` (`leninbot-llm-proxy.service`, 127.0.0.1:8110):
  **바이트 패스스루 키 주입 프록시**. 프로바이더 API 키는 이 서비스의 systemd
  credential에만 있고, 다른 서비스의 클라이언트는 placeholder 키(`via-llm-proxy`) +
  프록시 base_url로 구성된다. 요청/응답 본문은 건드리지 않고(스트리밍은 `aiter_raw`
  원시 바이트 그대로) auth 헤더만 교체하므로, LiteLLM 같은 *번역형* 프록시와 달리
  SSE·prompt cache·thinking 블록 계약에 회귀 여지가 없다. 키 제거는 완료됐다(아래
  "Enforcement — 키 제거 완료") — 키 없는 코드는 프로바이더를 직접 호출할 수 없다.

## Seam이 되는 지점

| 지점 | 커버 범위 |
|---|---|
| `llm.agent_loop.LoopState.add_cost` | 모든 툴-루프 라운드 (두 프로토콜 어댑터의 비용 이벤트가 이미 여기로 모임 — Claude·DeepSeek·Kimi Writer·GPT·Kimi·로컬) |
| `llm.agent_loop.run_tool_loop` 진입부 | 에이전트 턴당 1회 정책 체크 (`check_llm_call`) |
| `llm.call_registry.generate_sync` | 등록된 원샷 호출 전부 (gemini/deepseek/openai/claude/kimi executor) |
| `llm.instrumented_clients.AuditedGenAIClient` | graphiti-core가 소유하는 Gemini 추출·임베딩 SDK 호출 (임베딩 토큰은 SDK 미제공이라 `embed_content:estimated` 라벨의 보수적 추정치) |
| `browser.use_agent._AuditedBrowserChatMixin`, `telegram.commands.handle_photo` | browser-use 매 step과 Telegram vision 직접 호출 |

새 호출부를 만들 때: 루프면 `chat_with_tools`를, 원샷이면 registry `generate()`를
쓰는 한 자동으로 seam을 지난다. 그 밖의 직접 SDK 호출은 만들지 말 것.
`bot_config`의 SDK 객체는 애드혹 직접 사용도 놓치지 않도록 `AuditedAsyncAnthropic`/
`AuditedAsyncOpenAI`로 감싸지만, 툴 루프 요청은 `with_audit_owner(..., "loop")`로 소유자를
표시한다. 래퍼는 caller 헤더·DeepSeek thinking 기본값은 그대로 주입하면서 자체
`external_sdk` 정책/비용 행만 생략하고, `LoopState.add_cost`가 유일한 과금 행을 쓴다.
따라서 같은 비스트리밍 응답이 `external_sdk`와 `loop`에 중복 집계되지 않는다.

## API

```python
from llm.gateway import check_llm_call, record_llm_call, LLMGatewayDenied

check_llm_call(surface="loop|oneshot", caller=..., model=..., provider=None)
#   정책 위반 + enforce=true → LLMGatewayDenied. shadow에서는 would_deny 기록 후 통과.
#   내부 오류는 항상 fail-open (호출을 깨지 않는다).

record_llm_call(surface=..., caller=..., model=..., tokens_in=..., tokens_out=...,
                cache_read=..., cache_create=..., cost_usd=None, latency_ms=..., status="ok",
                token_semantics="anthropic|openai|gemini", estimate_cost=True)
#   cost_usd 미지정 시 provider_registry 가격표로 추정 (모르는 모델은 None — 날조 금지).
#   절대 raise하지 않는다.
```

## 감사 — 이중 싱크 (security_gateway/audit.py와 동일 구조)

1. `llm_gateway.audit` 로거의 `llm_call {json}` 라인 → journald (항상, 동기)
2. `llm_audit_log` PG 테이블 (메인 DB) — 백그라운드 워커 스레드, fire-and-forget.
   큐 2000건 초과분·싱크 실패분은 드롭 (로그만). `LENINBOT_LLM_AUDIT_DB=0`이면
   DB 싱크 생략 (유닛 테스트 러너가 설정).

### 감사 싱크 — 프록시가 유일한 DB 기록자 (2026-09-04)

이전에는 행을 만든 모든 프로세스가 직접 INSERT 했다. 그래서 서비스마다 전권 DB
비밀번호가 필요했고, 시크릿이 없는 애드혹 스크립트는 감사 행을 조용히 버렸다
(2026-09-03 재임베딩 스크립트가 그 예). 이제 `audit_sink.py` 하나가 양쪽 장부
(`llm_audit_log`, `tool_audit_log`)의 컬럼 화이트리스트·캡·INSERT를 갖고, 기록
경로는 세 가지 모드로 갈린다 (`audit_sink.mode()`):

| 모드 | 조건 | 동작 |
|---|---|---|
| `local` | `llm_proxy.app`가 import 시 `set_local_sink(True)` | 프록시 자신의 행은 직접 INSERT |
| `proxy` | 정책의 `proxy_base` 설정됨 (프로덕션 기본) | 워커가 최대 50행씩 `POST {proxy}/audit/{llm\|tool}` |
| `db` | proxy_base 없음 (테스트·독립 도구) | 예전처럼 직접 INSERT |

`LENINBOT_AUDIT_SINK=db|proxy`로 강제할 수 있다. 워커·큐·`flush_audit()` 계약은
그대로다 — 바뀐 것은 배치 드레인과 DB INSERT가 HTTP POST로 바뀐 것뿐이다.

프록시 쪽: `POST /audit/{kind}`는 본문 1 MB·200행 캡, 미지 컬럼은 400(스키마
드리프트는 테스트에서 드러나야지 null로 묻히면 안 된다), 문자열은 캡으로 잘라
저장, DB 장애는 503(클라이언트가 로그 남기고 드롭). `GET /audit/spend/today`는
클라이언트 프로세스의 예산 정책(`_today_spend`)이 읽는다 — 프록시가 권위 판정
지점이므로 클라이언트가 DB를 읽을 이유가 없어졌다. `/health`에 `audit_sink`
필드가 붙지만 상태 코드에는 영향 없다: DB 장애가 LLM 트래픽을 막으면 안 된다.

클라이언트 쪽: POST 타임아웃 3초, 연결 실패는 0.5초 후 1회 재시도(프록시
재시작 블립 흡수), 4xx는 재시도 없음. 실패 첫 회 WARNING, 반복은 DEBUG.
systemd 밖 프로세스(INVOCATION_ID 없음)가 낸 LLM 행은 `label`에 ` [adhoc]`이
붙어 비용 보고서에서 서비스와 구분된다.

**INSERT 전용 롤 (선택, 권장):** `.env`의 `AUDIT_DB_USER`와 credential
`AUDIT_DB_PASSWORD`가 있으면 프록시는 그 롤로 별도 커넥션 풀을 연다. 없으면
메인 DB 유저(`db_password`)로 INSERT 한다 — 동작은 같고 최소권한만 덜하다.
롤은 `schema_migrations.py --only audit-sink-role`이 만든다 (CREATE/ALTER ROLE +
두 표 INSERT + 시퀀스 + `llm_audit_log` SELECT(지출 합계용)). 순서: (1) sudo
`manage_secrets.py add AUDIT_DB_PASSWORD`, (2) `.env`에 `AUDIT_DB_USER=leninbot_audit`,
(3) 마이그레이션, (4) 유닛의 주석 처리된 `LoadCredentialEncrypted=audit_db_password`
해제 + daemon-reload + 프록시 재시작. 파일이 없는 LoadCredentialEncrypted는 유닛
시작을 실패시키므로 (1) 전에 (4)를 하면 안 된다.

읽기 쪽 운영 CLI(`scripts/security_gateway.py audit`, `llm_registry_cli`)는 여전히
DB 읽기 권한이 필요하다 — 싱크는 쓰기만 다룬다.

컬럼: ts, surface(loop|oneshot|external_sdk|browser_use|vision|proxy), caller(에이전트명/feature키), provider, model,
label(라운드 라벨), tokens_in/out, cache_read/create, cost_usd, latency_ms,
status(ok|error|denied|would_deny), error_excerpt.

토큰 의미론 주의: Anthropic 프로토콜은 tokens_in이 캐시 토큰을 **제외**하고,
OpenAI 호환과 Gemini는 prompt_tokens가 캐시 히트를 **포함**한다. DeepSeek/Kimi처럼
가격표가 프로토콜별로 겹치는 모델은 호출부가 `token_semantics`를 명시한다. Gemini
가격은 [Google AI Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)의
2026-08-05 standard tier를 기준으로 `provider_registry.GEMINI_PRICING`에 둔다.

GPT-5.6은 OpenAI 호환 의미론에서도 입력을 ordinary/cache-read/cache-write 세 종류로
나눈다. `usage.prompt_tokens_details.cache_write_tokens`를 `cache_create`에 보존하고,
ordinary input에서는 cache read와 write를 모두 뺀 뒤 각각 공식 단가로 다시 계산한다.
272K 입력 토큰을 초과하면 전체 요청에 long-context 단가(입력 2배·출력 1.5배)를
적용한다. 현재 Standard 단가는 `dev_docs/llm_provider_architecture.md`에 정리한다.

DeepSeek V4는 **시간대별 요금**이다: 2026-08-16 16:00 UTC(베이징 08-17 00:00)부터
평면 단가를 버리고 피크(UTC 01–04·06–10시)/오프피크(그 외, 피크의 절반) 티어로
바뀐다. 정적 표에 두지 않고 `provider_registry.deepseek_price_triple(model, now)`가
호출 시각으로 해석하며, `anthropic_pricing_table()`·`openai_compatible_pricing()`·
`gateway.estimate_cost_usd()`가 이를 경유한다. 컷오버 전은 옛 평면 단가. 큐레이터
레인은 17:00~23:20 UTC라 전 구간 오프피크다([[commulingo-curator-lanes]]).
2026-08-14에는 비용 절감을 위해 큐레이터 두 스펙(people/event)을 GPT-5.6 Luna로
옮겼지만, 한국어 공개 문서에서 외국어 토큰 혼입이 반복되어 2026-08-29부터 두 레인
모두 DeepSeek V4 Pro(`deepseek`/`deepseek_pro`)를 사용한다. 바인딩은
`bot_config.resolve_agent_tool_loop`가 스펙과 `config/agent_runtime.json`에서 해석한다.
DeepSeek 시간대 단가는 이 큐레이터들과 webchat 등 모든 DeepSeek 호출부의 과금에
계속 쓰인다.

## 정책 설정 계층 (mtime 핫리로드)

정책은 세 계층을 순서대로 병합한다.

1. `llm/gateway.py`의 코드 fallback
2. Git이 추적하는 `config/llm_gateway.defaults.json` — 권장 기본값과 배포 토폴로지
3. Git에서 제외한 `config/llm_gateway.local.json` — 해당 호스트의 가변 운영값

운영 CLI는 local 파일만 원자적으로 수정한다. 따라서 예산·차단 목록·kill switch를
바꿔도 작업트리가 더러워지지 않고, `git pull`이 호스트 정책을 덮어쓰지 않는다.
local 파일이 없거나 특정 key가 없으면 tracked default를 그대로 상속한다.

```json
{
  "enforce": true,                // false로 local override하면 shadow
  "block_all": false,             // kill switch
  "blocked_providers": [],
  "blocked_models": [],
  "daily_budget_usd": null,       // UTC 일일 총액 캡
  "daily_budget_per_provider": {} // {"claude": 20.0} 형태
}
```

예를 들어 한 호스트에서 일시적으로 shadow 모드와 일일 총액 cap만 적용하려면 local
파일은 아래 두 key만 가지면 된다.

```json
{
  "enforce": false,
  "daily_budget_usd": 25
}
```

- 예산 체크는 `llm_audit_log`의 당일 SUM(cost_usd) (60초 캐시).
- **DB 조회 실패 시 fail-open** — 봇 가용성이 enforcement보다 우선.
- 롤아웃은 툴 게이트웨이와 동일: shadow로 데이터를 쌓고, would_deny가 오탐 없음이
  확인되면 enforce로 올린다. **2026-08-05 enforce 전환 완료** — 차단 리스트·예산은
  전부 비활성이라 현재 거부되는 호출은 없고, 정책을 설정하는 즉시 강제된다(핫리로드).
  전환 검증: 프록시 스모크(`GET /anthropic/v1/models` → 200 통과) + 감사 행 확인.

**판정 지점은 둘, 로직은 하나, 권위는 프록시** (2026-08-05). 판정 로직은
`evaluate_policy()` 한 곳이고, in-process `check_llm_call`(빠른 실패 + caller 태깅)과
**프록시**가 같은 함수를 부른다. 프록시 쪽 판정이 권위다 — 키가 프록시 건너편에만
있으므로 클라이언트가 이 판정을 건너뛸 방법이 없다. OpenAI/Anthropic은 요청 본문,
Gemini는 `models/{model}:method` URL 경로(퍼센트 인코딩 포함)에서 모델을 읽는다.
원본 바이트는 그대로 전달한다. 라우트명은 정책 프로바이더명으로 매핑한다
(anthropic→claude, moonshot→kimi). enforce 시
403 + `surface=proxy` 거부 행을 기록한다. 허용된 upstream 요청도 `surface=proxy`의
비과금 transport 행(`estimate_cost=False`)을 남겨 in-process 감사와 대조할 수 있다.
이 transport 행은 **스트림이 실제로 끝난 시점**에 쓴다(2026-08-14,
`relay_and_record`): 헤더 도착 시점에 쓰면 중간에 끊긴 스트림이 영원히 ok로
남고 latency_ms가 헤더까지의 시간만 재기 때문이다. 이제 status가 스트림의 실제
결말을 반영하고(업스트림 중단 `stream aborted: …`와 클라이언트 이탈
`client disconnected mid-stream`을 error_excerpt로 구분), latency_ms는 스트림
전체 시간이다. 헤더 도착·스트림 종료는 각각 journald 로그 라인도 남긴다.

## 운영 CLI

```bash
venv/bin/python scripts/llm_gateway_cli.py status        # 정책 + 오늘 스펜드 + 오늘 거부(denied/would_deny) 건수
venv/bin/python scripts/llm_gateway_cli.py spend --days 7  # 일별 합계 + 기간 총계 포함
venv/bin/python scripts/llm_gateway_cli.py tail -n 30
venv/bin/python scripts/llm_gateway_cli.py set daily_budget_usd 25   # local override, 핫리로드
venv/bin/python scripts/llm_gateway_cli.py unset daily_budget_usd    # tracked default 상속
```

DB 명령은 자격증명 필요: `DB_PASSWORD="$(cat /run/credentials/leninbot-api.service/db_password)"`.

## 테이블 생성

`scripts/schema_migrations.py --only llm-audit-log` (2026-08-04 프로덕션 적용 완료).

## 프록시 라우팅 (2026-08-04 가동)

effective policy의 `"proxy_base": "http://127.0.0.1:8110"`이 스위치다. 저장소의
권장값은 `config/llm_gateway.defaults.json`에 있고, 다른 배포 토폴로지가 필요한
호스트만 `config/llm_gateway.local.json`에서 override한다.
설정되면 (env의 명시적 base URL 오버라이드가 없는 한) 관리형 클라이언트 전부가
프록시 경유로 구성된다: bot_config의 6개 클라이언트(Claude·OpenAI·DeepSeek 양栈·
Kimi 양栈), registry 원샷 executor 5종(gemini 포함), writer 클라이언트, browser
워커/browser-use의 provider 챗, Telegram vision. `null`로 되돌리고 재시작하면 직접 호출로 복귀한다
(롤백 경로). 클라이언트 구성이 임포트 시점이라 반영은 서비스 재시작.

### DeepSeek thinking 기본값 주입 (2026-08-14)

DeepSeek V4는 요청이 아무 말도 안 하면 thinking ON이고, 추론이 max_tokens를 reply와
나눠 쓰므로 텍스트만 원한 호출이 예산 전부를 추론에 태우고 본문 없는 200을 받는다.
관리 경로는 전부 명시한다(래퍼 `thinking_off`, 루프 파라미터, registry 스펙 —
`research_markdown_translation`도 2026-08-14부터 명시적 enabled). 문제는 **애드혹
스크립트가 반복적으로 누락**하는 것 — AST 컨포먼스 테스트는 temp_dev/저장소 밖을
안 보고 런타임엔 아무것도 못 막는다. 그래서 키가 있는 유일한 지점인 프록시가
백스톱이다: deepseek 라우트의 completion 경로(`chat/completions`·`v1/messages`)로
온 JSON object 본문에 `thinking` 키가 없으면 `{"type": "disabled"}`를 주입한다
(`apply_deepseek_thinking_default`). 키가 있으면(값 불문) 바이트 그대로 통과 —
바이트 패스스루 원칙의 유일한 본문 예외이며, 주입 시 transport 감사 행 label에
` +think-off-default`가 붙고 journald 로그가 남는다. thinking ON이 필요한 애드혹
호출은 명시적으로 `thinking: {"type": "enabled"}`를 보내면 된다.

Registry executor는 `resolve_provider_connection()`으로 base URL과 credential을 동시에
해석한다. 따라서 keyless gateway mode의 placeholder가 정상 credential로 취급되고,
archival translation preflight도 로컬 provider key를 잘못 요구하지 않는다. 반대로 명시적
direct endpoint override에는 실제 provider key가 필요해 placeholder가 외부 endpoint로
전송되지 않는다.

라우트: `/{provider}/{path}` → upstream `/{path}`. provider는 anthropic / deepseek /
moonshot / openai / gemini다. KG와 Writer도 각각 공용 gemini/anthropic route와 key를
사용한다. `/health`는 모든 route에 쓸 키가 있을 때만 200, 아니면 503이다.
proxy unit의 `ExecStartPost=wait_llm_proxy_ready.py`가 200까지 기다리므로 소비 unit의
`Wants/After=leninbot-llm-proxy.service`는 실제 readiness 뒤 시작을 보장한다.

### 잔액·비용 통합 조회

`scripts/llm-balances [--days 1..30] [--json]`은 provider 공식 재무 정보와
`llm_audit_log`의 로컬 비용 추정액을 한 표에 표시한다. DeepSeek
`GET /user/balance`와 Kimi `GET /v1/users/me/balance`는 기본 inference key로 실시간
잔액을 읽는다. OpenAI `/v1/organization/costs`와 Claude
`/v1/organizations/cost_report`는 각각 선택적 `OPENAI_ADMIN_KEY`와
`ANTHROPIC_ADMIN_KEY`가 있을 때만 공식 기간 비용을 읽고, 없으면 로컬 감사액만
표시한다. Gemini는 일반 API key만으로 통합 가능한 잔액 API가 없어 로컬 감사액만
표시하며 로컬 Qwen은 무과금이다.

같은 보고서는 owner 전용 Telegram 명령 `/llm_balance [1~30]`에서도 조회할 수 있다.
기간 기본값은 30일이며 `/` 자동완성 메뉴와 `/help`에 노출된다. Telegram 서비스는
자신의 읽기 전용 DB 연결로 로컬 감사액을 집계하고 공식 값은 localhost LLM proxy의
고정 billing route를 통해 읽는다.

Admin credential은 범용 `/{provider}/{path}` map에 등록하지 않는다. 로컬 전용
`/billing/{deepseek|kimi|openai|claude}`가 provider별 고정 GET 경로와 날짜 파라미터만
만들 수 있으므로, 비용 조회를 위해 넣은 조직 Admin 키로 다른 관리 API를 호출할 수
없다. Admin 키가 credstore에 없으면 구조화된 `credential_missing`을 반환하며
`/health` readiness에는 영향을 주지 않는다.

로컬 비용 집계는 `surface=proxy`의 비과금 transport 행과 과거
`external_sdk` 기본 wrapper caller(`openai_client`, `deepseek_anthropic_direct` 등)를
제외한다. 2026-08-29 이전 툴 루프가 같은 응답을 SDK와 loop 양쪽에 기록한 기간까지
30일 합계에 포함하더라도 중복 과금하지 않기 위한 호환 필터다. feature 이름을 가진
진짜 direct SDK 호출(예: `kg_graphiti`)은 계속 포함한다.

## Enforcement — 키 제거 완료 (2026-08-05)

`scripts/remove_llm_provider_keys.sh`(root)로 13개 서비스에서 anthropic/deepseek/
moonshot/openai 키 credential을 주석 처리했다. **이제 실키는 leninbot-llm-proxy에만
있고, 키 없는 코드는 프로바이더를 직접 호출할 수 없다.** 검증: keyless 서비스 전부
정상 기동, KG init 성공, 웹챗 라이브 스트리밍이 프록시 로그(`POST
/deepseek/anthropic/v1/messages → 200`)와 `llm_audit_log` 라운드 행으로 확인됨.

- 사전 조치: `.env`의 `OPENAI_API_KEY=via-llm-proxy` + `OPENAI_BASE_URL=프록시`
  (graphiti-core 내부 reranker — env로 AsyncOpenAI를 만드는 경로 — 를 프록시로).
  프록시 자신은 credential 파일 직독이라 이 env에 오염되지 않는다.
- **롤백**: 각 파일 옆의 `.bak-llmkeys` 백업 복원 + `systemctl daemon-reload` + 재시작.
- 키를 유지하는 예외: `leninbot-llm-proxy`(보관소)와
  `research-document-translation`(직접 클라이언트 일회성 스크립트)뿐이다.
- **gemini도 편입 완료 (2026-08-05 2차)**: graphiti 추출·임베딩은 `client=`로
  프록시 경유 `genai.Client`를 주입받고, browser-use vision 폴백(ChatGoogle
  `http_options` / ChatOpenAI `base_url`)도 프록시 경유. gemini 키 제거 후
  keyless 상태에서 KG 스모크 ok=true + 프록시 로그의
  `POST /gemini/.../batchEmbedContents → 200`으로 검증. ad-hoc KG 스크립트는
  `GEMINI_API_KEY="$(cat /run/credentials/leninbot-llm-proxy.service/gemini_api_key)"`.

## Seam 밖에 남은 호출

- **수동 maintenance/일회성 스크립트** — `research-document-translation`,
  `scripts/classify_untyped_entities.py`, `skills/kg-maintenance/scripts/*`는 운영 상주
  서비스가 아니며 operator가 명시적으로 provider credential을 전달해 실행하는 경로다.
  서비스 keyless 경계에는 포함되지 않는다. CommUlingo Wikipedia-evidence 사건 링크
  스크립트는 `commulingo_event_evidence_labels` registry executor로 이전되어 이 예외에서
  제외됐다.
- **Codex CLI 위임** (GPT Pro 구독 — 과금 자체가 API 밖)
- 로컬 임베딩 서버 (BGE-M3, :8100 — LLM 아님, 비용 없음. 프록시는 :8110)

Razvedchik의 독자 `cloud_llm.py`는 2026-08-05 삭제했다. 댓글·관찰·답글·디브리핑은
각각 registry executor feature로 이동해 다른 원샷 호출과 같은 정책·감사·프록시 경로를 쓴다.

## 테스트

`tests/test_llm_gateway.py`: provider 추론, 비용 추정(양쪽 토큰 의미론·GPT-5.6
cache-write/장문 티어·dated 모델 프리픽스·미지 모델 None), 정책
(shadow/enforce/예산/fail-open), LoopState seam 전달. 루프 계약 회귀와 SDK 래퍼의
단일 감사 소유권은 `test_*_loop_rounds.py`가 잡는다.
`tests/test_llm_balances.py`는 Admin 키 고정 경로, provider 응답 정규화, 과거 중복
wrapper 제외 SQL, query-db TSV 파싱과 CLI/Telegram 표시 계약을 검증한다.

## 짧게 사는 프로세스의 감사 유실 (2026-08-06)

DB 싱크 워커는 daemon 스레드다. 오래 사는 서비스에서는 문제가 없지만, 배치
스크립트처럼 한 번 돌고 끝나는 프로세스는 워커가 큐를 비우기 전에 죽어서 행이
통째로 사라졌다. 프록시 쪽 행(`surface=proxy`)은 남으니 지출 자체는 보존되지만,
`caller`를 들고 있는 것은 in-process 행이라 CLI 실행만 비용 귀속이 빠졌다.

`gateway.flush_audit(timeout)`이 큐가 빌 때까지 기다린다. 워커가 처음 뜰 때
`atexit`에 자동으로 걸리므로 호출부가 따로 부를 필요는 없다. 대기 상한은
`LENINBOT_LLM_AUDIT_FLUSH_SECONDS`(기본 5초)이고, 시간을 넘기면 남은 행 수를
경고로 남기고 포기한다 — 종료 경로를 붙잡고 있는 것보다 낫다.
