# LLM 게이트웨이 (llm/gateway.py + llm_proxy/)

최종 확인: 2026-08-05 (키 격리, 외부 SDK 계측, readiness 보강).

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
  SSE·prompt cache·thinking 블록 계약에 회귀 여지가 없다. 최종 단계(아래 "남은
  enforcement 단계")까지 가면 키 없는 코드는 프로바이더를 직접 호출할 수 없게 된다.

## Seam이 되는 지점

| 지점 | 커버 범위 |
|---|---|
| `agent_loop.LoopState.add_cost` | 모든 툴-루프 라운드 (두 프로토콜 어댑터의 비용 이벤트가 이미 여기로 모임 — Claude·DeepSeek·Kimi Writer·GPT·Kimi·로컬) |
| `agent_loop.run_tool_loop` 진입부 | 에이전트 턴당 1회 정책 체크 (`check_llm_call`) |
| `llm.call_registry.generate_sync` | 등록된 원샷 호출 전부 (gemini/deepseek/openai/claude/kimi executor) |
| `llm.instrumented_clients.AuditedGenAIClient` | graphiti-core가 소유하는 Gemini 추출·임베딩 SDK 호출 (임베딩 토큰은 SDK 미제공이라 `embed_content:estimated` 라벨의 보수적 추정치) |
| `browser.use_agent._AuditedBrowserChatMixin`, `telegram.commands.handle_photo` | browser-use 매 step과 Telegram vision 직접 호출 |

새 호출부를 만들 때: 루프면 `chat_with_tools`를, 원샷이면 registry `generate()`를
쓰는 한 자동으로 seam을 지난다. 그 밖의 직접 SDK 호출은 만들지 말 것.

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
   큐 2000건 초과분·DB 실패분은 드롭 (로그만). `LENINBOT_LLM_AUDIT_DB=0`이면
   DB 싱크 생략 (유닛 테스트 러너가 설정; ad-hoc read-only 가드 소음 방지에도 사용 가능).

컬럼: ts, surface(loop|oneshot|external_sdk|browser_use|vision|proxy), caller(에이전트명/feature키), provider, model,
label(라운드 라벨), tokens_in/out, cache_read/create, cost_usd, latency_ms,
status(ok|error|denied|would_deny), error_excerpt.

토큰 의미론 주의: Anthropic 프로토콜은 tokens_in이 캐시 토큰을 **제외**하고,
OpenAI 호환과 Gemini는 prompt_tokens가 캐시 히트를 **포함**한다. DeepSeek/Kimi처럼
가격표가 프로토콜별로 겹치는 모델은 호출부가 `token_semantics`를 명시한다. Gemini
가격은 [Google AI Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)의
2026-08-05 standard tier를 기준으로 `provider_registry.GEMINI_PRICING`에 둔다.

## 정책 (config/llm_gateway.json, mtime 핫리로드)

```json
{
  "enforce": false,               // false = shadow (would_deny 기록만)
  "block_all": false,             // kill switch
  "blocked_providers": [],
  "blocked_models": [],
  "daily_budget_usd": null,       // UTC 일일 총액 캡
  "daily_budget_per_provider": {} // {"claude": 20.0} 형태
}
```

- 예산 체크는 `llm_audit_log`의 당일 SUM(cost_usd) (60초 캐시).
- **DB 조회 실패 시 fail-open** — 봇 가용성이 enforcement보다 우선.
- 롤아웃은 툴 게이트웨이와 동일: shadow로 데이터를 쌓고, would_deny가 오탐 없음이
  확인되면 enforce로 올린다.

**판정 지점은 둘, 로직은 하나, 권위는 프록시** (2026-08-05). 판정 로직은
`evaluate_policy()` 한 곳이고, in-process `check_llm_call`(빠른 실패 + caller 태깅)과
**프록시**가 같은 함수를 부른다. 프록시 쪽 판정이 권위다 — 키가 프록시 건너편에만
있으므로 클라이언트가 이 판정을 건너뛸 방법이 없다. OpenAI/Anthropic은 요청 본문,
Gemini는 `models/{model}:method` URL 경로(퍼센트 인코딩 포함)에서 모델을 읽는다.
원본 바이트는 그대로 전달한다. 라우트명은 정책 프로바이더명으로 매핑한다
(anthropic→claude, moonshot→kimi). enforce 시
403 + `surface=proxy` 거부 행을 기록한다. 허용된 upstream 요청도 `surface=proxy`의
비과금 transport 행(`estimate_cost=False`)을 남겨 in-process 감사와 대조할 수 있다.

## 운영 CLI

```bash
venv/bin/python scripts/llm_gateway_cli.py status        # 정책 + 오늘 스펜드
venv/bin/python scripts/llm_gateway_cli.py spend --days 7
venv/bin/python scripts/llm_gateway_cli.py tail -n 30
venv/bin/python scripts/llm_gateway_cli.py set daily_budget_usd 25   # 핫리로드
```

DB 명령은 자격증명 필요: `DB_PASSWORD="$(cat /run/credentials/leninbot-api.service/db_password)"`.

## 테이블 생성

`scripts/schema_migrations.py --only llm-audit-log` (2026-08-04 프로덕션 적용 완료).

## 프록시 라우팅 (2026-08-04 가동)

`config/llm_gateway.json`의 `"proxy_base": "http://127.0.0.1:8110"`이 스위치다.
설정되면 (env의 명시적 base URL 오버라이드가 없는 한) 관리형 클라이언트 전부가
프록시 경유로 구성된다: bot_config의 6개 클라이언트(Claude·OpenAI·DeepSeek 양栈·
Kimi 양栈), registry 원샷 executor 5종(gemini 포함), writer 클라이언트, browser
워커/browser-use의 provider 챗, Telegram vision. `null`로 되돌리고 재시작하면 직접 호출로 복귀한다
(롤백 경로). 클라이언트 구성이 임포트 시점이라 반영은 서비스 재시작.

라우트: `/{provider}/{path}` → upstream `/{path}`. provider는 anthropic / deepseek /
moonshot / openai / gemini다. KG와 Writer도 각각 공용 gemini/anthropic route와 key를
사용한다. `/health`는 모든 route에 쓸 키가 있을 때만 200, 아니면 503이다.
proxy unit의 `ExecStartPost=wait_llm_proxy_ready.py`가 200까지 기다리므로 소비 unit의
`Wants/After=leninbot-llm-proxy.service`는 실제 readiness 뒤 시작을 보장한다.

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
  `scripts/classify_untyped_entities.py`, `scripts/commulingo_event_evidence_links.py`,
  `skills/kg-maintenance/scripts/*`는 운영 상주 서비스가 아니며 operator가 명시적으로
  provider credential을 전달해 실행하는 경로다. 서비스 keyless 경계에는 포함되지 않는다.
- **Codex CLI 위임** (GPT Pro 구독 — 과금 자체가 API 밖)
- 로컬 임베딩 서버 (BGE-M3, :8100 — LLM 아님, 비용 없음. 프록시는 :8110)

Razvedchik의 독자 `cloud_llm.py`는 2026-08-05 삭제했다. 댓글·관찰·답글·디브리핑은
각각 registry executor feature로 이동해 다른 원샷 호출과 같은 정책·감사·프록시 경로를 쓴다.

## 테스트

`tests/test_llm_gateway.py` (허메틱 22케이스): provider 추론, 비용 추정(양쪽 토큰
의미론·dated 모델 프리픽스·미지 모델 None), 정책(shadow/enforce/예산/fail-open),
LoopState seam 전달. 루프 계약 회귀는 기존 `test_*_loop_rounds.py`가 잡는다.
