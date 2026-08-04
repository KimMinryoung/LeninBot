# LLM 게이트웨이 (llm/gateway.py)

최종 확인: 2026-08-04 (도입).

모든 LLM API 호출이 지나는 단일 seam. 툴 보안 게이트웨이(`security_gateway/`)의
LLM 버전으로, 같은 패턴을 따른다: 단일 관문 + 이중 싱크 감사 + shadow→enforce 롤아웃.
데이터 모델은 LiteLLM proxy의 spend-log/예산 설계를 차용하되 **in-process**로 구현했다 —
프록시 홉을 두면 스트리밍 idle guard·prompt cache 브레이크포인트·thinking 리플레이가
전부 그 홉을 통과해야 해서 어댑터가 지키는 프로바이더 계약에 회귀 위험이 크다.

## Seam이 되는 세 지점

| 지점 | 커버 범위 |
|---|---|
| `agent_loop.LoopState.add_cost` | 모든 툴-루프 라운드 (두 프로토콜 어댑터의 비용 이벤트가 이미 여기로 모임 — Claude·DeepSeek·Kimi Writer·GPT·Kimi·로컬) |
| `agent_loop.run_tool_loop` 진입부 | 에이전트 턴당 1회 정책 체크 (`check_llm_call`) |
| `llm.call_registry.generate_sync` | 등록된 원샷 호출 전부 (gemini/deepseek/openai/claude/kimi executor) |

새 호출부를 만들 때: 루프면 `chat_with_tools`를, 원샷이면 registry `generate()`를
쓰는 한 자동으로 seam을 지난다. 그 밖의 직접 SDK 호출은 만들지 말 것.

## API

```python
from llm.gateway import check_llm_call, record_llm_call, LLMGatewayDenied

check_llm_call(surface="loop|oneshot", caller=..., model=..., provider=None)
#   정책 위반 + enforce=true → LLMGatewayDenied. shadow에서는 would_deny 기록 후 통과.
#   내부 오류는 항상 fail-open (호출을 깨지 않는다).

record_llm_call(surface=..., caller=..., model=..., tokens_in=..., tokens_out=...,
                cache_read=..., cache_create=..., cost_usd=None, latency_ms=..., status="ok")
#   cost_usd 미지정 시 provider_registry 가격표로 추정 (모르는 모델은 None — 날조 금지).
#   절대 raise하지 않는다.
```

## 감사 — 이중 싱크 (security_gateway/audit.py와 동일 구조)

1. `llm_gateway.audit` 로거의 `llm_call {json}` 라인 → journald (항상, 동기)
2. `llm_audit_log` PG 테이블 (메인 DB) — 백그라운드 워커 스레드, fire-and-forget.
   큐 2000건 초과분·DB 실패분은 드롭 (로그만). `LENINBOT_LLM_AUDIT_DB=0`이면
   DB 싱크 생략 (유닛 테스트 러너가 설정; ad-hoc read-only 가드 소음 방지에도 사용 가능).

컬럼: ts, surface(loop|oneshot), caller(에이전트명/feature키), provider, model,
label(라운드 라벨), tokens_in/out, cache_read/create, cost_usd, latency_ms,
status(ok|error|denied|would_deny), error_excerpt.

토큰 의미론 주의: Anthropic 프로토콜은 tokens_in이 캐시 토큰을 **제외**하고,
OpenAI 호환은 prompt_tokens가 캐시 히트를 **포함**한다. 비용 추정이 이를 구분한다.

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

## Seam 밖에 남은 호출 (알려진 사각지대)

- **graphiti-core 내부 OpenAI 호출** (KG 추출/임베딩) — 라이브러리 내부 클라이언트
- **browser-use 내부 호출** — 라이브러리가 자체 클라이언트 소유
- **razvedchik의 자체 cloud 클라이언트** (`agents/razvedchik/cloud_llm.py`)
- **Codex CLI 위임** (GPT Pro 구독 — 과금 자체가 API 밖)
- 로컬 임베딩 서버 (BGE-M3 — LLM 아님, 비용 없음)

razvedchik은 registry executor로 옮기면 seam에 들어온다. graphiti/browser-use는
env base_url로 로컬 프록시를 세워야만 잡히는데, 그 비용이 현재로선 수확보다 크다.

## 테스트

`tests/test_llm_gateway.py` (허메틱 22케이스): provider 추론, 비용 추정(양쪽 토큰
의미론·dated 모델 프리픽스·미지 모델 None), 정책(shadow/enforce/예산/fail-open),
LoopState seam 전달. 루프 계약 회귀는 기존 `test_*_loop_rounds.py`가 잡는다.
