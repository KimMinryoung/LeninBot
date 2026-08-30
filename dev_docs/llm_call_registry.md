# LLM 호출 레지스트리 (llm/call_registry.py)

최종 확인: 2026-08-05.

프로젝트의 LLM API 호출은 두 층으로 관리한다.

| 층 | 대상 | 설정 파일 | 반영 시점 |
|---|---|---|---|
| 에이전트 루프 | analyst/scout/diary/browser/... (claude_loop·openai_tool_loop 경유) | `config/agent_runtime.json` + `llm/runtime_profile.py` | 서비스 재시작 |
| **원샷 호출** | 요약·분류·번역·critic 등 단발 호출 | `config/llm_call_sites.json` + `llm/call_registry.py` | **핫리로드 (mtime)** |

## 원샷 레지스트리

기능 키별 JSON 항목: `{provider, model, temperature, max_tokens, timeout, json_mode, managed, env, note}`.

모델 해석 우선순위: **레거시 env(`env` 배열) > 제네릭 env `LLM_SITE_<KEY>_MODEL` > JSON > 콜사이트 기본값**.

`managed` 값:
- `executor` — `generate()/generate_sync()`가 직접 실행 (gemini/deepseek/openai/claude/kimi 지원, system·json_mode·timeout 옵션)
- `model-only` — 모델명만 여기서 조회, 실행은 자체 클라이언트 (KG graphiti, razvedchik, writer 경량 별칭)
- `external` — 정보 등재만 (vision 폴백처럼 실행 구조가 특수한 곳)

실패 시 항상 `None` 반환 — 콜사이트가 자체 폴백(추출식 요약, 기본 라벨, 스킵)을 유지한다.

원샷 executor의 endpoint와 credential은 공개 함수
`resolve_provider_connection(provider)`가 한 번에 해석한다. 이 함수는 direct mode에서는
실제 provider key를 요구하고, `proxy_base`가 설정된 keyless 서비스에서는 proxy route와
`via-llm-proxy` placeholder를 함께 반환한다. `deepseek_anthropic`처럼 명시적 direct-base
환경변수를 지원하는 경로는 그 override가 있을 때 proxy를 우회하므로 실제 키가 없으면
즉시 실패한다. 호출부가 private key/route 표를 직접 읽거나 base와 key를 따로 해석하지
않는다.

## 운영 CLI

```bash
python scripts/llm_registry_cli.py list              # 원샷 + 에이전트 루프 통합 조회
python scripts/llm_registry_cli.py show <feature>    # 원본 + env 오버라이드 반영 유효값
python scripts/llm_registry_cli.py set <feature> <key> <value>   # 핫리로드 반영
python scripts/llm_registry_cli.py add <feature> --provider gemini --model ... 
python scripts/llm_registry_cli.py agent-show <agent>            # 에이전트 루프 설정 조회
python scripts/llm_registry_cli.py agent-set <agent> <key> <value>
#   key: provider|model|budget_usd|max_rounds — agent_runtime.json 수정, 핫리로드
#   저장 후 런타임 로더로 재검증, 실패 시 자동 원복. model 별칭 오타는 경고
```

## 새 호출부 등록 방법

1. `scripts/llm_registry_cli.py add my_feature --provider gemini --model gemini-3.1-flash-lite --temperature 0 --max-tokens 256 --note "설명"`
2. 코드에서: `from llm.call_registry import generate` → `await generate("my_feature", prompt, system=...)` (sync는 `generate_sync`)
3. 실패(None) 폴백을 콜사이트에 마련할 것.

## 출력 예산 보장 (2026-08-30)

추론이 켜진 호출(DeepSeek `thinking: enabled`, GPT-5.6 reasoning)은 추론이 max_tokens를 다 먹으면 본문이 비었거나 잘린 채 200이 돌아온다. 이걸 그대로 돌려주면 추론은 비용만 쓰고 결과를 못 낸 것이다 — 1925 대회 번역에서 20k 예산이 통째로 추론에 들어가 본문 0자가 다섯 청크였다. 그래서 executor가 보장한다 (`_with_output_budget`):

- 길이 때문에 멈췄고(`stop_reason=max_tokens` / `finish_reason=length`) **본문이 비었거나 추론이 켜진 호출**이면 max_tokens를 2배로 늘려 다시 부른다. 최대 2단계, 상한 `OUTPUT_BUDGET_CAP=65536`.
- 상한까지 늘려도 완결되지 않으면 `OutputBudgetExhausted`를 던진다. `generate_sync`는 이를 실패로 기록하고(`status=error`) 경고 로그에 원인을 남긴 뒤 None을 돌려준다 — 조용한 빈 문자열이 아니다.
- 추론이 꺼진 호출이 본문을 낸 채 길이에 걸린 것은 호출부가 정한 길이 상한일 수 있으므로 경고만 남기고 그대로 돌려준다.
- 적용 executor: `deepseek_anthropic`, `deepseek`/`kimi`/`openai`(OpenAI 호환). 테스트: `tests/test_call_registry_output_budget.py`.

registry 항목의 max_tokens는 여전히 첫 시도 예산이다. 추론 호출부는 넉넉히 잡는 것이 맞고(archival은 48000), 보장 로직은 그 추정이 빗나간 청크를 구제하는 안전망이다.

## 주의

- **Kimi K3 제약**: temperature=1만 허용(그 외 400) — executor가 kimi provider에서는 temperature를 자동 생략한다. 추론 모델이라 max_tokens에 추론분 여유 필요. Kimi 경로 등재: `kimi_chat_model`(텔레그램/웹챗 티어, bot_config), `writer_main_kimi`(writer 메인 선택지) — 둘 다 model-only, 임포트 시점 해석.
- `writer/models.py`의 critic/research 별칭은 **임포트 시점**에 해석된다 — 바꾸면 writer 서비스 재시작 필요 (다른 executor 사이트는 핫리로드).
- KG graphiti 모델(kg_extraction_*, kg_embedding)도 KG 서비스 초기화 시점 해석 — 반영은 재시작 또는 KG unhealthy→재초기화 시.
- `vision_fallback`은 조회용 등재만 — 실제 모델은 bot_config 티어 시스템이 결정.
