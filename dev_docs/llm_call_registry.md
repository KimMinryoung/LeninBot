# LLM 호출 레지스트리 (llm/call_registry.py)

최종 확인: 2026-09-23.

프로젝트의 LLM API 호출은 두 층으로 관리한다.

| 층 | 대상 | 설정 파일 | 반영 시점 |
|---|---|---|---|
| 에이전트 루프 | analyst/scout/diary/browser/... (claude_loop·openai_tool_loop 경유) | `config/agent_runtime.json` + `llm/runtime_profile.py` | 서비스 재시작 |
| **원샷 호출** | 요약·분류·번역·critic 등 단발 호출 | `config/llm_call_sites.json` + `llm/call_registry.py` | **핫리로드 (mtime)** |

## 원샷 레지스트리

기능 키별 JSON 항목: `{provider, model, temperature, max_tokens, timeout, json_mode, managed, env, note}`.

모델 해석 우선순위: **레거시 env(`env` 배열) > 제네릭 env `LLM_SITE_<KEY>_MODEL` > JSON > 콜사이트 기본값**.
선택값은 `tier:high|medium|low`(Claude/OpenAI의 `frontier` 포함)를 권장한다. 우선순위 적용 뒤 `provider_registry.current_text_model()`이 실제 ID로 해석하므로 예전 ID를 담은 env도 해당 티어의 현재 모델로 바뀐다. 특수 모델(임베딩·Jev 등)은 ID를 그대로 둔다.

`managed` 값:
- `executor` — `generate()/generate_sync()`가 직접 실행 (gemini/deepseek/openai/claude/kimi 지원, system·json_mode·timeout 옵션). provider가 `openrouter`/`typesafe`인 항목은 `decide()`가 실행한다(아래)
- `model-only` — 모델명만 여기서 조회, 실행은 자체 클라이언트 (KG graphiti, razvedchik, writer 경량 별칭)
- `external` — 정보 등재만 (vision 폴백처럼 실행 구조가 특수한 곳)

`generate()` 계열은 실패 시 `None`을 반환하며 콜사이트가 후속 처리를 결정한다.
System One 호출의 결과·오류 계약과 분류 재시도 정책은 아래를 따른다.

### System One 판정 호출 (`decide()`)

TypeSafe Jev는 텍스트를 생성하지 않고 typed 판정을 돌려주는 모델이라 `generate()`가 아닌
별도 진입점을 쓴다. 적용 범위와 장애 정책은 [Jev 연동](jev_system_one_adoption.md)을 따른다.

- 항목: `{"provider": "openrouter"|"typesafe", "model": "typesafe/jev-1.13"|"jev-1.13.0", "timeout", "kind": "system_one", "note"}`.
  `openrouter`는 `POST /api/alpha/decisions`, `typesafe`는 `POST /v1/systemone` — body·answers는 동일.
  현재 Jev 항목은 `typesafe`(직접 API, `jev-1.13.0` 고정)다; `openrouter`는 예비 경로로 코드와
  프록시 라우트에 남아 있다. `kind`는 표시용이며 실행은 provider가 결정한다.
- `decide_detailed(feature, state, questions, label=) → DecisionResult`, `decide_sync(...) → Decision | None`,
  `async decide(...)`. `state`는 문자열 또는 JSON 구조(질문에 필요한 필드만), `questions`는
  `{key: {"type": "noul"|"choice"|"score", "instructions": str|dict|list, "criteria": dict|list}}`.
  choice는 2..255 옵션 dict, score는 2..10 단계 list. 중첩 criteria 값은 OpenRouter가 문자열만 받으므로
  JSON 문자열로 직렬화해 보낸다. 구조화된 instructions도 같은 방식으로 직렬화하며 `fan_out`에서도 지원한다.
- `Decision.noul(key)/choice(key)/score(key)/confidence(key)`는 없는 키에 `None`, `probabilities(key)`는 `{}`.
  confidence는 분포의 집중도에서 계산한 값이며 선택지의 확률이나 실제 정답률과 동일하지 않다.
  응답의 실제 `model`(예: `typesafe/jev-1.13-20260917`)과 `usage`, 지연, 비용을 담는다.
- 감사: `check_llm_call` → 호출 → `record_llm_call`. OpenRouter가 `usage.cost`를 주면 그 값을, 없으면
  gateway의 `SYSTEM_ONE_PRICING`(입력 $0.042/M, 출력 0)으로 추정. 실패는 status=error 행 하나.
- 실패(`4xx/5xx`, 전송 오류, 질문 형식 오류, 비-System One 항목)는 예외 없이 `decision=None`/`error_kind`로
  돌아온다. 이후 처리는 콜사이트별 정책을 따른다. CommuLingo editor 분류는 초안을 보존하고 재시도하며
  LLM 분류로 대체하지 않는다. 429·5xx·연결 거부/끊김은 한 번 더 시도한다(Retry-After
  존중, 최대 2초 대기; 항목 `retries`로 조정, 기본 1). 읽기 타임아웃은 재시도하지 않는다 — 첫 요청이 이미 처리됐을
  수 있고 게이트가 단계를 두 배로 세우게 된다. `async decide()`의 바깥 timeout은 항목의 시도 수 전체를 덮는다.
- 현재 등록 항목은 `config/llm_call_sites.json`을 따른다. 게이트의 `enabled`/`enforce`/`thresholds`는 핫리로드된다.
- 스모크: `venv/bin/python scripts/smoke_jev.py` (항목 `system_one_smoke`, 실제 외부 호출).

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
venv/bin/python scripts/llm_registry_cli.py add <feature> --provider gemini --tier low
python scripts/llm_registry_cli.py agent-show <agent>            # 에이전트 루프 설정 조회
python scripts/llm_registry_cli.py agent-set <agent> <key> <value>
#   key: provider|model|budget_usd|max_rounds — agent_runtime.json 수정, 핫리로드
#   저장 후 런타임 로더로 재검증, 실패 시 자동 원복. model 별칭 오타는 경고
```

## 새 호출부 등록 방법

1. `venv/bin/python scripts/llm_registry_cli.py add my_feature --provider gemini --tier low --temperature 0 --max-tokens 256 --note "설명"`
2. 코드에서: `from llm.call_registry import generate` → `await generate("my_feature", prompt, system=...)` (sync는 `generate_sync`)
3. 실패(None) 폴백을 콜사이트에 마련할 것.

## 출력 예산 보장 (2026-08-30)

추론이 켜진 호출(DeepSeek `thinking: enabled`, GPT-5.6/GPT-6 reasoning)은 추론이 max_tokens를 다 먹으면 본문이 비었거나 잘린 채 200이 돌아온다. 이걸 그대로 돌려주면 추론은 비용만 쓰고 결과를 못 낸 것이다 — 1925 대회 번역에서 20k 예산이 통째로 추론에 들어가 본문 0자가 다섯 청크였다. 그래서 executor가 보장한다 (`_with_output_budget`):

- 길이 때문에 멈췄고(`stop_reason=max_tokens` / `finish_reason=length`) **본문이 비었거나 추론이 켜진 호출**이면 max_tokens를 2배로 늘려 다시 부른다. 최대 2단계, 상한 `OUTPUT_BUDGET_CAP=65536`.
- 상한까지 늘려도 완결되지 않으면 `OutputBudgetExhausted`를 던진다. `generate_sync`는 이를 실패로 기록하고(`status=error`) 경고 로그에 원인을 남긴 뒤 None을 돌려준다 — 조용한 빈 문자열이 아니다.
- 추론이 꺼진 호출이 본문을 낸 채 길이에 걸린 것은 호출부가 정한 길이 상한일 수 있으므로 경고만 남기고 그대로 돌려준다.
- 적용 executor: `deepseek_anthropic`, `deepseek`/`kimi`/`openai`(OpenAI 호환). 테스트: `tests/test_call_registry_output_budget.py`.

registry 항목의 max_tokens는 여전히 첫 시도 예산이다. 추론 호출부는 넉넉히 잡는 것이 맞고(archival은 48000), 보장 로직은 그 추정이 빗나간 청크를 구제하는 안전망이다.

## 주의

- **Kimi K3 제약**: temperature=1만 허용(그 외 400) — executor가 kimi provider에서는 temperature를 자동 생략한다. 추론 모델이라 max_tokens에 추론분 여유 필요. Kimi 경로 등재: `kimi_chat_model`(텔레그램/웹챗 티어, bot_config), `writer_main_kimi`(writer 메인 선택지) — 둘 다 model-only, 임포트 시점 해석.
- `writer/models.py`의 critic/research 별칭은 **임포트 시점**에 해석된다 — 바꾸면 writer 서비스 재시작 필요 (다른 executor 사이트는 핫리로드).
- KG graphiti 모델(kg_extraction_*, kg_embedding)도 KG 서비스 초기화 시점 해석 — 반영은 재시작 또는 KG unhealthy→재초기화 시.
- `kg_document_extraction` (2026-09-03): 발행 문서 → fact 추출 (gemini-3.5-flash-lite, json_mode). `KG_DOC_EXTRACT_LLM=1`일 때만 `jobs/kg_sync_documents`·리서치 발행 훅이 호출한다. 백필 ≈$1 일회, 이후 월 <$0.05.
- `vision_fallback`은 조회용 등재만 — 실제 모델은 bot_config 티어 시스템이 결정.

2026-09-10부터 DeepSeek executor 모델 ID는 `deepseek-flash`(V4.1 Flash)다. Writer의 model-only 기본 선택은 `deepseek_flash`이며 옛 Pro 선택 호환은 Writer 입력 경계에서 처리한다.

## 역할극 자동 판정

`roleplay_scene_adjudication`은 Jev 고정 선택지 분류이며, `roleplay_duration_estimate`는 단일 사건 소요 분만 생성하는 별도 호출이다. 수치 계산·저장은 코드가 수행한다. 계약과 실패 정책은 [roleplay_jev.md](roleplay_jev.md)를 따른다.

`roleplay_scene_consistency`는 역할극의 초안과 질적 정산 결과의 서술 모순 검토용이다.
기존 연기 모델과 동일한 DeepSeek Anthropic endpoint를 강제하며 전체 메모 대신 이번 변경만 전달한다.
현재 enabled=true이며 비활성화하면 실행은 거절한다. 사건·활동 분류와 계산은 맡기지 않는다.
