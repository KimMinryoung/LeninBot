# LLM Provider Architecture — 2026-03-28

## 개요

Claude API 사용량 과다로 인한 불안정성에 대응하여, OpenAI API를 대체 provider로 사용할 수 있는 어댑터 패턴을 구현함.
런타임에 `/provider` 커맨드로 Claude↔OpenAI를 즉시 전환 가능.

---

## Provider 구조

```
telegram_bot.py
  _chat_with_tools()  ← 단일 진입점 (대화 + 태스크)
       │
       ├─ provider == "claude"
       │    └─ claude_loop.chat_with_tools(client=_claude, ...)
       │         └─ Anthropic Messages API (client.messages.create)
       │
       ├─ provider == "openai"
       │    └─ openai_tool_loop.chat_with_tools(client=_openai_client, ...)
       │         └─ OpenAI Chat Completions API (client.chat.completions.create)
       │
       └─ agent.provider == "moon"  (태스크 워커에서만)
            └─ openai_tool_loop.chat_with_tools(base_url=MOON_BASE, ...)
                 └─ httpx → 로컬 LLM (llama-server, Qwen 등)
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `claude_loop.py` | Claude 전용 tool-use 루프. Anthropic Messages API. 변경 없음. |
| `openai_tool_loop.py` | OpenAI + 로컬 LLM 겸용 tool-use 루프. SDK mode / httpx mode 이중 지원. |
| `telegram_bot.py` | provider dispatch, 모델 티어 해석, `/provider` `/fallback` `/config` 커맨드 |
| `config.json` | `provider`, `chat_model`, `task_model` 등 런타임 설정 영속화 (.gitignore) |

## openai_tool_loop.py — 이중 모드

```python
chat_with_tools(
    messages,
    client=AsyncOpenAI(),  # SDK mode (OpenAI 공식 API)
    # 또는
    base_url="http://...",  # httpx mode (로컬 LLM)
    model="gpt-5.4",
    tools=tool_defs,        # Anthropic 포맷 → 내부 자동 변환
    ...
)
```

- **SDK mode** (`client` 전달): `openai.AsyncOpenAI` 사용. 비용 추적 + 예산 제한 적용.
- **httpx mode** (`base_url` 전달): 로컬 LLM용. 비용 무시. Moon PC 호환.
- 두 모드 모두 동일한 인터페이스, 동일한 에러 복구 로직 공유.

## 모델 티어 시스템

config에 구체적 모델명 대신 `high`/`medium`/`low` 티어를 저장.
provider에 따라 자동 매핑:

| 티어 | Claude | OpenAI |
|------|--------|--------|
| high | opus (claude-opus-4-6) | gpt-5.4 |
| medium | sonnet (claude-sonnet-4-6) | gpt-5.4-mini |
| low | haiku (claude-haiku-4-5) | gpt-5.4-nano |

`/config` 패널에서는 `high (gpt-5.4)` 형태로 티어 + 실제 모델명 동시 표시.

## Tool 포맷 변환

Anthropic과 OpenAI의 tool 정의 포맷이 다름. `openai_tool_loop.py`의 `_convert_tools()`가 자동 변환:

```
Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    ↓ _convert_tool_anthropic_to_openai()
OpenAI:    {"type": "function", "function": {"name": "...", "parameters": {...}}}
```

- `cache_control` 등 Anthropic 전용 키 자동 제거
- `additionalProperties: false`인 도구는 `strict: true` 자동 활성화

## Chat Completions API 선택 이유

OpenAI는 Responses API(`/v1/responses`)와 Chat Completions API(`/v1/chat/completions`) 두 가지를 제공.
**Chat Completions API를 선택한 이유:**
- 업계 표준 — OpenAI, Claude, llama-server, vLLM, Ollama 모두 호환
- 로컬 LLM(Moon PC)과 동일한 코드 경로 공유 가능
- Responses API는 OpenAI 전용 (provider 독립성 상실)

## 에러 복구 메커니즘 (openai_tool_loop.py)

claude_loop.py의 견고한 복구 전략을 OpenAI 어댑터에 포팅:

| 메커니즘 | 설명 |
|----------|------|
| `_normalize_messages()` | 외부 히스토리에서 tool 프로토콜 블록 제거. 텍스트 전용 시작. |
| `_validate_tool_results()` | 매 라운드 tool_call → tool result 매칭 검증. 누락 시 자동 주입. |
| `_strip_tool_protocol()` | Nuclear recovery. 모든 tool 프로토콜을 한국어 라벨 텍스트로 변환. |
| `_build_tc_list()` | Malformed tool_call (id/name 누락, arguments JSON 파싱 실패) 스킵. |
| `_dump_messages_for_debug()` | API 에러 시 메시지 구조 로깅. |
| Auto-recovery cascade | API 에러 → strip → 재시도 1회 (parallel_tool_calls=False로 복잡도 감소) |
| Safety net | 모든 tool_call에 result가 있는지 검증, 누락 시 synthetic error 주입 |
| Forced final response | 예산/라운드 초과 시: preflight 검증 → 최종 API 호출 → 실패 시 strip 후 재시도 |
| Escalation hint | 강제 종료 시 `[CONTINUE_TASK: ...]` 패턴 안내 |
| Result truncation | 50KB 초과 tool result 자동 잘림 |

## Vision (이미지 분석) provider 분기

`telegram_bot.py`의 `handle_photo()`에서 provider별 분기:
- **Claude**: `type: "image"`, `source.type: "base64"` (Anthropic Vision 포맷)
- **OpenAI**: `type: "image_url"`, `url: "data:{media_type};base64,..."` (OpenAI Vision 포맷)

## 경량 보조 호출 (요약/압축/반성)

`telegram_bot.py`의 3곳에서 `_claude.messages.create(model=haiku)`를 직접 호출:
- 대화 청크 요약 (line ~793)
- 히스토리 압축 (line ~853)
- 경험 반성 (line ~2121)

이들은 provider 분기를 하지 않음. 로컬 LLM 우선 시도 후 Claude light(haiku) fallback.
Claude API 키가 있는 한 provider 설정과 무관하게 동작.

## 가격 정보

`openai_tool_loop.py`의 `OPENAI_PRICING` 상수 (2026-03-28 기준):

| 모델 | Input/1M | Output/1M | Cached Input/1M |
|------|----------|-----------|-----------------|
| gpt-5.4 | $2.50 | $15.00 | $0.25 |
| gpt-5.4-mini | $0.75 | $4.50 | $0.075 |
| gpt-5.4-nano | $0.20 | $1.25 | $0.02 |

`claude_loop.py`의 `PRICING` (Sonnet 4.6 기준):
- Input $3.00, Output $15.00, Cache creation $3.75, Cache read $0.30

## 텔레그램 커맨드

| 커맨드 | 기능 |
|--------|------|
| `/provider` | Claude↔OpenAI 토글. 모델 티어는 유지. |
| `/fallback` | high↔low 티어 토글. provider 무관. |
| `/config` | 패널에서 provider, 모델 티어, 예산, 라운드 수 개별 설정. |

## 알려진 제한사항

1. OpenAI provider에서 strict mode 미사용 — 25개 도구 모두 optional 파라미터 있어 strict 요건 불충족. non-strict 모드로 동작. 파라미터 hallucination 가능성은 있으나 실사용에서 큰 문제 없음.
2. 경량 보조 호출이 항상 Claude API 사용 — provider="openai"여도 요약/압축/반성은 Claude haiku. Claude 키 없으면 로컬 LLM만 사용 가능.
