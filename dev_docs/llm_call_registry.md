# LLM 호출 레지스트리 (llm/call_registry.py)

최종 확인: 2026-07-17.

프로젝트의 LLM API 호출은 두 층으로 관리한다.

| 층 | 대상 | 설정 파일 | 반영 시점 |
|---|---|---|---|
| 에이전트 루프 | analyst/scout/diary/browser/... (claude_loop·openai_tool_loop 경유) | `config/agent_runtime.json` + `runtime_profile.py` | 서비스 재시작 |
| **원샷 호출** | 요약·분류·번역·critic 등 단발 호출 | `config/llm_call_sites.json` + `llm/call_registry.py` | **핫리로드 (mtime)** |

## 원샷 레지스트리

기능 키별 JSON 항목: `{provider, model, temperature, max_tokens, timeout, json_mode, managed, env, note}`.

모델 해석 우선순위: **레거시 env(`env` 배열) > 제네릭 env `LLM_SITE_<KEY>_MODEL` > JSON > 콜사이트 기본값**.

`managed` 값:
- `executor` — `generate()/generate_sync()`가 직접 실행 (gemini/deepseek/openai/claude 지원, system·json_mode·timeout 옵션)
- `model-only` — 모델명만 여기서 조회, 실행은 자체 클라이언트 (KG graphiti, razvedchik, writer 경량 별칭)
- `external` — 정보 등재만 (vision 폴백처럼 실행 구조가 특수한 곳)

실패 시 항상 `None` 반환 — 콜사이트가 자체 폴백(추출식 요약, 기본 라벨, 스킵)을 유지한다.

## 운영 CLI

```bash
python scripts/llm_registry_cli.py list              # 원샷 + 에이전트 루프 통합 조회
python scripts/llm_registry_cli.py show <feature>    # 원본 + env 오버라이드 반영 유효값
python scripts/llm_registry_cli.py set <feature> <key> <value>   # 핫리로드 반영
python scripts/llm_registry_cli.py add <feature> --provider gemini --model ... 
```

## 새 호출부 등록 방법

1. `scripts/llm_registry_cli.py add my_feature --provider gemini --model gemini-3.1-flash-lite --temperature 0 --max-tokens 256 --note "설명"`
2. 코드에서: `from llm.call_registry import generate` → `await generate("my_feature", prompt, system=...)` (sync는 `generate_sync`)
3. 실패(None) 폴백을 콜사이트에 마련할 것.

## 주의

- `writer/models.py`의 critic/research 별칭은 **임포트 시점**에 해석된다 — 바꾸면 writer 서비스 재시작 필요 (다른 executor 사이트는 핫리로드).
- KG graphiti 모델(kg_extraction_*, kg_embedding)도 KG 서비스 초기화 시점 해석 — 반영은 재시작 또는 KG unhealthy→재초기화 시.
- `vision_fallback`은 조회용 등재만 — 실제 모델은 bot_config 티어 시스템이 결정.
