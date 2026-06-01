# Developer Documentation Index

이 디렉터리는 현재 코드와 운영 구조를 설명하는 문서만 유지한다. 완료된 인수인계, 과거 리팩터링 계획, 실제 라우트와 맞지 않는 API 초안은 보존하지 않는다.

## Core Runtime

| 문서 | 용도 |
|---|---|
| `project_state.md` | 전체 서비스, 데이터 저장소, systemd 단위, 운영 진입점 |
| `multi_agent_architecture.md` | Telegram orchestrator, delegated agents, task queue, Redis/DB context |
| `llm_provider_architecture.md` | Claude/OpenAI/DeepSeek/local provider 라우팅과 모델 티어 |
| `tool_allowlist_current_state.md` | 전역 도구 레지스트리와 채널/에이전트별 도구 가시성 |
| `mcp_gateway.md` | Codex/Claude Code 같은 개발용 MCP client에 노출하는 읽기 중심 gateway |
| `hot_reload_prompts.md` | 런타임 prompt overlay와 재시작 필요 경계 |

## Domain Subsystems

| 문서 | 용도 |
|---|---|
| `api_reference.md` | `api.py` FastAPI 라우트와 인증 |
| `secret_management.md` | systemd credential 기반 시크릿 로딩 |
| `knowledge_graph_design.md` | Neo4j/Graphiti KG 런타임 구조 |
| `knowledge_graph_schema.md` | KG typed entity/edge schema |
| `vector_corpus_reingestion_handoff.md` | Windows GPU PC에서 vector corpus 재등록 시 필요한 metadata/chunking 인수인계 |
| `mission_state_machine.md` | Telegram mission context lifecycle |
| `autonomous_project.md` | hourly autonomous project loop |
| `skill_import_design.md` | 외부 skill import/conversion 설계 |
| `x402_design.md` | Base USDC x402 payment demo/runtime |

## 문서 유지 원칙

- 코드의 현재 ownership을 먼저 확인한다. 주요 진입점은 `api.py`, `telegram/bot.py`, `telegram/tasks.py`, `agents/`, `runtime_tools/`, `bot_config.py`, `autonomous_project.py`, `kg_runtime/`, `graph_memory/`이다.
- 계획 문서는 구현이 끝나면 완료 기록으로 남기지 말고, 해당 주제의 현재 설계 문서에 흡수한다.
- 날짜가 붙은 handoff 문서는 장기 보존하지 않는다. 필요한 운영 지식만 주제별 문서로 옮긴다.
- 실제 라우트, config key, systemd unit, tool name을 쓸 때는 코드에서 다시 확인한다.
