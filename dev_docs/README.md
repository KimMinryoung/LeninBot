# Developer Documentation Index

이 디렉터리는 현재 코드와 운영 구조를 설명하는 문서만 유지한다. 완료된 인수인계, 과거 리팩터링 계획, 실제 라우트와 맞지 않는 API 초안은 보존하지 않는다.

## Core Runtime

| 문서 | 용도 |
|---|---|
| `project_state.md` | 전체 서비스, 데이터 저장소, systemd 단위, 운영 진입점 |
| `multi_agent_architecture.md` | Telegram orchestrator, delegated agents, task queue, Redis/DB context |
| `agent_tool_matrix.md` | Specialist agent별 실행 가능 tool 목록 |
| `llm_provider_architecture.md` | Claude/OpenAI/DeepSeek/local provider 라우팅과 모델 티어 |
| `llm_gateway.md` | 모든 LLM 호출의 정책·감사 seam과 key-injection proxy, 로컬 운영 오버라이드 |
| `tool_allowlist_current_state.md` | 전역 도구 레지스트리와 채널/에이전트별 도구 가시성 |
| `tool_gateway.md` | runtime tool visibility, dispatch, security/audit facade |
| `tool_security_gateway_improvements.md` | tool/security gateway 보안 검토 결과와 우선순위별 개선 체크리스트 |
| `mcp_gateway.md` | Codex/Claude Code 같은 개발용 MCP client에 노출하는 읽기 중심 gateway |
| `hot_reload_prompts.md` | 런타임 prompt overlay와 재시작 필요 경계 |

## Domain Subsystems

| 문서 | 용도 |
|---|---|
| `api_reference.md` | `services/api.py` FastAPI 라우트와 인증 |
| `secret_management.md` | systemd credential 기반 시크릿 로딩 |
| `db_migration_plan.md` | DB 인프라 현황 (로컬 leninbot-pg 구성·백업 체계·스탠바이 구축 기록) + Supabase 이탈 기록과 남은 단계 |
| `standby_operations.md` | 스트리밍 스탠바이 활용법, 승격 런북, 재시드 절차 |
| `monitoring.md` | 감시·알림 체계 — 외부 워치독, 복제 점검, 알림 채널, 사각지대 |
| `knowledge_graph_design.md` | Neo4j/Graphiti KG 런타임 구조 |
| `knowledge_graph_schema.md` | KG typed entity/edge schema |
| `translation_pipeline.md` | 사료(RU·ZH→KO)·사이트(KO→EN) 번역 파이프라인, 번역 메모리, 검증 레이어 |
| `vector_corpus_reingestion_handoff.md` | Windows GPU PC에서 vector corpus 재등록 시 필요한 metadata/chunking 인수인계 |
| `mission_state_machine.md` | Telegram mission context lifecycle |
| `autonomous_project.md` | hourly autonomous project loop |
| `skill_import_design.md` | 외부 skill import/conversion 설계 |
| `x402_design.md` | Base USDC x402 payment demo/runtime |

## 문서 유지 원칙

- 코드의 현재 ownership을 먼저 확인한다. 주요 진입점은 `services/api.py`, `telegram/bot.py`, `telegram/tasks.py`, `agents/`, `runtime_tools/`, `bot_config.py`, `jobs/autonomous_project.py`, `kg_runtime/`, `graph_memory/`이다.
- 계획 문서는 구현이 끝나면 완료 기록으로 남기지 말고, 해당 주제의 현재 설계 문서에 흡수한다.
- 날짜가 붙은 handoff 문서는 장기 보존하지 않는다. 필요한 운영 지식만 주제별 문서로 옮긴다.
- 실제 라우트, config key, systemd unit, tool name을 쓸 때는 코드에서 다시 확인한다.

## Top-level Code Layout

| 경로 | 역할 |
|---|---|
| `services/` | FastAPI·embedding 서비스 진입점과 web/A2A/email/image 서비스 구현 |
| `jobs/` | systemd timer나 수동 명령이 실행하는 autonomous/experience 작업 |
| `llm/` | provider adapter, 공용 tool-loop engine, gateway, model/runtime profile |
| `telegram/`, `agents/`, `runtime_tools/` | Telegram orchestration, agent specs, runtime tool implementations |
| `scripts/`, `deploy/`, `systemd/` | 운영·검증 스크립트, 배포 자산, unit 원본 |

프로젝트 루트에는 저장소 메타데이터, 환경·dependency 예제, 배포 진입 스크립트와 아직 별도 도메인 패키지가 없는 공용 compatibility module만 둔다.
