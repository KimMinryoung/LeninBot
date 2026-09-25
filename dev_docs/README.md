# Developer Documentation Index

이 디렉터리는 현재 코드와 운영 구조를 설명하는 문서만 유지한다. 완료된 인수인계, 과거 리팩터링 계획, 실제 라우트와 맞지 않는 API 초안은 보존하지 않는다.

## 시작 경로

새 작업은 먼저 [전체 런타임·요청 경로](project_state.md)에서 서비스 소유권과 데이터 흐름을 확인한다. 아래 표에서 작업 주제에 맞는 상세 문서로 이동한 뒤, 실제 코드·설정과 대조한다. 문서는 설계의 길잡이이며 실행 코드와 설치된 설정의 현재 상태를 대신하지 않는다. 프런트엔드와 Nginx 설정의 원본은 별도 저장소 `/home/grass/frontend`에 있다. 이 저장소는 프런트엔드 없이 단독 클론으로도 편집·단위 테스트가 가능하다. 클라우드 세션 설정과 범위는 `AGENTS.md`의 Cloud sessions 절을 따른다.

| 작업 주제 | 먼저 볼 문서 |
|---|---|
| 공개 사이트, Nginx, 프런트엔드 프록시, API 경로 | [전체 구조](project_state.md#공개-진입점과-프록시-경계) → [API 경계·인증](api_reference.md) |
| 모델 선택·호출·키 주입 프록시 | [Provider 구조](llm_provider_architecture.md) → [LLM 게이트웨이](llm_gateway.md) → [원샷 레지스트리](llm_call_registry.md) |
| 검색·본문 추출·유료 예산 | [웹 검색 게이트웨이](web_research.md) → [도구 실행 경계](tool_gateway.md) |
| Telegram 위임·작업·도구 권한 | [Multi-agent](multi_agent_architecture.md) → [도구 가시성](tool_allowlist_current_state.md) → [실행·보안 게이트웨이](tool_gateway.md) |
| DB·백업·스탠바이·장애 알림 | [PostgreSQL 운영](db_migration_plan.md) → [스탠바이](standby_operations.md) → [감시](monitoring.md) |
| KG·코퍼스 | [KG 런타임](knowledge_graph_design.md) → [KG 스키마](knowledge_graph_schema.md) / [코퍼스 재등록](vector_corpus_reingestion.md) |
| CommuLingo 편집·자동 파이프라인 | [편집 저장 계약](commulingo_editorial.md) → [파이프라인](commulingo_pipeline.md) |
| 역할극 | [인물 모델](roleplay_persona_design.md) → [장면 정산](roleplay_postdraft.md) → [게임 규칙](roleplay_game_balance.md) |
| 개발용 MCP 조회 | [MCP 게이트웨이](mcp_gateway.md) |

그 밖의 작업은 아래 전체 목록에서 찾는다. 코드 변경 뒤에는 해당 문서의 소유권·설정·운영 경계를 갱신하고, 문서 링크와 관련 검증 명령을 확인한다.

## Core Runtime

| 문서 | 용도 |
|---|---|
| `project_state.md` | 전체 서비스·공개 요청 경로(Nginx→프런트엔드→API)·내부 프록시·데이터 저장소·systemd 단위 |
| `multi_agent_architecture.md` | Telegram orchestrator, delegated agents, task queue, Redis/DB context |
| `agent_tool_matrix.md` | Specialist agent별 실행 가능 tool 목록 |
| `llm_provider_architecture.md` | Claude/OpenAI/DeepSeek/local provider 라우팅과 모델 티어 |
| `llm_gateway.md` | 모든 LLM 호출의 정책·감사 seam과 key-injection proxy, 로컬 운영 오버라이드 |
| `tool_allowlist_current_state.md` | 전역 도구 레지스트리와 채널/에이전트별 도구 가시성 |
| `tool_gateway.md` | runtime tool visibility, dispatch, security/audit facade |
| `security_gateway.md` | 실행 시점 인자 검증·권한·rate limit·idempotency·감사 |
| `web_research.md` | 검색·Extract 전용 게이트웨이, 키 격리, 공용 일일 예산과 사용량 집계 |
| `llm_call_registry.md` | 원샷 호출 등록과 실행 정책·핫리로드 |
| `jev_system_one_adoption.md` | Jev 분류·인용 판정, 장애 정책, 라우팅·KG 적용 범위와 평가 한계 |
| `mcp_gateway.md` | Codex/Claude Code 같은 개발용 MCP client에 노출하는 읽기 중심 gateway |
| `hot_reload_prompts.md` | 런타임 prompt overlay와 재시작 필요 경계 |
| `roleplay_game_balance.md` | 역할극 회복 이벤트·활동 비용·반복 보상 제한·극단값 완화·질병 상태 |
| `roleplay_jev.md` | 예조프 JEV 자동 분류·LLM 시간 추정·코드 수치 계산 |
| `roleplay_progression.md` | 예조프 장면 시간·고립 부담·예정 사건·자발적 행동 |
| `roleplay_postdraft.md` | 초안 생성 후 사건·시간 정산, 임시 기록·선택적 서술 검토의 활성화 상태 |
| `roleplay_persona_design.md` | 예조프 역할극의 행동 명세·자료 판독·검토 기준 |

## Domain Subsystems

| 문서 | 용도 |
|---|---|
| `mail_briefing.md` | 메일 원문 캐시, 본문 확인 이력과 Telegram 브리핑 전달 기록 |
| `writer_runtime.md` | 개인 소설 작업 공간의 도구·문맥·캐시·퇴고 |
| `api_reference.md` | `services/api.py` FastAPI 라우트와 인증 |
| `secret_management.md` | systemd credential 기반 시크릿 로딩 |
| `db_migration_plan.md` | 현재 PostgreSQL 구성·백업·복구·쓰기 가드와 이전 후 미확인 항목 (파일명은 기존 참조 유지) |
| `standby_operations.md` | 스트리밍 스탠바이 활용법, 승격 런북, 재시드 절차 |
| `monitoring.md` | 감시·알림 체계 — 외부 워치독, 복제 점검, 알림 채널, 사각지대 |
| `knowledge_graph_design.md` | Neo4j/Graphiti KG 런타임 구조 |
| `knowledge_graph_schema.md` | KG typed entity/edge schema |
| `translation_pipeline.md` | 사료(RU/ZH/EN/DE/FR/IT→KO)·사이트(KO→EN) 공통 실행, 검증·캐시·TM, 원문 최신성, DB 적용 상태와 평가 |
| `vector_corpus_reingestion.md` | 코퍼스 manifest·metadata·chunking·재등록과 감사 절차 |
| `mission_state_machine.md` | Telegram mission context lifecycle |
| `commulingo_editorial.md` | 공통 편집 저장·revision·출처·직접 도구의 제안과 검토 계약 |
| `commulingo_pipeline.md` | 인물·용어 editor 운영, 원문 캐시·부분 수정·독립 검토·자동 공개·복구 |
| `autonomous_project.md` | hourly autonomous project loop |
| `skill_import_design.md` | 외부 skill import/conversion 설계 |
| `x402_design.md` | Base USDC x402 결제 데모의 실행 흐름·와이어 형식·감사 경계 |

## 문서 유지 원칙

- 코드의 현재 ownership을 먼저 확인한다. 주요 진입점은 `services/api.py`, `telegram/bot.py`, `telegram/tasks.py`, `agents/`, `runtime_tools/`, `bot_config.py`, `jobs/autonomous_project.py`, `kg_runtime/`, `graph_memory/`이다.
- 계획 문서는 구현이 끝나면 완료 기록으로 남기지 말고, 해당 주제의 현재 설계 문서에 흡수한다. 미완료 작업도 별도 로드맵보다 해당 경계의 문서에 둔다.
- 날짜가 붙은 handoff 문서는 장기 보존하지 않는다. 필요한 운영 지식만 주제별 문서로 옮긴다.
- 메모리는 사용자 선호·현재 미완료 작업·문서 진입점만 간결하게 유지한다. 완료 배포 로그와 코드에서 조회할 모델·요율·행수는 중복 저장하지 않는다.
- 코드 기본값, 로컬 설정, 당시 운영 관찰을 구분한다. 재확인하지 않은 해지·활성화·승인 대기를 현재 사실처럼 갱신하지 않는다.
- 실제 라우트, config key, systemd unit, tool name을 쓸 때는 코드에서 다시 확인한다.
- Python 변경을 운영에 적용하기 전에 `scripts/run_unit_tests.sh`를 실행한다. 이 명령은 테스트 전에 `scripts/check_python_names.py`로 저장소의 Python 파일에서 미정의 이름과 문법 오류를 검사한다. 의존성은 `requirements.txt`의 `pyflakes`다. import 대상의 실제 존재 여부와 실행 중 동적 이름 해석은 이 검사 범위 밖이므로, 변경한 서비스의 모듈 import와 관련 실행 경로도 확인한다.

## Top-level Code Layout

| 경로 | 역할 |
|---|---|
| `services/` | FastAPI·embedding 서비스 진입점과 web/A2A/email/image 서비스 구현 |
| `jobs/` | systemd timer나 수동 명령이 실행하는 autonomous/experience 작업 |
| `llm/` | provider adapter, 공용 tool-loop engine, gateway, model/runtime profile |
| `translation_runtime/` | 사료·사이트 번역 공통 실행, 구조 검증, 원자적 파일 저장 |
| `telegram/`, `agents/`, `runtime_tools/` | Telegram orchestration, agent specs, runtime tool implementations |
| `scripts/`, `deploy/`, `systemd/` | 운영·검증 스크립트, 배포 자산, unit 원본 |

프로젝트 루트에는 저장소 메타데이터, 환경·dependency 예제, 배포 진입 스크립트와 아직 별도 도메인 패키지가 없는 공용 compatibility module만 둔다.
