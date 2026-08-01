# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Evolve Cyber-Lenin from a linear RAG chatbot into the most intelligent autonomous agent possible, based on CLAW (Planner + ReAct Executor + Critic triad) and other state-of-the-art agent architectures (Reflexion, LATS, Plan-and-Execute). The evolution is incremental — each phase adds a capability layer while preserving existing functionality.

## Development Documentation

- `dev_docs/project_state.md` — Current project state, architecture, capabilities, and limitations snapshot
- `dev_docs/knowledge_graph_design.md` — Knowledge graph 설계, 인프라, 제약사항, 변경이력
- `dev_docs/autonomous_project.md` — 자율 프로젝트 루프 (T0 pilot) — 티어 시스템, 상태기계, 스키마, 운영 CLI, 설계 결정
- `dev_docs/secret_management.md` — systemd-creds 기반 시크릿 관리 — 3-tier 분류, 서비스별 최소권한, 로테이션 플로우, 운영 CLI
- `dev_docs/db_migration_plan.md` — DB 인프라 — 로컬 leninbot-pg(pgvector/pg17, 메인+writer DB) 구성, R2 백업 3종, 스탠바이 구축 기록, Supabase 이탈 기록과 남은 단계(pgBackRest PITR)
- `dev_docs/security_gateway.md` — 툴 보안 게이트웨이 — execute_tool 단일 seam, 통합 정책/권한 통제, tool_audit_log 감사 로깅, shadow→enforce 롤아웃, 운영 CLI
- `dev_docs/agent_improvement_roadmap.md` — 에이전트 지능(CLAW/Reflexion/Plan-and-Execute) + 메모리 개선 단계별 로드맵
- `dev_docs/llm_call_registry.md` — LLM 원샷 호출 통합 레지스트리 (config/llm_call_sites.json, 핫리로드, 운영 CLI)
- `dev_docs/standby_operations.md` — 스트리밍 스탠바이(hel1) 활용법·승격 런북·재시드 절차
- `dev_docs/monitoring.md` — 감시·알림 체계 (VM 밖 워치독 Worker, 복제 점검, 알림 채널, 사각지대)

Update these documents as phases are completed.

## Project Overview


1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

4. Goal-Driven Execution
Define success criteria. Loop until verified.
Transform tasks into verifiable goals.


## Environment
- Always activate the virtual environment before running Python commands.

## CommuLingo registries (사전 분류값)
- 역할 범주, 용어 범주, 자동링크 예외, 폐기 id 리다이렉트는 모두 DB 테이블이다 (`commulingo_role_categories`, `commulingo_term_categories`, `commulingo_link_blocklist`, `commulingo_id_redirects`). 값을 바꾸거나 추가할 때는 **코드가 아니라 테이블**에 INSERT/UPDATE 한다 — 양쪽 저장소 어디도 커밋·배포가 필요 없다.
- `runtime_tools/commulingo_people.py`의 `_TERM_CATEGORY_FALLBACK`은 DB 접근 실패 시에만 쓰이는 사본이다. 여기를 고쳐도 정상 상태에서는 아무 효과가 없다. 폴백이 실제로 쓰이면 경고 로그가 남는다.
- `_NATIONALITY_CODES`는 예외적으로 코드에 남아 있다: 국기 SVG가 프론트엔드 이미지에 구워져 있어 어디에 두든 배포가 필요하고, 프론트엔드 `data/commulingo/flag-icons.js`와 쌍으로 관리된다. 국적을 추가하면 두 저장소 + SVG 자산 세 곳이다.
- 코드 사본이 테이블과 어긋났는지는 프론트엔드의 `scripts/check-commulingo-code-db-drift.js`가 검사한다.

## Testing
- 테스트 결과는 반드시 사용자가 직접 확인할 수 있도록 CLI에 전체 출력하거나 파일로 저장할 것. 요약만 하지 말 것.
