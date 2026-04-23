# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Evolve Cyber-Lenin from a linear RAG chatbot into the most intelligent autonomous agent possible, based on CLAW (Planner + ReAct Executor + Critic triad) and other state-of-the-art agent architectures (Reflexion, LATS, Plan-and-Execute). The evolution is incremental — each phase adds a capability layer while preserving existing functionality.

## Development Documentation

- `dev_docs/project_state.md` — Current project state, architecture, capabilities, and limitations snapshot
- `dev_docs/knowledge_graph_design.md` — Knowledge graph 설계, 인프라, 제약사항, 변경이력
- `dev_docs/autonomous_project.md` — 자율 프로젝트 루프 (T0 pilot) — 티어 시스템, 상태기계, 스키마, 운영 CLI, 설계 결정
- `dev_docs/secret_management.md` — systemd-creds 기반 시크릿 관리 — 3-tier 분류, 서비스별 최소권한, 로테이션 플로우, 운영 CLI

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

## Testing
- 테스트 결과는 반드시 사용자가 직접 확인할 수 있도록 CLI에 전체 출력하거나 파일로 저장할 것. 요약만 하지 말 것.
