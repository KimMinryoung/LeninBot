# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Evolve Cyber-Lenin from a linear RAG chatbot into the most intelligent autonomous agent possible, based on CLAW (Planner + ReAct Executor + Critic triad) and other state-of-the-art agent architectures (Reflexion, LATS, Plan-and-Execute). The evolution is incremental — each phase adds a capability layer while preserving existing functionality.

## Development Documentation

- `temp_dev/project_state.md` — Current project state, architecture, capabilities, and limitations snapshot
- `temp_dev/agent_evolution_plan.md` — Full phased plan for agent evolution (Phase 0-6)
- `temp_dev/to_agent_AI.txt` — Original CLAW architecture discussion and Gemini's advice
- `temp_dev/knowledge_graph_design.md` — Knowledge graph 설계, 인프라, 제약사항, 변경이력

Always consult `temp_dev/agent_evolution_plan.md` before starting new work to understand the current phase and what comes next. Update these documents as phases are completed.

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
- Always activate the virtual environment before running Python commands: `source .venv/Scripts/activate`

## Caution
- When running bash commands on Windows, use /dev/null for output redirection, not nul.