# Repository Instructions

Before making non-trivial changes, read `dev_docs/README.md` first, then read the relevant design document under `dev_docs/`.

Treat `dev_docs/` as the current developer documentation. If code changes alter architecture, API routes, service ownership, runtime config, tool allow-lists, KG behavior, or autonomous project behavior, update the relevant document in the same change.

Legacy `CLAUDE.md` is for Claude Code only. For Codex, follow this `AGENTS.md`.

## Important Docs

- `dev_docs/README.md` — documentation index and maintenance rules
- `dev_docs/project_state.md` — current service/runtime map
- `dev_docs/api_reference.md` — FastAPI routes
- `dev_docs/multi_agent_architecture.md` — agents and task runtime
- `dev_docs/llm_provider_architecture.md` — provider/model routing
- `dev_docs/tool_allowlist_current_state.md` — tool visibility boundaries
- `dev_docs/secret_management.md` — credential handling
- `dev_docs/knowledge_graph_design.md` — KG runtime boundary
- `dev_docs/autonomous_project.md` — autonomous loop design

## Local Rules

Use the project virtualenv before Python commands.
Prefer small, surgical changes.
Do not treat old handoff notes as source of truth.
