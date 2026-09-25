# Repository notes

- Use the project virtualenv (`venv/`) for Python commands.
- When running as root, edit project files and run Git as `grass`.
- Start at `dev_docs/README.md` (its **시작 경로** section), then use `dev_docs/project_state.md` for the whole-system map and the task-specific docs linked by the index. Read the relevant docs and update them when changing the behavior they describe. Verify current behavior against code and deployed config where relevant.
- For project inspection, prefer `scripts/mcp-gateway --list-tools` and its default `inspect` profile. The `operator` profile supports guarded read-only SQL through `scripts/query-db`. See `dev_docs/mcp_gateway.md`.
- `skills/` contains Leninbot runtime skills, including operational scripts; these are separate from Codex development skills.

## Cloud sessions and lone clones

This repository works without the server: no frontend checkout, `.env`, DB, Redis, Neo4j or secrets. In a Claude Code cloud session the SessionStart hook in `.claude/settings.json` runs `scripts/cloud_setup.sh` (only when `CLAUDE_CODE_REMOTE=true`), which builds `venv/` from `requirements.lock.txt` without torch. Elsewhere, run it by hand.

- Available: editing code, configs and docs, and `scripts/run_unit_tests.sh` (the whole suite passes in a fresh clone).
- Missing frontend: the CommuLingo contract JSON falls back to `config/commulingo_contracts/` copies (`ops/paths.py`). Archival specs write paths as `${FRONTEND_DIR}/…`/`${PROJECT_ROOT}/…`; never add host paths.
- Not available: `scripts/mcp-gateway`, `scripts/query-db`, service restarts, archival or DB translation runs, anything needing the embedding server or live APIs. Verify those on the server.
- After changing production dependencies, refresh the lock with `scripts/freeze_requirements_lock.sh` on the server. After the frontend changes a contract JSON, run `scripts/sync_commulingo_contracts.py`.
