# Repository notes

- Use the project virtualenv (`venv/`) for Python commands.
- When running as root, edit project files and run Git as `grass`.
- Start at `dev_docs/README.md` (its **시작 경로** section), then use `dev_docs/project_state.md` for the whole-system map and the task-specific docs linked by the index. Read the relevant docs and update them when changing the behavior they describe. Verify current behavior against code and deployed config where relevant.
- For project inspection, prefer `scripts/mcp-gateway --list-tools` and its default `inspect` profile. The `operator` profile supports guarded read-only SQL through `scripts/query-db`. See `dev_docs/mcp_gateway.md`.
- `skills/` contains Leninbot runtime skills, including operational scripts; these are separate from Codex development skills.
