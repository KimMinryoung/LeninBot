# Repository notes

- Use the project virtualenv (`venv/`) for Python commands.
- When running as root, edit project files and run Git as `grass`.
- Current developer documentation is indexed in `dev_docs/README.md`. Read the documents relevant to the task and update them when changing the behavior they describe. Verify current behavior against code.
- For project inspection, prefer `scripts/mcp-gateway --list-tools` and its default `inspect` profile. The `operator` profile supports guarded read-only SQL through `scripts/query-db`. See `dev_docs/mcp_gateway.md`.
- `skills/` contains Leninbot runtime skills, including operational scripts; these are separate from Codex development skills.
