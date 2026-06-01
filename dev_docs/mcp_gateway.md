# MCP Gateway

최종 확인 기준: 2026-06-01 코드 트리.

`mcp_gateway.server` is an inbound MCP server for local developer/operator clients. Its purpose is to give tools like Codex or Claude Code a typed, narrow path into project state without exposing raw shell, broad DB credentials, filesystem writes, publishing, service restart, payment, or send capabilities.

The current gateway is a minimal stdio JSON-RPC MCP implementation using standard `Content-Length` framing, with newline-delimited JSON retained for manual probes. It supports `initialize`, `tools/list`, `tools/call`, and `ping`. It deliberately avoids adding a new Python package dependency.

## Entrypoint

Run on demand from the project root:

```bash
venv/bin/python -m mcp_gateway.server
```

For humans, use the wrapper script:

```bash
scripts/mcp-gateway --help
scripts/mcp-gateway --list-tools
scripts/mcp-gateway --profile operator --list-tools
```

The default profile is `inspect`. Set `MCP_GATEWAY_PROFILE=operator` or pass `--profile operator` only for trusted local operator sessions that need read-only SQL diagnostics. The old `readonly` name is accepted as a compatibility alias for `inspect`, but new client configs should use `inspect`.

## Discovery

Codex sessions in this repository should discover the gateway through `AGENTS.md`, which points to this document and the `scripts/mcp-gateway` wrapper. The LeninBot `programmer` agent also embeds the same guidance in `agents/programmer.py` and `codex_exec_loop.py`, so Codex CLI tasks delegated by the multi-agent runtime are told that the gateway exists.

This discovery text does not by itself register an MCP server inside every external client. Codex CLI, Claude Code, or another MCP client still needs a client-side MCP server config if it should connect to the gateway as a live MCP server. Use the snippets below for that configuration.

## Codex Registration

Register the inspect profile for the current Unix user:

```bash
codex mcp add leninbot-inspect -- /home/grass/leninbot/scripts/mcp-gateway
```

Register the operator profile only for trusted users that should have guarded read-only SQL:

```bash
codex mcp add leninbot-operator -- /home/grass/leninbot/scripts/mcp-gateway --profile operator
```

The LeninBot programmer path runs under the `grass` service user, so register the same servers for that account when Codex CLI tasks delegated by the multi-agent runtime should see them:

```bash
sudo -u grass codex mcp add leninbot-inspect -- /home/grass/leninbot/scripts/mcp-gateway
sudo -u grass codex mcp add leninbot-operator -- /home/grass/leninbot/scripts/mcp-gateway --profile operator
```

Verify a specific registration without dumping unrelated MCP config:

```bash
codex mcp get leninbot-inspect
codex mcp get leninbot-operator
sudo -u grass codex mcp get leninbot-inspect
sudo -u grass codex mcp get leninbot-operator
```

## Client Config

Use this command for normal Codex/Claude Code style development sessions:

```json
{
  "command": "/home/grass/leninbot/scripts/mcp-gateway",
  "args": []
}
```

Use the operator profile only when the client should be allowed to run guarded read-only SQL through `scripts/query-db`:

```json
{
  "command": "/home/grass/leninbot/scripts/mcp-gateway",
  "args": ["--profile", "operator"]
}
```

Human quick checks:

```bash
/home/grass/leninbot/scripts/mcp-gateway --list-tools
/home/grass/leninbot/scripts/mcp-gateway --profile operator --list-tools
```

## Profiles

| Profile | Purpose | Additional risk boundary |
|---|---|---|
| `inspect` | Developer context, docs, task/report/corpus status, selected runtime search/fetch | No raw SQL and no writes |
| `operator` | Local operator diagnostics and bounded maintenance | Adds `readonly_query_db`, `bounded_query_db`, and `kg_maintenance_run` |

Both profiles use explicit allow-lists in `mcp_gateway/policy.py`. The gateway never exports `runtime_tools.registry.TOOLS` wholesale.

## Exposed Tool Families

Gateway-local inspection tools:

- `gateway_status`
- `list_mcp_tools`
- `search_dev_docs`
- `get_project_runtime_summary`
- `list_recent_tasks`
- `get_task_status`
- `list_recent_task_reports`
- `corpus_metadata_audit`
- `kg_integrity_check`

Selected runtime tools:

- `vector_search`
- `knowledge_graph_search`
- `fetch_url`

Operator-only:

- `readonly_query_db`
- `bounded_query_db`
- `kg_maintenance_run`

`readonly_query_db` delegates to `scripts/query-db`, preserving the existing guard that allows only a single `SELECT`, `WITH`, `SHOW`, or `EXPLAIN` diagnostic and runs it in a read-only transaction through `scripts/psql-supabase`.

`bounded_query_db` delegates to the existing `runtime_tools.db` `query_db` handler. It allows one SQL statement, blocks `DROP` and `TRUNCATE`, and rolls back `UPDATE`/`DELETE` statements that would affect 10 or more rows. This is the MCP path for small operator-approved DB corrections. Use domain-specific tools such as `edit_content` outside MCP when cache invalidation or publication side effects matter.

## KG Maintenance

`kg_integrity_check` is available in the default `inspect` profile and runs `scripts/check_kg_integrity.py`. It is read-only and can optionally run an end-to-end KG search smoke query.

`kg_maintenance_run` is available only in the `operator` profile. It exposes bounded script-backed actions instead of arbitrary Cypher:

- `backup` -> `skills/kg-maintenance/scripts/backup_kg.py`
- `duplicate_candidates` -> `skills/kg-maintenance/scripts/dedup_entities.py`
- `merge_exact_name_dupes` -> `skills/kg-maintenance/scripts/merge_exact_name_dupes.py`
- `cleanup_orphans` -> `skills/kg-maintenance/scripts/cleanup_orphans.py`
- `classify_untyped` -> `scripts/classify_untyped_entities.py`
- `full_cleanup` -> `skills/kg-maintenance/scripts/run_cleanup.py`

Mutating actions default to dry-run. To apply changes, callers must pass `execute=true` and `confirm=APPLY_KG_MAINTENANCE`. The wrapper runs a KG backup first for direct mutating actions that do not already include backup in their own pipeline.

## Non-Goals

The first gateway version does not create tasks, edit files, publish content, send email/A2A messages, restart services, sign/pay transactions, or expose arbitrary unbounded SQL/Cypher. KG mutation is limited to the bounded `operator` maintenance scripts above.

## Verification

Run:

```bash
venv/bin/python scripts/smoke_mcp_gateway.py
```

The smoke test verifies profile separation, the `readonly` compatibility alias, KG maintenance visibility, forbidden tool absence, MCP schema conversion, CLI help/list output, and the stdio `initialize`/`tools/list`/`tools/call` path.
