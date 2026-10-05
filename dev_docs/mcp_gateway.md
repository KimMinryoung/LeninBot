# MCP Gateway

2026-10-05 stdio에서 KG 무결성·엔티티 검색과 operator 진단 5종을 읽기 전용 검증했다. KG mutation은 실행하지 않았다.

`mcp_gateway.server` is an inbound MCP server for local developer/operator clients. Its purpose is to give tools like Codex or Claude Code a typed, narrow path into project state without exposing raw shell, broad DB credentials, filesystem writes, publishing, service restart, payment, or send capabilities. The default `inspect` profile is read-only; `operator` additionally exposes read-only SQL diagnostics and guarded KG maintenance.

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

Codex sessions in this repository should discover the gateway through `AGENTS.md`, which points to this document and the `scripts/mcp-gateway` wrapper. The LeninBot `programmer` agent also embeds the same guidance in `agents/programmer.py` and `llm/codex_exec_loop.py`, so Codex CLI tasks delegated by the multi-agent runtime are told that the gateway exists.

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

Claude Code on the production server registers both profiles in local scope
(this project on this host only, not committed to the public repo):

```bash
claude mcp add --scope local leninbot-inspect -- /home/grass/leninbot/scripts/mcp-gateway
claude mcp add --scope local leninbot-operator -- /home/grass/leninbot/scripts/mcp-gateway --profile operator
```

Python-backed DB tools (`list_recent_tasks`, `get_task_status`, …) run outside
a systemd service and log in through `db.py`'s read-only `leninbot_ro`
fallback (`dev_docs/db_migration_plan.md`); without its password file they fail
with `Missing database configuration: DB_PASSWORD`.

Human quick checks:

```bash
/home/grass/leninbot/scripts/mcp-gateway --list-tools
/home/grass/leninbot/scripts/mcp-gateway --profile operator --list-tools
```

## Profiles

| Profile | Purpose | Additional risk boundary |
|---|---|---|
| `inspect` | Developer context, docs, task/report/corpus status, selected runtime search/fetch | No raw SQL and no writes |
| `operator` | Local operator diagnostics and bounded maintenance | Adds `readonly_query_db`, `tool_usage_report`, `trace_tool_run`, five operational diagnostics below, and `kg_maintenance_run`; only the repository owner UID may select this profile |

Both profiles use explicit allow-lists sourced from `tool_gateway.profiles` and exposed through compatibility names in `mcp_gateway/policy.py`. The gateway never exports `runtime_tools.registry.TOOLS` wholesale.

## Exposed Tool Families

Gateway-local inspection tools:

- `gateway_status`
- `list_mcp_tools`
- `list_runtime_tool_profiles` — lists runtime subjects and their allowed tools, including orchestrator, specialist agents, web/A2A/roleplay surfaces, and optionally MCP profiles
- `search_dev_docs`
- `get_project_runtime_summary`
- `list_recent_tasks`
- `get_task_status` — supports `field`, `offset`, and `max_chars` for paginating long `content`, `result`, or `tool_log` fields
- `list_recent_task_reports`
- `corpus_metadata_audit`
- `kg_integrity_check`

Selected runtime tools:

- `vector_search`
- `read_corpus_passage` — UUID-based same-document/layer context; default ±1, max ±3, total 20,000 chars
- `knowledge_graph_search`
- `fetch_url`

Operator-only:

- `readonly_query_db`
- `tool_usage_report`
- `trace_tool_run`
- `kg_maintenance_run`
- `service_health_snapshot`
- `get_service_logs`
- `get_effective_runtime_config`
- `pipeline_status`
- `usage_and_budget_report`

`readonly_query_db` delegates to `scripts/query-db`, preserving the existing guard that allows only a single `SELECT`, `WITH`, `SHOW`, or `EXPLAIN` diagnostic and runs it in a read-only transaction, logged in as the read-only role `leninbot_ro` (password file `~/.config/leninbot/db_ro_password` from `scripts/setup_readonly_db_role.sh`; no DB_PASSWORD, sudo or credstore).

Every `tools/call` checks the profile allow-list and the shared security gateway authorization policy. Allowed and denied calls both emit `tool_audit_log` rows with `interface=mcp`, the local UID and profile; arguments use the common recursive redactor. The `operator` profile is downgraded to `inspect` when the MCP server process is not running as the repository owner UID. This is an account boundary for the local stdio server, not a separate operator credential.

`bounded_query_db` is no longer exposed by MCP. Use a domain-specific operator command for corrections so its cache and publication side effects are handled explicitly.

## Audit diagnostics

`tool_usage_report(days=7, tool_name?, interface?, agent_name?)` aggregates the current
UTC window and the immediately preceding equal window. `days` must be 1–90 (the query
therefore reads at most 180 days for comparison). Each tool/interface/agent group has
call counts, status counts, latency p50/p95, and changes from the preceding window.
`runtime`, `test`, and `unknown` are separate partitions; a missing previous denominator
produces null percentage change. No metadata or labels are inferred for old rows.

`trace_tool_run(request_id=... OR task_id=..., limit=200, offset=0)` returns audit order
(`ts`, then row ID), descendant request IDs and parent links, status/decision/latency,
and current persisted Telegram task states for tasks on that page. It requires exactly
one identifier, at most 200 rows/page and offset 0–100,000. `next_offset` is null at the
end. Parent links are returned even when the parent's audit is absent. Arguments and
error excerpts are never returned; a hash of the redacted argument summary is used
internally for `same_call_ordinal`. `possible_repeat` is only a hint, since repeated
calls and redacted/truncated arguments cannot establish retry intent. `replay_suppressed`
identifies dispatcher deduplication. Missing task state is unknown. Audit delivery is
best effort, spool replay may duplicate rows, concurrent calls are not necessarily
serial, and a still-running trace may grow between pages.

Both tools are operator-only at the MCP profile, handler and authorization boundaries;
SQL runs in a read-only transaction with a 5-second timeout. They are absent from the
global runtime registry and public web/A2A surfaces. Runtime profile listings include
static registration/exposure/injection/retirement details, not session availability.

## KG Maintenance

`kg_integrity_check` is available in the default `inspect` profile and runs `scripts/check_kg_integrity.py`. It is read-only and can optionally run an end-to-end KG search smoke query.

`kg_maintenance_run` is available only in the `operator` profile. It exposes bounded script-backed actions instead of arbitrary Cypher:

- `backup` -> `skills/kg-maintenance/scripts/backup_kg.py`
- `duplicate_candidates` -> `skills/kg-maintenance/scripts/dedup_entities.py`
- `merge_exact_name_dupes` -> `skills/kg-maintenance/scripts/merge_exact_name_dupes.py`
- `cleanup_orphans` -> `skills/kg-maintenance/scripts/cleanup_orphans.py`
- `classify_untyped` -> `scripts/classify_untyped_entities.py`
- `full_cleanup` -> `skills/kg-maintenance/scripts/run_cleanup.py`

Mutating actions default to dry-run. To apply changes, callers must pass `execute=true` and `confirm=APPLY_KG_MAINTENANCE`. Before every mutating action, including `full_cleanup`, the wrapper runs a KG backup and checks that fresh entity, edge and mention JSON artifacts can be parsed. It refuses mutation if the backup or validation fails.

## Non-Goals

The first gateway version does not create tasks, edit files, publish content, send email/A2A messages, restart services, sign/pay transactions, or expose arbitrary unbounded SQL/Cypher. KG mutation is limited to the bounded `operator` maintenance scripts above.

## Verification

Run:

```bash
venv/bin/python scripts/smoke_mcp_gateway.py
```

The smoke test verifies profile separation, the `readonly` compatibility alias, runtime tool-profile listing, KG maintenance visibility, forbidden tool absence, MCP schema conversion, CLI help/list output, and the stdio `initialize`/`tools/list`/`tools/call` path.

## Operational diagnostics

All five tools are operator-only in profile, handler and shared authorization policy,
and absent from the runtime/public web registry. Calls use the usual MCP audit sink.
No retry, publication, translation execution, notifications or service restart occurs.

| Tool | Arguments | Observation |
|---|---|---|
| `service_health_snapshot` | optional `services` array (schema lists allowed names) | Systemd daemon properties plus PostgreSQL SELECT, Redis PING, Neo4j RETURN, embedding/API/proxy/web health endpoints |
| `get_service_logs` | required `service`; `hours_back=1` (1–168), optional `level`, `query`, `limit=200` (1–1000) | Allowlisted journal, masked text, 60,000-character row budget |
| `get_effective_runtime_config` | optional `surface`, `agent` | Allowlisted provider/model/budget/token/round fields and source chains; shared with `scripts/model_runtime_audit.py` |
| `pipeline_status` | `pipeline=all\|commulingo\|translation\|kg_sync`, `limit=20` (1–100) | Queue/lease/stall/error data, last systemd run, next timer run and recent failure log excerpts |
| `usage_and_budget_report` | `days=7` (1–30), optional `provider` | Official provider balances/costs, local estimates, separate UTC-today LLM/web budgets |

Health distinguishes `collection_status=error` (unknown observation) from an observed
`status=unhealthy` (e.g. inactive unit or HTTP 503). Each observation has UTC time and
elapsed milliseconds. Composite results retain successful components and set `partial`
on collection failures; these return MCP `isError=true` and audit `error`. A successfully
observed outage alone is a successful diagnostic. Systemd and HTTP probes time out after
5 seconds; journal after 15; SQL uses read-only transactions and a 5-second statement
timeout. Collector deadlines bound the response while underlying reads retain their own
timeouts. MCP defaults libpq connection timeout to 5 seconds.

Log `level` means journal severity threshold (error includes more severe priorities).
`query` is a case-insensitive literal substring within the latest `limit` records,
not an unbounded historical search. Empty results are successful; inaccessible journals
are failures. Known environment secrets, labelled credentials, bearer tokens and URL
passwords are masked. This is best-effort masking of arbitrary free text, not a guarantee
that an unlabelled secret from another process can be recognized.

Runtime configuration describes the current MCP/CLI process, including process-load
`config.json`, hot-reloaded agent overlays and environment overrides. Running application
service configuration is `unknown`: file contents and systemd active state do not attest
that a process applied those values. The tool never dumps all configuration or environments.

Translation status covers scheduled public research documents and posts/diary/curation
DB translations. It uses the same freshness predicates and fingerprint-aware cooldown
as execution. Missing and changed-source counts can overlap; current validation failure
records include retry time. Archival translation batches, manual static-page translation
and manual Markdown-file translation are explicitly excluded. CommuLingo stall means an
expired running lease or a ready/deferred job eligible for over 24 hours. Recent failure
logs search the last 1,000 journal records within 168 hours; they are not complete history.
KG status reuses `kg_runtime.metrics.sync_metrics` and its stale/incomplete classification.

Official costs retain provider-reported currency and window. Local LLM estimates cover
rolling days for providers without official cost reporting; web usage covers UTC calendar
days including today. Official balance is never inferred from spend. Query failure stays
unknown, not zero. Today's LLM accounting comes from the live proxy `/audit/spend/today`,
while its cap comes from this process's `llm.gateway.load_policy` (live policy application
unknown); any remaining amount is labelled a calculation. Web today's cap/spend come from
live `/usage?days=1`, separately from the requested period. Global budget calculations
always include all providers, even with a provider filter. Uncapped or unknown remaining
amounts are null; a UTC midnight crossing rejects the inconsistent budget observation.

## Failure semantics and credential bootstrap

Local execution failures and timeouts return `ToolFailure`; invalid schemas, empty required
arguments and missing maintenance confirmation raise `ToolRejection`. Both set `isError`,
with audit `error` versus `rejected`. SQL wrapper policy refusals are rejections; connection
failures remain errors. Concatenating maintenance output preserves failure type, and backup
execution/artifact failures still prevent mutation.

`gateway_status` retains its existing fields and reports gateway scope only. Its count and
`write_tools_exposed` come from the actual catalog, and `mutable_tools` identifies potentially
mutating entries (including `kg_maintenance_run`, even though it defaults to dry-run).

At startup `mcp_gateway.credentials` honors existing environment/.env and explicit
`CREDENTIALS_DIRECTORY`. Without either KG setting, only the repository-owner UID may
read `neo4j_password` from current API then Telegram service credential mounts. It never
sets another service's credential directory or copies other credentials. The password is
kept in process environment, inherited by KG checker subprocesses, never placed in argv,
responses or files. Missing credential and permission denial are distinct from connection
failure. Offline help/list/docs remain usable. Existing Codex registrations remain valid;
start a new MCP connection to load changes. No application restart or migration is needed.
