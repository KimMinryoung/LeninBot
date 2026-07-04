# Tool Security Gateway

A single control plane that authorizes and audits **every tool call** Cyber-Lenin
makes to a database or external system. It consolidates policy that was previously
scattered across allow-lists, a smoke test, and ad-hoc interface gating, and adds
the two things that were missing: execution-time enforcement and a per-call audit log.

## Why

Tools reach Postgres/pgvector, Neo4j, Redis, and external systems (Tavily, X,
Telegram, R2, Replicate, IMAP/SMTP, blockchain RPC, browser, Moltbook/Mersoom).
Before this, "who may call what" lived in five+ places — `runtime_tools/allowlists.py`,
`agents/base.filter_tools`, `scripts/smoke_tool_allowlists.TOOL_RISK_CLASS`,
`mcp_gateway/policy.py`, and `web_chat`/`api` gating — all of them *pre-filters*
that only shape the tool list shown to the model. There was no central, queryable
policy, no enforcement at execution time, and no security audit trail.

## The seam

Every interface (Telegram, web chat, agents, autonomous, A2A) funnels model-emitted tool calls through `tool_gateway.dispatcher.execute_tools_batch()`, which calls `tool_gateway.dispatcher.execute_tool()`. That one execution function is where the security gateway is mounted.

```
interface boundary  →  installs CallerContext (contextvar) for its run
   chat loop        →  tool_gateway.dispatcher.execute_tools_batch
                     →  tool_gateway.dispatcher.execute_tool()    ← SECURITY GATEWAY
                          authorize(ctx, tool, args) → Decision
                          enforced-deny → skip handler, return denial
                          run handler (unchanged)
                          audit(...) → tool_audit_log + journal line
```

## Package layout (`security_gateway/`)

- **`context.py`** — `CallerContext(interface, agent_name, user_id, is_owner, task_id,
  session_id)` carried in a `contextvars.ContextVar`. `caller_scope(ctx)` is a
  context manager that installs it for a `with` block and restores the parent on
  exit (so a nested `run_agent` sub-call doesn't leak its identity back to the
  orchestrator). ContextVars are snapshotted into `asyncio.gather` children, so the
  parallel-batch path inherits the caller for free. Default when unset:
  `interface="unknown"` — fail-open, but still audited.
- **`policy.py`** — single source of truth. Holds `TOOL_RISK_CLASS` (moved here from
  the smoke test, which now imports it), the per-interface rules, the owner-gated
  classes, and the rate limits. `enforce_mode()` reads `gateway_enforce_mode` from
  `config.json` (TTL-cached, flips without restart). Optional
  `config/security_policy.json` overlay tunes owner-required classes and rate limits.
- **`gateway.py`** — `authorize(ctx, tool) -> Decision`. Pure apart from a
  Redis-backed sliding-window rate counter (degrades open if Redis is down). Fails
  open on any internal error.
- **`audit.py`** — `audit(...)`: redacts+truncates args, emits a structured JSON log
  line (always), and enqueues a best-effort row to `tool_audit_log` via a single
  background worker thread. Never blocks the event loop, never raises into tool
  execution. Owns `ensure_tool_audit_log_table()` (the `tool-audit-log` migration).

## Policy rules (in order)

1. **Unknown risk class** → never blocked (a dynamic/future tool is allowed, audited).
2. **Interface restriction** → web chat may only call `read`/`fetch`/`wallet_read`
   classes. **Always enforced** (it mirrors the existing pre-filter, so enforcing
   here changes nothing observable — it just adds defense-in-depth + audit).
3. **Owner-gating** → `pay`/`send`/`execute`/`admin` require `is_owner`. **Shadow by
   default** (non-owner call allowed but logged `shadow_deny`); blocks only in enforce.
4. **Rate limit** → per `(caller, risk_class)` sliding window on outbound/irreversible
   classes only: `pay` = 3/hour, `send` = 20/hour, `publish` = 20/hour. `execute` and
   `admin` are intentionally **uncapped** (risk is in the payload, not the call count;
   legitimate bulk runs are common). **Shadow by default**; blocks only in enforce.
   Denied calls don't consume a slot. In practice these classes are reachable only by
   the trusted owner/agent/autonomous paths (webchat/a2a never expose them).

Decision labels (also the audit `decision` value): `allow`, `deny`, `shadow_deny`.

## Caller attribution

| Interface | Where set | is_owner |
|---|---|---|
| `telegram` / `agent` | `telegram/bot._chat_with_tools` via `tool_gateway.security`; standalone roleplay bot also uses `tool_gateway.security` with `agent_name=roleplay` | `True` (owner's gated channel) |
| `webchat` | `web_chat._run_llm` via `tool_gateway.security` | `False` |
| `a2a` | `a2a_handler._run_llm` via `tool_gateway.security` | `False` |
| `unknown` | unannotated direct callers | `False` |

Unannotated callers fall to `unknown`/fail-open and are still audited — they can be
annotated incrementally.

## Audit table

`tool_audit_log` (applied via `scripts/schema_migrations.py --only tool-audit-log`,
no startup DDL): `ts, interface, agent_name, user_id, is_owner, task_id, tool_name,
risk_class, decision, enforced, deny_reason, args_summary (redacted+truncated),
result_status (ok|error|denied), latency_ms, error_excerpt`. Indexed on `ts`,
`(tool_name, ts)`, `(decision, ts)`, `(interface, ts)`.

The table is append-only at the database layer. The migration installs triggers that
block `UPDATE`, `DELETE`, and `TRUNCATE`. A direct administrator maintenance
transaction can override this only by explicitly setting
`SET LOCAL leninbot.audit_log_mutation_approved = on`; normal runtime paths do not
set that flag.

## Operator CLI

```
venv/bin/python scripts/security_gateway.py policy
venv/bin/python scripts/security_gateway.py check --interface webchat --tool send_email
venv/bin/python scripts/security_gateway.py check --interface telegram --tool transfer_usdc --owner
venv/bin/python scripts/security_gateway.py audit --since 24h --decision deny --limit 50
```

## Rollout

Ships in **shadow** mode: new owner-gating and rate-limit rules log what they *would*
block without changing behavior. Watch `scripts/security_gateway.py audit --decision
shadow_deny` (and the `tool_audit shadow_deny` journal lines) to confirm no legitimate
flow trips, then flip `gateway_enforce_mode` to `enforce` (via `/config` or
`bot_config.set_gateway_enforce_mode`). Web-chat interface restriction is enforced
from day one.

## Invariants

- **Fail-open on internal error.** A broken gateway logs a warning and lets the tool
  run. It must never take down the agent.
- **Audit is non-fatal.** Both sinks swallow errors; a DB outage drops audit rows
  (logged) but never blocks or fails a tool call.
- **Defense-in-depth, not a replacement.** Pre-filters (orchestrator/agent/web
  allow-lists) still shape what the model sees; the gateway is the second,
  centralized, audited check at execution time.

## Tests

- `scripts/smoke_security_gateway.py` — policy, interface restriction, owner-gating
  (shadow vs enforce), rate limiting, redaction, fail-open.
- `scripts/smoke_tool_allowlists.py` — now validates the same `TOOL_RISK_CLASS` the
  gateway uses (single source of truth).

## Out of scope (future)

Low-level connector wrapping (`db.py` / `kg_runtime` / HTTP clients) and routing the
inbound `mcp_gateway` through this same policy. The tool layer is where capability is
granted, so it is the right first control plane.
