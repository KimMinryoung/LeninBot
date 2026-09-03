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
                          authorize(..., consume_rate_limit=False) → preflight Decision
                          validate schema/argument policy
                          lookup/reserve tool_idempotency for stable scopes
                          authorize(..., consume_rate_limit=True) → atomic Redis decision
                          run handler; persist succeeded/outcome_unknown
                          audit(...) → tool_audit_log + journal line
```

## Package layout (`security_gateway/`)

- **`context.py`** — `CallerContext(interface, agent_name, user_id, is_owner, task_id,
  session_id, request_id, parent_request_id, scope_type, scope_id, chat_log_id)`
  carried in a `contextvars.ContextVar`. `new_run_context()` allocates a fresh
  `request_id`, inherits the active caller's user/session/business scope, and
  points a nested run's `parent_request_id` at the active run. `scope_type` and
  `scope_id` identify the durable business object independently of channel;
  `chat_log_id` remains the web-chat-specific fast join.
  `caller_scope(ctx)` is a
  context manager that installs it for a `with` block and restores the parent on
  exit (so a nested `run_agent` sub-call doesn't leak its identity back to the
  orchestrator). ContextVars are snapshotted into `asyncio.gather` children, so the
  parallel-batch path inherits the caller for free. Default when unset:
  `interface="unknown"` — policy rules still run and the call is audited.
- **`policy.py`** — single source of truth. Holds `TOOL_RISK_CLASS` (moved here from
  the smoke test, which now imports it), the per-interface rules, the owner-gated
  classes, the owner-gated tools, the per-tool caller allow-lists, and the rate
  limits. `enforce_mode()` reads `gateway_enforce_mode` from
  `config.json` (TTL-cached, flips without restart). Optional
  `config/security_policy.json` overlay tunes owner-required classes and tools,
  the caller allow-lists (`tool_caller_allowlist`, a tool→callers map; an entry
  set to anything other than a list clears the restriction) and rate limits.
- **`gateway.py`** — `authorize(ctx, tool, consume_rate_limit=...) -> Decision`.
  Redis Lua performs one atomic sliding-window decision. Capped side effects fail
  closed when Redis is unavailable (`deny`, `rule=rate_store`); internal policy errors
  also fail closed (`rule=error`).
- **`idempotency.py`** — stable task/session-scoped Postgres reservation and outcome
  store for side-effect tools (`running`, `succeeded`, `outcome_unknown`).
- **`audit.py`** — `audit(...)`: redacts+truncates args, emits a structured JSON log
  line (always), and enqueues a best-effort row to `tool_audit_log` via a single
  background worker thread. Never blocks the event loop, never raises into tool
  execution. Owns `ensure_tool_audit_log_table()` (the `tool-audit-log` migration).

## Policy rules (in order)

1. **Unknown risk class** → always denied until explicitly classified (`rule=taxonomy`).
2. **Interface restriction** → public webchat and A2A may only call
   `read`/`fetch`/`wallet_read` classes. **Always enforced**, independently of
   shadow/enforce rollout mode. This mirrors the profile pre-filters and prevents
   a stale profile or provider-emitted hidden tool name from reaching a handler.
3. **Owner-gating** → `pay`/`send`/`execute`/`admin` require `is_owner`. **Shadow by
   default** (non-owner call allowed but logged `shadow_deny`); blocks only in enforce.
4. **Owner-gated tools** (`rule=owner_tool`) → individual tools requiring `is_owner`
   when their whole risk class should not be owner-only. Currently the six CommuLingo
   narrow writers, which are `write`, a class that legitimately runs elsewhere. Same
   shadow/enforce behaviour as rule 3.
5. **Caller allow-list** (`rule=caller`) → tools that name the callers permitted to run
   them, matched on `agent_name` falling back to `interface`. The CommuLingo writers
   allow `commulingo_curator`, `analyst` and `telegram` (the orchestrator, which writes
   with no agent name), the three callers `tool_audit_log` shows have ever written.
   **Always enforced**, like rule 2 and for the same reason: it mirrors a profile
   pre-filter. Rule 4 alone would not hold here, because the roleplay bot runs on a
   private allow-listed channel and correctly declares `is_owner`.
6. **Rate limit** → per `(caller, risk_class)` atomic Redis sliding window on outbound/irreversible
   classes only: `pay` = 3/hour, `send` = 20/hour, `publish` = 20/hour. `execute` and
   `admin` are intentionally **uncapped** (risk is in the payload, not the call count;
   legitimate bulk runs are common). **Shadow by default**; blocks only in enforce.
   Denied calls don't consume a slot. In practice these classes are reachable only by
   the trusted owner/agent/autonomous paths (webchat/a2a never expose them).

Decision labels (also the audit `decision` value): `allow`, `deny`, `shadow_deny`.

## Caller attribution

| Interface | Where set | is_owner |
|---|---|---|
| `telegram` / `agent` | `telegram/bot._chat_with_tools`; direct messages use `telegram_message`, durable workers/verifiers use `telegram_task`, and nested `run_agent` calls inherit the parent request | `True` (owner's gated channel) |
| `webchat` | `services.web_chat._run_llm` via `tool_gateway.security` | `False` |
| `a2a` | `services.a2a_handler._run_llm`; A2A context/task/message IDs map to session/scope/user fields | `False` |
| `autonomous` | autonomous project ticks and scheduled CommuLingo maintainers; project ticks use `autonomous_project`, maintainers use `maintenance_job` | `True` |
| `system:writer` | `/writer` main run plus diagnosis/revision children under `writer_project` | `True` |
| `unknown` | unannotated direct callers | `False` |

Unannotated callers fall to `unknown` and are still audited — they can be
annotated incrementally.

Writer-specific local tools are included in the taxonomy even though they are not
part of the global runtime registry: `search_manuscript` is `read`, while
`append_to_manuscript` and `replace_in_manuscript` are `write`. This keeps audit
rows policy-stable for the personal fiction workspace.

## Outbound URL boundary

Publicly reachable `fetch_url` calls share the same local-network exclusion policy in
`content_fetch/url_security.py`. Only HTTP(S) on ports 80, 443, 8080, and 8443 is
accepted. Literal addresses and every DNS answer must be globally routable; localhost,
private, loopback, link-local, reserved, and mixed public/private answers are rejected.
The policy is re-applied to each requests redirect, Playwright main navigation and
subrequest, and failure-diagnosis DNS/TCP/HTTP probe. Remote Tavily retrieval may remain
a fallback, but no local socket is opened before this validation succeeds.

## Audit table

**Write path (2026-09-04):** the worker thread in `audit.py` no longer inserts
itself; it batches rows to the LLM proxy's audit sink (`POST /audit/tool`,
`audit_sink.py` — see `dev_docs/llm_gateway.md` "감사 싱크"). The proxy is the
only process that writes `tool_audit_log`, optionally through an INSERT-only
role, so tool callers need no DB password and ad-hoc runs are audited too.
With no `proxy_base` configured (tests, standalone tools) the worker inserts
directly as before.

`tool_audit_log` (applied via `scripts/schema_migrations.py --only tool-audit-log`,
no startup DDL): `ts, interface, agent_name, user_id, is_owner, task_id, session_id,
request_id, parent_request_id, scope_type, scope_id, chat_log_id, tool_name,
risk_class, decision, enforced, deny_reason, args_summary (redacted+truncated),
result_status, latency_ms, `error_excerpt`. Indexed on `ts`,
`(tool_name, ts)`, `(decision, ts)`, `(interface, ts)`, and partial
`(request_id, ts)`, `(parent_request_id, ts)`, `(scope_type, scope_id, ts)`, and
`(chat_log_id, ts)` indexes. Apply the cross-runtime correlation columns with:

```
venv/bin/python scripts/schema_migrations.py --only tool-audit-run-correlation
```

For public web chat, `chat_logs.request_id` stores the same run ID. A normal
turn reserves `chat_logs.id` before model execution and inserts the row with
that ID after the answer completes; regeneration reuses the row being replaced.
This makes `tool_audit_log.request_id = chat_logs.request_id` the run-level join
and `tool_audit_log.chat_log_id = chat_logs.id` the message-level join, including
detached browser observers. Apply both sides before deploying the code:

```
venv/bin/python scripts/schema_migrations.py --only web-chat-audit-correlation
```

Every exit from `execute_tool` writes exactly one row, including the ones that
never reach the handler. `result_status` is one of `ok`, `error`, `rejected`,
`outcome_unknown`, `denied`, `invalid_args`, `deduplicated`,
`deduplicated_durable`, `idempotency_unavailable`, `unknown_tool`, or
`gateway_error`. The last two, plus `idempotency_unavailable`, cover the
fail-closed blocks where the gateway or the idempotency store was itself
unavailable; they are recorded as a synthetic `deny` because the caller was
refused whatever the preflight decision had said.

`error` used to mean only "the handler raised". Handlers that catch their own
exception and return the explanation as text — most of `runtime_tools/` does,
so the model gets a readable message instead of a stack trace — were audited as
`ok`, and the eight Tavily quota rejections of 2026-07-27 were indistinguishable
from real search hits. A handler now reports that verdict by returning
`tool_gateway.results.ToolFailure(...)`, a `str` subclass: every consumer still
sees the same text, the dispatcher audits `error` and flags the tool_result as
an error for the model, and a side-effect tool holding a durable reservation
falls back to `outcome_unknown` exactly as a raised exception would.

Use `ToolFailure` for a call that could not do its job (external system failed,
dependency unavailable, exception swallowed). Do NOT use it for a well-formed
call with a negative answer — "no documents found", "person not in the
registry" — those are successful lookups, and marking them errors would bury
the real failures.

`rejected` is the mirror correction. Handlers that enforce their own domain
rules raise to unwind, and the generic `except` filed all of it as `error`:
4,025 CommuLingo curator rows over 30 days — duplicate candidates, bios past
the length limit — sat next to genuine breakage. `tool_gateway.results.
ToolRejection` (a `ValueError` subclass, so the `except ValueError` catches
that already sit between handler and dispatcher keep working) audits `rejected`
and hands the model the message verbatim, without the "external outcome may be
unknown" prefix a real exception earns. The tool_result stays flagged as an
error so the model changes course instead of repeating the call. Raise it only
when the refusal preceded any side effect; a tool holding a durable
reservation still falls back to `outcome_unknown`, because only a terminal
status releases the reservation and the safe terminal status for an
irreversible tool is the one that refuses a blind retry.

The table is append-only at the database layer. The migration installs triggers that
block `UPDATE`, `DELETE`, and `TRUNCATE`. A direct administrator maintenance
transaction can override this only by explicitly setting
`SET LOCAL leninbot.audit_log_mutation_approved = on`; normal runtime paths do not
set that flag.

## Durable idempotency table

`tool_idempotency` is applied explicitly with
`scripts/schema_migrations.py --only tool-idempotency`. The primary key is a hash of
stable task/session scope, tool name, and normalized arguments. Successful results are
reused across process restarts. A handler exception is conservative: the external system
may have committed before the response was lost, so status becomes `outcome_unknown` and
a retry is blocked until an operator verifies the target. Calls without stable scope use
loop-local dedupe only.

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
`bot_config.set_gateway_enforce_mode`). Public webchat and A2A interface restrictions
are enforced from day one.

## Invariants

- **Fail-closed control plane.** Broken policy/pre-check, unknown taxonomy, invalid
  arguments, unavailable capped rate store, and unavailable scoped idempotency store
  return a denied/error tool result and never call the handler.
- **Audit is non-fatal.** Both sinks swallow errors; a DB outage drops audit rows
  (logged) but never blocks or fails a tool call.
- **Defense-in-depth, not a replacement.** Pre-filters (orchestrator/agent/web/A2A
  allow-lists) still shape what the model sees; the gateway is the second,
  centralized, audited check at execution time.

## Tests

- `scripts/smoke_security_gateway.py` — policy, interface restriction, owner-gating
  (shadow vs enforce), rate limiting, redaction, fail-closed, and run-local
  side-effect idempotency, schema/default normalization, nested non-finite number rejection, atomic rate-store failures,
  durable success reuse, and `outcome_unknown` replay suppression.
- `scripts/smoke_tool_allowlists.py` — validates the same `TOOL_RISK_CLASS` and checks
  that orchestrator schemas and executable handlers share one filtered set. Web chat is
  checked per persona, not only for the default one: a tool injected for a single
  persona (`read_persona_context` for Gramsci) shipped unregistered, so the model saw it
  and the gateway denied every call (2026-07-26 → fixed 2026-08-03).
- `scripts/smoke_url_security.py` — validates unsafe addresses, mixed DNS answers,
  redirects, and diagnostic fail-closed behavior.

## Out of scope (future)

Low-level connector wrapping (`db.py` / `kg_runtime` / HTTP clients) and routing the
inbound `mcp_gateway` through this same policy. The tool layer is where capability is
granted, so it is the right first control plane.
