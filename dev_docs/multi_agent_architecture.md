# Multi-Agent Architecture

최종 확인 기준: 2026-05-09 코드 트리.

Telegram is the full multi-agent runtime. The orchestrator receives user messages, answers directly when appropriate, or creates database-backed tasks for specialist agents. PostgreSQL is the durable task record; Redis holds live progress and shared mission state.

## High-Level Flow

```
Telegram user
  -> telegram/bot.py orchestrator
      -> direct answer through provider tool loop
      -> delegate/multi_delegate/run_agent tools
          -> task_store.py / telegram_tasks
          -> telegram/tasks.py worker
              -> agents/<name>.py AgentSpec
              -> runtime_tools filtered by allow-list
              -> task result in DB
      -> orchestrator report callback to user
```

The orchestrator has a limited tool set and does not directly edit code/files. Code work is delegated to `programmer`, whose current spec is a Codex CLI handoff rather than a normal broad Python tool set.

## Registered Agents

Registered in `agents/__init__.py`:

| Agent | Role | Current notes |
|---|---|---|
| `programmer` | code writing, modification, debugging | delegated to Codex CLI; normal tool list only includes `list_agent_tools` |
| `analyst` | information analysis and KG cross-validation | uses search, corpus, KG, report/private-report tools |
| `scout` | external platform reconnaissance and web patrol | uses web/fetch/social/platform tools |
| `browser` | browser automation | talks to `browser/worker.py` through browser-use tooling |
| `visualizer` | image prompt/design specialist | image/media and publication support |
| `diary` | scheduled diary writer and published diary maintenance owner | terminal `save_diary` is allowed only for the configured scheduled prompt `[diary] Write a periodic diary entry`; other diary tasks run autonomously with maintenance tools such as `edit_content`, including diary edit/delete/unpublish actions |
| `stasova` | publication OpSec reviewer | small read/fetch/write tool set and low budget |
| `diplomat` | A2A and email communications | external communications tools |
| `autonomous_project` | scheduled long-term project agent | T0 research and cyber-lenin.com publication tools |

Each agent is an `AgentSpec` with prompt IR or legacy prompt, tools, finalization tools, terminal tools, provider override, budget, max rounds, and political-line inclusion flag. The current executable tool matrix is maintained in `dev_docs/agent_tool_matrix.md`.

## Runtime Overlay

`config/agent_runtime.json` can override registered specs without editing Python. The example file shows supported keys:

- `provider`
- `model`
- `budget_usd`
- `max_rounds`
- `finalization_tools`
- `terminal_tools`
- `skip_orchestrator_report`

`agents/runtime_config.py` reloads this overlay when specs are requested.

## Prompt And Tool Payload Efficiency

System prompts are rendered from provider-aware prompt IR and kept as stable as possible so provider prompt caching can work. Tool definitions are filtered by allow-list before every run, then provider loops compact human-readable tool and schema `description` strings before sending them to the model. The compaction preserves tool names, schema structure, parameter types, enums, defaults, and required keys; it only shortens long explanatory prose.

## Tool Isolation

Tool visibility is filtered at dispatch:

1. `runtime_tools/registry.py` builds the global tool definitions and handlers.
2. `runtime_tools/allowlists.py` selects the Telegram orchestrator tools through `tool_gateway.selection`.
3. `AgentSpec.tools` selects specialist tools through `AgentSpec.filter_tools()` and `tool_gateway.selection`.
4. `web_chat.py` defines a separate public web-chat allow-list and uses `tool_gateway.selection` before injecting web-only safe handlers.
5. Provider loops dispatch model-emitted tool calls through `tool_gateway.dispatcher`, whose `execute_tool()` implementation preserves the `security_gateway` authorization/audit check.

Empty `AgentSpec.tools` is fail-closed. A tool must be present in the global registry and the relevant allow-list to be callable.

## Task Lifecycle

Durable task states live in `telegram_tasks`:

```
pending -> processing -> done
                    \-> failed
                    \-> handed_off
blocked -> pending  (synthesis task after parallel subtasks finish)
```

`telegram/tasks.py` polls with `FOR UPDATE SKIP LOCKED`, uses a bounded semaphore, and unblocks synthesis tasks when all sibling subtasks are terminal.

### Delegation

`delegate` creates one pending task linked to the current mission. The task content includes the orchestrator instruction plus recent conversation context. The worker runs `process_task()`, saves result/tool log, and triggers an orchestrator callback unless the agent/spec path suppresses it.

### Parallel Delegation

`multi_delegate` creates N subtasks sharing a `plan_id` plus one blocked synthesis task. When subtasks finish, the synthesis task receives a `<subtask-results>` block and produces a combined report.

### Synchronous Sub-Agent

`run_agent` runs a bounded in-turn specialist call for narrow analysis paths. It does not replace durable delegated tasks for work that needs continuity.

### Post-Hoc Verification (Critic)

Every completed task passes through `_run_verification()` (`telegram/tasks.py`), an independent LLM critic that re-checks the executor's report with its own tools before the orchestrator relays it. Rollout follows the standard shadow→enforce pattern via the `task_verification_mode` config key (`off | shadow | enforce`, default `shadow`, flips live without restart):

- **Policy.** A delegation may carry an explicit `verification` object (`checks`/`urls`/`log_service`/`log_grep`/`retry_limit`/`required`) on `delegate`/`multi_delegate`; without one, a per-agent default applies — programmer gets `task_report` + `server_logs`, analyst/scout/diplomat get `task_report`, all other agents skip (`_DEFAULT_VERIFICATION_POLICIES`). `verification: {required: false}` opts a task out. Skipped tasks are marked `passed` so `verification_status` never rots at `pending`.
- **Verifier runtime.** The critic runs on the **low tier** of the executor's provider (codex/moon executors are verified by the task provider — an independent judge), budget-capped at $0.15, with a read-only tool surface (`read_self`, `read_file`, `search_files`, `list_directory`, `fetch_url`). `restart_service` is added only in enforce mode.
- **Verdict flow.** The verdict + details persist to `telegram_tasks.verification_status`/`verification_details`; a FAIL is surfaced in the orchestrator report callback (in shadow mode as an advisory caveat for the user) and in the system alert.
- **Enforce mode** additionally feeds a FAIL into `_maybe_redelegate_after_verification_failure()`: bounded auto-retry (`retry_limit`, chain-depth guard) that re-delegates with a "take a DIFFERENT approach" instruction, or the restart-handoff path when the verifier determines a telegram restart is required.

Smoke test: `scripts/smoke_task_verification.py` (hermetic — stubbed LLM + captured SQL).

## Context Assembly

Agent tasks receive structured context rather than a passive chat dump:

| Section | Source |
|---|---|
| current state | recent completed/in-progress/pending tasks |
| mission context | `telegram_mission_events` |
| agent execution history | recent completed tasks by same agent type |
| task chain | Redis `task_result:*` and DB fallback |
| agent board | Redis `board:{mission_id}` |
| diary activity preflight | scheduled diary-writing prompt only: latest diary anchor plus recent Telegram context, completed tasks/reports, public or staged research documents, and autonomous project state are injected automatically so new entries can focus on the period since the last diary |
| diary web-chat preflight | scheduled diary-writing prompt only: recent public web `chat_logs` are injected automatically so correction, omission, non-publication, and topic-priority instructions from web chat reach the next scheduled diary run |
| task | orchestrator delegation text |

Agents can call chat-reading tools when they need the original timestamped user messages. Shared chat-audience guidance requires Telegram and web chat to remain separate sources when they are read; it is not an instruction to always read both channels.

For diary tasks, the runtime distinguishes new diary writing by exact scheduled prompt rather than edit/delete keywords. `telegram.diary_mode.is_diary_writing_task()` treats a diary task as a new-entry run only when its task text matches an enabled diary schedule prompt, with `[diary] Write a periodic diary entry` as the default. Only that mode receives the injected diary activity and web-chat preflight blocks. Other diary-agent tasks receive normal task context and must use tools autonomously for the requested maintenance or inspection work.

The diary prompt treats finance/securities data as background context by default during scheduled writing: it may be collected for judgment, but routine stock prices, tickers, index moves, and market fluctuations should not appear in diary prose unless directly necessary to explain the actual events, decisions, or political-economic contradiction of the period. `telegram/bot.py` also guards the `save_diary` handler and rejects publication attempts unless the task matches the configured scheduled diary-writing prompt. Published diary edits use `edit_content(content_type="diary", id=...)`; deletion and non-publication/unpublish requests use `edit_content(content_type="diary", id=..., action="delete"|"unpublish", confirm=true)`. Because `ai_diary` has no private/unpublished status column, unpublish removes the row from public diary storage after clearing publication audit FK references and invalidating caches.

## Redis Runtime State

| Key pattern | Purpose |
|---|---|
| `task:{id}:progress` | incremental tool-call log for restart recovery |
| `task:{id}:state` | live round/cost/status metadata |
| `active_tasks` | IDs currently processing |
| `board:{mission_id}` | inter-agent mission bulletin board |
| `task_result:{task_id}` | 7-day task-chain summaries |

Redis failures are intended to degrade live continuity, not crash task execution.

`telegram_tasks.tool_log` is treated as append-only execution evidence once populated. The task success path sets it initially and appends later retry/resume logs instead of replacing it. A Telegram schema trigger blocks clearing, replacing, deleting rows with non-empty tool logs, or truncating `telegram_tasks` unless an administrator explicitly sets `SET LOCAL leninbot.task_tool_log_mutation_approved = on` in a maintenance transaction.

## Restart Handoff

If a service restart interrupts a processing task, startup recovery marks the parent as `handed_off` and creates a child task with the saved Redis progress and restart metadata. Guards prevent infinite handoff loops by age, attempt count, and depth.

Programmer-triggered restarts should use the runtime restart tool path so syntax/import checks and handoff metadata are recorded before the process exits.
