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
| `diary` | scheduled diary writer | terminal `save_diary`, skips routine orchestrator report |
| `stasova` | publication OpSec reviewer | small read/fetch/write tool set and low budget |
| `diplomat` | A2A and email communications | external communications tools |
| `autonomous_project` | scheduled long-term project agent | T0 research and cyber-lenin.com publication tools |

Each agent is an `AgentSpec` with prompt IR or legacy prompt, tools, finalization tools, terminal tools, provider override, budget, max rounds, and political-line inclusion flag.

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

## Tool Isolation

Tool visibility is filtered at dispatch:

1. `runtime_tools/registry.py` builds the global tool definitions and handlers.
2. `runtime_tools/allowlists.py` selects the Telegram orchestrator tools.
3. `AgentSpec.tools` selects specialist tools.
4. `web_chat.py` defines a separate public web-chat allow-list.

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

## Context Assembly

Agent tasks receive structured context rather than a passive chat dump:

| Section | Source |
|---|---|
| current state | recent completed/in-progress/pending tasks |
| mission context | `telegram_mission_events` |
| agent execution history | recent completed tasks by same agent type |
| task chain | Redis `task_result:*` and DB fallback |
| agent board | Redis `board:{mission_id}` |
| task | orchestrator delegation text |

Agents can call chat-reading tools when they need the original timestamped user messages.

## Redis Runtime State

| Key pattern | Purpose |
|---|---|
| `task:{id}:progress` | incremental tool-call log for restart recovery |
| `task:{id}:state` | live round/cost/status metadata |
| `active_tasks` | IDs currently processing |
| `board:{mission_id}` | inter-agent mission bulletin board |
| `task_result:{task_id}` | 7-day task-chain summaries |

Redis failures are intended to degrade live continuity, not crash task execution.

## Restart Handoff

If a service restart interrupts a processing task, startup recovery marks the parent as `handed_off` and creates a child task with the saved Redis progress and restart metadata. Guards prevent infinite handoff loops by age, attempt count, and depth.

Programmer-triggered restarts should use the runtime restart tool path so syntax/import checks and handoff metadata are recorded before the process exits.
