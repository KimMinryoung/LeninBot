# Multi-Agent Architecture

## Overview

Cyber-Lenin uses a hierarchical multi-agent system built on the orchestrator-worker pattern. A single orchestrator handles user conversation and dispatches tasks to specialized agents. PostgreSQL is the system of record; Redis provides live state that survives process restarts.

```
User (Telegram)
    |
Orchestrator (telegram_bot.py)
    |--- direct response (simple queries)
    |--- delegate() ---------> Agent (programmer/analyst/scout/browser/visualizer)
    |--- multi_delegate() ---> [Agent A, Agent B, ...] --> Synthesis Agent
    |
    <--- _orchestrator_report_task() --- Agent result
```

## Agents

All agents are `AgentSpec` instances registered in `agents/__init__.py`. Each spec defines: name, system prompt, allowed tools, budget, and max rounds.

| Agent | Budget | Rounds | Role | Key Tools |
|-------|--------|--------|------|-----------|
| **programmer** | $1.50 | 50 | Code writing, modification, debugging | read_file, write_file, patch_file, execute_python, restart_service |
| **analyst** | $1.00 | 50 | Information analysis, KG cross-validation | knowledge_graph_search, vector_search, web_search, fetch_url |
| **scout** | $1.00 | 30 | Platform surveillance, web patrol | execute_python, web_search, fetch_url, download_image |
| **browser** | $1.50 | 30 | Login, forms, multi-page navigation | browse_web, fetch_url, check_inbox |
| **visualizer** | $1.00 | 40 | Image generation (constructivist style) | generate_image, read_file, download_image |

### Tool Isolation

The orchestrator **cannot** access programming tools (`read_file`, `write_file`, `patch_file`, `list_directory`, `execute_python`). It must delegate to the programmer agent. Each agent only sees tools listed in its spec — `AgentSpec.filter_tools()` enforces this at dispatch time.

### Agent System Prompts

Each agent's prompt is assembled from shared blocks (`agents/base.py`) + agent-specific instructions:

- `CONTEXT_AWARENESS_BLOCK` — explains `<current_state>`, `<mission-context>`, `<agent-execution-history>`, `<task>` XML sections
- `MISSION_GUIDELINES_BLOCK` — save_finding, write_kg, budget management rules
- `CONTEXT_FOOTER` — current datetime, system alerts

## Task Lifecycle

### State Machine

```
pending ──> processing ──> done ──> (verification: passed/failed)
                |──> failed
                |──> handed_off (interrupted by restart)
blocked ──> pending (when subtask group completes)
```

### Task Worker (`telegram_tasks.py:task_worker`)

- Polls PostgreSQL for `status='pending'` tasks using `FOR UPDATE SKIP LOCKED`
- Bounded concurrency via `asyncio.Semaphore` (default 2, max 8)
- Before each poll, unblocks synthesis tasks whose subtasks are all complete
- Poll interval: 5s (idle), 1s (at capacity), 10s (on error)

### Delegation Flow

1. User message arrives at orchestrator
2. Orchestrator decides to delegate via `delegate(agent, task, context)` tool
3. `_exec_delegate()` (`self_tools.py`) creates a DB row in `telegram_tasks` with `status='pending'`
4. Resolves or auto-creates a mission to link the task to
5. Injects recent conversation (last 6 messages) into delegation content
6. Task worker picks up the task, runs `process_task()`
7. On completion, `_orchestrator_report_task()` runs a new orchestrator turn to interpret results and respond to the user

### Parallel Delegation (`multi_delegate`)

1. Creates N subtasks with `plan_role='subtask'`, all sharing the same `plan_id`
2. Creates one synthesis task with `plan_role='synthesis'`, `status='blocked'`
3. Worker unblocks synthesis when all subtasks reach terminal state
4. Synthesis task receives `<subtask-results>` block with all subtask outputs

## Memory Systems

### 1. Redis — Live Execution State (`redis_state.py`)

Redis holds ephemeral state that must survive process death but is not the long-term record.

**Task Progress** — `task:{id}:progress` (LIST)
- Every tool call during agent execution is appended: tool name, input, result, round number, timestamp
- Written incrementally in `claude_loop.py` and `openai_tool_loop.py` after each tool result
- On restart recovery, parent's progress is formatted as `<parent-execution-log>` and injected into child task content
- Cleared after task completes and tool_log is saved to PostgreSQL

**Live Task State** — `task:{id}:state` (HASH)
- Updated per API round: round number, cumulative cost, status, agent type
- Used for real-time monitoring

**Active Task Registry** — `active_tasks` (SET)
- Task IDs currently being processed
- `register_active_task()` on pickup, `unregister_active_task()` on completion/failure
- Merged with in-memory set during shutdown for checkpoint coverage

All keys have 24h TTL. All operations are fail-safe (try-except, never crashes the bot).

### 2. PostgreSQL — System of Record

| Table | Purpose |
|-------|---------|
| `telegram_tasks` | Task queue: content, status, result, tool_log, agent_type, mission_id, restart state, verification state, metadata (JSONB) |
| `telegram_chat_history` | User-bot conversation: role, content, created_at. Indexed on (user_id, id DESC) |
| `chat_history_summaries` | Chunked summaries (10 msgs each, max 3 in context). Created by background `_maybe_summarize_chunk()` |
| `telegram_missions` | Multi-task campaigns: title, status (active/done), user_id |
| `telegram_mission_events` | Mission timeline: source, event_type, content, created_at |
| `ai_diary` | Daily experience summaries from experience_writer/diary_writer |

### 3. Chat Context for Orchestrator (`_load_context_with_summaries`)

Each orchestrator turn loads conversation history in layers:

1. **Summary preamble** — up to 3 chunk summaries (covering older messages) injected as a single context block, not fake conversation pairs
2. **Raw messages** — last 30 messages always loaded regardless of summary coverage, with exact timestamps `[YYYY-MM-DD HH:MM]` on user messages
3. **Time-gap annotations** — gaps of 1h+ annotated inline: `[2026-04-03 14:30 (8시간 경과)]`
4. **Temporal awareness** — system prompt instructs the model to treat post-gap messages as potential context switches

Summaries that overlap with raw messages are excluded (no duplication). Orphaned summaries (referencing deleted chat rows) are auto-purged.

### 4. Agent Context Isolation (`process_task` sections)

When an agent task executes, its context is assembled from:

| Section | Source | Content |
|---------|--------|---------|
| Current State | `build_current_state(user_id)` | Completed (24h), in-progress, pending tasks overview |
| Mission Context | `get_mission_events(mission_id)` | Last 20 mission timeline events |
| Execution History | `telegram_tasks` query | Last 3 completed/handed_off tasks by same agent type, with observation masking |
| Task Content | Orchestrator delegation | The actual task instructions wrapped in `<task>` |

**No passive chat dump** — agents rely on the orchestrator's delegation message as primary context. If the delegation is unclear, agents can call the `read_user_chat` tool on demand to read the user's actual timestamped messages.

**Observation masking** for execution history:
- Newest task: full tool log (8000 chars)
- Middle tasks: actions only, results masked
- Oldest task: summary only, no tool log

### 5. Mission System (`telegram_mission.py`)

Missions are auto-created when the orchestrator delegates work. They track multi-step campaigns:

- `add_mission_event(mission_id, source, event_type, content)` — records decisions, task completions, findings
- Events visible to all tasks linked to the same mission
- Orchestrator uses mission status to judge completion: close only when all goals are met

### 6. Knowledge Graph (Neo4j via Graphiti)

Long-term factual memory shared across all agents:

- `write_kg(content, group_id)` — extract entities/relationships into graph
- `knowledge_graph_search(query)` — vector + graph traversal
- Group IDs: `geopolitics_conflict`, `diplomacy`, `economy`, `korea_domestic`, `agent_knowledge`
- Design docs: `dev_docs/knowledge_graph_design.md`, `dev_docs/knowledge_graph_schema.md`

### 7. Experience & Diary

- **experience_writer**: Aggregates daily learnings into searchable experience entries
- **diary_writer**: Periodic self-reflection summaries stored in `ai_diary` table
- `recall_experience(query)`: Semantic search over accumulated insights, injected into orchestrator context when relevant

## Restart Recovery

### Problem

When the programmer agent restarts the telegram service, the process dies mid-task. The in-flight conversation and tool execution state would be lost.

### Solution: Redis + DB Handoff

**Before restart:**
1. `persist_task_restart_state()` saves restart phase to task metadata
2. Tool progress has been incrementally written to Redis throughout execution

**On startup:**
1. `recover_processing_tasks_on_startup()` finds interrupted `processing` tasks
2. Guards: age > 60min → close as failed; handoff count >= 2 → close; depth >= 5 → close
3. Creates child task with:
   - `<parent-execution-log>` from Redis (what tools were already called)
   - `[restart already completed by parent task]` marker
   - Restart state metadata (phase, attempt count, should_skip_restart)
4. Parent marked as `handed_off`
5. Redis progress cleared only after child is in DB

**Child task behavior:**
- Sees parent's tool call history — does not repeat completed work
- Restart markers prevent re-triggering restart
- Proceeds to verification/commit phase

### File-to-Service Mapping

The programmer agent and `restart_service` tool include explicit mapping so the agent restarts the correct service:

| Service | Files |
|---------|-------|
| telegram | telegram_bot.py, telegram_commands.py, telegram_tasks.py, telegram_tools.py, telegram_mission.py, claude_loop.py, openai_tool_loop.py, self_tools.py, shared.py, agents/*.py, redis_state.py, chatbot.py |
| api | api.py |
| browser | browser_worker.py |
| all | db.py, embedding_server.py, or files shared by multiple services |

## Browser Worker

The browser agent can run in a separate process communicating via Unix socket:

- **Socket**: `/tmp/leninbot-browser.sock`
- **Protocol**: JSON request/response (cmd: "task" or "ping")
- **Timeout**: 180s per task
- **Fallback**: If worker unreachable, executes in-process in the main telegram_bot

Separate systemd service (`leninbot-browser`) allows independent restarts without affecting the telegram service.

## Verification & Auto-Retry

After task completion, optional verification can run:

1. **Auto checks**: task_report quality, URL accessibility
2. **LLM verification**: Verifier agent evaluates result against original task, can check server logs
3. **Auto-retry on failure**: Creates child task with parent's summary + failure details, marked `[AUTO-RETRY]`
4. **Guards**: Chain depth limit, restart phase blocking, max retry attempts

## System Monitoring

`system_monitor()` runs as a background loop (120s interval):

- **KG (Neo4j)**: Checks connection, broadcasts alert on down/restored
- **Redis**: Checks connection, broadcasts alert on down/restored
- Alerts injected into orchestrator's system prompt via `_system_alerts`

## Infrastructure

| Component | Runtime | Notes |
|-----------|---------|-------|
| Telegram bot | systemd (direct Python) | Main orchestrator + task worker |
| API server | systemd (uvicorn) | FastAPI REST endpoints |
| Browser worker | systemd (direct Python) | Playwright Chromium automation |
| Embedding server | systemd (direct Python) | BGE-M3 on port 8100 |
| Neo4j | Docker Compose | Knowledge graph |
| Redis | Docker Compose | Live task state |
| PostgreSQL | Supabase (remote) | System of record |
| Local LLM | systemd (llama-server) | Qwen 3.5-4B fallback on port 11435 |

Deployment via `deploy.sh`: git pull, conditional pip install, selective systemd restart.
