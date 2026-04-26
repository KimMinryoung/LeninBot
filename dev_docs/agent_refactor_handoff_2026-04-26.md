# Agent Refactor Handoff - 2026-04-26

This note captures the completed work in the current session and the next
cleanup steps for the Cyber-Lenin agent/runtime refactor.

## Current State

Branch: `main`

Latest pushed commit at handoff:

- `64c5760 refactor(tasks): extract context assembly`

The working tree was clean before this handoff document was added. A follow-up
continuation later on 2026-04-26 added local, uncommitted cleanup work described
below.

Services were restarted after the continuation work:

- `leninbot-telegram.service`
- `leninbot-api.service`
- `leninbot-browser.service`

## Completed In This Session

### Provider and Model Fixes

- `01be467 fix(tasks): render context for task provider`
  - Task context now follows the actual provider format.
  - Claude keeps XML blocks.
  - OpenAI, DeepSeek, and local providers receive Markdown context.

- `1d27177 fix(autonomous): render tick prompt for provider`
  - Autonomous project tick prompts now render Markdown for OpenAI/DeepSeek/local.
  - Claude keeps the legacy XML prompt structure.
  - Confirmed `autonomous/openai/high` resolves to `gpt-5.5`.

- `2a22325 fix(webchat): render system prompt for provider`
  - Web chat system prompt now renders as XML for Claude and Markdown for
    OpenAI/DeepSeek.

- `226cc5e fix(chat): preserve context on recovery retry`
  - Telegram chat tool-pair recovery retry now preserves mission/current-state/
    autonomous context and the mission handler.

### Structural Refactors

- `96863cb refactor(runtime): centralize provider profiles`
  - Added `runtime_profile.py`.
  - Introduced `RuntimeProfile` and `resolve_runtime_profile()`.
  - Centralizes provider, model, prompt format, rounds, token limit, and budget
    resolution for `chat`, `task`, `webchat`, and `autonomous`.
  - Migrated:
    - `telegram/bot.py` LLM dispatch
    - `web_chat.py`
    - `autonomous_project.py`
    - task-agent model resolution paths

- `6d2a8d1 refactor(prompts): centralize context formatting`
  - Added `prompt_context.py`.
  - Centralized `uses_xml()`, `wrap_context_block()`, `wrap_task_content()`,
    and `fenced_text()`.
  - Migrated task, Redis context, autonomous prompt, and web chat provider-format
    checks to the common helper.

- `64c5760 refactor(tasks): extract context assembly`
  - Extracted task context assembly from `process_task()` into
    `_build_task_context_content()`.
  - The execution loop, retry handling, persistence, verification, and reporting
    behavior were intentionally left unchanged.

### Follow-up Continuation (Uncommitted)

- Added config runtime apply metadata in `bot_config._CONFIG_META`:
  - `applies_to`
  - `restart_required`
- Updated `/config` UI in `telegram/commands.py` to display immediate vs
  restart-required semantics from metadata instead of hardcoded webchat checks.
- Fixed `/config` boolean callback conversion for values such as
  `autonomous_active`.
- Further decomposed `telegram/tasks.py`:
  - `_run_task_llm()`
  - `_persist_task_success()`
  - `_handle_task_failure()`
- Added `scripts/smoke_runtime.py` for dependency-light runtime/profile and
  prompt-context smoke checks.
- Moved provider-native rendering for these context blocks into
  `prompt_context.py`:
  - `mission-context`
  - `subtask-results`
  - `agent-execution-history`
- Updated A2A path to use `resolve_runtime_profile()` for model/limit/budget
  resolution and fixed provider fallback when OpenAI is selected but unavailable.
- Fixed browser worker task adapter to accept/pass `terminal_tools`, matching the
  `process_task()` chat-loop contract.

## Verification Already Run

The following checks passed during the session:

- `python -m py_compile` or `./venv/bin/python -m py_compile` on changed modules.
- `git diff --check`.
- Runtime profile smoke test with provider/model assertions:
  - DeepSeek chat resolves to `deepseek-v4-pro`.
  - OpenAI task resolves to `gpt-5.5`.
  - Autonomous OpenAI high resolves to `gpt-5.5`.
  - Claude webchat uses XML prompt format.
  - Claude task tier override `sonnet` resolves to Claude Sonnet.
- Prompt/context smoke tests:
  - Claude task wrapping returns `<task>...</task>`.
  - OpenAI/DeepSeek task wrapping returns Markdown.
  - Redis board/task-chain context renders XML vs Markdown correctly.
  - Autonomous prompt renders XML for Claude and Markdown for OpenAI.
- Follow-up continuation checks:
  - `./venv/bin/python -m py_compile` on changed modules.
  - `git diff --check`.
  - `./venv/bin/python scripts/smoke_runtime.py`.
  - A2A provider fallback smoke.
  - `_persist_task_success()` smoke with Redis reachable.
  - Restarted and confirmed active: telegram, API, browser.

No dedicated test suite was found in the repo.

## Remaining Work

Continue one cleanup step at a time.

### 1. Config Runtime Semantics

Status: initial metadata pass completed in the follow-up continuation.

Goal: make it explicit which config changes apply live and which require service
restart.

Suggested implementation:

- Add restart/apply metadata to `bot_config._CONFIG_META`, for example:
  - `applies_to`: `telegram`, `api`, `autonomous-next-tick`
  - `restart_required`: `telegram`, `api`, `none`, or a list
- Update `/config` UI in `telegram/commands.py` to use metadata instead of
  ad hoc checks like `if key in ("webchat_provider", "webchat_model")`.
- Current likely semantics:
  - Telegram chat/task settings apply to the running Telegram process after
    `/config` mutates in-memory `_config`.
  - Web chat settings require `leninbot-api.service` restart because API imports
    `bot_config` once.
  - Autonomous active flag is disk-reloaded on each one-shot tick.
  - Autonomous provider/model settings are read by each one-shot process, so
    they apply on the next timer tick.
- Consider whether to make all runtime config reads reload-aware later. Do not
  do that in the same patch unless the blast radius is deliberately accepted.

### 2. Process Task Further Decomposition

Status: initial LLM/success/failure extraction completed in the follow-up
continuation. `process_task()` is smaller now, but still owns reporting and
visualizer delivery responsibilities. Next low-risk splits:

- `_deliver_visualizer_outputs()` for generated image sending.
- `_notify_task_complete()` for callback/system-alert reporting.
- Keep behavioral changes out of the first extraction pass.

### 3. Prompt Context Builder Cleanup

`prompt_context.py` now provides primitive helpers only. Next step is to move
more repeated block construction into it or a sibling module.

Candidates:

- `mission-context`
- `agent-execution-history`
- `subtask-results`
- `agent-board`
- `task-chain`

Initial migration completed for mission context, subtask results, and execution
history. Still consider moving Redis-rendered `agent-board` and `task-chain`
helpers into `prompt_context.py`. Keep each migration small and backed by smoke
checks for XML and Markdown output.

### 4. Tests

There is no obvious repo test suite. Add focused smoke tests or a small script
before broad refactors.

Pytest is not installed. Added lightweight `scripts/smoke_runtime.py` that can
run with `./venv/bin/python`.

### 5. A2A and Secondary Paths

This session focused on:

- Telegram chat
- Telegram background tasks
- Web chat
- Autonomous project loop

Still audit:

- `a2a_handler.py` — initial runtime-profile migration completed.
- browser worker path — `terminal_tools` adapter compatibility fixed.
- email bridge LLM path, if any
- reflection/compression paths using light models
