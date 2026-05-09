# Hot-Reloaded Prompt Policy

최종 확인 기준: 2026-05-09 코드 트리.

Prompt text has a hot-reload layer, but tool definitions and Python agent specs do not hot-reload.

## Hot-Reloaded Files

`AgentSpec.render_prompt()` reads these files at render time:

- `identity/political_line.md`
- `identity/agent_prompts/<agent_name>.md`

Default paths can be overridden with:

- `POLITICAL_LINE_PATH`
- `AGENT_PROMPT_DIR`

Edits to these Markdown files affect the next LLM call that renders the agent prompt. No service restart is needed for prompt text alone.

## Prompt Assembly

For prompt-IR agents, `agents/base.py` builds the final prompt in this order:

1. political line section, if `include_political_line=True`
2. per-agent runtime prompt overlay, if `identity/agent_prompts/<agent>.md` exists
3. built-in agent identity/preamble/sections from Python
4. provider-specific rendering via `llm/prompt_renderer.py`

Claude renders XML-oriented structure. OpenAI, DeepSeek, local, Moon, and Codex-oriented paths render Markdown/local structure.

Per-turn volatile state such as current time, current model, task content, mission context, and system alerts is injected by callers as runtime context rather than baked into the system prompt. This keeps system prompts more cacheable.

## Requires Restart

Restart the owning service when changing:

- Python files in `agents/`, `telegram/`, `runtime_tools/`, `llm/`, `kg_runtime/`, `graph_memory/`, or API modules
- `config/agent_runtime.json` if the running path does not call the registry reload point before use
- systemd unit files or credentials
- third-party SDK/env configuration read at process startup

In practice:

- Telegram agent behavior: restart `leninbot-telegram.service` for Python/tool changes.
- Public web chat behavior: restart `leninbot-api.service` for `api.py`, `web_chat.py`, or webchat provider setting changes.
- Browser automation worker changes: restart `leninbot-browser.service`.

## What Belongs in Hot-Reload Files

Good candidates:

- operating policy
- tone and audience guidance
- task-specific heuristics
- temporary caution text
- source-handling rules

Keep these in Python/spec/config instead:

- tool allow-lists
- budgets and max rounds
- provider and model defaults
- terminal/finalization tools
- DB schema and runtime logic

This boundary keeps prompt tuning fast while preserving reviewable capability changes.
