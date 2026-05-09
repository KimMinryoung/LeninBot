# Tool Allow-List Current State

최종 확인 기준: 2026-05-09 코드 트리.

Tool visibility is intentionally split by execution surface. There is no single JSON file that owns every allow-list.

## Sources of Truth

| Layer | File | Purpose |
|---|---|---|
| Global registry | `runtime_tools/registry.py` | all possible tool definitions and handlers |
| Telegram orchestrator | `runtime_tools/allowlists.py` | tools visible to the top-level Telegram orchestrator |
| Specialist agents | `agents/*.py` | `AgentSpec.tools` per agent |
| Agent runtime overlay | `config/agent_runtime.json` | provider/model/budget/finalization/terminal overrides, not normal tools |
| Public web chat | `web_chat.py` | `_WEB_ALLOWED_TOOLS` plus `WEB_READ_SELF_TOOL` |

## Global Registry

`runtime_tools/registry.py` starts with core tools such as `vector_search`, `knowledge_graph_search`, and `web_search`, then appends focused tool families:

- `runtime_tools/filesystem.py`: programmer-style filesystem and Python execution tools
- `runtime_tools/fetch.py`: URL/file/document fetch and conversion tools
- `runtime_tools/media.py`: image generation and browser automation tools
- `runtime_tools/social.py`: social/platform tools
- `runtime_tools/a2a.py`: A2A client tool
- additional self, DB, research, private-report, post-edit, crypto, and broadcast tools are imported into the final registry in the lower part of `runtime_tools/registry.py`

A tool is callable only if both its definition and handler are present after filtering.

## Telegram Orchestrator

The orchestrator allow-list is `ORCHESTRATOR_TOOL_NAMES` in `runtime_tools/allowlists.py`. This surface should remain small: delegation, mission/context operations, safe recall/search, and user-facing coordination. It should not expose broad filesystem or code execution tools.

When adding a new orchestrator tool:

1. Add the definition and handler to the global registry.
2. Add the name to `ORCHESTRATOR_TOOL_NAMES`.
3. Confirm the tool is safe for a top-level conversational loop.
4. Run or update `scripts/smoke_tool_allowlists.py`.

## Specialist Agents

Each `AgentSpec` declares its own `tools` list. Current registered agents are:

- `programmer`
- `analyst`
- `scout`
- `browser`
- `visualizer`
- `diary`
- `stasova`
- `diplomat`
- `autonomous_project`

`AgentSpec.filter_tools()` is fail-closed. If a tool name is absent from the spec, that agent cannot call it even if the global registry contains it.

`finalization_tools` and `terminal_tools` are special execution controls, not general allow-list replacements:

- `finalization_tools` remain callable after budget/round exhaustion so an agent can persist work.
- `terminal_tools` end the loop after a successful call and use the tool result as the task report.

## Public Web Chat

Public web chat is not the Telegram orchestrator. `web_chat.py` builds `_web_tools` from `_WEB_ALLOWED_TOOLS` and adds `WEB_READ_SELF_TOOL`. This keeps anonymous public users away from Telegram-only, email, code, filesystem, and broad operational tools.

When changing public web tools, review `scripts/smoke_webchat_security.py` and the frontend caller behavior.

## Agent Runtime Config Is Not the Tool Registry

`config/agent_runtime.json` overlays execution settings onto registered agents. It can adjust finalization and terminal tool controls, but normal tool ownership should stay in Python specs so capability boundaries are reviewed with code.

## Change Checklist

- Update `runtime_tools/registry.py` for new global tools.
- Update exactly the relevant allow-list/spec.
- Keep public web chat and Telegram orchestrator separate.
- Add smoke coverage when a new tool can write, publish, send, browse, pay, or execute.
- Update this document only after verifying names in code.
