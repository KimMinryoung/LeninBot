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
| Inbound MCP gateway | `mcp_gateway/policy.py` | profile-based allow-lists for developer/operator MCP clients |

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

`check_wallet` is intentionally exposed to public web chat as a read-only wallet visibility tool. It can show public wallet address/balance information, but it must not expose private keys, credential paths, signing, transfer, swap, or payment capabilities. Payment tools such as `pay_and_fetch`, code execution tools, filesystem write tools, email/A2A send tools, and publishing tools must remain absent from the web-chat allow-list.

When changing public web tools, review `scripts/smoke_webchat_security.py` and the frontend caller behavior.

## Inbound MCP Gateway

`mcp_gateway.server` is an on-demand stdio MCP server for local developer/operator clients such as Codex or Claude Code. It is not the Telegram orchestrator and not the public web chat API.

The gateway is fail-closed by profile. The old `readonly` profile name is accepted only as an alias for `inspect`:

- `inspect` is the default profile. It exposes gateway-local inspection tools, `kg_integrity_check`, and selected read-only runtime tools: `vector_search`, `knowledge_graph_search`, and `fetch_url`.
- `operator` includes all `inspect` tools and adds `readonly_query_db`, which shells out to `scripts/query-db` so SQL remains limited to a single `SELECT`/`WITH`/`SHOW`/`EXPLAIN` diagnostic in a read-only transaction. It also adds `bounded_query_db`, which reuses the existing `runtime_tools.db` guard for one-statement DB work, and `kg_maintenance_run`, which exposes only bounded KG maintenance scripts and requires an explicit confirmation string for mutating runs.

The gateway must not export the global registry wholesale. High-risk runtime tools such as filesystem writes, arbitrary Python execution, service restart, email/A2A send, publishing, payment/signing, arbitrary KG writes/Cypher, and broad or unguarded DB mutation remain absent from MCP profiles.

When changing MCP tools or profiles, update `mcp_gateway/policy.py` and run `scripts/smoke_mcp_gateway.py`.

## Agent Runtime Config Is Not the Tool Registry

`config/agent_runtime.json` overlays execution settings onto registered agents. It can adjust finalization and terminal tool controls, but normal tool ownership should stay in Python specs so capability boundaries are reviewed with code.

## Change Checklist

- Update `runtime_tools/registry.py` for new global tools.
- Update exactly the relevant allow-list/spec.
- Keep public web chat and Telegram orchestrator separate.
- Keep MCP profiles separate from public web chat and Telegram orchestrator.
- Add smoke coverage when a new tool can write, publish, send, browse, pay, or execute.
- Treat read-only wallet visibility separately from signing/payment capability; `check_wallet` may be public, but transfer/swap/pay tools may not.
- Update this document only after verifying names in code.
