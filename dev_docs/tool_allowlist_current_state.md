# Tool Allow-List Current State

최종 확인 기준: 2026-07-25 코드 트리.

Tool visibility is intentionally split by execution surface. There is no single JSON file that owns every allow-list.

## Sources of Truth

| Layer | File | Purpose |
|---|---|---|
| Global registry | `runtime_tools/registry.py` | all possible tool definitions and handlers |
| Telegram orchestrator | `tool_gateway/profiles.py` (`TELEGRAM_ORCHESTRATOR_TOOLS`; compatibility alias in `runtime_tools/allowlists.py`) | schemas and executable handlers available to the top-level Telegram orchestrator |
| Specialist agents | `agents/*.py` | `AgentSpec.tools` per agent |
| Agent runtime overlay | `config/agent_runtime.json` | provider/model/budget/finalization/terminal overrides, not normal tools |
| Public web chat | `services/web_chat.py` | persona-specific allowed tools plus web-only `WEB_READ_SELF_TOOL` / `WEB_PERSONA_CONTEXT_TOOL` |
| Roleplay bot | `telegram/roleplay_bot.py` | `_TOOL_NAMES` — its own narrow read-only set, independent of the orchestrator |
| Inbound MCP gateway | `mcp_gateway/policy.py` | profile-based allow-lists for developer/operator MCP clients |
| Runtime tool gateway | `tool_gateway/` | named surface profiles, shared visibility filtering, batch dispatch, and security/audit integration |

## Global Registry

`runtime_tools/registry.py` starts with core tools such as `vector_search`, `knowledge_graph_search`, and `web_search`, then appends focused tool families:

`web_search` keeps one stable tool name and schema across every allow-listed
surface, while `runtime_tools/web_search.py` owns the provider chain. The default
order is Tavily then Brave (`WEB_SEARCH_PROVIDERS=tavily,brave`); a provider
error opens a short process-local circuit and the next configured provider is
tried once. Empty successful results do not fan out to another billed provider.
Tavily calls explicitly disable automatic depth promotion, omit answer/raw-page
payloads, report credit usage to logs, and use two focused chunks per source
only for explicitly requested advanced searches.

- `runtime_tools/filesystem.py`: programmer-style filesystem and Python execution tools
- `runtime_tools/fetch.py`: URL/file/document fetch and conversion tools
- `runtime_tools/media.py`: image generation and browser automation tools
- `runtime_tools/social.py`: social/platform tools
- `runtime_tools/a2a.py`: A2A client tool
- additional self, DB, research, private-report, post-edit, crypto, and broadcast tools are imported into the final registry in the lower part of `runtime_tools/registry.py`

A tool is callable only if both its definition and handler are present after filtering.

## Telegram Orchestrator

The orchestrator allow-list is `ORCHESTRATOR_TOOL_NAMES` in `runtime_tools/allowlists.py`. `build_orchestrator_toolset()` applies that list to both schemas and handlers, so a provider-emitted hidden tool name has no executable handler. This surface should remain small: delegation, mission/context operations, safe recall/search, and user-facing coordination. It should not expose broad filesystem or code execution tools.

CommUlingo is the bounded content-write exception. The orchestrator sees `commulingo_people` plus seven target-specific writes (`commulingo_person_create`, `commulingo_person_update`, `commulingo_section_save`, `commulingo_event_link`, `commulingo_office_row_save`, `commulingo_term_create`, `commulingo_term_update`). The global registry no longer contains a generic union write tool. The narrow tools are shared global-registry tools and still execute through `tool_gateway.dispatcher`, security authorization/audit, and the common CommUlingo normalization/transaction core.

When adding a new orchestrator tool:

1. Add the definition and handler to the global registry.
2. Add the name to `ORCHESTRATOR_TOOL_NAMES`.
3. Confirm the tool is safe for a top-level conversational loop.
4. Run or update `scripts/smoke_tool_allowlists.py`.

## Read Tool Pagination

Long body reads should keep per-call character limits and expose continuation parameters instead of forcing agents to rely on one large result. `fetch_url` uses `max_chars` + `offset`; its local HTTP paths also reject unsafe schemes/ports, non-global DNS destinations, and unsafe redirects before connecting. `read_self` detail reads use `max_chars` + `offset`; `check_inbox` list mode returns bounded previews with `folder`/`uid`, and a single-email read uses `folder` + `uid` + `body_offset` + `body_max_chars`. `read_file`/`read_document` use their existing line or character pagination parameters.

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
- `analyst` — uses `commulingo_people` plus the same six target-specific CommUlingo writes
- `commulingo_curator` — research/read tools plus target-specific `commulingo_person_create`, `commulingo_person_update`, `commulingo_section_save`, `commulingo_event_link`, and `commulingo_term_create`; every scheduled stage filters that list again so only its applicable terminal write tool(s) are visible. New-person discovery uses a runner-local typed `commulingo_candidate_select` terminal and has no database write tool.

`AgentSpec.filter_tools()` is fail-closed and delegates the actual schema/handler filtering to `tool_gateway.selection.filter_agent_tools()`. If a tool name is absent from the spec, that agent cannot call it even if the global registry contains it.

`edit_content` includes diary maintenance actions in addition to field edits. For `content_type="diary"`, `action="delete"` and `action="unpublish"` are destructive public-removal operations and require `confirm=true`; unpublish removes the row from `ai_diary` because the diary table has no private status field. The diary and analyst agents can see `edit_content`, but routine diary ownership remains with the diary agent.

`finalization_tools` and `terminal_tools` are special execution controls, not general allow-list replacements:

- `finalization_tools` remain callable after budget/round exhaustion so an agent can persist work.
- `terminal_tools` end the loop after a successful call and use the tool result as the task report.

## Public Web Chat

Public web chat is not the Telegram orchestrator. `services/web_chat.py` builds tools from the active `PersonaSpec.allowed_tools`, excluding shared registry `read_self` and injecting web-only handlers where allowed. `WEB_READ_SELF_TOOL` is the public-safe Cyber-Lenin self-inspection surface. `WEB_PERSONA_CONTEXT_TOOL` is a persona-bound dossier reader: the handler closes over the active persona and resolves reads only under `identity/web_personas/<context_dir>/knowledge`, so one persona cannot request another persona's notes. This keeps anonymous public users away from Telegram-only, email, code, filesystem, and broad operational tools.

`check_wallet` is intentionally exposed to public web chat as a read-only wallet visibility tool. It can show public wallet address/balance information, but it must not expose private keys, credential paths, signing, transfer, swap, or payment capabilities. Payment tools such as `pay_and_fetch`, code execution tools, filesystem write tools, email/A2A send tools, and publishing tools must remain absent from the web-chat allow-list.

Every public persona also receives the common web runtime rules from
`services/web_personas.py`. They require exact named-entity/date anchors in the
first verification query, forbid treating an irrelevant or empty search result
as evidence, and forbid claiming an external mutation unless a successful
action-capable tool actually performed it. Because this surface is read-only,
stored/public-content mutation requests are finalized fail-closed without
repeating identifying details or claiming that an operator was contacted.

After successful `web_search` or `fetch_url` use, the final response may cite
only URLs extracted from those successful tool results. The response format is
body markers `[^1]`, `[^2]`, and so on, followed by URL-only definitions:

```markdown
주장 본문.[^1]

[^1]: https://example.com/source
```

Titles, publishers, dates, descriptions, numbered source lists, raw body URLs,
Markdown source links, and model-invented URLs are not valid citation
definitions. `services/web_chat.py` normalizes the final persisted/SSE answer
to this contract after the provider loop returns.

When changing public web tools, review `scripts/smoke_webchat_security.py`, persona-specific tool labels in `services/web_chat.py`, and the frontend caller behavior.

Current public persona-specific additions:

- `gramsci`: `vector_search`, `web_search`, `fetch_url`, and `read_persona_context`. Gramsci primary writings should come from `vector_search(layer="core_theory", author="Gramsci")`; `services/web_chat.py` also performs a bounded server-side preflight vector lookup for Gramsci theory/concept triggers. The persona dossier is supplemental reading protocol and strategy scaffolding.

## Public A2A

Inbound A2A is an unauthenticated public surface and is read-only. Every A2A skill
profile may expose only `read`, `fetch`, or `wallet_read` risk classes;
`geopolitical-analysis` uses search/read tools and does not receive
`write_kg_structured`. `security_gateway` applies the same class restriction at
execution time even if profile selection or provider output is misconfigured.

## Roleplay Bot

`leninbot-roleplay.service` (`telegram/roleplay_bot.py`) is a separate identity, not the Cyber-Lenin orchestrator. Its allow-list is the inline `_TOOL_NAMES` list and is intentionally minimal and read-only: `vector_search`, `knowledge_graph_search`, `web_search`, `fetch_url`. These are selected from the global registry (`TOOLS` / `TOOL_HANDLERS`) at import time. No task execution, KG writes, filesystem, code, email/A2A, or publishing tools are exposed. Adding tools here is deliberate — keep this surface to safe read-only knowledge lookups.

## Inbound MCP Gateway

`mcp_gateway.server` is an on-demand stdio MCP server for local developer/operator clients such as Codex or Claude Code. It is not the Telegram orchestrator and not the public web chat API.

The gateway is fail-closed by profile. The old `readonly` profile name is accepted only as an alias for `inspect`:

- `inspect` is the default profile. It exposes gateway-local inspection tools, `kg_integrity_check`, and selected read-only runtime tools: `vector_search`, `knowledge_graph_search`, and `fetch_url`.
- `operator` includes all `inspect` tools and adds `readonly_query_db`, which shells out to `scripts/query-db` so SQL remains limited to a single `SELECT`/`WITH`/`SHOW`/`EXPLAIN` diagnostic in a read-only transaction. It also adds `bounded_query_db`, which reuses the existing `runtime_tools.db` guard for one-statement DB work, and `kg_maintenance_run`, which exposes only bounded KG maintenance scripts and requires an explicit confirmation string for mutating runs.

The gateway must not export the global registry wholesale. High-risk runtime tools such as filesystem writes, arbitrary Python execution, service restart, email/A2A send, publishing, payment/signing, arbitrary KG writes/Cypher, and broad or unguarded DB mutation remain absent from MCP profiles.

When changing MCP tools or profiles, update `mcp_gateway/policy.py` and run `scripts/smoke_mcp_gateway.py`. MCP catalog/handler filtering uses `tool_gateway.selection`, but MCP remains a separate inbound surface rather than the runtime agent gateway.

## Runtime Tool Gateway

`tool_gateway` is the internal runtime facade for tool selection and dispatch. `tool_gateway.profiles` owns reusable surface allow-lists for the Telegram orchestrator, web personas, A2A skills, the standalone roleplay bot, and MCP profiles. The MCP `list_runtime_tool_profiles` inspection tool reports those profiles plus specialist `AgentSpec.tools` for humans and developer agents. `runtime_tools/allowlists.py`, `services/web_personas.py`, `services/a2a_handler.py`, `telegram/roleplay_bot.py`, and `mcp_gateway/policy.py` keep compatibility aliases where needed. `AgentSpec.tools` remains agent-local. Provider loops import `execute_tools_batch()` through `tool_gateway.dispatcher`, whose `execute_tool()` implementation runs `security_gateway.authorize()` and audit per call.

Audit attribution is cross-surface rather than web-session-specific. Every
annotated LLM/tool loop has a unique `request_id`; nested loops use
`parent_request_id`; durable work is grouped by `scope_type/scope_id` such as
`telegram_message`, `telegram_task`, `autonomous_project`, `maintenance_job`,
`writer_project`, `web_chat_turn`, or `a2a_task`. Channel-specific IDs remain
available as secondary fields where they exist.

This does not make all surfaces share one allow-list. It centralizes the mechanics while preserving separate orchestrator, agent, webchat, A2A, and MCP boundaries. See `dev_docs/tool_gateway.md`.

## Agent Runtime Config Is Not the Tool Registry

`config/agent_runtime.json` overlays execution settings onto registered agents. It can adjust finalization and terminal tool controls, but normal tool ownership should stay in Python specs so capability boundaries are reviewed with code.

## Change Checklist

- Update `runtime_tools/registry.py` for new global tools.
- Update exactly the relevant allow-list/spec.
- Keep public web chat, public A2A, and Telegram orchestrator separate.
- Keep every public A2A profile read-only; KG mutation belongs to trusted internal runtimes.
- Keep MCP profiles separate from public web chat, public A2A, and Telegram orchestrator.
- Add smoke coverage when a new tool can write, publish, send, browse, pay, or execute.
- Treat read-only wallet visibility separately from signing/payment capability; `check_wallet` may be public, but transfer/swap/pay tools may not.
- Update this document only after verifying names in code.
