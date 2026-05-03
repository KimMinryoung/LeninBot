# Tool Allow-list Current State

Date: 2026-05-03

This note records the current runtime tool-visibility structure after the
published-content editing/cache-purge cleanup. It is intentionally descriptive,
not a full redesign.

## Summary

Tool allow-lists are not currently managed from one JSON document. They are
split across four runtime paths:

1. Global tool registry: `telegram/tools.py`
2. Telegram orchestrator allow-list: `_ORCHESTRATOR_TOOLS` inside
   `telegram/bot.py::_chat_with_tools`
3. Specialist agent allow-lists: `AgentSpec.tools` in `agents/*.py`
4. Public web chat allow-list: `_WEB_ALLOWED_TOOLS` plus `WEB_READ_SELF_TOOL`
   in `web_chat.py`

`config/agent_runtime.json` does not own tool allow-lists. It overlays provider,
model, budget, max rounds, finalization tools, terminal tools, and related
runtime settings onto registered `AgentSpec` objects.

## Global Registry

`telegram/tools.py` builds the full set of possible tools and handlers:

- base tools such as `web_search`, `fetch_url`, file tools, etc.
- self tools from `self_tools.py`
- DB/query/publishing tools
- post/research/private-report tools
- crypto and channel tools
- image/browser/Moltbook/Mersoom/A2A tools

This registry is not an allow-list. It is the superset from which each runtime
selects.

## Orchestrator

The Telegram orchestrator does not use `AgentSpec`. It has a local curated set
inside `telegram/bot.py::_chat_with_tools`:

```python
_ORCHESTRATOR_TOOLS = {
    "delegate", "multi_delegate",
    "mission",
    "web_search", "fetch_url",
    "knowledge_graph_search", "vector_search",
    "get_finance_data",
    "check_wallet", "swap_eth_to_usdc", "transfer_usdc", "pay_and_fetch",
    "broadcast_to_channel",
    "recall_experience", "save_self_analysis",
    "write_kg_structured",
    "read_self",
    "list_agent_tools",
    "run_agent",
}
```

The orchestrator intentionally cannot use programming/file tools directly. Code
work is delegated to `programmer`. Routine public-content edits should be
delegated to the content-owning agent:

- diary entries -> `diary`
- research documents, task reports, blog posts, curations -> `analyst`

The orchestrator should not invent filesystem paths when delegating. Delegation
context should carry user-provided identifiers such as public URL, slug,
post_id, DB document identifier, error text, command output, or visible symptom.
Specialist agents inspect code/repositories themselves when they need code
context.

## Specialist Agents

Specialist agents are registered in `agents/__init__.py` as `AgentSpec`
instances. Each spec declares a static `tools=[...]` list. Task execution calls:

```python
spec.filter_tools(BASE_TOOLS, BASE_HANDLERS)
```

from `agents/base.py`, so each agent only sees tools listed in its spec. If
`tools=[]`, the current `AgentSpec` implementation treats that as "all tools
allowed"; avoid using empty lists for normal in-process agents.

Current notable ownership:

- `analyst`: research/publication tools, `edit_public_post`, `edit_research`,
  private report tools, analysis/retrieval/KG tools
- `diary`: `save_diary`, `edit_public_post`, diary/retrieval/KG tools
- `programmer`: currently `provider="codex"` in `config/agent_runtime.json`;
  delegated programmer tasks bypass the LeninBot tool loop and run Codex CLI.
  Its non-Codex `AgentSpec.tools` is currently only `["list_agent_tools"]` for
  routing introspection.
- `autonomous_project`: project-state tools plus publishing/editing tools for
  cyber-lenin.com
- `diplomat`: email/A2A tools
- `scout`, `browser`, `visualizer`: external collection, browser automation, and
  image workflows respectively

## Web Chat

Public web chat is not an `AgentSpec`.

Entry path:

```text
api.py POST /chat -> web_chat.handle_web_chat()
```

`web_chat.py` defines its own system prompt and tool filter:

```python
_WEB_ALLOWED_TOOLS = {
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
}

_web_tools = [t for t in TOOLS if t.get("name") in _WEB_ALLOWED_TOOLS] + [WEB_READ_SELF_TOOL]
```

`WEB_READ_SELF_TOOL` is a web-safe replacement for normal `read_self`; its
handler redacts private logs, credentials, raw local paths, operational traces,
and private task report bodies.

This means web chat has a fourth allow-list path separate from both
orchestrator and `AgentSpec`.

## Introspection

`self_tools.py` now exposes:

- `get_agent_tool_manifest(...)`: Python function for runtime tool visibility
- `list_agent_tools`: tool wrapper returning JSON

Important detail: orchestrator tool visibility is only exact from an active
orchestrator tool context, because the actual orchestrator allow-list is local
to `_chat_with_tools`. The tool wrapper captures the current `merged_tools`
closure and reports that. Direct Python calls outside an active orchestrator
context can report specialist-agent manifests, but cannot truthfully reconstruct
the active orchestrator list unless it is passed in.

## Current Design Debt

The clean direction is probably not a standalone JSON allow-list file for every
runtime. Keeping specialist ownership in `AgentSpec.tools` is still readable and
keeps prompt/tool ownership together.

The two awkward cases are:

1. Orchestrator is not an `AgentSpec`; its allow-list lives inside
   `_chat_with_tools`.
2. Web chat is not an `AgentSpec`; it has bespoke prompt/tool filtering in
   `web_chat.py`.

Future cleanup options:

- Keep specialist allow-lists in `AgentSpec.tools`.
- Consider an `ORCHESTRATOR` spec-like object only if it can avoid making the
  existing `_chat_with_tools` path harder to reason about.
- Consider a `WEB_CHAT` spec-like object later, but preserve the web-safe
  `read_self` boundary. Either keep the special handler override or split it
  into a separate `web_read_self` tool.
- Do not move web chat hastily: it has public-safety redaction requirements that
  are different from Telegram and specialist task execution.

