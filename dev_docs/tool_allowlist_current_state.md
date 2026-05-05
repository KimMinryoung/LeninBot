# Tool Allow-list Current State

Date: 2026-05-05

This note records the current runtime tool-visibility structure after the
published-content editing/cache-purge cleanup. It is intentionally descriptive,
not a full redesign.

## Summary

Tool allow-lists are not currently managed from one JSON document. They are
split across four runtime paths:

1. Global tool registry: `runtime_tools/registry.py`
2. Telegram orchestrator allow-list: `ORCHESTRATOR_TOOL_NAMES` in
   `runtime_tools/allowlists.py`
3. Specialist agent allow-lists: `AgentSpec.tools` in `agents/*.py`
4. Public web chat allow-list: `_WEB_ALLOWED_TOOLS` plus `WEB_READ_SELF_TOOL`
   in `web_chat.py`

`config/agent_runtime.json` does not own tool allow-lists. It overlays provider,
model, budget, max rounds, finalization tools, terminal tools, and related
runtime settings onto registered `AgentSpec` objects.

## Global Registry

`runtime_tools/registry.py` builds the full set of possible tools and handlers. Some
tool families now live in focused modules that export both tool definitions and
handler maps:

- base tools such as `web_search`, `fetch_url`, file tools, etc.
- self tools from `self_tools.py`
- DB/query/publishing tools
- post/research/private-report tools
- crypto and channel tools
- `runtime_tools/fetch.py`: URL/file/document fetch and conversion tools
- `runtime_tools/filesystem.py`: programmer filesystem and Python execution tools
- `runtime_tools/media.py`: image generation and browser automation tools
- `runtime_tools/social.py`: Moltbook and Mersoom tools
- `runtime_tools/a2a.py`: A2A client tool

This registry is not an allow-list. It is the superset from which each runtime
selects.

## Orchestrator

The Telegram orchestrator does not use `AgentSpec`. It has a curated set in
`runtime_tools/allowlists.py`:

```python
ORCHESTRATOR_TOOL_NAMES = {
    "delegate", "multi_delegate",
    "mission",
    "web_search", "fetch_url", "fetch_x_post",
    "knowledge_graph_search", "vector_search",
    "get_finance_data",
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

`fetch_x_post` is intentionally not enabled for public web chat yet. It consumes
the server-side X API quota, so exposing it publicly should wait for explicit
web rate limiting and abuse controls.

This means web chat has a fourth allow-list path separate from both
orchestrator and `AgentSpec`.

## Introspection

`self_tools.py` now exposes:

- `get_agent_tool_manifest(...)`: Python function for runtime tool visibility
- `list_agent_tools`: tool wrapper returning JSON

Important detail: when called from an active orchestrator tool context, the tool
wrapper reports the exact `merged_tools` closure. Outside that context,
`get_agent_tool_manifest(agent="orchestrator")` reconstructs the static
orchestrator view from `runtime_tools.allowlists.select_orchestrator_tools()`.

`scripts/smoke_tool_allowlists.py` verifies:

- every global tool definition has a handler, except dynamic closure-backed
  tools (`mission`, `run_agent`)
- orchestrator allow-list names exist in the global registry
- specialist `AgentSpec.tools` names exist in the global registry or in known
  dynamic task/project tool sets (`save_finding`, autonomous project state tools)
- terminal/finalization tools remain inside each agent's declared allow-list
- web chat's bespoke allow-list resolves to real global tools plus its safe
  `read_self` override

## Current Design Debt

The clean direction is probably not a standalone JSON allow-list file for every
runtime. Keeping specialist ownership in `AgentSpec.tools` is still readable and
keeps prompt/tool ownership together.

The remaining awkward case is:

1. Web chat is not an `AgentSpec`; it has bespoke prompt/tool filtering in
   `web_chat.py`.

Future cleanup options:

- Keep specialist allow-lists in `AgentSpec.tools`.
- Keep the orchestrator allow-list in `runtime_tools/allowlists.py` unless a
  future `ORCHESTRATOR` spec-like object can avoid making `_chat_with_tools`
  harder to reason about.
- Consider a `WEB_CHAT` spec-like object later, but preserve the web-safe
  `read_self` boundary. Either keep the special handler override or split it
  into a separate `web_read_self` tool.
- Do not move web chat hastily: it has public-safety redaction requirements that
  are different from Telegram and specialist task execution.
