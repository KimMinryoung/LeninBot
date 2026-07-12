# Runtime Tool Gateway

최종 확인 기준: 2026-07-12 코드 트리.

`tool_gateway` is the runtime facade for LeninBot tool visibility and dispatch. It is separate from `mcp_gateway`: MCP is an inbound developer/operator server, while `tool_gateway` is the internal control-plane boundary used when Cyber-Lenin, specialist agents, web chat, or A2A model loops call runtime tools.

The gateway now owns reusable surface profiles for orchestrator, web persona, A2A, roleplay, and MCP tool sets. Specialist agents still keep role-specific `AgentSpec.tools` next to their prompts; the gateway reads those through `AgentSpec.filter_tools()`. Runtime callers route through named gateway helpers so capability selection, batching, security authorization, and audit have one obvious entrypoint.

## Runtime Flow

```text
interface / agent context
  -> tool_gateway.selection builds visible tools + handlers
  -> provider loop receives compacted tool schemas
  -> model emits tool_use / tool_calls
  -> tool_gateway.dispatcher.execute_tools_batch
      -> tool_gateway.dispatcher.execute_tool
          -> security_gateway.authorize
          -> selected TOOL_HANDLERS[name](**args)
          -> security_gateway.audit
```

Read-only/idempotent tools in consecutive batches may run concurrently through the existing parallel-safe list. Other tools run sequentially in model emission order.

## Package Layout

| Module | Role |
|---|---|
| `tool_gateway.profiles` | Named source-of-truth tool profiles for orchestrator, web persona, A2A, roleplay, and MCP surfaces |
| `tool_gateway.selection` | Common allow-list filtering helpers for tool schemas and handlers |
| `tool_gateway.dispatcher` | Runtime dispatch implementation for `execute_tool`, `execute_tools_batch`, and prompt schema compaction |
| `tool_gateway.security` | Runtime adapter/re-export for `security_gateway` caller context, authorization, and audit |
| `tool_gateway.inference` | Default and resolved delegated-agent input/output ceilings, continuation count, round/budget envelope, replay-safe reads, and provider thinking/budget policy |

## Current Sources Of Truth

The gateway is a facade, not a wholesale policy rewrite. These modules still own their established contracts:

| Concern | Current owner | Gateway usage |
|---|---|---|
| Global tool definitions and handlers | `runtime_tools/registry.py` | Inputs to selection helpers |
| Telegram orchestrator allow-list | `tool_gateway.profiles.TELEGRAM_ORCHESTRATOR_TOOLS` | `runtime_tools/allowlists.py` keeps a compatibility alias and uses `tool_gateway.selection.select_tools_by_name` |
| Specialist agent allow-lists | `agents/*.py` `AgentSpec.tools` | `AgentSpec.filter_tools()` delegates to `tool_gateway.selection.filter_agent_tools` |
| Web chat persona tool set | `tool_gateway.profiles.WEB_*_TOOLS` | `web_personas.py` aliases profile values; `web_chat.py` uses `tool_gateway.selection.build_toolset` before injecting web-only safe tools |
| A2A skill tool sets | `tool_gateway.profiles.A2A_*_TOOLS` | `a2a_handler.py` aliases profile values and uses `tool_gateway.selection.build_toolset` |
| Roleplay Telegram tool set | `tool_gateway.profiles.ROLEPLAY_TELEGRAM_TOOLS` | `telegram/roleplay_bot.py` uses `tool_gateway.selection.build_toolset` and `tool_gateway.security` caller attribution |
| MCP profile allow-lists | `tool_gateway.profiles.MCP_*` | `mcp_gateway/policy.py` keeps compatibility aliases; MCP remains a separate inbound surface; `list_runtime_tool_profiles` exposes runtime allow-list inspection through MCP |
| Execution authorization and audit | `security_gateway/` | Called from `tool_gateway.dispatcher.execute_tool()` for every executed tool |
| Delegated-agent inference envelope | `tool_gateway.inference` + `config/agent_runtime.json` | Resolves one policy before task provider dispatch; all normal provider loops receive the same input/output/continuation settings |

## Invariants

- Tool visibility remains fail-closed: callers only see explicitly allowed tools.
- Reusable surface allow-lists should be added to `tool_gateway.profiles`; keep agent-specific `AgentSpec.tools` beside agent definitions unless there is a concrete reuse reason.
- A visible tool must still have a registered handler to execute.
- The execution-time `security_gateway` remains defense-in-depth, not a replacement for allow-lists.
- MCP gateway is not the runtime gateway; keep the names and docs distinct.
- `tool_loop_common` re-exports dispatcher functions only for compatibility; new runtime imports should use `tool_gateway.dispatcher`.
- Gateway-managed input overflow must not silently drop conversation text. Large results from explicitly replay-safe read-only tools become replay checkpoints that preserve the preceding tool call and its exact arguments; the model must re-run that call when it needs the complete source. Side-effecting tools are never marked for replay.
- Output continuation is bounded by policy and does not consume an ordinary tool round solely because a completion hit its output ceiling.

## Verification

For changes that touch this boundary, run at minimum:

```bash
venv/bin/python scripts/smoke_tool_allowlists.py
venv/bin/python scripts/smoke_security_gateway.py
venv/bin/python scripts/smoke_mcp_gateway.py
python3 -m py_compile tool_gateway/*.py runtime_tools/allowlists.py agents/base.py claude_loop.py openai_tool_loop.py web_chat.py a2a_handler.py mcp_gateway/tools.py
```
