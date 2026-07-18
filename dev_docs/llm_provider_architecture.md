# LLM Provider Architecture

최종 확인 기준: 2026-07-17 코드 트리.

LeninBot은 provider와 모델 티어를 런타임 설정으로 해석한다. Telegram chat, background task, autonomous loop, public web chat은 서로 다른 provider 설정을 가질 수 있다.

## Config Sources

| Source | Owner | Purpose |
|---|---|---|
| `config.json` | `bot_config.py` | mutable runtime config saved by Telegram `/config` |
| `config/agent_runtime.json` | `agents/runtime_config.py` | per-agent provider/model/budget/round/input-output/continuation/thinking/tool overlay |
| systemd credentials / env | `secrets_loader.py` | API keys and provider endpoint secrets |

`bot_config.py` defaults are authoritative when `config.json` is missing a key.

## Provider Paths

```
Claude
  bot_config._claude
  -> claude_loop.chat_with_tools()
  -> Anthropic Messages API

OpenAI
  bot_config._openai_client
  -> openai_tool_loop.chat_with_tools(client=...)
  -> OpenAI Chat Completions API

Kimi
  bot_config._kimi_client
  -> openai_tool_loop.chat_with_tools(client=...)
  -> Moonshot OpenAI-compatible Chat Completions API

DeepSeek
  bot_config._deepseek_anthropic_client
  -> claude_loop.chat_with_tools(client=...)
  -> DeepSeek Anthropic-compatible Messages API

DeepSeek web chat / browser automation
  bot_config._deepseek_anthropic_client
  -> claude_loop.chat_with_tools(client=..., thinking={"type": "disabled"})
  -> DeepSeek Anthropic-compatible Messages API

DeepSeek roleplay bot (leninbot-roleplay.service)
  bot_config._deepseek_anthropic_client
  -> claude_loop.chat_with_tools(client=..., thinking={"type": "enabled"}, output_config={"effort": "high"})
  -> DeepSeek Anthropic-compatible Messages API

Personal fiction writer (/writer)
  creative_writer._client()
  -> claude_loop.chat_with_tools(client=..., model="claude-fable-5", writer tools)
  -> Anthropic Messages API or provider-compatible Messages endpoint, no shared provider tier

Local
  llm.client backend
  -> openai_tool_loop.chat_with_tools(base_url=...)
  -> OpenAI-compatible local server
```

OpenAI-compatible providers share `openai_tool_loop.py`. Claude uses `claude_loop.py` because Anthropic tool-use message structure is different. Kimi K3 uses `MOONSHOT_API_KEY`, defaults to `https://api.moonshot.ai/v1`, and sends model ID `kimi-k3`. K3 always reasons and currently accepts only `reasoning_effort=max`; the Kimi path preserves the API's `reasoning_content` in replayed assistant messages across tool and continuation rounds while keeping it out of the user-facing answer, as required by Moonshot's multi-turn/tool-call protocol. Budget accounting uses Moonshot's launch pricing ($3/M cache-miss input, $0.30/M cache-hit input, $15/M output). The official API reference is [Kimi API Quickstart](https://platform.kimi.ai/docs/api/quickstart).

Kimi의 개별 호출이 prompt-side `400 content_filter` 또는 `high risk`로 거부되면 `openai_tool_loop.py`의 API-call 경계에서 같은 메시지와 도구 스키마를 DeepSeek Pro의 OpenAI-compatible endpoint로 재시도하며, 해당 tool loop의 남은 호출도 DeepSeek에 고정한다. 이 경계는 일반 라운드뿐 아니라 도구/예산 한도 뒤의 forced-final 및 finalization-tool 후속 호출에도 공통 적용되므로 이미 실행한 도구를 처음부터 반복하지 않고 provider별 reasoning/tool replay 규약도 섞지 않는다. 로그에는 원본/폭백 provider와 model 및 필터 오류가 기록된다. 잘못된 파라미터·스키마 같은 다른 400은 이 폭백 조건에 포함하지 않는다. DeepSeek credential이 없으면 기존 Kimi 오류 처리가 그대로 적용된다.
Telegram chat, background tasks, A2A (`leninbot-a2a-api.service`), public web chat, browser worker tasks, browser-use automation, and the hourly autonomous project loop use DeepSeek's Anthropic-compatible API when `provider=deepseek`, so tool inputs arrive as structured `tool_use.input` blocks instead of OpenAI-compatible `function.arguments` JSON strings. Agent/task DeepSeek paths resolve `thinking_policy=tool_loop` through `tool_gateway.inference`; the current `DEEPSEEK_TOOL_THINKING_MODE` default is off, and operators can enable it centrally without per-call overrides. `claude_loop.py` preserves `thinking` and `redacted_thinking` assistant content blocks in replayed tool-call turns so DeepSeek receives the reasoning payload it requires on follow-up requests. Public web chat and browser automation deliberately keep DeepSeek Flash in non-thinking mode (`thinking={"type": "disabled"}`): web chat does it for lower latency, while browser automation does it because browser-use relies on forced structured tool calls and DeepSeek does not support that path with thinking enabled. DeepSeek Anthropic-compatible messages do not support image content, so browser-use runs DeepSeek as a non-vision DOM/tool controller first and retries with the configured Google/OpenAI vision fallback only if that primary attempt fails.

The roleplay bot (`leninbot-roleplay.service`, `telegram/roleplay_bot.py`) is a separate runtime, not the Cyber-Lenin orchestrator, and does not read `config.json`'s `provider`/`chat_model` keys. It pins DeepSeek directly: `_deepseek_anthropic_client` + `claude_loop.chat_with_tools`, model `deepseek-v4-flash` (via `_resolve_deepseek_model("deepseek_flash")`), with thinking **enabled** (`output_config.effort=high`). Thinking is on for answer quality; because it goes through `claude_loop`, the reasoning stays in replay-only `thinking` blocks and never appears in the user-facing reply — which is why the roleplay bot uses the Anthropic-compatible path rather than the OpenAI-compatible loop (the latter prepends reasoning to the reply). The bot ignores the global `DEEPSEEK_THINKING_MODE` env and sets its thinking inline.

The personal fiction writer (`/writer`, `api_routes/writer.py`, `novel_writer_api.py`, `creative_writer.py`) is separate from normal provider routing. Its role-level inference envelopes are centralized in `writer.config.WRITER_CALL_POLICIES` (input/output ceilings, rounds, output continuations, thinking policy); the heavy `main` and `revision` roles import the shared 160k/32k gateway defaults, while `diagnosis`, `line_edit`, and `research` retain smaller explicit role limits; input overflow uses durable summaries plus chapter-boundary anchor replay rather than silent truncation. It is not exposed through Telegram `/config`, `webchat_provider`, `webchat_model`, personas, tools, or the task/autonomous tiers. It uses the shared `claude_loop.chat_with_tools` path with `model="claude-fable-5"` by default and can route explicitly selected DeepSeek or Kimi K3 writer models through their Anthropic-compatible clients. Kimi Writer uses `https://api.moonshot.ai/anthropic`, model `kimi-k3`, and K3's default thinking; this preserves the Writer's existing Messages tool protocol, replayed thinking blocks, streaming idle retry, and 1-hour cache system blocks. It sends project instructions and manuscript context as separate prompt-cache system blocks, stores isolated project state in `writer_projects`, `writer_messages`, `writer_manuscripts`, searchable manuscript chunks, and revision metadata, and requires writer access authentication on its API routes. The dedicated process is `novel-writer-api.service` (`uvicorn novel_writer_api:app`, port 8001); frontend `/api/proxy/writer` targets that service. The public frontend path uses the existing admin login session and injects the backend admin key server-side; direct API callers may use `X-Writer-Key` (falling back to `ADMIN_API_KEY` when `WRITER_ACCESS_KEY` is unset). `WRITER_ANTHROPIC_API_KEY` can provide a separate paid Anthropic key; if absent, it falls back to `ANTHROPIC_API_KEY`.

## Runtime Config Keys

| Key | Values | Applies to |
|---|---|---|
| `provider` | `claude`, `openai`, `deepseek`, `kimi`, `local` | Telegram chat |
| `chat_model` | `high`, `medium`, `low` | Telegram chat tier |
| `task_provider` | `default`, `claude`, `openai`, `deepseek`, `kimi`, `local` | delegated/background tasks |
| `task_model` | `high`, `medium`, `low` | task tier |
| `webchat_provider` | `claude`, `openai`, `deepseek`, `kimi` | public web chat, API restart required |
| `webchat_model` | `high`, `medium`, `low` | public web chat tier |
| `autonomous_provider` | `default`, `claude`, `openai`, `deepseek`, `kimi`, `local` | hourly autonomous loop |
| `autonomous_model` | `high`, `medium`, `low` | autonomous tier |

`task_provider=default` inherits `provider`. `autonomous_provider=default` inherits the resolved task provider. Public web chat deliberately does not allow `local`.

Browser-use automation has its own environment overrides because it runs inside the separate `leninbot-browser.service` process:

| Env | Default | Meaning |
|---|---|---|
| `BROWSER_USE_PROVIDER` | `deepseek` | Primary browser-use provider; supported values are `deepseek`, `google`, and `openai`. Claude/Anthropic is deliberately mapped back to DeepSeek for cost control. |
| `BROWSER_USE_MODEL` | provider default | Optional primary browser-use model override. |
| `BROWSER_USE_VISION` | `auto` | `auto` disables screenshots for DeepSeek and enables them for Google/OpenAI; explicit true/false values override this. |
| `BROWSER_USE_VISION_FALLBACK` | `auto` | Enables one retry with a vision-capable provider after a failed DeepSeek non-vision browser-use attempt; set to `off` to disable. |
| `BROWSER_USE_VISION_FALLBACK_PROVIDER` | `google` | Vision retry provider; supported values are `google` and `openai`. |
| `BROWSER_USE_VISION_FALLBACK_MODEL` | provider default | Optional vision fallback model override. |

DeepSeek general/roleplay thinking is controlled by the following environment variables. Delegated-agent `thinking_policy=tool_loop` instead uses `DEEPSEEK_TOOL_THINKING_MODE` and `DEEPSEEK_TOOL_THINKING_EFFORT`; an explicit per-agent `thinking_policy` of `thinking`, `disabled`, or `model_default` overrides that tool-loop choice through the gateway. Browser is explicitly `disabled` in `config/agent_runtime.json` because forced structured browser calls do not support the thinking path.

DeepSeek Anthropic-compatible thinking is controlled by environment variables:

| Env | Default | Meaning |
|---|---|---|
| `DEEPSEEK_THINKING_MODE` | `thinking` | `thinking` enables DeepSeek thinking; `thinking_max` forces max effort; `disabled`/`non-thinking` disables it |
| `DEEPSEEK_THINKING_EFFORT` | `high` | `high` or `max`; invalid values fall back to `high` |

## Tier Resolution

| Tier | Claude | OpenAI | DeepSeek | Kimi | Local |
|---|---|---|---|---|---|
| `high` | `claude-opus-4-8` alias | `gpt-5.5` | `deepseek-v4-pro` | `kimi-k3` | local backend model |
| `medium` | `claude-sonnet-5` alias | `gpt-5.5-mini` | `deepseek-v4-flash` | `kimi-k3` | local backend model |
| `low` | `claude-haiku-4-5` alias | `gpt-5.5-nano` | `deepseek-v4-flash` | `kimi-k3` | local backend model |

Claude aliases are resolved lazily through Anthropic Models API and cached in-process. OpenAI, DeepSeek, Kimi, and local model IDs resolve synchronously from maps or local backend config. Kimi currently has one model in all three tiers, so the tier changes budget/display grouping but not the upstream model ID.

Kimi non-secret and secret settings:

| Setting | Default | Meaning |
|---|---|---|
| `MOONSHOT_API_KEY` | unset | Moonshot API token; client stays disabled until present |
| `MOONSHOT_BASE_URL` | `https://api.moonshot.ai/v1` | OpenAI-compatible API root |
| `MOONSHOT_ANTHROPIC_BASE_URL` | `https://api.moonshot.ai/anthropic` | Anthropic-compatible API root used by Novel Writer |
| `KIMI_MIN_OUTPUT_TOKENS` | `16384` | Minimum chat/web completion budget; K3 reasoning and visible output share this ceiling |

After adding the credential and restarting the relevant service, choose `kimi` through Telegram `/provider` or set `provider`, `task_provider`, `autonomous_provider`, or `webchat_provider` in runtime config. No live API call is made while the key is absent.

## Agent Overrides

Each `AgentSpec` may set `provider` and `model`. `None` means follow task config. Its inference envelope is resolved centrally by `tool_gateway.inference`: `max_input_tokens`, `max_output_tokens`, `max_output_continuations`, `thinking_policy`, `thinking_budget_tokens`, `max_rounds`, and `budget_usd`. Provider wrappers receive that single resolved policy rather than choosing token or thinking settings independently. `thinking` maps to DeepSeek thinking mode, Claude extended thinking with the configured budget, and GPT `reasoning_effort=high`; `disabled` maps to explicit non-thinking behavior where the provider supports it, while `model_default` omits provider reasoning controls. Current example overlay (`config/agent_runtime.json.example`) pins:

| Agent | Default override |
|---|---|
| `programmer` | `provider="codex"`; Codex CLI owns the actual code execution tool loop |
| `autonomous_project` | DeepSeek Pro, lower budget, publication finalization tools |
| `browser`, `scout`, `stasova`, `diary` | DeepSeek by default |
| `analyst`, `diplomat`, `visualizer` | inherit task config unless overridden |

`AgentSpec.effective_provider()` renders `codex`, `moon`, and `local` prompts as local/Markdown format. Claude gets XML-style rendering; OpenAI-compatible providers get Markdown rendering through `llm/prompt_renderer.py`.

The browser worker accepts tier names, legacy model aliases such as `deepseek_flash`, and official API model IDs, but normalizes them before sending requests upstream. `config/agent_runtime.json` should prefer official API IDs for browser automation because that worker calls DeepSeek's Anthropic-compatible endpoint directly.

## Model Context Injection

`get_current_model_selection(kind, provider_override=None)` returns provider, tier, alias, model ID, display name, and resolution status for `chat`, `task`, `autonomous`, and `webchat`. Telegram and task prompts inject this metadata so the model sees the actual runtime model selection.

Use `scripts/model_runtime_audit.py` to print the current surface-level and per-agent provider/model/budget/input-output/continuation/thinking policy. Add `--json` for a full machine-readable snapshot.

Do not hardcode model names in prompts or docs beyond describing current maps. Use `bot_config.py` as source of truth.

## Error Recovery and Tool Conversion

For delegated agents, both Anthropic and OpenAI-compatible loops enforce the gateway input ceiling. When tool output growth crosses it, large prior results from explicitly replay-safe read-only tools are replaced in the request by explicit replay checkpoints while their preceding tool calls retain the exact tool name and arguments. This allows complete source material to be retrieved again instead of silently truncating it. Write, publish, send, execute, and other side-effecting results are never checkpointed with a replay instruction. Output-length stops continue from the exact cutoff up to the configured bounded continuation count.

Provider-facing tool definitions are compacted before API calls: long human-readable `description` strings in tool definitions and nested schemas are shortened, while tool names, schema keys, types, enums, defaults, and required fields are preserved. This reduces prompt overhead without changing execution capabilities.

`openai_tool_loop.py` converts Anthropic-style tool definitions to Chat Completions function tools and normalizes malformed tool-call messages. It also handles:

- tool-call/result pairing validation
- text-only recovery after provider protocol errors
- result truncation for large tool output, with a larger cap for pagination-capable read tools (`fetch_url`, `read_file`, `read_document`, `read_self`) so their own offset/next-hint contracts remain usable
- forced final response after budget/round exhaustion

`claude_loop.py` owns the Anthropic-native equivalent and pricing/cost accounting for Claude calls and non-web DeepSeek agent-harness calls. DeepSeek OpenAI-compatible DSML argument spillover is treated as provider serialization leakage, not as an autonomous publication policy or content gate.

Both Anthropic-native Claude calls and Anthropic-compatible DeepSeek calls retry transient provider failures at the API-call boundary: connection/timeouts, 408/409/429, 5xx, and 529 are retried up to three attempts with a short backoff. Non-transient protocol/auth/schema errors are not retried. Streaming callers can opt into a provider idle timeout; `/writer` uses it so a DeepSeek stream that returns HTTP 200 but then produces no text/final event is converted into a transient timeout and retried server-side. For streaming callers such as `/writer`, retry progress can be surfaced as `provider_retry`; final `done` still comes from the successful response, and already-executed local tools are not duplicated because retries happen before each model response is processed.
