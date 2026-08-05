# LLM Provider Architecture

최종 확인 기준: 2026-07-25 코드 트리.

LeninBot은 provider와 모델 티어를 런타임 설정으로 해석한다. Telegram chat, background task, autonomous loop, public web chat은 서로 다른 provider 설정을 가질 수 있다.

## Config Sources

| Source | Owner | Purpose |
|---|---|---|
| `config.json` | `bot_config.py` | mutable runtime config saved by Telegram `/config` |
| `config/agent_runtime.json` | `agents/runtime_config.py` | per-agent provider/model/budget/round/input-output/continuation/thinking/tool overlay |
| `llm/provider_registry.py` | provider loops + `bot_config.py` | canonical model IDs, tier keys, display names, pricing, Kimi request/fallback options |
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

2026-08-04부터 모든 루프 라운드와 registry 원샷 호출은 LLM 게이트웨이(`llm/gateway.py`)를 지난다. 2026-08-05에는 graphiti Gemini, browser-use 매 step, Telegram vision까지 계측을 확장했다. 모든 provider 실키는 key-injection proxy에만 있고 소비 서비스는 proxy readiness 뒤 시작한다. 계측 지점과 정책은 `llm_gateway.md`.

OpenAI-compatible providers share `openai_tool_loop.py`. Claude uses `claude_loop.py` because Anthropic tool-use message structure is different. Since 2026-08-04 both modules are protocol adapters over the single loop engine `agent_loop.run_tool_loop`, which owns the shared control flow (round loop, cancel checks, budget accounting/warnings, tool-batch execution and the missing-result safety net, terminal-tool short-circuit, length continuation, forced-final with finalization tools and the followup-skip heuristic). Provider mechanics — message shapes, streaming/idle guards, cost math, protocol recovery — stay per-module, so control-flow fixes now land once instead of being mirrored by hand. Tool arguments must decode to an object on normal and forced-final rounds; malformed arguments receive an error tool-result and are never executed. Kimi K3 uses `MOONSHOT_API_KEY`, defaults to `https://api.moonshot.ai/v1`, and sends model ID `kimi-k3`. K3 always reasons and currently accepts only `reasoning_effort=max`; the Kimi path preserves the API's `reasoning_content` in replayed assistant messages across tool and continuation rounds while keeping it out of the user-facing answer, as required by Moonshot's multi-turn/tool-call protocol. Budget accounting uses Moonshot's launch pricing ($3/M cache-miss input, $0.30/M cache-hit input, $15/M output). The official API reference is [Kimi API Quickstart](https://platform.kimi.ai/docs/api/quickstart).

(Kimi content-filter 시 DeepSeek으로 요청 단위 스위칭하던 폴백 계약은 2026-08-04 제거됨 — Kimi 미사용 상태에서 루프 복잡도만 키우고 있었다. Kimi의 `reasoning_effort=max`, `max_tokens`, reasoning replay 옵션은 여전히 `llm.provider_registry.kimi_openai_tool_options()`가 소유하며 Telegram, public web chat, A2A가 같은 설정을 사용한다.)
Telegram chat, background tasks, A2A (`leninbot-a2a-api.service`), public web chat, browser worker tasks, browser-use automation, and the hourly autonomous project loop use DeepSeek's Anthropic-compatible API when `provider=deepseek`, so tool inputs arrive as structured `tool_use.input` blocks instead of OpenAI-compatible `function.arguments` JSON strings. Agent/task DeepSeek paths resolve `thinking_policy=tool_loop` through `tool_gateway.inference`; the current `DEEPSEEK_TOOL_THINKING_MODE` default is off, and operators can enable it centrally without per-call overrides. `claude_loop.py` preserves `thinking` and `redacted_thinking` assistant content blocks in replayed tool-call turns so DeepSeek receives the reasoning payload it requires on follow-up requests. Public web chat and browser automation deliberately keep DeepSeek Flash in non-thinking mode (`thinking={"type": "disabled"}`): web chat does it for lower latency, while browser automation does it because browser-use relies on forced structured tool calls and DeepSeek does not support that path with thinking enabled. DeepSeek Anthropic-compatible messages do not support image content, so browser-use runs DeepSeek as a non-vision DOM/tool controller first and retries with the configured Google/OpenAI vision fallback only if that primary attempt fails.

The roleplay bot (`leninbot-roleplay.service`, `telegram/roleplay_bot.py`) is a separate runtime, not the Cyber-Lenin orchestrator, and does not read `config.json`'s `provider`/`chat_model` keys. It pins the DeepSeek provider/model while routing through the proxy: `_deepseek_anthropic_client` + `claude_loop.chat_with_tools`, model `deepseek-v4-flash` (via `_resolve_deepseek_model("deepseek_flash")`), with thinking **enabled** (`output_config.effort=high`). Thinking is on for answer quality; because it goes through `claude_loop`, the reasoning stays in replay-only `thinking` blocks and never appears in the user-facing reply — which is why the roleplay bot uses the Anthropic-compatible path rather than the OpenAI-compatible loop (the latter prepends reasoning to the reply). The bot ignores the global `DEEPSEEK_THINKING_MODE` env and sets its thinking inline.

The personal fiction writer (`/writer`, `api_routes/writer.py`, `novel_writer_api.py`, `creative_writer.py`) is separate from normal provider selection but not from gateway enforcement. Its role-level inference envelopes are centralized in `writer.config.WRITER_CALL_POLICIES` (input/output ceilings, rounds, output continuations, thinking policy); the heavy `main` and `revision` roles import the shared 160k/32k gateway defaults, while `diagnosis`, `line_edit`, and `research` retain smaller explicit role limits; input overflow uses durable summaries plus chapter-boundary anchor replay rather than silent truncation. It uses the shared `claude_loop.chat_with_tools` path with `model="claude-fable-5"` by default and can route explicitly selected DeepSeek or Kimi K3 writer models through their Anthropic-compatible clients. The Claude writer client uses the proxy's shared `anthropic` route and `ANTHROPIC_API_KEY`. Kimi Writer uses the proxy's Moonshot Anthropic-compatible route, model `kimi-k3`, and K3's default thinking. The dedicated process is `novel-writer-api.service` and starts after the proxy is ready.

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
| `high` | `claude-opus-5` alias | `gpt-5.6-sol` | `deepseek-v4-pro` | `kimi-k3` | local backend model |
| `medium` | `claude-sonnet-5` alias | `gpt-5.6-terra` | `deepseek-v4-flash` | `kimi-k3` | local backend model |
| `low` | `claude-haiku-4-5` alias | `gpt-5.6-luna` | `deepseek-v4-flash` | `kimi-k3` | local backend model |

Claude aliases are resolved lazily through Anthropic Models API and cached in-process. OpenAI, DeepSeek, Kimi, and local model IDs resolve synchronously from `llm/provider_registry.py` or local backend config. GPT-5.6 uses the official Sol/Terra/Luna family mapping and does not silently fall back to an older GPT generation when a model is unavailable. Kimi currently has one model in all three tiers, so the tier changes budget/display grouping but not the upstream model ID.

Current registry pricing follows the provider documentation: GPT-5.6 Sol/Terra/Luna
are respectively `$5/$30`, `$2.50/$15`, `$1/$6` per million uncached input/output
tokens (cached input `$0.50/$0.25/$0.10`). Claude Opus 5 is `$5/$25` and Sonnet 5
uses its `$2/$10` launch price through 2026-08-31, then `$3/$15`; the 1-hour cache
write/read rows are derived from those rates. Sources: [OpenAI GPT-5.6 models](https://developers.openai.com/api/docs/models), [Claude models](https://platform.claude.com/docs/en/about-claude/models/overview), [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing).

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

The browser worker accepts tier names, legacy model aliases such as `deepseek_flash`, and official API model IDs, but normalizes them before sending requests upstream. `config/agent_runtime.json` should prefer official API IDs. browser-use calls the provider through the LLM proxy and `_AuditedBrowserChatMixin` applies policy/usage audit to every agent step.

## Model Context Injection

`get_current_model_selection(kind, provider_override=None)` returns provider, tier, alias, model ID, display name, and resolution status for `chat`, `task`, `autonomous`, and `webchat`. Telegram and task prompts inject this metadata so the model sees the actual runtime model selection.

Use `scripts/model_runtime_audit.py` to print the current surface-level and per-agent provider/model/budget/input-output/continuation/thinking policy. Add `--json` for a full machine-readable snapshot.

Do not hardcode model names in prompts or docs beyond describing current maps. Use `bot_config.py` as source of truth.

## Error Recovery and Tool Conversion

For delegated agents, both Anthropic and OpenAI-compatible loops enforce the gateway input ceiling. When tool output growth crosses it, large prior results from explicitly replay-safe read-only tools are replaced in the request by explicit replay checkpoints while their preceding tool calls retain the exact tool name and arguments. This allows complete source material to be retrieved again instead of silently truncating it. Write, publish, send, execute, and other side-effecting results are never checkpointed with a replay instruction. Output-length stops continue from the exact cutoff up to the configured bounded continuation count.

Provider-facing tool definitions are compacted before API calls: long human-readable `description` strings in tool definitions and nested schemas are shortened, while tool names, schema keys, types, enums, defaults, and required fields are preserved. This reduces prompt overhead without changing execution capabilities.

`openai_tool_loop.py` converts Anthropic-style tool definitions to Chat Completions function tools and normalizes malformed tool-call messages. It also handles:

- tool-call/result pairing validation
- text-only recovery only for positively identified tool protocol 400s before any
  side-effect succeeds; general 400/auth/schema errors are not rewritten or replayed
- no model replay after a successful side-effect if a later protocol/transport error
  occurs; the loop returns the completed-work fallback report instead
- malformed/non-object function arguments as synthetic errors, never `{}` handler calls
- result truncation for large tool output, with a larger cap for pagination-capable read tools (`fetch_url`, `read_file`, `read_document`, `read_self`) so their own offset/next-hint contracts remain usable
- forced final response after budget/round exhaustion

`claude_loop.py` owns the Anthropic-native equivalent and pricing/cost accounting for Claude calls and non-web DeepSeek agent-harness calls. The round/forced-final control flow of both loops is the shared engine `agent_loop.run_tool_loop` (see above); `tests/test_claude_loop_rounds.py`, `tests/test_openai_loop_rounds.py`, and `tests/test_agent_loop_engine.py` pin its contracts. DeepSeek OpenAI-compatible DSML argument spillover is treated as provider serialization leakage, not as an autonomous publication policy or content gate.

Both Anthropic-native Claude calls and Anthropic-compatible DeepSeek calls retry transient provider failures at the API-call boundary: connection/timeouts, 408/409/429, 5xx, and 529 are retried up to three attempts with a short backoff. Non-transient protocol/auth/schema errors are not retried. Streaming callers can opt into a provider idle timeout; `/writer` uses it so a DeepSeek stream that returns HTTP 200 but then produces no text/final event is converted into a transient timeout and retried server-side. For streaming callers such as `/writer`, retry progress can be surfaced as `provider_retry`; final `done` still comes from the successful response, and already-executed local tools are not duplicated because retries happen before each model response is processed.

## 2026-07-25 Migration Checklist

- [x] Claude high tier를 `claude-opus-5`로 교체하고 Models API resolution을 확인
- [x] OpenAI high/medium/low와 browser OpenAI tiers를 각각 GPT-5.6
  Sol/Terra/Luna로 교체
- [x] old-generation automatic GPT fallback을 제거해 요청 모델과 과금 모델의
  불일치를 차단
- [x] 모델 ID, tier, display name, provider pricing, Kimi tool options를
  `llm/provider_registry.py`로 중앙화
- [x] Kimi content-filter fallback을 Telegram, web chat, A2A에 동일하게 적용
- [x] 네 provider의 current credential에서 Models API 가용성을 확인하고 runtime,
  gateway, DeepSeek harness smoke를 통과
