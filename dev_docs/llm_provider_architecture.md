# LLM Provider Architecture

최종 확인 기준: 2026-05-09 코드 트리.

LeninBot은 provider와 모델 티어를 런타임 설정으로 해석한다. Telegram chat, background task, autonomous loop, public web chat은 서로 다른 provider 설정을 가질 수 있다.

## Config Sources

| Source | Owner | Purpose |
|---|---|---|
| `config.json` | `bot_config.py` | mutable runtime config saved by Telegram `/config` |
| `config/agent_runtime.json` | `agents/runtime_config.py` | per-agent provider/model/budget/round/tool overlay |
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

DeepSeek
  bot_config._deepseek_anthropic_client
  -> claude_loop.chat_with_tools(client=...)
  -> DeepSeek Anthropic-compatible Messages API

DeepSeek web chat / browser automation
  bot_config._deepseek_anthropic_client
  -> claude_loop.chat_with_tools(client=..., thinking={"type": "disabled"})
  -> DeepSeek Anthropic-compatible Messages API

Local
  llm.client backend
  -> openai_tool_loop.chat_with_tools(base_url=...)
  -> OpenAI-compatible local server
```

OpenAI-compatible providers share `openai_tool_loop.py`. Claude uses `claude_loop.py` because Anthropic tool-use message structure is different.
Telegram chat, background tasks, A2A, public web chat, browser worker tasks, browser-use automation, and the hourly autonomous project loop use DeepSeek's Anthropic-compatible API when `provider=deepseek`, so tool inputs arrive as structured `tool_use.input` blocks instead of OpenAI-compatible `function.arguments` JSON strings. Agent/task DeepSeek paths enable thinking mode by default and send `output_config.effort` explicitly; `claude_loop.py` preserves `thinking` and `redacted_thinking` assistant content blocks in replayed tool-call turns so DeepSeek receives the reasoning payload it requires on follow-up requests. Public web chat and browser automation deliberately keep DeepSeek Flash in non-thinking mode (`thinking={"type": "disabled"}`): web chat does it for lower latency, while browser automation does it because browser-use relies on forced structured tool calls and DeepSeek does not support that path with thinking enabled. DeepSeek Anthropic-compatible messages do not support image content, so browser-use runs DeepSeek as a non-vision DOM/tool controller first and retries with the configured Google/OpenAI vision fallback only if that primary attempt fails.

## Runtime Config Keys

| Key | Values | Applies to |
|---|---|---|
| `provider` | `claude`, `openai`, `deepseek`, `local` | Telegram chat |
| `chat_model` | `high`, `medium`, `low` | Telegram chat tier |
| `task_provider` | `default`, `claude`, `openai`, `deepseek`, `local` | delegated/background tasks |
| `task_model` | `high`, `medium`, `low` | task tier |
| `webchat_provider` | `claude`, `openai`, `deepseek` | public web chat, API restart required |
| `webchat_model` | `high`, `medium`, `low` | public web chat tier |
| `autonomous_provider` | `default`, `claude`, `openai`, `deepseek`, `local` | hourly autonomous loop |
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

DeepSeek Anthropic-compatible thinking is controlled by environment variables:

| Env | Default | Meaning |
|---|---|---|
| `DEEPSEEK_THINKING_MODE` | `thinking` | `thinking` enables DeepSeek thinking; `thinking_max` forces max effort; `disabled`/`non-thinking` disables it |
| `DEEPSEEK_THINKING_EFFORT` | `high` | `high` or `max`; invalid values fall back to `high` |

## Tier Resolution

| Tier | Claude | OpenAI | DeepSeek | Local |
|---|---|---|---|---|
| `high` | `claude-opus-4-7` alias | `gpt-5.5` | `deepseek-v4-pro` | local backend model |
| `medium` | `claude-sonnet-4-6` alias | `gpt-5.5-mini` | `deepseek-v4-flash` | local backend model |
| `low` | `claude-haiku-4-5` alias | `gpt-5.5-nano` | `deepseek-v4-flash` | local backend model |

Claude aliases are resolved lazily through Anthropic Models API and cached in-process. OpenAI, DeepSeek, and local model IDs resolve synchronously from maps or local backend config.

## Agent Overrides

Each `AgentSpec` may set `provider` and `model`. `None` means follow task config. Current example overlay (`config/agent_runtime.json.example`) pins:

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

Use `scripts/model_runtime_audit.py` to print the current surface-level and per-agent provider/model/budget policy. Add `--json` for a full machine-readable snapshot.

Do not hardcode model names in prompts or docs beyond describing current maps. Use `bot_config.py` as source of truth.

## Error Recovery and Tool Conversion

`openai_tool_loop.py` converts Anthropic-style tool definitions to Chat Completions function tools and normalizes malformed tool-call messages. It also handles:

- tool-call/result pairing validation
- text-only recovery after provider protocol errors
- result truncation for large tool output
- forced final response after budget/round exhaustion

`claude_loop.py` owns the Anthropic-native equivalent and pricing/cost accounting for Claude calls and non-web DeepSeek agent-harness calls. DeepSeek OpenAI-compatible DSML argument spillover is treated as provider serialization leakage, not as an autonomous publication policy or content gate.
