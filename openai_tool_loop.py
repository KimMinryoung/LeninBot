"""openai_tool_loop.py — OpenAI-compatible tool-use loop.

Two modes:
  1. **SDK mode** (client=AsyncOpenAI): OpenAI 공식 API. 비용 추적 + 예산 제한.
  2. **httpx mode** (base_url=...): llama-server, vLLM 등 로컬 LLM. 비용 무시.

claude_loop.py와 동일한 인터페이스를 제공하되, OpenAI 호환 API
(/v1/chat/completions with function calling)를 사용한다.

사용 예 (SDK mode — OpenAI 공식 API):
    from openai import AsyncOpenAI
    from openai_tool_loop import chat_with_tools

    result = await chat_with_tools(
        messages=history,
        client=AsyncOpenAI(),
        model="gpt-5.4",
        tools=tool_defs,          # Anthropic 포맷 → 내부에서 자동 변환
        tool_handlers=handlers,
        system_prompt="...",
    )

사용 예 (httpx mode — 로컬 LLM):
    result = await chat_with_tools(
        messages=history,
        base_url="http://127.0.0.1:8080",
        model="qwen3.5-9b",
        tools=tool_defs,
        tool_handlers=handlers,
        system_prompt="...",
    )
"""

import json
import logging
import httpx

logger = logging.getLogger(__name__)


# ── Pricing Constants (USD per million tokens) ────────────────────────
OPENAI_PRICING = {
    "gpt-5.4": {
        "input": 2.50 / 1_000_000,
        "output": 15.00 / 1_000_000,
        "cached_input": 0.25 / 1_000_000,
    },
    "gpt-5.4-mini": {
        "input": 0.75 / 1_000_000,
        "output": 4.50 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
    },
    "gpt-5.4-nano": {
        "input": 0.20 / 1_000_000,
        "output": 1.25 / 1_000_000,
        "cached_input": 0.02 / 1_000_000,
    },
}
# Fallback pricing for unknown models (use gpt-5.4 rates)
_DEFAULT_PRICING = OPENAI_PRICING["gpt-5.4"]


def _calculate_cost(usage, model: str) -> float:
    """Calculate USD cost from an OpenAI usage object."""
    pricing = OPENAI_PRICING.get(model, _DEFAULT_PRICING)
    cost = 0.0

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Check for cached tokens in prompt_tokens_details
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    non_cached_input = prompt_tokens - cached_tokens
    cost += non_cached_input * pricing["input"]
    cost += cached_tokens * pricing["cached_input"]
    cost += completion_tokens * pricing["output"]
    return cost


# ── Anthropic → OpenAI tool format conversion ────────────────────────

def _convert_tool_anthropic_to_openai(tool: dict) -> dict:
    """Anthropic tool definition → OpenAI function tool definition.

    Anthropic: {"name", "description", "input_schema": {type, properties, required}}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters": {...}, "strict": bool}}
    """
    params = tool.get("input_schema", {"type": "object", "properties": {}})
    func_def: dict = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": params,
    }
    # Enable strict mode if schema has additionalProperties: false
    if params.get("additionalProperties") is False:
        func_def["strict"] = True
    return {
        "type": "function",
        "function": func_def,
    }


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert a list of Anthropic-format tools to OpenAI format.

    Auto-detects: if already in OpenAI format (has "type": "function"), pass through.
    """
    converted = []
    for t in tools:
        if t.get("type") == "function":
            converted.append(t)
        elif "input_schema" in t:
            converted.append(_convert_tool_anthropic_to_openai(t))
        else:
            converted.append(t)
    return converted


# ── Message normalization ─────────────────────────────────────────────

def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message history to plain text OpenAI chat format.

    Strips any Anthropic-specific block structures (tool_use, tool_result)
    and converts them to readable text, keeping only role + content strings.
    """
    clean: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            clean.append({"role": "user", "content": str(raw)})
            continue

        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system", "tool"):
            role = "assistant" if role in ("model", "bot") else "user"

        content = raw.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}), ensure_ascii=False)[:500]
                        parts.append(f"[tool call: {name}({inp})]")
                    elif btype == "tool_result":
                        parts.append(block.get("content", ""))
                    else:
                        parts.append(str(block))
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content)

        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n" + text
        else:
            clean.append({"role": role, "content": text})

    return clean


# ── Core API call (httpx — for local LLM) ────────────────────────────

async def _call_api(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = 300,
) -> dict:
    """Single call to OpenAI-compatible /v1/chat/completions."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── Core API call (SDK — for OpenAI official API) ────────────────────

async def _call_sdk(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
):
    """Single call via openai.AsyncOpenAI SDK. Returns ChatCompletion object."""
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return await client.chat.completions.create(**kwargs)


# ── Tool-use loop ────────────────────────────────────────────────────

async def chat_with_tools(
    messages: list[dict],
    *,
    client=None,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen3.5-9b",
    tools: list[dict],
    tool_handlers: dict,
    system_prompt: str,
    max_rounds: int = 30,
    max_tokens: int = 4096,
    log_event=None,
    budget_usd: float = 0.0,
    budget_tracker: dict | None = None,
    on_progress=None,
) -> str:
    """Call OpenAI-compatible LLM with tools, execute tool calls, loop until text response.

    Interface mirrors claude_loop.chat_with_tools() for drop-in use.

    Args:
        client: openai.AsyncOpenAI instance for SDK mode. If None, uses httpx with base_url.
        base_url: httpx mode endpoint (ignored if client is provided).
        budget_usd: Max USD budget. Enforced in SDK mode; ignored in httpx mode (local LLM = free).
    """
    sdk_mode = client is not None

    # Budget validation (SDK mode only)
    if sdk_mode and budget_usd > 0:
        try:
            budget_usd = float(budget_usd)
        except (TypeError, ValueError):
            budget_usd = 0.30
        if budget_usd <= 0:
            budget_usd = 0.01

    openai_tools = _convert_tools(tools)
    working_msgs = _normalize_messages(messages)

    # Prepend system prompt
    if system_prompt:
        working_msgs.insert(0, {"role": "system", "content": system_prompt})

    tool_call_log = []
    tool_work_details = []
    total_cost = 0.0
    budget_warning_sent = False

    for round_num in range(1, max_rounds + 1):
        try:
            if sdk_mode:
                response = await _call_sdk(
                    client=client,
                    model=model,
                    messages=working_msgs,
                    tools=openai_tools if round_num <= max_rounds else None,
                    max_tokens=max_tokens,
                )
                # Extract from SDK response object
                choice = response.choices[0]
                message = choice.message
                finish_reason = choice.finish_reason or "stop"
                content_text = message.content or ""
                tool_calls = message.tool_calls

                # Cost tracking
                if response.usage:
                    round_cost = _calculate_cost(response.usage, model)
                    total_cost += round_cost
                    logger.debug("Round %d cost: $%.4f (total: $%.4f / $%.2f)", round_num, round_cost, total_cost, budget_usd)
                    if on_progress:
                        try:
                            await on_progress("budget", f"[{round_num}] ${total_cost:.3f}/${budget_usd:.2f}")
                        except Exception:
                            pass
            else:
                data = await _call_api(
                    base_url=base_url,
                    model=model,
                    messages=working_msgs,
                    tools=openai_tools if round_num <= max_rounds else None,
                    max_tokens=max_tokens,
                )
                choice = data["choices"][0]
                message = choice["message"]
                finish_reason = choice.get("finish_reason", "stop")
                content_text = message.get("content", "") or ""
                tool_calls = message.get("tool_calls")
        except Exception as e:
            logger.error("OpenAI API call failed at round %d: %s", round_num, e)
            if log_event:
                log_event("error", "openai_loop", f"API call failed: {e}")
            raise

        # No tool calls → return text response
        # finish_reason: "stop" (normal), "length" (truncated), "tool_calls" (has tools),
        #                "content_filter" (blocked)
        if finish_reason != "tool_calls" or not tool_calls:
            if finish_reason == "length":
                logger.warning("Response truncated by max_completion_tokens (%d) at round %d", max_tokens, round_num)
            elif finish_reason == "content_filter":
                logger.warning("Response blocked by content filter at round %d", round_num)
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": total_cost,
                    "rounds_used": round_num,
                    "was_interrupted": False,
                    "tool_work_details": list(tool_work_details),
                })
            return content_text.strip() if content_text.strip() else "응답을 생성하지 못했습니다."

        # Budget check (SDK mode only)
        budget_exceeded = sdk_mode and budget_usd > 0 and total_cost >= budget_usd

        # Has tool calls → execute them
        # Build assistant message for history
        if sdk_mode:
            tc_list = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        else:
            tc_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]

        assistant_msg = {"role": "assistant", "content": content_text, "tool_calls": tc_list}
        working_msgs.append(assistant_msg)

        if on_progress and content_text.strip():
            try:
                await on_progress("thinking", f"[{round_num}] {content_text.strip()}")
            except Exception:
                pass

        for tc_item in tc_list:
            tc_id = tc_item["id"]
            func_name = tc_item["function"]["name"]
            try:
                func_args = json.loads(tc_item["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                func_args = {}

            input_summary = json.dumps(func_args, ensure_ascii=False)

            if on_progress:
                try:
                    await on_progress("tool_call", f"[{round_num}] 🔧 {func_name}({input_summary})")
                except Exception:
                    pass

            handler = tool_handlers.get(func_name)
            if handler:
                logger.info("Tool call: %s(%s)", func_name, input_summary[:200])
                try:
                    result = await handler(**func_args)
                    is_error = False
                except Exception as e:
                    logger.error("Tool %s execution error: %s", func_name, e)
                    if log_event:
                        log_event("warning", "tool", f"Tool {func_name} failed: {e}")
                    result = f"Tool execution failed: {e}"
                    is_error = True
            else:
                result = f"Unknown tool: {func_name}"
                is_error = True

            if not isinstance(result, str) or result is None:
                result = str(result) if result is not None else "(no result)"

            # Append tool result as "tool" role message (OpenAI format)
            working_msgs.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

            if on_progress:
                status = "❌" if is_error else "✓"
                try:
                    await on_progress("tool_result", f"  {status} {result[:200]}")
                except Exception:
                    pass

            tool_call_log.append(f"  [{round_num}/{max_rounds}] {func_name}({input_summary})")
            tool_work_details.append(f"  [{round_num}] {func_name}({input_summary}) -> {result}")

        # Budget exceeded → break after processing tool calls
        if budget_exceeded:
            logger.warning("Budget exhausted: $%.4f >= $%.2f at round %d", total_cost, budget_usd, round_num)
            break

        # Budget warning at 80% (SDK mode)
        if sdk_mode and budget_usd > 0 and not budget_warning_sent and total_cost > budget_usd * 0.8:
            budget_warning_sent = True
            working_msgs.append({
                "role": "user",
                "content": (
                    f"[SYSTEM] 예산 80% 소진 (${total_cost:.3f}/${budget_usd:.2f}). "
                    "마무리하거나 request_continuation 도구를 사용하세요."
                ),
            })

    # Max rounds or budget exhausted → force final text response (no tools)
    budget_exhausted = sdk_mode and budget_usd > 0 and total_cost >= budget_usd
    limit_reason = "예산 소진" if budget_exhausted else "도구 호출 한도 도달"
    logger.warning(
        "Limit reached (rounds=%d/%d, budget=$%.4f/$%.2f). Forcing final response. Calls:\n%s",
        round_num, max_rounds, total_cost, budget_usd, "\n".join(tool_call_log),
    )
    working_msgs.append({
        "role": "user",
        "content": (
            f"[SYSTEM] {limit_reason} (비용: ${total_cost:.3f}/${budget_usd:.2f}, 라운드: {round_num}/{max_rounds}). "
            "추가 도구를 사용하지 말고, 지금까지 수집한 정보만으로 최선의 답변을 완성하세요."
        ),
    })

    try:
        if sdk_mode:
            response = await _call_sdk(
                client=client,
                model=model,
                messages=working_msgs,
                tools=None,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            if response.usage:
                total_cost += _calculate_cost(response.usage, model)
        else:
            data = await _call_api(
                base_url=base_url,
                model=model,
                messages=working_msgs,
                tools=None,
                max_tokens=max_tokens,
            )
            text = data["choices"][0]["message"].get("content", "")
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        text = f"⚠️ {limit_reason} 후 응답 생성 실패: {e}"

    if budget_tracker is not None:
        budget_tracker.update({
            "total_cost": total_cost,
            "rounds_used": round_num,
            "was_interrupted": True,
            "tool_work_details": list(tool_work_details),
        })
    return text.strip() if text.strip() else "응답을 생성하지 못했습니다."
