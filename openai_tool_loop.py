"""openai_tool_loop.py — Robust OpenAI-compatible tool-use loop.

Two modes:
  1. **SDK mode** (client=AsyncOpenAI): OpenAI 공식 API. 비용 추적 + 예산 제한.
  2. **httpx mode** (base_url=...): llama-server, vLLM 등 로컬 LLM. 비용 무시.

claude_loop.py와 동일한 인터페이스를 제공하되, OpenAI 호환 API
(/v1/chat/completions with function calling)를 사용한다.

Error recovery strategy (ported from claude_loop.py):
  - Pre-API validation: check message integrity each round
  - Auto-recovery: on API error, strip tool blocks and retry once
  - Nuclear recovery: _strip_tool_blocks() converts all tool protocol to text
  - Safety net: every tool_call gets a result (synthetic error if skipped)
  - Forced final response: escalation hints + preflight + last-resort strip
"""

import asyncio
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
_DEFAULT_PRICING = OPENAI_PRICING["gpt-5.4"]


def _calculate_cost(usage, model: str) -> float:
    """Calculate USD cost from an OpenAI usage object."""
    pricing = OPENAI_PRICING.get(model, _DEFAULT_PRICING)
    cost = 0.0

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

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
    """Anthropic tool definition → OpenAI function tool definition."""
    params = tool.get("input_schema", {"type": "object", "properties": {}})
    params = {k: v for k, v in params.items() if k != "cache_control"}
    func_def: dict = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": params,
    }
    if params.get("additionalProperties") is False:
        func_def["strict"] = True
    return {"type": "function", "function": func_def}


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tools to OpenAI format. Strips cache_control."""
    converted = []
    for t in tools:
        clean = {k: v for k, v in t.items() if k != "cache_control"}
        if clean.get("type") == "function":
            converted.append(clean)
        elif "input_schema" in clean:
            converted.append(_convert_tool_anthropic_to_openai(clean))
        else:
            converted.append(clean)
    return converted


# ── Message normalization & sanitization ──────────────────────────────

def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize inbound history to text-only OpenAI chat format.

    Root-cause fix: strips ALL tool protocol blocks from persisted/external
    history. Tool protocol for the current request is generated only inside
    chat_with_tools, eliminating cross-turn dangling tool mismatches.
    """
    clean: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            clean.append({"role": "user", "content": str(raw)})
            continue

        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "assistant" if role in ("model", "bot") else "user"

        # Skip "tool" role messages from external history entirely
        if raw.get("role") == "tool":
            continue

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
                        parts.append(f"[도구 호출: {name}({inp})]")
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        if isinstance(rc, str):
                            parts.append(f"[도구 결과: {rc[:2000]}]")
                    else:
                        parts.append(str(block))
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content)

        # Skip assistant messages that had tool_calls but no text
        # (they're just protocol, not useful context)
        if role == "assistant" and raw.get("tool_calls") and not text.strip():
            tool_calls = raw.get("tool_calls", [])
            tc_parts = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    tc_parts.append(f"[도구 호출: {fn.get('name', '?')}]")
            if tc_parts:
                text = "\n".join(tc_parts)

        if not text.strip():
            continue

        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n" + text
        else:
            clean.append({"role": role, "content": text})

    return clean


def _strip_tool_protocol(msgs: list[dict]) -> list[dict]:
    """Nuclear recovery: convert ALL tool protocol to plain text messages.

    Preserves tool call/result CONTENT as readable text so the model retains
    context from previous tool executions. Only the protocol structure
    (tool_calls, role:tool) is removed.
    """
    cleaned = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content") or ""

        if role == "tool":
            # Convert tool result to user text
            tc_id = msg.get("tool_call_id", "?")
            text = f"[도구 결과 (id={tc_id}): {content[:2000]}]"
            # Merge into previous or create new user message
            if cleaned and cleaned[-1]["role"] == "user":
                cleaned[-1]["content"] += "\n" + text
            else:
                cleaned.append({"role": "user", "content": text})
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # Convert tool calls to text
            tc_parts = []
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")[:500]
                    tc_parts.append(f"[도구 호출: {name}({args})]")
            text_content = content if isinstance(content, str) and content.strip() else ""
            if tc_parts:
                text_content = (text_content + "\n" + "\n".join(tc_parts)).strip()
            if text_content:
                cleaned.append({"role": "assistant", "content": text_content})
            continue

        # Regular message — pass through
        if isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in content
            )
            if text.strip():
                cleaned.append({"role": role, "content": text})

    # Merge consecutive same-role messages
    merged = []
    for msg in cleaned:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    # Ensure no empty messages
    return [m for m in merged if m.get("content", "").strip()]


def _validate_tool_results(msgs: list[dict]) -> list[str]:
    """Find tool_call IDs that are missing their tool result messages.

    Returns list of missing tool_call_ids.
    """
    # Collect all tool_call IDs from assistant messages
    expected_ids: list[str] = []
    for msg in msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    if tc_id:
                        expected_ids.append(tc_id)

    # Collect all resolved tool_call IDs
    resolved_ids = set()
    for msg in msgs:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                resolved_ids.add(tc_id)

    return [tid for tid in expected_ids if tid not in resolved_ids]


def _dump_messages_for_debug(msgs: list[dict], round_num: int, error: Exception):
    """Log concise message structure when API call fails."""
    lines = [f"=== API ERROR at round {round_num}: {error} ==="]
    lines.append(f"Total messages: {len(msgs)}")

    for idx, msg in enumerate(msgs):
        role = msg.get("role", "?")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tc_id = msg.get("tool_call_id")

        parts = [f"[{idx}] {role}"]
        if isinstance(content, str):
            parts.append(f"text({len(content)} chars)")
        elif content is None:
            parts.append("content=null")

        if tool_calls:
            tc_descs = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    tc_descs.append(f"{fn.get('name', '?')}(id={tc.get('id', '?')})")
            parts.append(f"tool_calls=[{', '.join(tc_descs)}]")

        if tc_id:
            parts.append(f"tool_call_id={tc_id}")

        lines.append("  " + " | ".join(parts))

    logger.error("\n".join(lines))


# ── Core API calls ───────────────────────────────────────────────────

async def _call_api(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: int = 300,
) -> dict:
    """Single call to OpenAI-compatible /v1/chat/completions (httpx)."""
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


async def _call_sdk(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    parallel_tool_calls: bool = True,
):
    """Single call via openai.AsyncOpenAI SDK."""
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = parallel_tool_calls

    return await client.chat.completions.create(**kwargs)


# ── Helper: unified API call with mode dispatch ──────────────────────

async def _api_call(sdk_mode, client, base_url, model, messages, tools, max_tokens,
                    parallel_tool_calls=True):
    """Dispatch to SDK or httpx based on mode."""
    if sdk_mode:
        return await _call_sdk(client, model, messages, tools, max_tokens,
                               parallel_tool_calls=parallel_tool_calls)
    else:
        return await _call_api(base_url, model, messages, tools, max_tokens)


def _extract_response(sdk_mode, response_or_data):
    """Extract (finish_reason, content_text, tool_calls, message, usage) from response."""
    if sdk_mode:
        choice = response_or_data.choices[0]
        return (
            choice.finish_reason or "stop",
            choice.message.content or "",
            choice.message.tool_calls,
            choice.message,
            response_or_data.usage,
        )
    else:
        choice = response_or_data["choices"][0]
        msg = choice["message"]
        return (
            choice.get("finish_reason", "stop"),
            msg.get("content", "") or "",
            msg.get("tool_calls"),
            msg,
            None,  # no usage tracking in httpx mode
        )


def _build_tc_list(sdk_mode, tool_calls) -> list[dict]:
    """Build a normalized tool_calls list from SDK objects or raw dicts."""
    result = []
    for tc in tool_calls:
        try:
            if sdk_mode:
                tc_id = tc.id
                name = tc.function.name
                arguments = tc.function.arguments
            else:
                tc_id = tc["id"]
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]

            if not tc_id or not name:
                logger.warning("Skipping malformed tool_call: missing id or name: %s", tc)
                continue

            result.append({
                "id": str(tc_id),
                "type": "function",
                "function": {"name": str(name), "arguments": str(arguments or "{}")},
            })
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning("Skipping malformed tool_call: %s — %s", tc, e)
            continue
    return result


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
    """
    sdk_mode = client is not None

    # ── Budget validation ──
    try:
        budget_usd = float(budget_usd)
    except (TypeError, ValueError):
        logger.warning("Invalid budget_usd=%r; falling back to 0.30", budget_usd)
        budget_usd = 0.30
    if budget_usd <= 0 and sdk_mode:
        logger.warning("Non-positive budget_usd=%s; clamping to 0.01", budget_usd)
        budget_usd = 0.01

    openai_tools = _convert_tools(tools)

    # ── Root-cause fix: start from text-only canonical history ──
    # Tool protocol blocks are generated only within this call.
    working_msgs = _normalize_messages(messages)

    if system_prompt:
        working_msgs.insert(0, {"role": "system", "content": system_prompt})

    tool_call_log = []
    tool_work_details = []
    total_cost = 0.0
    budget_warning_sent = False
    round_num = 0
    accumulated_text_parts: list[str] = []  # Collect text from tool_calls rounds

    for round_num in range(1, max_rounds + 1):

        # ── Pre-API validation: check tool result completeness ──
        missing_ids = _validate_tool_results(working_msgs)
        if missing_ids:
            logger.error("Pre-API check (round %d): %d missing tool results: %s",
                         round_num, len(missing_ids), missing_ids)
            # Inject synthetic error results for missing IDs
            for mid in missing_ids:
                working_msgs.append({
                    "role": "tool",
                    "tool_call_id": mid,
                    "content": "[tool result unavailable — auto-injected]",
                })
            # Re-check after injection
            still_missing = _validate_tool_results(working_msgs)
            if still_missing:
                logger.error("Still missing after injection: %s — stripping tool protocol",
                             still_missing)
                working_msgs = _strip_tool_protocol(working_msgs)
                if system_prompt:
                    working_msgs.insert(0, {"role": "system", "content": system_prompt})

        # ── API call with auto-recovery ──
        response = None
        try:
            response = await _api_call(
                sdk_mode, client, base_url, model, working_msgs,
                openai_tools, max_tokens,
            )
        except Exception as api_err:
            err_str = str(api_err)
            _dump_messages_for_debug(working_msgs, round_num, api_err)

            # Auto-recovery: strip tool protocol and retry once
            if "tool" in err_str.lower() or "400" in err_str or "invalid" in err_str.lower():
                logger.warning("Auto-recovery (round %d): stripping tool protocol and retrying",
                               round_num)
                stripped = _strip_tool_protocol(working_msgs)
                if system_prompt:
                    stripped.insert(0, {"role": "system", "content": system_prompt})
                try:
                    response = await _api_call(
                        sdk_mode, client, base_url, model, stripped,
                        openai_tools, max_tokens,
                        parallel_tool_calls=False,  # reduce complexity on retry
                    )
                    working_msgs = stripped
                    logger.info("Auto-recovery succeeded at round %d", round_num)
                except Exception as retry_err:
                    logger.error("Auto-recovery retry also failed: %s", retry_err)
                    # Force final response path with no tools
                    working_msgs = stripped
                    break
            else:
                if log_event:
                    log_event("error", "openai_loop", f"API call failed: {api_err}")
                raise

        if response is None:
            break

        finish_reason, content_text, tool_calls, message_obj, usage = \
            _extract_response(sdk_mode, response)

        # ── Cost tracking (SDK mode) ──
        if sdk_mode and usage:
            round_cost = _calculate_cost(usage, model)
            total_cost += round_cost
            logger.debug("Round %d cost: $%.4f (total: $%.4f / $%.2f)",
                         round_num, round_cost, total_cost, budget_usd)
            if on_progress:
                try:
                    await on_progress("budget", f"[{round_num}] ${total_cost:.3f}/${budget_usd:.2f}")
                except Exception:
                    pass

        # ── Handle refusal (OpenAI safety filter) ──
        refusal = None
        if sdk_mode and hasattr(message_obj, "refusal"):
            refusal = message_obj.refusal
        elif not sdk_mode and isinstance(message_obj, dict):
            refusal = message_obj.get("refusal")
        if refusal:
            logger.warning("Model refused request at round %d: %s", round_num, refusal)
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": total_cost, "rounds_used": round_num,
                    "was_interrupted": False, "tool_work_details": list(tool_work_details),
                })
            return f"⚠️ 모델이 요청을 거부했습니다: {refusal}"

        # ── No tool calls → return text response ──
        if finish_reason != "tool_calls" or not tool_calls:
            if finish_reason == "length":
                logger.warning("Response truncated by max_completion_tokens (%d) at round %d",
                               max_tokens, round_num)
                if log_event:
                    log_event("warning", "chat",
                              f"Response truncated ({max_tokens} tokens) at round {round_num}")
            elif finish_reason == "content_filter":
                logger.warning("Response blocked by content filter at round %d", round_num)
            # Combine accumulated text from tool_calls rounds with final response
            final_text = content_text.strip()
            if accumulated_text_parts:
                all_text = "\n".join(accumulated_text_parts)
                final_text = f"{all_text}\n\n{final_text}" if final_text else all_text
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": total_cost, "rounds_used": round_num,
                    "was_interrupted": False, "tool_work_details": list(tool_work_details),
                })
            return final_text if final_text else "응답을 생성하지 못했습니다."

        # ── Budget check ──
        budget_exceeded = sdk_mode and budget_usd > 0 and total_cost >= budget_usd
        if budget_exceeded:
            logger.warning("Budget exhausted: $%.4f >= $%.2f at round %d — "
                           "processing final tool calls before exit",
                           total_cost, budget_usd, round_num)

        # ── Build normalized tool_calls list (with malformed block skipping) ──
        tc_list = _build_tc_list(sdk_mode, tool_calls)
        if not tc_list:
            # All tool calls were malformed — return text content if any
            logger.warning("All tool_calls malformed at round %d — returning text", round_num)
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": total_cost, "rounds_used": round_num,
                    "was_interrupted": False, "tool_work_details": list(tool_work_details),
                })
            return content_text.strip() if content_text.strip() else "응답을 생성하지 못했습니다."

        # Accumulate substantial text from tool_calls rounds for final result
        if content_text.strip() and len(content_text.strip()) > 20:
            accumulated_text_parts.append(content_text.strip())

        # ── Append assistant message with tool_calls ──
        assistant_msg = {
            "role": "assistant",
            "content": content_text if content_text.strip() else None,
            "tool_calls": tc_list,
        }
        working_msgs.append(assistant_msg)

        if on_progress and content_text.strip():
            try:
                await on_progress("thinking", f"[{round_num}] {content_text.strip()}")
            except Exception:
                pass

        # ── Execute tool calls ──
        executed_ids: set[str] = set()
        for tc_item in tc_list:
            tc_id = tc_item["id"]
            func_name = tc_item["function"]["name"]

            try:
                func_args = json.loads(tc_item["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed arguments for %s: %s",
                               func_name, tc_item["function"]["arguments"][:200])
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
                    raw = handler(**func_args)
                    if asyncio.iscoroutine(raw) or asyncio.isfuture(raw):
                        result = await raw
                    else:
                        result = raw
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

            # Truncate oversized results to avoid context overflow
            if len(result) > 50000:
                result = result[:50000] + "\n... [truncated]"

            working_msgs.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })
            executed_ids.add(tc_id)

            if on_progress:
                status = "❌" if is_error else "✓"
                try:
                    await on_progress("tool_result", f"  {status} {result[:200]}")
                except Exception:
                    pass

            tool_call_log.append(f"  [{round_num}/{max_rounds}] {func_name}({input_summary})")
            tool_work_details.append(f"  [{round_num}] {func_name}({input_summary}) → {result}")

        # ── Safety net: ensure EVERY tool_call has a result ──
        for tc_item in tc_list:
            if tc_item["id"] not in executed_ids:
                logger.warning("Safety net: missing result for tool_call id=%s name=%s",
                               tc_item["id"], tc_item["function"]["name"])
                working_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc_item["id"],
                    "content": f"Tool execution skipped (internal error): "
                               f"no result for {tc_item['function']['name']}",
                })

        # ── Budget break AFTER tool results are properly appended ──
        if budget_exceeded:
            break

        # ── Budget warning at 80% ──
        if sdk_mode and budget_usd > 0 and not budget_warning_sent and total_cost > budget_usd * 0.8:
            budget_warning_sent = True
            working_msgs.append({
                "role": "system",
                "content": (
                    f"[BUDGET WARNING] 예산 80% 소진 (${total_cost:.3f}/${budget_usd:.2f}). "
                    "작업을 계속하라. 한도 도달 시 시스템이 자동 종료한다."
                ),
            })

        # ── Round limit warning 2 rounds before max ──
        if round_num == max_rounds - 2:
            working_msgs.append({
                "role": "system",
                "content": (
                    f"[SYSTEM] 라운드 한도 임박 ({round_num}/{max_rounds}). "
                    "다음 라운드가 마지막이다. 파일 저장 등 최종 도구 호출을 지금 하라."
                ),
            })

    # ══════════════════════════════════════════════════════════════════
    # Forced final response: max_rounds or budget exhausted
    # ══════════════════════════════════════════════════════════════════
    budget_exhausted = sdk_mode and budget_usd > 0 and total_cost >= budget_usd
    was_still_working = (
        response is not None
        and _extract_response(sdk_mode, response)[0] == "tool_calls"
    ) if response else False

    limit_reason = "예산 소진" if budget_exhausted else "도구 호출 한도 도달"
    log_detail = "\n".join(tool_call_log) if tool_call_log else ""
    logger.warning(
        "Limit reached (rounds=%d/%d, budget=$%.4f/$%.2f, still_working=%s). "
        "Forcing final response. Calls:\n%s",
        round_num, max_rounds, total_cost, budget_usd, was_still_working, log_detail,
    )

    # ── Escalation hint (like claude_loop.py) ──
    escalation_hint = ""
    if was_still_working:
        escalation_hint = (
            " 미완료 작업이 있으면 수행한 것, 못한 것, 다음에 해야 할 것을 명시하라. "
            "orchestrator가 재위임 여부를 판단한다."
        )

    working_msgs.append({
        "role": "user",
        "content": (
            f"[SYSTEM] {limit_reason} (비용: ${total_cost:.3f}/${budget_usd:.2f}, "
            f"라운드: {round_num}/{max_rounds}). "
            "추가 도구를 사용하지 말고, 지금까지 수행한 작업과 수집한 데이터를 있는 그대로 정리하라. "
            "보고서 형식이 아니어도 된다. 시행착오, 중간 결과, raw 데이터 모두 포함하라."
            + escalation_hint
        ),
    })

    # ── Preflight: validate tool result completeness ──
    missing_ids = _validate_tool_results(working_msgs)
    if missing_ids:
        logger.error("Forced-final preflight: %d missing tool results — stripping",
                     len(missing_ids))
        working_msgs = _strip_tool_protocol(working_msgs)
        if system_prompt:
            working_msgs.insert(0, {"role": "system", "content": system_prompt})
        # Re-append the limit message (was lost in strip)
        working_msgs.append({
            "role": "user",
            "content": f"[SYSTEM] {limit_reason}. 지금까지 수집한 정보만으로 답변하세요.",
        })

    # ── Final API call (no tools → forces text-only response) ──
    try:
        final_response = await _api_call(
            sdk_mode, client, base_url, model, working_msgs, None, max_tokens,
        )
        _, text, _, _, final_usage = _extract_response(sdk_mode, final_response)
        if sdk_mode and final_usage:
            total_cost += _calculate_cost(final_usage, model)
    except Exception as final_err:
        # ── Last resort: strip all tool protocol and retry ──
        _dump_messages_for_debug(working_msgs, -1, final_err)
        logger.warning("Forced response failed — retrying with stripped messages")
        stripped = _strip_tool_protocol(working_msgs)
        if system_prompt:
            stripped.insert(0, {"role": "system", "content": system_prompt})
        stripped.append({
            "role": "user",
            "content": f"[SYSTEM] {limit_reason}. 지금까지 수집한 정보만으로 답변하세요.",
        })
        try:
            last_response = await _api_call(
                sdk_mode, client, base_url, model, stripped, None, max_tokens,
            )
            _, text, _, _, last_usage = _extract_response(sdk_mode, last_response)
            if sdk_mode and last_usage:
                total_cost += _calculate_cost(last_usage, model)
        except Exception as e2:
            logger.error("Final stripped response also failed: %s", e2)
            if log_event:
                log_event("error", "final_response",
                          f"Final response failed even after strip: {e2}")
            if budget_tracker is not None:
                budget_tracker.update({
                    "total_cost": total_cost, "rounds_used": round_num,
                    "was_interrupted": was_still_working,
                    "tool_work_details": list(tool_work_details),
                })
            return f"⚠️ {limit_reason} 후 응답 생성 실패: {final_err}"

    final_text = text.strip()
    if accumulated_text_parts:
        all_text = "\n".join(accumulated_text_parts)
        final_text = f"{all_text}\n\n{final_text}" if final_text else all_text
    if budget_tracker is not None:
        budget_tracker.update({
            "total_cost": total_cost,
            "rounds_used": round_num,
            "was_interrupted": was_still_working,
            "tool_work_details": list(tool_work_details),
        })
    return final_text if final_text else "응답을 생성하지 못했습니다."
