"""claude_loop.py — Claude tool-use loop and message sanitization.

Extracted from telegram_bot.py. Dependencies injected via function parameters
to avoid circular imports.
"""

import json
import logging

logger = logging.getLogger(__name__)


# ── Pricing Constants (USD per million tokens) ──────────────────────
# Claude Sonnet 4.6 pricing — update when model pricing changes
PRICING = {
    "input": 3.00 / 1_000_000,
    "output": 15.00 / 1_000_000,
    "cache_creation": 3.75 / 1_000_000,
    "cache_read": 0.30 / 1_000_000,
}


def _calculate_cost(usage) -> float:
    """Calculate USD cost from a response.usage object."""
    cost = 0.0
    cost += getattr(usage, "input_tokens", 0) * PRICING["input"]
    cost += getattr(usage, "output_tokens", 0) * PRICING["output"]
    # Cache tokens (may not always be present)
    cost += getattr(usage, "cache_creation_input_tokens", 0) * PRICING["cache_creation"]
    cost += getattr(usage, "cache_read_input_tokens", 0) * PRICING["cache_read"]
    return cost


# ── Message Sanitization ─────────────────────────────────────────────

def sanitize_messages(msgs: list[dict], handle_server_tools: bool = True) -> list[dict]:
    """Ensure every tool_use/server_tool_use has a matching result.

    Single-pass replacement for the previous _validate_tool_results +
    _ensure_tool_results + _force_fix_tool_results triple.

    Rules:
    - tool_use → tool_result in the NEXT user message
    - server_tool_use → web_search_tool_result in the SAME assistant message
      (only when handle_server_tools=True)

    Missing results are injected as dummies to prevent 400 errors.

    Args:
        msgs: List of message dicts.
        handle_server_tools: If True, handle server_tool_use/web_search_tool_result.
            Set to False for local_agent which doesn't use server-side tools.
    """
    msgs = [dict(m) for m in msgs]
    injected = 0
    i = 0

    while i < len(msgs):
        msg = msgs[i]
        if msg.get("role") != "assistant":
            i += 1
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            i += 1
            continue

        # Collect custom tool_use IDs
        custom_ids = [
            b["id"] for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
        ]

        # Collect server_tool_use IDs, excluding those already resolved in-message
        unresolved_server = []
        if handle_server_tools:
            resolved_server = {
                b.get("tool_use_id") for b in content
                if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
            }
            unresolved_server = [
                b["id"] for b in content
                if isinstance(b, dict) and b.get("type") == "server_tool_use"
                and "id" in b and b["id"] not in resolved_server
            ]

        if not custom_ids and not unresolved_server:
            i += 1
            continue

        # Inject server tool dummies into the SAME assistant message
        if unresolved_server:
            server_dummies = [
                {"type": "web_search_tool_result", "tool_use_id": tid, "content": []}
                for tid in unresolved_server
            ]
            msgs[i] = {**msgs[i], "content": list(content) + server_dummies}
            injected += len(server_dummies)

        # Check next user message for existing custom tool_results
        resolved_custom: set = set()
        next_is_user = (i + 1 < len(msgs) and msgs[i + 1].get("role") == "user")
        next_content: list = []

        if next_is_user:
            nc = msgs[i + 1].get("content", [])
            if isinstance(nc, list):
                next_content = nc
                resolved_custom = {
                    b.get("tool_use_id") for b in nc
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }

        # Inject custom tool dummies into the NEXT user message
        missing_custom = [tid for tid in custom_ids if tid not in resolved_custom]
        if missing_custom:
            dummies = [{
                "type": "tool_result",
                "tool_use_id": tid,
                "content": "[tool result unavailable]",
                "is_error": True,
            } for tid in missing_custom]
            injected += len(dummies)

            if next_content:
                msgs[i + 1] = {**msgs[i + 1], "content": dummies + next_content}
            elif next_is_user:
                old = msgs[i + 1].get("content", "")
                msgs[i + 1] = {
                    "role": "user",
                    "content": dummies + [{"type": "text", "text": str(old)}],
                }
            else:
                msgs.insert(i + 1, {"role": "user", "content": dummies})

        # Skip past assistant + user pair
        i += 2 if (i + 1 < len(msgs) and msgs[i + 1].get("role") == "user") else 1

    if injected:
        logger.warning("sanitize_messages: injected %d dummy result(s)", injected)
    return msgs


# ── Token Estimation ─────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate for multilingual text (~3 chars/token for Korean+English mix)."""
    return len(text) // 3


# ── Chat with Tools Loop ─────────────────────────────────────────────

async def chat_with_tools(
    messages: list[dict],
    *,
    client,
    model: str,
    tools: list[dict],
    tool_handlers: dict,
    system_prompt: str,
    max_rounds: int = 50,
    max_tokens: int = 4096,
    log_event=None,
    budget_usd: float = 0.30,
    budget_tracker: dict | None = None,
) -> str:
    """Call Claude with tools, execute tool calls, loop until text response.

    Args:
        messages: Conversation history.
        client: Anthropic AsyncAnthropic client.
        model: Model ID string.
        tools: Tool definitions (Anthropic API format).
        tool_handlers: Dict mapping tool name → async handler function.
        system_prompt: System prompt text.
        max_rounds: Max tool-use rounds before forcing response.
        max_tokens: Max tokens for response.
        log_event: Optional callable(level, source, message, detail=None, task_id=None)
            for persistent error logging.
        budget_usd: Maximum USD budget for this call (default 0.30).
        budget_tracker: Optional dict — filled with {"total_cost", "rounds_used"} after return.
    """
    working_msgs = list(messages)
    tool_call_log = []
    total_cost = 0.0
    budget_warning_sent = False

    # Prompt caching: mark system prompt and tools as cacheable
    cached_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    # Mark last custom tool for caching (skip server-side tools like web_search)
    cached_tools = [dict(t) for t in tools]
    for i in range(len(cached_tools) - 1, -1, -1):
        if cached_tools[i].get("type", "").startswith("web_search"):
            continue  # server-side tool — can't add cache_control
        cached_tools[i] = {**cached_tools[i], "cache_control": {"type": "ephemeral"}}
        break

    response = None
    round_num = 0
    for round_num in range(1, max_rounds + 1):
        # Sanitize message structure before every API call
        working_msgs = sanitize_messages(working_msgs)

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system,
            tools=cached_tools,
            messages=working_msgs,
        )

        # Track cost
        if hasattr(response, "usage") and response.usage:
            round_cost = _calculate_cost(response.usage)
            total_cost += round_cost
            logger.debug("Round %d cost: $%.4f (total: $%.4f / $%.2f)", round_num, round_cost, total_cost, budget_usd)

        # If no custom tool use, extract and return text (check BEFORE budget)
        if response.stop_reason not in ("tool_use", "pause_turn"):
            if response.stop_reason == "max_tokens":
                logger.warning("Response truncated by max_tokens (%d) at round %d/%d", max_tokens, round_num, max_rounds)
                if log_event:
                    log_event("warning", "chat", f"Response truncated by max_tokens ({max_tokens}) at round {round_num}/{max_rounds}")
            text_parts = [b.text for b in response.content if b.type == "text"]
            if budget_tracker is not None:
                budget_tracker.update({"total_cost": total_cost, "rounds_used": round_num})
            return "\n".join(text_parts) if text_parts else "응답을 생성하지 못했습니다."

        # Budget exceeded → process this response's tool calls, then break
        budget_exceeded_this_round = total_cost >= budget_usd
        if budget_exceeded_this_round:
            logger.warning("Budget exhausted: $%.4f >= $%.2f at round %d — processing final tool calls before exit", total_cost, budget_usd, round_num)

        # Process tool calls
        assistant_content = []
        tool_results = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "server_tool_use":
                assistant_content.append({
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {block.name}(server-side)")
            elif block.type == "web_search_tool_result":
                assistant_content.append(block.model_dump())
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                # Execute custom tool
                handler = tool_handlers.get(block.name)
                if handler:
                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input, ensure_ascii=False)[:200])
                    try:
                        result = await handler(**block.input)
                        is_error = False
                    except Exception as e:
                        logger.error("Tool %s execution error: %s", block.name, e)
                        if log_event:
                            log_event("warning", "tool", f"Tool {block.name} failed: {e}")
                        result = f"Tool execution failed: {e}"
                        is_error = True
                else:
                    result = f"Unknown tool: {block.name}"
                    is_error = True
                # Guard: ensure result is a non-None string
                if not isinstance(result, str) or result is None:
                    result = str(result) if result is not None else "(no result)"
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
                # Log for diagnostics
                input_summary = json.dumps(block.input, ensure_ascii=False)
                if len(input_summary) > 120:
                    input_summary = input_summary[:120] + "..."
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {block.name}({input_summary})")

        # Safety net: ensure EVERY tool_use block has a matching tool_result
        resolved_ids = {r["tool_use_id"] for r in tool_results}
        for block in assistant_content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block["id"] not in resolved_ids:
                logger.warning("Safety net: missing tool_result for tool_use id=%s name=%s", block["id"], block.get("name"))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Tool execution skipped (internal error): no result was produced for {block.get('name', 'unknown')}",
                    "is_error": True,
                })

        # Append assistant message with tool_use + user message with tool_results
        working_msgs.append({"role": "assistant", "content": assistant_content})

        # Fix: inject dummy web_search_tool_result for any unresolved server_tool_use
        already_resolved_server_ids = {
            b.get("tool_use_id") for b in assistant_content
            if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
        }
        pending_server_ids = [
            b["id"] for b in assistant_content
            if isinstance(b, dict) and b.get("type") == "server_tool_use"
            and b["id"] not in already_resolved_server_ids
        ]
        if pending_server_ids:
            for tid in pending_server_ids:
                assistant_content.append({
                    "type": "web_search_tool_result",
                    "tool_use_id": tid,
                    "content": [],
                })
            working_msgs[-1] = {"role": "assistant", "content": assistant_content}
            logger.debug("Injected %d dummy web_search_tool_result(s) into assistant content", len(pending_server_ids))

        if tool_results:
            # Inject budget warning at 80% threshold (as separate text block in user message)
            if not budget_warning_sent and total_cost >= budget_usd * 0.8:
                budget_warning_sent = True
                tool_results.insert(0, {
                    "type": "text",
                    "text": (
                        f"[SYSTEM] 예산 80% 소진 (${total_cost:.3f}/${budget_usd:.2f}). "
                        "마무리하거나 request_continuation 도구를 사용하세요."
                    ),
                })
            working_msgs.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "pause_turn":
            working_msgs.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
        else:
            logger.warning("No tool_results and not pause_turn (stop_reason=%s); appending fallback user message", response.stop_reason)
            working_msgs.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})

        # Budget break AFTER tool results are properly appended
        if budget_exceeded_this_round:
            break

    # Limit reached (rounds or budget) — force final response
    budget_exhausted = total_cost >= budget_usd
    log_detail = "\n".join(tool_call_log) if tool_call_log else ""
    was_still_working = response.stop_reason in ("tool_use", "pause_turn") if response else False
    logger.warning(
        "Limit reached (rounds=%d/%d, budget=$%.4f/$%.2f, still_working=%s). Forcing final response. Calls:\n%s",
        round_num if response else 0, max_rounds, total_cost, budget_usd, was_still_working, log_detail,
    )

    # Inject a nudge so the model knows it must answer now
    escalation_hint = ""
    if was_still_working:
        escalation_hint = (
            " 미완료 작업이 있다면 request_continuation 도구가 있으면 사용하고, "
            "없으면 응답 맨 끝에 \"[CONTINUE_TASK: 남은 작업 설명]\" 형식으로 한 줄 추가하세요."
        )
    limit_reason = "예산 소진" if budget_exhausted else "도구 호출 한도 도달"
    working_msgs.append({
        "role": "user",
        "content": (
            f"[SYSTEM] {limit_reason} (비용: ${total_cost:.3f}/${budget_usd:.2f}, 라운드: {round_num if response else 0}/{max_rounds}). "
            "추가 도구를 사용하지 말고, 지금까지 수집한 정보만으로 최선의 답변을 완성하세요."
            + escalation_hint
        ),
    })
    # Sanitize: ensure ALL tool_use/server_tool_use have matching results
    working_msgs = sanitize_messages(working_msgs)
    try:
        final = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system,
            messages=working_msgs,  # no tools parameter — forces text-only response
        )
        if hasattr(final, "usage") and final.usage:
            total_cost += _calculate_cost(final.usage)
        if final.stop_reason == "max_tokens":
            logger.warning("Forced final response truncated by max_tokens (%d)", max_tokens)
        text_parts = [b.text for b in final.content if b.type == "text"]
        if budget_tracker is not None:
            budget_tracker.update({"total_cost": total_cost, "rounds_used": round_num if response else 0})
        return "\n".join(text_parts) if text_parts else "응답을 생성하지 못했습니다."
    except Exception as e:
        logger.error("Final forced response failed: %s", e)
        if log_event:
            log_event("error", "final_response", f"Final forced response failed: {e}")
        if budget_tracker is not None:
            budget_tracker.update({"total_cost": total_cost, "rounds_used": round_num if response else 0})
        return f"⚠️ {limit_reason} 후 응답 생성 실패: {e}"
