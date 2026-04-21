"""claude_loop.py — Claude tool-use loop.

Extracted from telegram_bot.py. Dependencies injected via function parameters
to avoid circular imports.

Strict-at-source policy: inputs are normalized once via `_normalize_initial_messages`
(flattened to text-only alternating history) before the loop starts. Tool protocol
blocks only ever originate from this loop's own response parsing, which guarantees
id/name correctness, 1:1 tool_use↔tool_result pairing (safety-netted), and strict
assistant→user alternation. No post-hoc sanitization is performed.
"""

import asyncio
import json
import logging

from tool_loop_common import (
    validate_budget, build_budget_tracker, emit_progress,
    update_redis_state, save_redis_progress, execute_tool,
    execute_tools_batch,
    build_limit_message, build_budget_warning, build_round_warning,
    EMPTY_RESPONSE_FALLBACK,
    check_cancelled, TaskCancelledError,
)

logger = logging.getLogger(__name__)


def dedupe_tools_by_name(tools: list[dict] | None) -> list[dict]:
    """Deduplicate tool schemas by name while preserving first occurrence.

    Anthropic can also reject malformed tool payloads, and sharing one dedupe
    path with OpenAI removes repeated failure classes across providers.
    """
    if not tools:
        return []

    deduped: list[dict] = []
    seen_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            deduped.append(tool)
            continue
        name = str(tool.get("name", "") or "").strip()
        if name and name in seen_names:
            logger.warning("Dropping duplicate tool definition: %s", name)
            continue
        if name:
            seen_names.add(name)
        deduped.append(tool)
    return deduped


# ── Cache TTL ────────────────────────────────────────────────────────
# Anthropic's default ephemeral cache lives 5 minutes. That's too short for a
# Telegram conversation where the user may read, think, multitask between
# turns. We use the 1-hour extended-TTL tier — the write premium goes from
# 1.25× to 2.0× base input, but any turn within an hour of the prior one
# hits cache_read (at 0.1× base) instead of paying full cache_creation again.
# Break-even after 2 reads; almost always a win for chat usage.
_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}


# ── Pricing Constants (USD per million tokens) ──────────────────────
# Per-tier list prices. Cache-creation shown for the **1-hour TTL** tier
# (matches what this loop writes). cache_read is identical across TTL tiers.
# Prefix-match picks by base model name so pinned-date variants reuse the row.
PRICING_TABLE = {
    "claude-opus-4-7": {
        "input":          15.00 / 1_000_000,
        "output":         75.00 / 1_000_000,
        "cache_creation": 30.00 / 1_000_000,   # 2.0× input (1h TTL)
        "cache_read":      1.50 / 1_000_000,
    },
    "claude-sonnet-4-6": {
        "input":           3.00 / 1_000_000,
        "output":         15.00 / 1_000_000,
        "cache_creation":  6.00 / 1_000_000,   # 2.0× input (1h TTL)
        "cache_read":      0.30 / 1_000_000,
    },
    "claude-haiku-4-5": {
        "input":           1.00 / 1_000_000,
        "output":          5.00 / 1_000_000,
        "cache_creation":  2.00 / 1_000_000,   # 2.0× input (1h TTL)
        "cache_read":      0.10 / 1_000_000,
    },
}

# Fallback when the model string doesn't match any known family — use Sonnet
# (middle tier) so we don't wildly under- or over-report on unknown variants.
PRICING = PRICING_TABLE["claude-sonnet-4-6"]


def _pricing_for(model: str) -> dict:
    """Pick the pricing row for a Claude model id. Matches by prefix so
    pinned-date variants (``claude-haiku-4-5-20251001``) reuse the family."""
    if not model:
        return PRICING
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]
    for base, price in PRICING_TABLE.items():
        if model.startswith(base + "-") or model.startswith(base + "."):
            return price
    return PRICING


def _calculate_cost(usage, model: str | None = None) -> float:
    """Calculate USD cost from a response.usage object for the given model."""
    p = _pricing_for(model) if model else PRICING
    cost = 0.0
    cost += getattr(usage, "input_tokens", 0) * p["input"]
    cost += getattr(usage, "output_tokens", 0) * p["output"]
    # Cache tokens (may not always be present)
    cost += getattr(usage, "cache_creation_input_tokens", 0) * p["cache_creation"]
    cost += getattr(usage, "cache_read_input_tokens", 0) * p["cache_read"]
    return cost


# ── Content helpers ──────────────────────────────────────────────────

def _to_block_dict(block):
    """Best-effort conversion of SDK block objects to plain dict."""
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        try:
            dumped = block.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(block, "type"):
        out = {"type": getattr(block, "type", None)}
        for key in ("id", "name", "input", "text", "tool_use_id", "content", "is_error"):
            if hasattr(block, key):
                out[key] = getattr(block, key)
        return out
    return None


def _coerce_text(value) -> str:
    """Convert arbitrary nested content to human-readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_text(v) for v in value]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        btype = value.get("type")
        if btype == "text":
            return str(value.get("text", ""))
        if "content" in value:
            return _coerce_text(value.get("content"))
        if "text" in value:
            return str(value.get("text", ""))
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _normalize_initial_messages(msgs: list[dict]) -> list[dict]:
    """Structural fix: normalize inbound history to text-only alternating chat.

    We intentionally remove all tool protocol blocks from persisted/external history.
    Tool protocol for the current request is generated only inside chat_with_tools,
    which eliminates cross-turn dangling tool_use/tool_result mismatches at the root.
    """
    clean: list[dict] = []
    for raw in msgs:
        if not isinstance(raw, dict):
            role = "user"
            text = _coerce_text(raw)
        else:
            role = raw.get("role", "user")
            if role not in ("user", "assistant"):
                role = "assistant" if role in ("model", "bot") else "user"
            text = _coerce_text(raw.get("content", ""))

        # Merge consecutive same-role into one message → strictly alternating.
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] = f"{clean[-1]['content']}\n{text}".strip()
        else:
            clean.append({"role": role, "content": text})

    return clean


def _with_message_cache_breakpoint(msgs: list[dict]) -> list[dict]:
    """Return a shallow copy of `msgs` with a cache_control breakpoint on the
    last stable assistant turn.

    Prompt caching on the Messages API requires an explicit breakpoint. Placing
    it on the most recent assistant message makes the entire prefix up to that
    point cacheable, so subsequent rounds in this call (and follow-up requests
    while the cache is warm) only re-process what came after.

    No-op when there is no assistant message yet (first turn). Leaves the input
    list and its inner dicts untouched.
    """
    if not msgs:
        return msgs
    result = list(msgs)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") != "assistant":
            continue
        msg = dict(result[i])
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content or "(계속)",
                 "cache_control": _CACHE_CONTROL_1H}
            ]
        elif isinstance(content, list) and content:
            new_content = list(content)
            last = new_content[-1]
            last = dict(last) if isinstance(last, dict) else {"type": "text", "text": str(last)}
            last["cache_control"] = _CACHE_CONTROL_1H
            new_content[-1] = last
            msg["content"] = new_content
        else:
            msg["content"] = [
                {"type": "text", "text": "(계속)",
                 "cache_control": _CACHE_CONTROL_1H}
            ]
        result[i] = msg
        break
    return result


def _append_user_text_message(msgs: list[dict], text: str):
    """Append user text while preserving role alternation."""
    if msgs and msgs[-1].get("role") == "user":
        prev = msgs[-1].get("content", "")
        if isinstance(prev, str):
            msgs[-1]["content"] = f"{prev}\n{text}".strip()
        elif isinstance(prev, list):
            msgs[-1]["content"].append({"type": "text", "text": text})
        else:
            msgs[-1]["content"] = f"{_coerce_text(prev)}\n{text}".strip()
    else:
        msgs.append({"role": "user", "content": text})


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
    on_progress=None,
    task_id: int | None = None,
    agent_name: str = "agent",
    mission_id: int | None = None,
    finalization_tools: list[str] | None = None,
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
        on_progress: Optional async callable(event: str, detail: str) for live progress.
            Events: "thinking" (model's intermediate text), "tool_call" (tool invoked),
            "tool_result" (tool finished), "budget" (budget status update).
    """
    budget_usd = validate_budget(budget_usd)

    # Per-agent-run provenance buffer for KG write/read trust tracking.
    from shared import init_provenance_buffer
    init_provenance_buffer(agent=agent_name, mission_id=mission_id)

    # Root-cause fix: start from text-only canonical history.
    # Tool protocol blocks are generated only within this call.
    working_msgs = _normalize_initial_messages(messages)
    tool_call_log = []
    tool_work_details = []  # result snippets for scratchpad
    total_cost = 0.0
    budget_warning_sent = False

    # Prompt caching: mark system prompt and tools as cacheable with the
    # 1-hour TTL tier (see _CACHE_CONTROL_1H rationale above).
    cached_system = [{"type": "text", "text": system_prompt, "cache_control": _CACHE_CONTROL_1H}]

    # Mark last tool for prompt caching
    cached_tools = [dict(t) for t in tools]
    if cached_tools:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": _CACHE_CONTROL_1H}

    response = None
    round_num = 0
    accumulated_text_parts: list[str] = []  # Collect text from tool_use rounds
    for round_num in range(1, max_rounds + 1):
        # ── Cancel check ──
        check_cancelled(task_id)

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system,
            tools=cached_tools,
            messages=_with_message_cache_breakpoint(working_msgs),
        )

        # Track cost. Log cache-token breakdown at INFO so prompt-caching
        # effectiveness is visible in journald without a debug rebuild — if
        # cache_read stays at 0 across rounds, something in the prefix is
        # drifting and the ephemeral cache can't latch on.
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            round_cost = _calculate_cost(usage, model)
            total_cost += round_cost
            logger.info(
                "Round %d usage: in=%d out=%d cache_create=%d cache_read=%d → $%.4f (total: $%.4f / $%.2f)",
                round_num,
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
                getattr(usage, "cache_read_input_tokens", 0),
                round_cost, total_cost, budget_usd,
            )
            await emit_progress(on_progress, "budget", f"[{round_num}] ${total_cost:.3f}/${budget_usd:.2f}")
            update_redis_state(task_id, round_num, total_cost)

        # If no custom tool use, extract and return text (check BEFORE budget)
        if response.stop_reason not in ("tool_use", "pause_turn"):
            if response.stop_reason == "max_tokens":
                logger.warning("Response truncated by max_tokens (%d) at round %d/%d", max_tokens, round_num, max_rounds)
                if log_event:
                    log_event("warning", "chat", f"Response truncated by max_tokens ({max_tokens}) at round {round_num}/{max_rounds}")
            text_parts = [b.text for b in response.content if b.type == "text"]
            # Combine accumulated text from tool_use rounds with final response
            all_text = accumulated_text_parts + text_parts
            if budget_tracker is not None:
                budget_tracker.update(build_budget_tracker(total_cost, round_num, False, tool_work_details))
            return "\n".join(all_text) if all_text else EMPTY_RESPONSE_FALLBACK

        # Budget exceeded → process this response's tool calls, then break
        budget_exceeded_this_round = total_cost >= budget_usd
        if budget_exceeded_this_round:
            logger.warning("Budget exhausted: $%.4f >= $%.2f at round %d — processing final tool calls before exit", total_cost, budget_usd, round_num)

        # First pass: build assistant_content and collect tool_uses to execute.
        assistant_content = []
        tool_uses_to_execute: list[tuple[str, str, dict]] = []
        for block in response.content:
            b = _to_block_dict(block) or {"type": getattr(block, "type", "unknown")}
            btype = b.get("type")

            if btype == "text":
                text = str(b.get("text", ""))
                assistant_content.append({"type": "text", "text": text})
                # Accumulate substantial text from tool_use rounds for final result
                if text.strip() and len(text.strip()) > 20:
                    accumulated_text_parts.append(text.strip())
                if text.strip():
                    await emit_progress(on_progress, "thinking", f"[{round_num}] {text.strip()}")
            elif btype in ("server_tool_use", "web_search_tool_result"):
                # Defensive fallback: server-side tools are no longer used
                # (replaced by Tavily client tool), but convert to text if
                # the API ever returns them unexpectedly.
                t = _coerce_text(b.get("content", "") or b.get("input", "") or b.get("name", ""))
                if t:
                    assistant_content.append({"type": "text", "text": f"[server: {t[:2000]}]"})
            elif btype == "tool_use":
                tid = str(b.get("id", "")).strip()
                tname = str(b.get("name", "")).strip()
                tinput = b.get("input", {}) if isinstance(b.get("input", {}), dict) else {}
                if not tid or not tname:
                    logger.warning("Skipping malformed tool_use block: %s", b)
                    continue

                assistant_content.append({
                    "type": "tool_use",
                    "id": tid,
                    "name": tname,
                    "input": tinput,
                })
                tool_uses_to_execute.append((tid, tname, tinput))
            else:
                # Preserve unknown future block types as text context.
                assistant_content.append({"type": "text", "text": _coerce_text(b)})

        # Second pass: execute tool calls. Consecutive read-only tools run in
        # parallel via execute_tools_batch; everything else stays sequential.
        tool_results = []
        if tool_uses_to_execute:
            exec_results = await execute_tools_batch(
                tool_uses_to_execute,
                tool_handlers,
                on_progress=on_progress,
                round_num=round_num,
                log_event=log_event,
            )
            for tid, tname, tinput, result, is_error in exec_results:
                input_summary = json.dumps(tinput, ensure_ascii=False)
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
                # Log for diagnostics (post-execution, in input order)
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {tname}({input_summary})")
                tool_work_details.append(f"  [{round_num}] {tname}({input_summary}) → {result}")
                save_redis_progress(task_id, round_num, tname, input_summary, result, is_error)

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
        # Note: server_tool blocks are already converted to text above,
        # so working_msgs never contain server_tool_use/web_search_tool_result.
        working_msgs.append({"role": "assistant", "content": assistant_content})

        if tool_results:
            # Warnings must be appended AFTER tool_result blocks, not prepended.
            # Claude requires tool_use ids to have tool_result blocks immediately
            # after in the next user turn — a text block before them triggers
            # "tool_use ids without tool_result blocks immediately after".
            if not budget_warning_sent and total_cost > budget_usd * 0.8:
                budget_warning_sent = True
                tool_results.append({"type": "text", "text": build_budget_warning(total_cost, budget_usd)})
            if round_num == max_rounds - 2:
                tool_results.append({"type": "text", "text": build_round_warning(round_num, max_rounds)})
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

    limit_reason = "예산 소진" if budget_exhausted else "도구 호출 한도 도달"

    # Build the finalization tool whitelist (subset of the agent's allowed
    # tools). If provided, expose only these tools on the forced-final call so
    # the agent can persist its work (e.g. save_diary) on the way out.
    final_tools = None
    final_tool_names: list[str] = []
    if finalization_tools:
        final_tool_names = [t["name"] for t in cached_tools if t.get("name") in set(finalization_tools)]
        if final_tool_names:
            final_tools = [dict(t) for t in cached_tools if t.get("name") in set(final_tool_names)]
            # Preserve prompt caching semantics on the filtered list. Must
            # use the same 1h TTL as cached_system — Anthropic processes
            # `tools` before `system`, and a longer TTL cannot follow a
            # shorter one, so mixing 5m (default ephemeral) here with a 1h
            # system block raises a 400 (the diary task forced-final path
            # hit this).
            final_tools[-1] = {**final_tools[-1], "cache_control": _CACHE_CONTROL_1H}

    _append_user_text_message(
        working_msgs,
        build_limit_message(
            limit_reason, total_cost, budget_usd,
            round_num if response else 0, max_rounds, was_still_working,
            finalization_tools=final_tool_names or None,
        ),
    )
    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": cached_system,
        "messages": _with_message_cache_breakpoint(working_msgs),
    }
    if final_tools:
        create_kwargs["tools"] = final_tools
    final = await client.messages.create(**create_kwargs)

    if hasattr(final, "usage") and final.usage:
        total_cost += _calculate_cost(final.usage, model)
    if final.stop_reason == "max_tokens":
        logger.warning("Forced final response truncated by max_tokens (%d)", max_tokens)

    # If the finalization call returned tool_use blocks, execute them once
    # and then make a plain text follow-up call to collect the final answer.
    final_tool_uses: list[tuple[str, str, dict]] = []
    final_assistant_content: list[dict] = []
    for block in final.content:
        b = _to_block_dict(block) or {"type": getattr(block, "type", "unknown")}
        btype = b.get("type")
        if btype == "text":
            final_assistant_content.append({"type": "text", "text": str(b.get("text", ""))})
        elif btype == "tool_use":
            tid = str(b.get("id", "")).strip()
            tname = str(b.get("name", "")).strip()
            tinput = b.get("input", {}) if isinstance(b.get("input", {}), dict) else {}
            if tid and tname and tname in set(final_tool_names):
                final_assistant_content.append({
                    "type": "tool_use", "id": tid, "name": tname, "input": tinput,
                })
                final_tool_uses.append((tid, tname, tinput))
            else:
                logger.warning("Forced-final: ignoring non-finalization tool_use name=%s", tname)

    if final_tool_uses:
        logger.info("Forced-final: executing %d finalization tool call(s)", len(final_tool_uses))
        exec_results = await execute_tools_batch(
            final_tool_uses,
            tool_handlers,
            on_progress=on_progress,
            round_num=(round_num if response else 0) + 1,
            log_event=log_event,
        )
        final_tool_results = []
        for tid, tname, tinput, result, is_error in exec_results:
            input_summary = json.dumps(tinput, ensure_ascii=False)
            tr = {"type": "tool_result", "tool_use_id": tid, "content": result}
            if is_error:
                tr["is_error"] = True
            final_tool_results.append(tr)
            tool_call_log.append(f"  [final] {tname}({input_summary})")
            tool_work_details.append(f"  [final] {tname}({input_summary}) → {result}")
            save_redis_progress(task_id, (round_num if response else 0) + 1, tname, input_summary, result, is_error)

        working_msgs.append({"role": "assistant", "content": final_assistant_content})
        working_msgs.append({"role": "user", "content": final_tool_results})

        # Plain text follow-up to collect the final answer.
        followup = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system,
            messages=_with_message_cache_breakpoint(working_msgs),  # no tools — force text
        )
        if hasattr(followup, "usage") and followup.usage:
            total_cost += _calculate_cost(followup.usage, model)
        followup_text_parts = [b.text for b in followup.content if b.type == "text"]
        text_parts = [tp["text"] for tp in final_assistant_content if tp.get("type") == "text"] + followup_text_parts
    else:
        text_parts = [b.text for b in final.content if b.type == "text"]

    all_text = accumulated_text_parts + text_parts
    if budget_tracker is not None:
        budget_tracker.update(build_budget_tracker(
            total_cost, round_num if response else 0, was_still_working, tool_work_details))
    return "\n".join(all_text) if all_text else EMPTY_RESPONSE_FALLBACK
