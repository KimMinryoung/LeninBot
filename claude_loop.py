"""claude_loop.py — Claude tool-use loop and message sanitization.

Extracted from telegram_bot.py. Dependencies injected via function parameters
to avoid circular imports.
"""

import asyncio
import json
import logging

from tool_loop_common import (
    validate_budget, build_budget_tracker, emit_progress,
    update_redis_state, save_redis_progress, execute_tool,
    build_limit_message, build_budget_warning, build_round_warning,
    EMPTY_RESPONSE_FALLBACK,
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


def _parse_json_string_maybe(text: str):
    s = text.strip()
    if not s or (not s.startswith("[") and not s.startswith("{")):
        return text
    try:
        return json.loads(s)
    except Exception:
        return text


def _normalize_content_for_role(role: str, content, handle_server_tools: bool):
    """Normalize message content into Anthropic-compatible shape for this role."""
    parsed = _parse_json_string_maybe(content) if isinstance(content, str) else content
    items = parsed if isinstance(parsed, list) else [parsed]
    blocks: list[dict] = []
    loose_text: list[str] = []

    for item in items:
        block = _to_block_dict(item)
        if block is None:
            t = _coerce_text(item)
            if t:
                loose_text.append(t)
            continue

        btype = block.get("type")
        if btype == "text":
            text = _coerce_text(block.get("text", "")).strip()
            if text:
                blocks.append({"type": "text", "text": text})
            continue

        if role == "assistant" and btype == "tool_use":
            tid = str(block.get("id", "")).strip()
            name = str(block.get("name", "")).strip()
            raw_input = block.get("input", {})
            tool_input = raw_input if isinstance(raw_input, dict) else {}
            if tid and name:
                blocks.append({"type": "tool_use", "id": tid, "name": name, "input": tool_input})
            else:
                t = _coerce_text(block)
                if t:
                    loose_text.append(t)
            continue

        if role == "assistant" and handle_server_tools and btype == "server_tool_use":
            tid = str(block.get("id", "")).strip()
            name = str(block.get("name", "")).strip()
            raw_input = block.get("input", {})
            tool_input = raw_input if isinstance(raw_input, dict) else {}
            if tid and name:
                blocks.append({"type": "server_tool_use", "id": tid, "name": name, "input": tool_input})
            continue

        if role == "assistant" and handle_server_tools and btype == "web_search_tool_result":
            tid = str(block.get("tool_use_id", "")).strip()
            raw_content = block.get("content", [])
            result_content = raw_content if isinstance(raw_content, list) else []
            if tid:
                blocks.append({"type": "web_search_tool_result", "tool_use_id": tid, "content": result_content})
            continue

        if role == "user" and btype == "tool_result":
            tid = str(block.get("tool_use_id", "")).strip()
            if tid:
                tr = {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": block.get("content", ""),
                }
                if bool(block.get("is_error", False)):
                    tr["is_error"] = True
                blocks.append(tr)
            continue

        # Unknown/incompatible blocks become text only.
        t = _coerce_text(block)
        if t:
            loose_text.append(t)

    if loose_text:
        text = "\n".join(t for t in loose_text if t).strip()
        if text:
            if blocks:
                blocks.insert(0, {"type": "text", "text": text})
            else:
                return text

    if blocks:
        return blocks
    return _coerce_text(parsed) or "(empty)"


def _canonicalize_messages(msgs: list[dict], handle_server_tools: bool = True):
    """Canonicalize transcript and guarantee tool pair invariants."""
    normalized: list[dict] = []
    for raw in msgs:
        if not isinstance(raw, dict):
            raw = {"role": "user", "content": _coerce_text(raw)}
        role = raw.get("role", "user")
        if role not in ("user", "assistant"):
            role = "assistant" if role in ("model", "bot") else "user"
        content = _normalize_content_for_role(role, raw.get("content", ""), handle_server_tools)
        normalized.append({"role": role, "content": content})

    # Ensure alternating roles; insert lightweight fillers when broken.
    aligned: list[dict] = []
    inserted_fillers = 0
    for msg in normalized:
        if aligned and aligned[-1]["role"] == msg["role"]:
            filler_role = "assistant" if msg["role"] == "user" else "user"
            aligned.append({"role": filler_role, "content": "(계속)"})
            inserted_fillers += 1
        aligned.append(msg)

    injected_results = 0
    rewritten_orphans = 0
    i = 0
    while i < len(aligned):
        msg = aligned[i]
        if msg.get("role") != "assistant":
            # Orphan tool_result blocks are converted to text to keep API-valid structure.
            content = msg.get("content", "")
            if isinstance(content, list):
                prev_tool_ids: set[str] = set()
                if i > 0 and aligned[i - 1].get("role") == "assistant":
                    prev_content = aligned[i - 1].get("content", [])
                    if isinstance(prev_content, list):
                        prev_tool_ids = {
                            str(b.get("id", "")).strip()
                            for b in prev_content
                            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
                        }

                fixed = []
                orphan_notes = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = str(b.get("tool_use_id", "")).strip()
                        if tid and tid in prev_tool_ids:
                            fixed.append(b)
                        else:
                            orphan_notes.append(f"[orphan tool_result ignored: {tid or '?'}]")
                            rewritten_orphans += 1
                    else:
                        fixed.append(b)

                if orphan_notes:
                    fixed.insert(0, {"type": "text", "text": "\n".join(orphan_notes)})
                msg["content"] = fixed if fixed else "(empty)"
            i += 1
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            i += 1
            continue

        tool_ids = [
            str(b.get("id", "")).strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        ]

        if handle_server_tools:
            resolved_server = {
                str(b.get("tool_use_id", "")).strip()
                for b in content
                if isinstance(b, dict) and b.get("type") == "web_search_tool_result" and b.get("tool_use_id")
            }
            pending_server = [
                str(b.get("id", "")).strip()
                for b in content
                if isinstance(b, dict) and b.get("type") == "server_tool_use" and b.get("id")
                and str(b.get("id", "")).strip() not in resolved_server
            ]
            for tid in pending_server:
                content.append({"type": "web_search_tool_result", "tool_use_id": tid, "content": []})
                injected_results += 1
            msg["content"] = content

        if not tool_ids:
            i += 1
            continue

        next_idx = i + 1
        if next_idx >= len(aligned) or aligned[next_idx].get("role") != "user":
            aligned.insert(next_idx, {"role": "user", "content": []})
            inserted_fillers += 1

        next_msg = aligned[next_idx]
        next_content = next_msg.get("content", [])
        if not isinstance(next_content, list):
            txt = _coerce_text(next_content)
            next_content = [{"type": "text", "text": txt}] if txt else []

        resolved_custom = {
            str(b.get("tool_use_id", "")).strip()
            for b in next_content
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
        }

        for tid in tool_ids:
            if tid and tid not in resolved_custom:
                next_content.insert(0, {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "[tool result unavailable — auto-injected]",
                    "is_error": True,
                })
                injected_results += 1

        next_msg["content"] = next_content
        aligned[next_idx] = next_msg
        i = next_idx + 1

    stats = {
        "injected_results": injected_results,
        "inserted_fillers": inserted_fillers,
        "rewritten_orphans": rewritten_orphans,
    }
    return aligned, stats


def _find_unresolved_tool_uses(msgs: list[dict]) -> list[tuple[int, list[str]]]:
    """Return unresolved custom tool_use IDs per assistant message index."""
    unresolved: list[tuple[int, list[str]]] = []
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        tool_ids = {
            str(b.get("id", "")).strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        }
        if not tool_ids:
            continue
        resolved = set()
        if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
            nc = msgs[i + 1].get("content", [])
            if isinstance(nc, list):
                resolved = {
                    str(b.get("tool_use_id", "")).strip()
                    for b in nc
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
                }
        missing = sorted(tid for tid in tool_ids if tid and tid not in resolved)
        if missing:
            unresolved.append((i, missing))
    return unresolved


def sanitize_messages(msgs: list[dict], handle_server_tools: bool = True) -> list[dict]:
    """Robustly canonicalize message history and inject missing tool results."""
    cleaned, stats = _canonicalize_messages(msgs, handle_server_tools=handle_server_tools)
    if any(stats.values()):
        logger.warning(
            "sanitize_messages: injected=%d fillers=%d orphan_rewrites=%d",
            stats["injected_results"], stats["inserted_fillers"], stats["rewritten_orphans"],
        )
    return cleaned


def validate_tool_pairs(msgs: list[dict]) -> list[dict]:
    """Strict pass: enforce immediate tool_use -> tool_result adjacency invariants."""
    cleaned, stats = _canonicalize_messages(msgs, handle_server_tools=True)
    unresolved = _find_unresolved_tool_uses(cleaned)
    if unresolved:
        logger.error("validate_tool_pairs: unresolved tool_use pairs after strict pass: %s", unresolved)
    elif any(stats.values()):
        logger.warning(
            "validate_tool_pairs: fixed pairs (injected=%d, fillers=%d, orphan_rewrites=%d)",
            stats["injected_results"], stats["inserted_fillers"], stats["rewritten_orphans"],
        )
    return cleaned


def _dump_messages_for_debug(msgs: list[dict], round_num: int, error: Exception):
    """Log detailed message structure when API call fails.

    Produces a concise per-message summary showing role, content block types,
    and tool_use/tool_result IDs — enough to pinpoint the mismatch.
    """
    lines = [f"=== API ERROR at round {round_num}: {error} ==="]
    lines.append(f"Total messages: {len(msgs)}")

    for idx, msg in enumerate(msgs):
        role = msg.get("role", "?")
        content = msg.get("content", "")

        if isinstance(content, str):
            lines.append(f"  [{idx}] {role}: text({len(content)} chars)")
            continue

        if not isinstance(content, list):
            lines.append(f"  [{idx}] {role}: <{type(content).__name__}>")
            continue

        block_descs = []
        for block in content:
            if not isinstance(block, dict):
                block_descs.append(f"<{type(block).__name__}>")
                continue
            btype = block.get("type", "?")
            if btype == "tool_use":
                block_descs.append(f"tool_use(id={block.get('id','?')}, name={block.get('name','?')})")
            elif btype == "tool_result":
                block_descs.append(f"tool_result(for={block.get('tool_use_id','?')})")
            elif btype == "server_tool_use":
                block_descs.append(f"server_tool_use(id={block.get('id','?')}, name={block.get('name','?')})")
            elif btype == "web_search_tool_result":
                block_descs.append(f"web_search_result(for={block.get('tool_use_id','?')})")
            elif btype == "text":
                text_preview = str(block.get("text", ""))[:60]
                block_descs.append(f"text({text_preview!r})")
            else:
                block_descs.append(f"{btype}({list(block.keys())})")
        lines.append(f"  [{idx}] {role}: [{', '.join(block_descs)}]")

    full_dump = "\n".join(lines)
    logger.error(full_dump)


def _strip_tool_blocks(msgs: list[dict]) -> list[dict]:
    """Nuclear recovery: convert all tool protocol blocks to plain text.

    Preserves tool call/result CONTENT as readable text so the model retains
    context from previous tool executions.  Only the protocol structure
    (tool_use/tool_result types) is removed.
    """
    cleaned = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            cleaned.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            cleaned.append({"role": role, "content": str(content)})
            continue

        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                t = _coerce_text(block)
                if t:
                    text_parts.append(t)
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = _coerce_text(block.get("input", {}))[:500]
                text_parts.append(f"[도구 호출: {name}({inp})]")
            elif btype == "tool_result":
                result_text = _coerce_text(block.get("content", ""))[:2000]
                text_parts.append(f"[도구 결과: {result_text}]")
            elif btype in ("server_tool_use", "web_search_tool_result"):
                # Server tool blocks: extract any useful text
                t = _coerce_text(block.get("content", "") or block.get("input", ""))
                if t:
                    text_parts.append(f"[검색 결과: {t[:2000]}]")
            else:
                t = _coerce_text(block)
                if t:
                    text_parts.append(t)

        text = "\n".join(t for t in text_parts if t).strip()
        if not text:
            text = "(빈 메시지)"

        cleaned.append({"role": role, "content": text})

    # Merge consecutive same-role messages
    merged = []
    for msg in cleaned:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n" + msg["content"]}
        else:
            merged.append(msg)

    # Ensure alternating user/assistant (API requirement)
    final = []
    for msg in merged:
        if final and final[-1]["role"] == msg["role"]:
            # Insert a filler of the opposite role
            filler_role = "assistant" if msg["role"] == "user" else "user"
            final.append({"role": filler_role, "content": "(계속)"})
        final.append(msg)

    logger.info("_strip_tool_blocks: %d msgs → %d msgs", len(msgs), len(final))
    return final


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

        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] = f"{clean[-1]['content']}\n{text}".strip()
        else:
            clean.append({"role": role, "content": text})

    # API expects alternating roles
    fixed: list[dict] = []
    for m in clean:
        if fixed and fixed[-1]["role"] == m["role"]:
            filler_role = "assistant" if m["role"] == "user" else "user"
            fixed.append({"role": filler_role, "content": "(계속)"})
        fixed.append(m)
    return fixed


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


def _prepare_messages_for_api(msgs: list[dict]) -> list[dict]:
    """Build API-safe messages from working transcript.

    Server tool blocks are converted to text at response-processing time,
    so working_msgs should not contain them.  The strip here is a defensive
    fallback.  Custom tool_use/tool_result blocks are preserved.
    """
    prepared = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, list):
            prepared.append({"role": role, "content": content})
            continue

        kept = []
        for b in content:
            if not isinstance(b, dict):
                kept.append({"type": "text", "text": _coerce_text(b)})
                continue
            btype = b.get("type")
            # Strip server tool protocol blocks entirely
            if btype in ("server_tool_use", "web_search_tool_result"):
                continue
            kept.append(b)

        if not kept:
            prepared.append({"role": role, "content": "(계속)"})
        else:
            prepared.append({"role": role, "content": kept})

    # Final strict pair validation on API-bound payload.
    prepared = validate_tool_pairs(prepared)
    return prepared


def _build_api_payload(working_msgs: list[dict], round_num: int) -> list[dict]:
    """Single path to produce API-safe payload from internal transcript."""
    api_msgs = _prepare_messages_for_api(working_msgs)
    unresolved_api = _find_unresolved_tool_uses(api_msgs)
    if unresolved_api:
        logger.error("Round %d API payload unresolved pairs: %s", round_num, unresolved_api)
        api_msgs = _strip_tool_blocks(api_msgs)
    return api_msgs


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

    # Root-cause fix: start from text-only canonical history.
    # Tool protocol blocks are generated only within this call.
    working_msgs = _normalize_initial_messages(messages)
    tool_call_log = []
    tool_work_details = []  # result snippets for scratchpad
    total_cost = 0.0
    budget_warning_sent = False

    # Prompt caching: mark system prompt and tools as cacheable
    cached_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    # Mark last tool for prompt caching
    cached_tools = [dict(t) for t in tools]
    if cached_tools:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    response = None
    round_num = 0
    accumulated_text_parts: list[str] = []  # Collect text from tool_use rounds
    for round_num in range(1, max_rounds + 1):
        unresolved = _find_unresolved_tool_uses(working_msgs)
        if unresolved:
            # Should not happen in normal flow; sanitize as a hard fail-safe.
            logger.error("Invariant broken before API call (round %d): %s", round_num, unresolved)
            working_msgs = validate_tool_pairs(working_msgs)
            unresolved = _find_unresolved_tool_uses(working_msgs)
            if unresolved:
                working_msgs = _strip_tool_blocks(working_msgs)

        try:
            api_msgs = _build_api_payload(working_msgs, round_num)

            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=cached_system,
                tools=cached_tools,
                messages=api_msgs,
            )
        except Exception as api_err:
            err_str = str(api_err)
            recovered = False
            if ("tool_use" in err_str and "tool_result" in err_str) or "tool" in err_str.lower() or "400" in err_str:
                # Auto-recovery: strict re-canonicalization first, then strip.
                _dump_messages_for_debug(api_msgs, round_num, api_err)
                logger.warning("Auto-recovery: retrying once with strict canonicalization")
                strict_msgs = validate_tool_pairs(working_msgs)
                unresolved = _find_unresolved_tool_uses(strict_msgs)
                if unresolved:
                    logger.warning("Strict retry still unresolved: %s", unresolved)
                    strict_msgs = _strip_tool_blocks(strict_msgs)
                try:
                    strict_api_msgs = _build_api_payload(strict_msgs, round_num)
                    response = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=cached_system,
                        tools=cached_tools,
                        messages=strict_api_msgs,
                    )
                    working_msgs = strict_msgs
                    recovered = True
                except Exception:
                    logger.warning("Auto-recovery strict retry failed; stripping tool blocks and forcing final response")
                    working_msgs = _strip_tool_blocks(working_msgs)
                    response = None  # ensure forced-response path runs
                    break
            if not recovered:
                raise

        # Track cost
        if hasattr(response, "usage") and response.usage:
            round_cost = _calculate_cost(response.usage)
            total_cost += round_cost
            logger.debug("Round %d cost: $%.4f (total: $%.4f / $%.2f)", round_num, round_cost, total_cost, budget_usd)
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

        # Process tool calls
        assistant_content = []
        tool_results = []
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
                # Execute tool via shared dispatch
                input_summary = json.dumps(tinput, ensure_ascii=False)
                await emit_progress(on_progress, "tool_call", f"[{round_num}] 🔧 {tname}({input_summary})")
                result, is_error = await execute_tool(tname, tinput, tool_handlers, log_event=log_event)
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": result,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                tool_results.append(tool_result_block)
                await emit_progress(on_progress, "tool_result", f"  {'❌' if is_error else '✓'} {result[:200]}")
                # Log for diagnostics
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {tname}({input_summary})")
                tool_work_details.append(f"  [{round_num}] {tname}({input_summary}) → {result}")
                save_redis_progress(task_id, round_num, tname, input_summary, result, is_error)
            else:
                # Preserve unknown future block types as text context.
                assistant_content.append({"type": "text", "text": _coerce_text(b)})

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
            # Inject budget warning at 80% threshold
            if not budget_warning_sent and total_cost > budget_usd * 0.8:
                budget_warning_sent = True
                tool_results.insert(0, {"type": "text", "text": build_budget_warning(total_cost, budget_usd)})
            # Inject round limit warning 2 rounds before max
            if round_num == max_rounds - 2:
                tool_results.insert(0, {"type": "text", "text": build_round_warning(round_num, max_rounds)})
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
    _append_user_text_message(
        working_msgs,
        build_limit_message(limit_reason, total_cost, budget_usd,
                            round_num if response else 0, max_rounds, was_still_working),
    )
    # Final preflight
    unresolved = _find_unresolved_tool_uses(working_msgs)
    if unresolved:
        logger.error("Forced-final preflight unresolved tool pairs: %s; stripping tool blocks", unresolved)
        working_msgs = _strip_tool_blocks(working_msgs)
    try:
        final_api_msgs = _build_api_payload(working_msgs, round_num if response else 0)
        final = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system,
            messages=final_api_msgs,  # no tools parameter — forces text-only response
        )
    except Exception as api_err:
        # Last resort: strip all tool blocks and retry with plain text
        _dump_messages_for_debug(working_msgs, -1, api_err)
        logger.warning("Forced response failed — retrying with stripped messages")
        stripped = _strip_tool_blocks(working_msgs)
        try:
            final = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=cached_system,
                messages=stripped,
            )
        except Exception as e2:
            logger.error("Final stripped response also failed: %s", e2)
            if log_event:
                log_event("error", "final_response", f"Final response failed even after strip: {e2}")
            if budget_tracker is not None:
                budget_tracker.update(build_budget_tracker(
                    total_cost, round_num if response else 0, was_still_working, tool_work_details))
            return f"⚠️ {limit_reason} 후 응답 생성 실패: {api_err}"

    if hasattr(final, "usage") and final.usage:
        total_cost += _calculate_cost(final.usage)
    if final.stop_reason == "max_tokens":
        logger.warning("Forced final response truncated by max_tokens (%d)", max_tokens)
    text_parts = [b.text for b in final.content if b.type == "text"]
    all_text = accumulated_text_parts + text_parts
    if budget_tracker is not None:
        budget_tracker.update(build_budget_tracker(
            total_cost, round_num if response else 0, was_still_working, tool_work_details))
    return "\n".join(all_text) if all_text else EMPTY_RESPONSE_FALLBACK
