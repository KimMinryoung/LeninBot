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
import re
import uuid
import httpx

from tool_loop_common import (
    validate_budget, build_budget_tracker, emit_progress,
    update_redis_state, save_redis_progress, execute_tool,
    execute_tools_batch,
    build_limit_message, build_budget_warning, build_round_warning,
    build_stripped_limit_message, EMPTY_RESPONSE_FALLBACK,
    check_cancelled, TaskCancelledError,
)

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
    """Convert Anthropic-format tools to OpenAI format and deduplicate by tool name.

    OpenAI rejects duplicated tool names with:
    `tools: Tool names must be unique.`
    Keep first occurrence to preserve the originally registered schema.
    """
    converted = []
    seen_names: set[str] = set()
    for t in tools:
        clean = {k: v for k, v in t.items() if k != "cache_control"}
        if clean.get("type") == "function":
            candidate = clean
            name = ((candidate.get("function") or {}).get("name") or "").strip()
        elif "input_schema" in clean:
            candidate = _convert_tool_anthropic_to_openai(clean)
            name = ((candidate.get("function") or {}).get("name") or "").strip()
        else:
            candidate = clean
            name = str(candidate.get("name", "") or "").strip()

        if name and name in seen_names:
            logger.warning("Dropping duplicate tool definition for OpenAI payload: %s", name)
            continue
        if name:
            seen_names.add(name)
        converted.append(candidate)
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


def _ensure_system_first(msgs: list[dict], system_prompt: str) -> list[dict]:
    """Remove all system messages from msgs and prepend a single system message.

    Prevents duplicate/misplaced system messages that cause Jinja template
    errors on local LLMs (e.g. Qwen's "System message must be at the beginning").
    """
    cleaned = [m for m in msgs if m.get("role") != "system"]
    if system_prompt:
        cleaned.insert(0, {"role": "system", "content": system_prompt})
    return cleaned


def _estimate_tokens(msgs: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token for mixed content."""
    total = 0
    for m in msgs:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            total += sum(len(str(b)) for b in content) // 4
        # tool_calls add overhead
        if m.get("tool_calls"):
            total += sum(len(json.dumps(tc)) for tc in m["tool_calls"]) // 4
    return total


def _truncate_to_context(msgs: list[dict], context_limit: int, max_tokens: int) -> list[dict]:
    """Trim older messages to fit within context_limit, reserving space for output.

    Keeps the system message (index 0) and trims from the front of the
    conversation (oldest messages first), preserving the most recent exchanges.
    """
    if context_limit <= 0:
        return msgs

    budget = context_limit - max_tokens - 200  # 200 token safety margin
    if budget <= 0:
        budget = context_limit // 2

    est = _estimate_tokens(msgs)
    if est <= budget:
        return msgs

    # Keep system message + trim oldest non-system messages
    system_msg = msgs[0] if msgs and msgs[0].get("role") == "system" else None
    rest = msgs[1:] if system_msg else list(msgs)

    # Remove oldest messages until we fit
    while rest and _estimate_tokens(([system_msg] if system_msg else []) + rest) > budget:
        rest.pop(0)

    result = ([system_msg] if system_msg else []) + rest
    if len(result) != len(msgs):
        logger.warning("Context truncation: %d → %d messages (est %d → %d tokens, budget %d)",
                       len(msgs), len(result), est,
                       _estimate_tokens(result), budget)
    return result


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
    enable_thinking: bool = False,
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
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

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
                    parallel_tool_calls=True, enable_thinking=False):
    """Dispatch to SDK or httpx based on mode."""
    if sdk_mode:
        return await _call_sdk(client, model, messages, tools, max_tokens,
                               parallel_tool_calls=parallel_tool_calls)
    else:
        return await _call_api(base_url, model, messages, tools, max_tokens,
                               enable_thinking=enable_thinking)


# <think>...</think> emitted by Qwen/Deepseek reasoning models. llama-server
# with --reasoning-format deepseek splits these into a separate
# reasoning_content field, but with older server versions or when the flag
# isn't set the tags leak into the main content and end up shown to the user.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Qwen's tool_call marker when the server fails to parse it natively. The
# model emits `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` in
# content and llama-server forwards it verbatim instead of populating the
# tool_calls field — our loop then treats it as plain text and drops the
# intended call silently.
_INLINE_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _rescue_inline_tool_calls(content: str) -> tuple[str, list[dict] | None]:
    """Parse `<tool_call>…</tool_call>` blocks embedded in the content string.

    Returns (content_without_tags, tool_calls_or_None). The content keeps any
    prose outside the tags so the loop can still accumulate reasoning text.
    """
    if not content or "<tool_call>" not in content:
        return content, None

    calls: list[dict] = []
    for m in _INLINE_TOOL_CALL_RE.finditer(content):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            logger.warning("Inline <tool_call> block had invalid JSON: %s", e)
            continue
        name = (payload.get("name") or payload.get("function") or "").strip()
        args = payload.get("arguments") if "arguments" in payload else payload.get("args")
        if not name:
            continue
        if args is None:
            args = {}
        # Normalize: arguments may come back as a JSON string instead of an
        # object when the model double-encoded the call.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({
            "id": f"inline_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })

    if not calls:
        return content, None

    cleaned = _INLINE_TOOL_CALL_RE.sub("", content).strip()
    return cleaned, calls


def _extract_response(sdk_mode, response_or_data):
    """Extract (finish_reason, content_text, tool_calls, message, usage) from response.

    Post-processing applied before returning:
      1. Strip `<think>…</think>` blocks leaked into content (Qwen with older
         llama-server builds where reasoning isn't split into its own field).
      2. Rescue `<tool_call>…</tool_call>` blocks that llama-server forwarded
         as text instead of parsing as a proper tool_calls field. When a
         rescue happens we also flip finish_reason to "tool_calls" so the
         surrounding loop dispatches the call instead of treating the turn
         as a plain text reply.
    """
    if sdk_mode:
        choice = response_or_data.choices[0]
        return (
            choice.finish_reason or "stop",
            choice.message.content or "",
            choice.message.tool_calls,
            choice.message,
            response_or_data.usage,
        )

    choice = response_or_data["choices"][0]
    msg = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")

    # Strip <think> tags if leaked into content
    if content and "<think>" in content:
        stripped = _THINK_TAG_RE.sub("", content).strip()
        if stripped != content:
            logger.info("Stripped <think> tags from content (original %d → %d chars)",
                        len(content), len(stripped))
            content = stripped

    # Rescue inline <tool_call> blocks only when the server didn't already
    # surface structured tool_calls — otherwise we'd double-dispatch.
    if not tool_calls and content and "<tool_call>" in content:
        cleaned, rescued = _rescue_inline_tool_calls(content)
        if rescued:
            logger.warning(
                "Rescued %d inline <tool_call> block(s) from content — "
                "llama-server did not surface them in tool_calls",
                len(rescued),
            )
            content = cleaned
            tool_calls = rescued
            finish_reason = "tool_calls"

    return finish_reason, content, tool_calls, msg, None


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
    max_rounds: int = 50,
    max_tokens: int = 4096,
    log_event=None,
    budget_usd: float = 0.30,
    budget_tracker: dict | None = None,
    on_progress=None,
    task_id: int | None = None,
    context_limit: int = 0,
    enable_thinking: bool = False,
    agent_name: str = "agent",
    mission_id: int | None = None,
    finalization_tools: list[str] | None = None,
    api_semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Call OpenAI-compatible LLM with tools, execute tool calls, loop until text response.

    Interface mirrors claude_loop.chat_with_tools() for drop-in use.

    api_semaphore: if provided, acquired/released around each individual LLM
    API call (not the entire tool loop).  This prevents deadlocks when a tool
    handler (e.g. run_agent) recursively invokes chat_with_tools on the same
    single-slot backend.
    """
    sdk_mode = client is not None

    # Wrap _api_call with per-call semaphore so the lock is held only during
    # the HTTP POST, not during tool execution between rounds.
    async def _guarded_api_call(*a, **kw):
        if api_semaphore is not None:
            async with api_semaphore:
                return await _api_call(*a, **kw)
        return await _api_call(*a, **kw)

    budget_usd = validate_budget(budget_usd)

    # Per-agent-run provenance buffer for KG write/read trust tracking.
    from shared import init_provenance_buffer
    init_provenance_buffer(agent=agent_name, mission_id=mission_id)

    openai_tools = _convert_tools(tools)

    # ── Root-cause fix: start from text-only canonical history ──
    # Tool protocol blocks are generated only within this call.
    working_msgs = _normalize_messages(messages)
    working_msgs = _ensure_system_first(working_msgs, system_prompt)

    # ── Context window management for local LLMs ──
    if context_limit > 0:
        working_msgs = _truncate_to_context(working_msgs, context_limit, max_tokens)

    tool_call_log = []
    tool_work_details = []
    total_cost = 0.0
    budget_warning_sent = False
    round_num = 0
    accumulated_text_parts: list[str] = []  # Collect text from tool_calls rounds

    for round_num in range(1, max_rounds + 1):

        # ── Cancel check ──
        check_cancelled(task_id)

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
                working_msgs = _ensure_system_first(working_msgs, system_prompt)

        # ── API call with auto-recovery ──
        response = None
        try:
            response = await _guarded_api_call(
                sdk_mode, client, base_url, model, working_msgs,
                openai_tools, max_tokens, enable_thinking=enable_thinking,
            )
        except Exception as api_err:
            err_str = str(api_err)
            _dump_messages_for_debug(working_msgs, round_num, api_err)

            # Auto-recovery: strip tool protocol and retry once
            if "tool" in err_str.lower() or "400" in err_str or "invalid" in err_str.lower():
                logger.warning("Auto-recovery (round %d): stripping tool protocol and retrying",
                               round_num)
                stripped = _strip_tool_protocol(working_msgs)
                stripped = _ensure_system_first(stripped, system_prompt)
                try:
                    response = await _guarded_api_call(
                        sdk_mode, client, base_url, model, stripped,
                        openai_tools, max_tokens,
                        parallel_tool_calls=False, enable_thinking=enable_thinking,
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
            await emit_progress(on_progress, "budget", f"[{round_num}] ${total_cost:.3f}/${budget_usd:.2f}")
            update_redis_state(task_id, round_num, total_cost)

        # ── Handle refusal (OpenAI safety filter) ──
        refusal = None
        if sdk_mode and hasattr(message_obj, "refusal"):
            refusal = message_obj.refusal
        elif not sdk_mode and isinstance(message_obj, dict):
            refusal = message_obj.get("refusal")
        if refusal:
            logger.warning("Model refused request at round %d: %s", round_num, refusal)
            if budget_tracker is not None:
                budget_tracker.update(build_budget_tracker(total_cost, round_num, False, tool_work_details))
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
            all_parts = accumulated_text_parts + ([content_text.strip()] if content_text.strip() else [])
            final_text = "\n".join(all_parts)
            if budget_tracker is not None:
                budget_tracker.update(build_budget_tracker(total_cost, round_num, False, tool_work_details))
            return final_text if final_text else EMPTY_RESPONSE_FALLBACK

        # ── Budget check (matches claude_loop.py) ──
        budget_exceeded = total_cost >= budget_usd
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
                budget_tracker.update(build_budget_tracker(total_cost, round_num, False, tool_work_details))
            return content_text.strip() if content_text.strip() else EMPTY_RESPONSE_FALLBACK

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

        if content_text.strip():
            await emit_progress(on_progress, "thinking", f"[{round_num}] {content_text.strip()}")

        # ── Build the batch (id, name, parsed_args) for parallel-aware exec ──
        batch: list[tuple[str, str, dict]] = []
        for tc_item in tc_list:
            tc_id = tc_item["id"]
            func_name = tc_item["function"]["name"]
            try:
                func_args = json.loads(tc_item["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed arguments for %s: %s",
                               func_name, tc_item["function"]["arguments"][:200])
                func_args = {}
            batch.append((tc_id, func_name, func_args))

        # ── Execute tool calls (parallel for read-only batches) ──
        executed_ids: set[str] = set()
        if batch:
            exec_results = await execute_tools_batch(
                batch,
                tool_handlers,
                on_progress=on_progress,
                round_num=round_num,
                log_event=log_event,
            )
            for tc_id, func_name, func_args, result, is_error in exec_results:
                input_summary = json.dumps(func_args, ensure_ascii=False)
                working_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })
                executed_ids.add(tc_id)
                tool_call_log.append(f"  [{round_num}/{max_rounds}] {func_name}({input_summary})")
                tool_work_details.append(f"  [{round_num}] {func_name}({input_summary}) → {result}")
                save_redis_progress(task_id, round_num, func_name, input_summary, result, is_error)

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
        if not budget_warning_sent and total_cost > budget_usd * 0.8:
            budget_warning_sent = True
            working_msgs.append({"role": "user", "content": build_budget_warning(total_cost, budget_usd)})

        # ── Round limit warning 2 rounds before max ──
        if round_num == max_rounds - 2:
            working_msgs.append({"role": "user", "content": build_round_warning(round_num, max_rounds)})

    # ══════════════════════════════════════════════════════════════════
    # Forced final response: max_rounds or budget exhausted
    # ══════════════════════════════════════════════════════════════════
    budget_exhausted = total_cost >= budget_usd
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

    # Build the finalization tool whitelist so the agent can still persist its
    # work (e.g. save_diary) on the way out. Filter by name against the
    # already-converted OpenAI tool list.
    finalization_names: list[str] = []
    final_openai_tools = None
    if finalization_tools:
        allowed_set = set(finalization_tools)
        final_openai_tools = [
            t for t in openai_tools
            if t.get("function", {}).get("name") in allowed_set
        ]
        finalization_names = [
            t["function"]["name"] for t in final_openai_tools
        ]
        if not final_openai_tools:
            final_openai_tools = None

    working_msgs.append({
        "role": "user",
        "content": build_limit_message(
            limit_reason, total_cost, budget_usd,
            round_num, max_rounds, was_still_working,
            finalization_tools=finalization_names or None,
        ),
    })

    # ── Preflight: validate tool result completeness ──
    missing_ids = _validate_tool_results(working_msgs)
    if missing_ids:
        logger.error("Forced-final preflight: %d missing tool results — stripping",
                     len(missing_ids))
        working_msgs = _strip_tool_protocol(working_msgs)
        working_msgs = _ensure_system_first(working_msgs, system_prompt)
        # Re-append the limit message (was lost in strip)
        working_msgs.append({
            "role": "user",
            "content": build_stripped_limit_message(limit_reason),
        })

    # ── Final API call: expose finalization tools if any, else plain text ──
    try:
        final_response = await _guarded_api_call(
            sdk_mode, client, base_url, model, working_msgs, final_openai_tools, max_tokens,
        )
        _, text, final_tool_calls, _, final_usage = _extract_response(sdk_mode, final_response)
        if sdk_mode and final_usage:
            total_cost += _calculate_cost(final_usage, model)

        # If the agent called finalization tools, execute them and do a
        # text-only follow-up to collect the final answer.
        if final_tool_calls and finalization_names:
            final_tc_list = _build_tc_list(sdk_mode, final_tool_calls)
            if final_tc_list:
                working_msgs.append({
                    "role": "assistant",
                    "content": text if text.strip() else None,
                    "tool_calls": final_tc_list,
                })
                allowed = set(finalization_names)
                final_batch: list[tuple[str, str, dict]] = []
                for tc_item in final_tc_list:
                    fname = tc_item["function"]["name"]
                    if fname not in allowed:
                        # Inject an error result so the protocol stays valid.
                        working_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc_item["id"],
                            "content": f"Tool {fname} blocked: budget exhausted.",
                        })
                        continue
                    try:
                        fargs = json.loads(tc_item["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        fargs = {}
                    final_batch.append((tc_item["id"], fname, fargs))

                if final_batch:
                    logger.info("Forced-final: executing %d finalization tool call(s)", len(final_batch))
                    final_exec = await execute_tools_batch(
                        final_batch, tool_handlers,
                        on_progress=on_progress, round_num=round_num + 1, log_event=log_event,
                    )
                    for tc_id, fname, fargs, result, is_error in final_exec:
                        input_summary = json.dumps(fargs, ensure_ascii=False)
                        working_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result,
                        })
                        tool_call_log.append(f"  [final] {fname}({input_summary})")
                        tool_work_details.append(f"  [final] {fname}({input_summary}) → {result}")
                        save_redis_progress(task_id, round_num + 1, fname, input_summary, result, is_error)

                # Plain text follow-up to collect the final answer.
                followup = await _guarded_api_call(
                    sdk_mode, client, base_url, model, working_msgs, None, max_tokens,
                )
                _, followup_text, _, _, followup_usage = _extract_response(sdk_mode, followup)
                if sdk_mode and followup_usage:
                    total_cost += _calculate_cost(followup_usage, model)
                text = (text or "") + ("\n" if text and followup_text else "") + (followup_text or "")
    except Exception as final_err:
        # ── Last resort: strip all tool protocol and retry ──
        _dump_messages_for_debug(working_msgs, -1, final_err)
        logger.warning("Forced response failed — retrying with stripped messages")
        stripped = _strip_tool_protocol(working_msgs)
        if system_prompt:
            stripped.insert(0, {"role": "system", "content": system_prompt})
        stripped.append({
            "role": "user",
            "content": build_stripped_limit_message(limit_reason),
        })
        try:
            last_response = await _guarded_api_call(
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
                budget_tracker.update(build_budget_tracker(
                    total_cost, round_num, was_still_working, tool_work_details))
            return f"⚠️ {limit_reason} 후 응답 생성 실패: {final_err}"

    all_parts = accumulated_text_parts + ([text.strip()] if text.strip() else [])
    final_text = "\n".join(all_parts)
    if budget_tracker is not None:
        budget_tracker.update(build_budget_tracker(
            total_cost, round_num, was_still_working, tool_work_details))
    return final_text if final_text else EMPTY_RESPONSE_FALLBACK
