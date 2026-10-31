"""Staged in-call context compaction for the provider tool loops.

Both protocol adapters (``llm/claude_loop.py`` for the Anthropic message
shape, ``llm/openai_tool_loop.py`` for the OpenAI shape) call this module when
the estimated request input exceeds ``max_input_tokens``. Stages run in order
and stop as soon as the request fits:

A. Replay-safe large tool results (``tool_gateway.inference``) become an
   explicit replay instruction. The source can be recovered by repeating the
   preceding read call with the same name and arguments.
B. Other large tool results, except those of the newest round, are cut to a
   head excerpt with a marker that states the length and that the omitted part
   cannot be recovered by repeating the call.
C. The oldest whole tool rounds are removed, keeping the newest
   ``KEEP_RECENT_ROUNDS`` rounds intact. A deterministic structural digest
   (tool name, argument excerpt, status, result excerpt, assistant text
   excerpt) is appended to the first user message. The digest is built from
   the removed protocol blocks, never by a model, so it cannot invent results.

Cuts always land on round boundaries, so every remaining ``tool_use`` keeps its
``tool_result`` (and every OpenAI ``tool_calls`` entry keeps its ``tool``
message). Inputs are never mutated; callers receive new lists.
"""

from __future__ import annotations

import json
from collections.abc import Callable

REPLAY_STUB_MIN_CHARS = 800
EXCERPT_CHARS = 600
KEEP_RECENT_ROUNDS = 2
DIGEST_MAX_ENTRIES = 40
DIGEST_ARG_CHARS = 200
DIGEST_RESULT_CHARS = 240
DIGEST_TEXT_CHARS = 300
CHECKPOINT_PREFIX = "[Input checkpoint:"

REPLAY_STUB_TEXT = (
    "[Input checkpoint: prior tool result omitted from replay to stay within the "
    "input ceiling. The source remains stored; repeat the preceding read-only tool "
    "call with the exact same tool name and arguments to recover it.]"
)

# fits(messages) -> (fits, estimated_tokens)
FitFn = Callable[[list[dict]], tuple[bool, int]]


def _is_checkpoint(text: str) -> bool:
    return text.lstrip().startswith(CHECKPOINT_PREFIX)


def excerpt_marker(original_len: int) -> str:
    return (
        f"\n[Input checkpoint: tool result shortened from {original_len} chars to the "
        f"first {EXCERPT_CHARS} to stay within the input ceiling. This tool is not "
        "safely repeatable, so the omitted part cannot be recovered by re-running it; "
        "rely on this excerpt and on findings already stated.]"
    )


def _excerpt(text: str) -> str:
    return text[:EXCERPT_CHARS] + excerpt_marker(len(text))


def _one_line(value, limit: int) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = " ".join(value.split())
    if len(value) > limit:
        return value[:limit] + "…"
    return value


def _block_text(content) -> str:
    """Coerce Anthropic block content (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "content" in block:
                    parts.append(_block_text(block.get("content")))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def build_digest(entries: list[dict], removed_rounds: int) -> str:
    """Render removed rounds as a structural digest.

    ``entries`` items are ``{"kind": "call", "name", "args", "result",
    "is_error"}`` or ``{"kind": "text", "text"}`` in original order.
    """
    lines = []
    omitted = max(0, len(entries) - DIGEST_MAX_ENTRIES)
    for entry in entries[omitted:]:
        if entry["kind"] == "call":
            status = "error" if entry.get("is_error") else "ok"
            result = entry.get("result") or ""
            if _is_checkpoint(result):
                result = "(already omitted earlier)"
            lines.append(
                f"- {entry['name']}({_one_line(entry.get('args', ''), DIGEST_ARG_CHARS)}) "
                f"→ {status}: {_one_line(result, DIGEST_RESULT_CHARS)}"
            )
        else:
            text = _one_line(entry.get("text", ""), DIGEST_TEXT_CHARS)
            if text:
                lines.append(f"- assistant said: {text}")
    header = (
        f"[Input checkpoint: {removed_rounds} earlier tool round(s) of this call were "
        "removed to stay within the input ceiling. Structural digest generated from the "
        "removed protocol blocks (not a model summary; excerpts only). Repeat a read-only "
        "call to recover its full output; side-effect calls listed here already ran and "
        "must not be repeated blindly."
    )
    if omitted:
        header += f" The oldest {omitted} entries are not listed."
    return header + "]\n" + "\n".join(lines)


# ── Anthropic message shape ─────────────────────────────────────────


def _anthropic_tool_names(messages: list[dict]) -> dict[str, str]:
    names = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names[str(block.get("id") or "")] = str(block.get("name") or "")
    return names


def _anthropic_has_tool_results(message: dict) -> bool:
    content = message.get("content")
    return message.get("role") == "user" and isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _anthropic_rewrite_results(messages, should_rewrite, rewrite, fits):
    """Rewrite tool_result bodies oldest-first until the request fits."""
    names = _anthropic_tool_names(messages)
    ok, estimate = fits(messages)
    for index, message in enumerate(messages):
        if ok:
            break
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        changed = False
        blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = _block_text(block.get("content"))
                name = names.get(str(block.get("tool_use_id") or ""))
                if (
                    len(text) > REPLAY_STUB_MIN_CHARS
                    and not _is_checkpoint(text)
                    and should_rewrite(index, name)
                ):
                    block = dict(block)
                    block["content"] = rewrite(text)
                    changed = True
            blocks.append(block)
        if changed:
            messages[index] = {**message, "content": blocks}
            ok, estimate = fits(messages)
    return ok, estimate


def _anthropic_digest_entries(segment: list[dict]) -> list[dict]:
    calls: dict[str, dict] = {}
    entries: list[dict] = []
    for message in segment:
        content = message.get("content")
        role = message.get("role")
        if role == "assistant":
            if isinstance(content, str):
                entries.append({"kind": "text", "text": content})
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    entries.append({"kind": "text", "text": block.get("text", "")})
                elif block.get("type") == "tool_use":
                    entry = {"kind": "call", "name": str(block.get("name") or "?"),
                             "args": block.get("input", {}), "result": "(no result)",
                             "is_error": False}
                    calls[str(block.get("id") or "")] = entry
                    entries.append(entry)
        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    entry = calls.get(str(block.get("tool_use_id") or ""))
                    if entry is not None:
                        entry["result"] = _block_text(block.get("content"))
                        entry["is_error"] = bool(block.get("is_error"))
    return entries


def _append_text(message: dict, text: str) -> dict:
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": list(content) + [{"type": "text", "text": text}]}
    base = content if isinstance(content, str) else _block_text(content)
    return {**message, "content": f"{base}\n\n{text}".strip()}


def _drop_oldest_rounds(messages, anchor, round_starts, entries_fn, fits):
    """Remove whole rounds after ``anchor`` until the request fits.

    ``round_starts`` are indexes of assistant messages that begin a round and
    directly follow a completed one. The newest KEEP_RECENT_ROUNDS stay.
    """
    ok, estimate = fits(messages)
    candidates = round_starts[1:len(round_starts) - KEEP_RECENT_ROUNDS + 1] \
        if len(round_starts) > KEEP_RECENT_ROUNDS else []
    for removed, cut in enumerate(candidates, start=1):
        if ok:
            break
        segment = messages[anchor + 1:cut]
        digest = build_digest(entries_fn(segment), removed)
        trial = messages[:anchor] + [_append_text(messages[anchor], digest)] + messages[cut:]
        ok, estimate = fits(trial)
        if ok:
            return trial, ok, estimate
    if not ok and candidates:
        # Nothing fit: return the most compacted form so the caller's ceiling
        # error reports the smallest achievable estimate.
        return trial, ok, estimate
    return messages, ok, estimate


def compact_anthropic_messages(
    messages: list[dict], fits: FitFn, *, is_replay_safe: Callable[[str | None], bool],
) -> tuple[list[dict], int]:
    """Apply stages A→C to an Anthropic-shape message list."""
    ok, estimate = fits(messages)
    if ok:
        return messages, estimate
    working = [dict(m) for m in messages]

    # A. replay-safe → replay stub (newest result last, by order).
    ok, estimate = _anthropic_rewrite_results(
        working, lambda _i, name: is_replay_safe(name), lambda _t: REPLAY_STUB_TEXT, fits,
    )
    if ok:
        return working, estimate

    # B. other large results → excerpt, except the newest round.
    result_indexes = [i for i, m in enumerate(working) if _anthropic_has_tool_results(m)]
    newest = result_indexes[-1] if result_indexes else -1
    ok, estimate = _anthropic_rewrite_results(
        working, lambda i, name: i != newest and not is_replay_safe(name), _excerpt, fits,
    )
    if ok:
        return working, estimate

    # C. drop oldest rounds behind the first user message.
    if not working or working[0].get("role") != "user":
        return working, estimate
    round_starts = [
        i for i in range(1, len(working))
        if working[i].get("role") == "assistant" and working[i - 1].get("role") == "user"
    ]
    working, ok, estimate = _drop_oldest_rounds(
        working, 0, round_starts, _anthropic_digest_entries, fits,
    )
    return working, estimate


# ── OpenAI message shape ────────────────────────────────────────────


def _openai_tool_names(messages: list[dict]) -> dict[str, str]:
    names = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                names[str(call.get("id") or "")] = str(function.get("name") or "")
    return names


def _openai_rewrite_results(messages, should_rewrite, rewrite, fits):
    names = _openai_tool_names(messages)
    ok, estimate = fits(messages)
    for index, message in enumerate(messages):
        if ok:
            break
        if message.get("role") != "tool":
            continue
        text = str(message.get("content", ""))
        name = names.get(str(message.get("tool_call_id") or ""))
        if len(text) > REPLAY_STUB_MIN_CHARS and not _is_checkpoint(text) and should_rewrite(index, name):
            messages[index] = {**message, "content": rewrite(text)}
            ok, estimate = fits(messages)
    return ok, estimate


def _openai_digest_entries(segment: list[dict]) -> list[dict]:
    calls: dict[str, dict] = {}
    entries: list[dict] = []
    for message in segment:
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if content:
                entries.append({"kind": "text", "text": _block_text(content)})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                entry = {"kind": "call", "name": str(function.get("name") or "?"),
                         "args": function.get("arguments", ""), "result": "(no result)",
                         "is_error": False}
                calls[str(call.get("id") or "")] = entry
                entries.append(entry)
        elif role == "tool":
            entry = calls.get(str(message.get("tool_call_id") or ""))
            if entry is not None:
                entry["result"] = str(message.get("content", ""))
    return entries


def compact_openai_messages(
    messages: list[dict], fits: FitFn, *, is_replay_safe: Callable[[str | None], bool],
) -> tuple[list[dict], int]:
    """Apply stages A→C to an OpenAI-shape message list."""
    ok, estimate = fits(messages)
    if ok:
        return messages, estimate
    working = [dict(m) for m in messages]

    ok, estimate = _openai_rewrite_results(
        working, lambda _i, name: is_replay_safe(name), lambda _t: REPLAY_STUB_TEXT, fits,
    )
    if ok:
        return working, estimate

    # Newest round = tool messages after the last assistant carrying tool_calls.
    last_call_index = max(
        (i for i, m in enumerate(working) if m.get("role") == "assistant" and m.get("tool_calls")),
        default=-1,
    )
    ok, estimate = _openai_rewrite_results(
        working,
        lambda i, name: i < last_call_index and not is_replay_safe(name),
        _excerpt,
        fits,
    )
    if ok:
        return working, estimate

    anchor = next((i for i, m in enumerate(working) if m.get("role") == "user"), None)
    if anchor is None:
        return working, estimate
    round_starts = [
        i for i in range(anchor + 1, len(working))
        if working[i].get("role") == "assistant"
        and working[i - 1].get("role") in ("user", "tool")
    ]
    working, ok, estimate = _drop_oldest_rounds(
        working, anchor, round_starts, _openai_digest_entries, fits,
    )
    return working, estimate
