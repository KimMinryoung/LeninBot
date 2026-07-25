"""Runtime tool dispatch implementation.

Provider loops call this module for schema compaction, single tool execution,
and batch execution. Security authorization and audit remain enforced per call
through ``security_gateway``.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import time
from typing import Any

from tool_gateway.validation import (
    ToolArgumentValidationError,
    tool_schema_map,
    validate_tool_arguments,
)

logger = logging.getLogger(__name__)

_TOOL_DESC_LIMIT = 360
_SCHEMA_DESC_LIMIT = 160

SIDE_EFFECT_RISK_CLASSES = frozenset({
    "admin",
    "browser",
    "delegate",
    "execute",
    "file_write",
    "media",
    "pay",
    "publish",
    "send",
    "state",
    "write",
})


def is_side_effect_tool(name: str, risk_class: str | None = None) -> bool:
    """Return whether a tool may mutate state or cause an external effect.

    If the risk registry itself is unavailable, classify the tool as
    side-effecting. Recovery and idempotency must degrade toward safety.
    """
    if risk_class is None:
        try:
            from security_gateway.policy import risk_class as _risk_class

            risk_class = _risk_class(name)
        except Exception:
            return True
    return risk_class in SIDE_EFFECT_RISK_CLASSES


def _idempotency_key(name: str, args: dict) -> str:
    payload = json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{name}:{payload}"


def _compact_text(text: str, limit: int) -> str:
    """Collapse long tool/schema descriptions while preserving their meaning."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)].rstrip()
    split_at = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
    if split_at >= int(limit * 0.55):
        cut = cut[: split_at + 1].rstrip()
    return cut + "…"


def _compact_schema_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "description" and isinstance(item, str):
                out[key] = _compact_text(item, _SCHEMA_DESC_LIMIT)
            else:
                out[key] = _compact_schema_descriptions(item)
        return out
    if isinstance(value, list):
        return [_compact_schema_descriptions(item) for item in value]
    return value


def compact_tool_definitions(tools: list[dict]) -> list[dict]:
    """Return provider payload tool definitions with compact descriptions.

    Tool names, schema keys, types, enums, defaults, and required lists are left
    intact. Only human-readable description strings are shortened to reduce
    prompt overhead from large tool surfaces.
    """
    compacted: list[dict] = []
    for tool in tools or []:
        t = copy.deepcopy(tool)
        if isinstance(t.get("description"), str):
            t["description"] = _compact_text(t["description"], _TOOL_DESC_LIMIT)
        if isinstance(t.get("input_schema"), dict):
            t["input_schema"] = _compact_schema_descriptions(t["input_schema"])
        fn = t.get("function")
        if isinstance(fn, dict):
            if isinstance(fn.get("description"), str):
                fn["description"] = _compact_text(fn["description"], _TOOL_DESC_LIMIT)
            if isinstance(fn.get("parameters"), dict):
                fn["parameters"] = _compact_schema_descriptions(fn["parameters"])
        compacted.append(t)
    return compacted


def _record_tool_provenance(name: str, args: dict, result: str) -> None:
    """Record external-source tool calls and KG reads into the per-run buffer."""
    try:
        from provenance.runtime import get_provenance_buffer
        buf = get_provenance_buffer()
        if buf is None:
            return
        if name == "fetch_url":
            buf.record_external(name, f"url:{args.get('url', '')}")
        elif name == "web_search":
            buf.record_external(name, f"web_search:{args.get('query', '')}")
        elif name == "convert_document":
            buf.record_external(name, f"document:{args.get('file_path', '')}")
        elif name == "check_inbox":
            buf.record_external(name, "imap_inbox")
        elif name in ("read_file", "search_files"):
            path = args.get("path") or ""
            if "data/downloads" in path or "data/converted" in path:
                buf.record_external(name, f"file:{path}")
        elif name == "knowledge_graph_search":
            buf.record_kg_read(result)
    except Exception:
        pass


_HANDLER_KWARGS_CACHE: dict[int, tuple[set[str] | None, frozenset[str]]] = {}


def _inspect_handler_kwargs(handler: Any) -> tuple[set[str] | None, frozenset[str]]:
    """Return (accepted_names, required_names) for a handler."""
    target = handler
    if not inspect.isfunction(target) and not inspect.ismethod(target):
        target = getattr(handler, "__wrapped__", handler)

    # Cache by the code object, not the handler instance: per-request closures
    # (e.g. writer tools) are freed and their id() reused by other handlers,
    # which poisoned an id(handler)-keyed cache with the wrong signature. Code
    # objects are module-lifetime constants shared by every closure instance.
    code = getattr(target, "__code__", None)
    cache_key = id(code) if code is not None else None
    if cache_key is not None:
        hit = _HANDLER_KWARGS_CACHE.get(cache_key)
        if hit is not None:
            return hit

    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        result: tuple[set[str] | None, frozenset[str]] = (None, frozenset())
        if cache_key is not None:
            _HANDLER_KWARGS_CACHE[cache_key] = result
        return result

    accepts_var_kw = False
    names: set[str] = set()
    required: set[str] = set()
    for pname, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(pname)
            if p.default is inspect.Parameter.empty:
                required.add(pname)

    result = (None if accepts_var_kw else names, frozenset(required))
    if cache_key is not None:
        _HANDLER_KWARGS_CACHE[cache_key] = result
    return result


async def execute_tool(
    name: str,
    args: dict,
    handlers: dict,
    *,
    log_event=None,
    idempotency_cache: dict[str, tuple[str, bool]] | None = None,
    tool_schema: dict | None = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    """Validate, authorize, deduplicate, and execute one tool handler."""
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}", True

    args = dict(args or {})
    started = time.perf_counter()
    gw_ctx = None
    gw_decision = None
    try:
        from tool_gateway.security import audit as _gw_audit
        from tool_gateway.security import authorize as _gw_authorize
        from tool_gateway.security import get_caller as _gw_caller

        gw_ctx = _gw_caller()
        gw_decision = _gw_authorize(
            gw_ctx,
            name,
            args,
            consume_rate_limit=False,
        )
    except Exception as exc:
        msg = f"Tool '{name}' blocked: security gateway pre-check failed: {exc}"
        logger.error(msg)
        if log_event:
            log_event("error", "tool", msg)
        return msg, True

    if gw_decision.denied:
        msg = (
            f"Tool '{name}' denied by security gateway "
            f"(caller={gw_ctx.label()}): {gw_decision.reason}"
        )
        logger.warning(msg)
        if log_event:
            log_event("warning", "tool", msg)
        try:
            _gw_audit(gw_ctx, name, args, gw_decision, result_status="denied", latency_ms=0)
        except Exception as exc:
            logger.warning("gateway denial audit failed (ignored) for %s: %s", name, exc)
        return msg, True

    try:
        args = validate_tool_arguments(
            name,
            args,
            schema=tool_schema,
            risk_class=gw_decision.risk_class,
        )
        accepted, required = _inspect_handler_kwargs(handler)
        if accepted is not None:
            unknown = sorted(key for key in args if key not in accepted)
            if unknown:
                raise ToolArgumentValidationError(
                    f"unknown arguments {unknown}; accepted: {sorted(accepted)}"
                )
        # A **kwargs handler cannot say which names it rejects, but its required
        # params are still knowable. This check used to sit inside the branch
        # above, so an empty payload sailed past it and died on a raw TypeError
        # ("missing 3 required positional arguments") that the model could not act
        # on — the call was simply lost.
        missing = sorted(key for key in required if key not in args)
        if missing:
            hint = f"; accepted: {sorted(accepted)}" if accepted is not None else ""
            raise ToolArgumentValidationError(
                f"missing required arguments {missing}{hint}"
            )
    except ToolArgumentValidationError as exc:
        msg = f"Tool '{name}' arguments rejected: {exc}"
        logger.warning(msg)
        if log_event:
            log_event("warning", "tool", msg)
        try:
            _gw_audit(
                gw_ctx,
                name,
                args,
                gw_decision,
                result_status="invalid_args",
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_excerpt=str(exc),
            )
        except Exception as audit_exc:
            logger.warning("gateway validation audit failed for %s: %s", name, audit_exc)
        return msg, True

    side_effect = is_side_effect_tool(name, gw_decision.risk_class)
    cache_key = _idempotency_key(name, args) if side_effect else None
    if cache_key is not None and idempotency_cache is not None:
        cached = idempotency_cache.get(cache_key)
        if cached is not None:
            result, is_error = cached
            logger.warning("Suppressed duplicate side-effect tool call: %s", name)
            try:
                _gw_audit(
                    gw_ctx,
                    name,
                    args,
                    gw_decision,
                    result_status="deduplicated",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                logger.warning("gateway dedupe audit failed for %s: %s", name, exc)
            return result, is_error

    durable_scope = None
    durable_record = None
    stable_scope_expected = bool(
        getattr(gw_ctx, "task_id", None) or getattr(gw_ctx, "session_id", None)
    )
    if side_effect:
        try:
            from security_gateway.idempotency import lookup, scope_for_context

            durable_scope = scope_for_context(gw_ctx)
            if durable_scope:
                durable_record = await asyncio.to_thread(
                    lookup,
                    durable_scope,
                    name,
                    args,
                )
        except Exception as exc:
            if stable_scope_expected:
                msg = f"Tool '{name}' blocked: durable idempotency store unavailable: {exc}"
                logger.error(msg)
                try:
                    _gw_audit(
                        gw_ctx,
                        name,
                        args,
                        gw_decision,
                        result_status="idempotency_unavailable",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        error_excerpt=str(exc),
                    )
                except Exception:
                    pass
                return msg, True

    if durable_record is not None:
        if durable_record.status == "succeeded":
            result = durable_record.result_text or "(previous call succeeded)"
            status = "deduplicated_durable"
            is_error = False
        else:
            result = (
                f"Tool '{name}' was not retried because an earlier scoped call is "
                f"{durable_record.status}. Its external outcome may be unknown; "
                "verify the target system before any manual retry."
            )
            status = "outcome_unknown"
            is_error = True
        try:
            _gw_audit(
                gw_ctx,
                name,
                args,
                gw_decision,
                result_status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_excerpt=result if is_error else None,
            )
        except Exception as exc:
            logger.warning("gateway durable dedupe audit failed for %s: %s", name, exc)
        return result, is_error

    try:
        gw_decision = _gw_authorize(gw_ctx, name, args, consume_rate_limit=True)
    except Exception as exc:
        msg = f"Tool '{name}' blocked: security gateway rate pre-check failed: {exc}"
        logger.error(msg)
        return msg, True
    if gw_decision.denied:
        msg = (
            f"Tool '{name}' denied by security gateway "
            f"(caller={gw_ctx.label()}): {gw_decision.reason}"
        )
        try:
            _gw_audit(
                gw_ctx,
                name,
                args,
                gw_decision,
                result_status="denied",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            logger.warning("gateway denial audit failed for %s: %s", name, exc)
        return msg, True

    if side_effect and durable_scope:
        try:
            from security_gateway.idempotency import reserve

            durable_record = await asyncio.to_thread(
                reserve,
                durable_scope,
                name,
                args,
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            msg = f"Tool '{name}' blocked: durable idempotency reservation failed: {exc}"
            logger.error(msg)
            return msg, True
        if not durable_record.acquired:
            if durable_record.status == "succeeded":
                result = durable_record.result_text or "(previous call succeeded)"
                is_error = False
                status = "deduplicated_durable"
            else:
                result = (
                    f"Tool '{name}' was not retried because another scoped call is "
                    f"{durable_record.status}; verify the external target before retrying."
                )
                is_error = True
                status = "outcome_unknown"
            try:
                _gw_audit(
                    gw_ctx, name, args, gw_decision, result_status=status,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_excerpt=result if is_error else None,
                )
            except Exception as exc:
                logger.warning("gateway reservation conflict audit failed for %s: %s", name, exc)
            return result, is_error

    logger.info("Tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])
    try:
        raw = handler(**args)
        if asyncio.iscoroutine(raw) or asyncio.isfuture(raw):
            result = await raw
        else:
            result = raw
        is_error = False
    except Exception as exc:
        logger.error("Tool %s execution error: %s", name, exc)
        if log_event:
            log_event("warning", "tool", f"Tool {name} failed: {exc}")
        result = f"Tool execution failed; external outcome may be unknown: {exc}"
        is_error = True
        if durable_record is not None and durable_record.acquired:
            try:
                from security_gateway.idempotency import mark_outcome_unknown

                await asyncio.to_thread(mark_outcome_unknown, durable_record, result)
            except Exception as store_exc:
                logger.error("failed to persist outcome_unknown for %s: %s", name, store_exc)

    if not isinstance(result, str):
        result = str(result) if result is not None else "(no result)"

    read_self_type = args.get("content_type") or args.get("source")
    if read_self_type == "static_pages":
        read_self_type = "static_page"
    allow_full_content_read = (
        name == "read_self"
        and read_self_type in {"diary", "static_page"}
        and args.get("max_chars") is None
    )
    allow_complete_chat_turns = name == "read_self" and read_self_type == "chat_logs"
    paginated_large_result = name in {"fetch_url", "read_file", "read_document", "read_self"}
    result_cap = 120000 if paginated_large_result else 50000
    if len(result) > result_cap and not (allow_full_content_read or allow_complete_chat_turns):
        result = result[:result_cap] + f"\n... [truncated by tool loop at {result_cap} chars]"

    if not is_error:
        _record_tool_provenance(name, args, result)
        if cache_key is not None and idempotency_cache is not None:
            idempotency_cache[cache_key] = (result, False)
        if durable_record is not None and durable_record.acquired:
            try:
                from security_gateway.idempotency import mark_succeeded

                await asyncio.to_thread(mark_succeeded, durable_record, result)
            except Exception as exc:
                logger.error("failed to persist idempotency success for %s: %s", name, exc)

    try:
        if gw_ctx is not None:
            _gw_audit(
                gw_ctx,
                name,
                args,
                gw_decision,
                result_status=(
                    "outcome_unknown"
                    if is_error and durable_record is not None
                    else "error" if is_error else "ok"
                ),
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_excerpt=result if is_error else None,
            )
    except Exception as exc:
        logger.warning("gateway audit failed (ignored) for %s: %s", name, exc)

    return result, is_error


PARALLEL_SAFE_TOOLS = frozenset({
    "fetch_url",
    "fetch_x_post",
    "web_search",
    "vector_search",
    "knowledge_graph_search",
    "read_file",
    "search_files",
    "list_directory",
    "convert_document",
    "get_finance_data",
    "check_wallet",
    "recall_experience",
    "read_self",
})


async def _emit_progress(on_progress, event: str, detail: str) -> None:
    if on_progress is None:
        return
    try:
        await on_progress(event, detail)
    except Exception:
        pass


async def execute_tools_batch(
    tool_uses: list[tuple[str, str, dict]],
    handlers: dict,
    *,
    on_progress=None,
    round_num: int,
    log_event=None,
    idempotency_cache: dict[str, tuple[str, bool]] | None = None,
    tool_definitions: list[dict] | None = None,
    parallel_safe: frozenset = PARALLEL_SAFE_TOOLS,
) -> list[tuple[str, str, dict, str, bool]]:
    """Execute a sequence of (id, name, input) tool calls."""
    n = len(tool_uses)
    results: list = [None] * n
    schemas = tool_schema_map(tool_definitions)

    async def _run_one(idx: int, tid: str, tname: str, tinput: dict):
        input_summary = json.dumps(tinput, ensure_ascii=False)
        await _emit_progress(on_progress, "tool_call", f"[{round_num}] 🔧 {tname}({input_summary})")
        result, is_error = await execute_tool(
            tname,
            tinput,
            handlers,
            log_event=log_event,
            idempotency_cache=idempotency_cache,
            tool_schema=schemas.get(tname),
            tool_call_id=tid,
        )
        await _emit_progress(on_progress, "tool_result", f"  {'❌' if is_error else '✓'} {tname}: {result[:200]}")
        results[idx] = (tid, tname, tinput, result, is_error)

    i = 0
    while i < n:
        tid_i, tname_i, tinput_i = tool_uses[i]
        if tname_i in parallel_safe:
            j = i
            tasks = []
            while j < n and tool_uses[j][1] in parallel_safe:
                tid_j, tname_j, tinput_j = tool_uses[j]
                tasks.append(_run_one(j, tid_j, tname_j, tinput_j))
                j += 1
            await asyncio.gather(*tasks)
            i = j
        else:
            await _run_one(i, tid_i, tname_i, tinput_i)
            i += 1

    return results  # type: ignore[return-value]


__all__ = [
    "PARALLEL_SAFE_TOOLS",
    "SIDE_EFFECT_RISK_CLASSES",
    "compact_tool_definitions",
    "execute_tool",
    "execute_tools_batch",
    "is_side_effect_tool",
]
