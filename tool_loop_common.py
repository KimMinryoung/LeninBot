"""tool_loop_common.py — Shared utilities for claude_loop.py and openai_tool_loop.py.

Centralizes duplicated logic so improvements to tool execution, budget tracking,
progress events, and Redis state management only need to be made in one place.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


# ── Budget ───────────────────────────────────────────────────────────

def validate_budget(budget_usd) -> float:
    """Validate and clamp budget_usd to a safe positive float."""
    try:
        budget_usd = float(budget_usd)
    except (TypeError, ValueError):
        logger.warning("Invalid budget_usd=%r; falling back to 0.30", budget_usd)
        budget_usd = 0.30
    if budget_usd <= 0:
        logger.warning("Non-positive budget_usd=%s; clamping to 0.01", budget_usd)
        budget_usd = 0.01
    return budget_usd


def build_budget_tracker(
    total_cost: float, rounds_used: int,
    was_interrupted: bool, tool_work_details: list,
) -> dict:
    """Build the budget_tracker dict returned to callers."""
    return {
        "total_cost": total_cost,
        "rounds_used": rounds_used,
        "was_interrupted": was_interrupted,
        "tool_work_details": list(tool_work_details),
    }


# ── Progress events ──────────────────────────────────────────────────

async def emit_progress(on_progress, event: str, detail: str):
    """Null-safe async progress callback wrapper."""
    if on_progress is None:
        return
    try:
        await on_progress(event, detail)
    except Exception:
        pass


# ── Task cancellation ─────────────────────────────────────────────────

class TaskCancelledError(Exception):
    """Raised when a task is cancelled via Redis signal."""
    pass


def check_cancelled(task_id: int | None):
    """Check if a task has been cancelled. Raises TaskCancelledError if so."""
    if task_id is None:
        return
    try:
        from redis_state import get_redis
        r = get_redis()
        if r.exists(f"task:{task_id}:cancel"):
            r.delete(f"task:{task_id}:cancel")
            raise TaskCancelledError(f"Task #{task_id} cancelled by user")
    except TaskCancelledError:
        raise
    except Exception:
        pass


def request_cancel(task_id: int):
    """Set a cancel flag in Redis for a running task."""
    try:
        from redis_state import get_redis
        r = get_redis()
        r.set(f"task:{task_id}:cancel", "1", ex=300)  # 5 min TTL
    except Exception:
        pass


# ── Redis state ──────────────────────────────────────────────────────

def update_redis_state(task_id: int | None, round_num: int, total_cost: float):
    """Update live task state in Redis (non-fatal on failure)."""
    if task_id is None:
        return
    try:
        from redis_state import set_task_state
        set_task_state(task_id, round_num, total_cost, status="running")
    except Exception:
        pass


def save_redis_progress(
    task_id: int | None, round_num: int,
    tool_name: str, input_summary: str, result: str, is_error: bool,
):
    """Persist tool execution to Redis incrementally (survives process death)."""
    if task_id is None:
        return
    try:
        from redis_state import save_task_progress
        save_task_progress(task_id, round_num, tool_name, input_summary, result, is_error)
    except Exception:
        pass


# ── Tool execution ───────────────────────────────────────────────────

async def execute_tool(
    name: str, args: dict, handlers: dict, *, log_event=None,
) -> tuple[str, bool]:
    """Execute a tool handler by name. Returns (result_str, is_error).

    Handles sync/async dispatch, error catching, str guard, and 50K truncation.
    """
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}", True

    logger.info("Tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])
    try:
        raw = handler(**args)
        if asyncio.iscoroutine(raw) or asyncio.isfuture(raw):
            result = await raw
        else:
            result = raw
        is_error = False
    except Exception as e:
        logger.error("Tool %s execution error: %s", name, e)
        if log_event:
            log_event("warning", "tool", f"Tool {name} failed: {e}")
        result = f"Tool execution failed: {e}"
        is_error = True

    # Guard: ensure result is a non-None string
    if not isinstance(result, str):
        result = str(result) if result is not None else "(no result)"
    # Truncate oversized results to avoid context overflow
    if len(result) > 50000:
        result = result[:50000] + "\n... [truncated]"

    return result, is_error


# ── Parallel batch execution ─────────────────────────────────────────

# Tools that are safe to run concurrently with each other within a single
# round. They must be:
#   * Read-only (no disk/DB/network writes that could race)
#   * Idempotent (multiple calls produce the same result)
#   * Free of order-dependent side effects on other tools in the same batch
#
# Anything not on this list is run sequentially in original emission order.
# Tools that download files, execute code, send messages, modify files,
# spawn agents, or perform financial transactions are intentionally excluded.
PARALLEL_SAFE_TOOLS = frozenset({
    "fetch_url",
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


async def execute_tools_batch(
    tool_uses: list[tuple[str, str, dict]],
    handlers: dict,
    *,
    on_progress=None,
    round_num: int,
    log_event=None,
    parallel_safe: frozenset = PARALLEL_SAFE_TOOLS,
) -> list[tuple[str, str, dict, str, bool]]:
    """Execute a sequence of (id, name, input) tool calls.

    Consecutive parallel-safe tools are run concurrently via ``asyncio.gather``;
    any tool not on the safe-list breaks the batch and runs sequentially in its
    original position. Results are returned in input order so the caller can
    build tool_result blocks / messages without re-sorting.
    """
    n = len(tool_uses)
    results: list = [None] * n

    async def _run_one(idx: int, tid: str, tname: str, tinput: dict):
        input_summary = json.dumps(tinput, ensure_ascii=False)
        await emit_progress(on_progress, "tool_call",
                            f"[{round_num}] 🔧 {tname}({input_summary})")
        result, is_error = await execute_tool(tname, tinput, handlers, log_event=log_event)
        await emit_progress(on_progress, "tool_result",
                            f"  {'❌' if is_error else '✓'} {result[:200]}")
        results[idx] = (tid, tname, tinput, result, is_error)

    i = 0
    while i < n:
        tid_i, tname_i, tinput_i = tool_uses[i]
        if tname_i in parallel_safe:
            # Greedily collect the longest run of consecutive parallel-safe tools.
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


# ── System messages (forced final response, warnings) ────────────────

def build_limit_message(
    limit_reason: str, total_cost: float, budget_usd: float,
    round_num: int, max_rounds: int, was_still_working: bool,
) -> str:
    """Build the [SYSTEM] message injected when budget/round limit is reached."""
    escalation_hint = ""
    if was_still_working:
        escalation_hint = (
            " 미완료 작업이 있으면 수행한 것, 못한 것, 다음에 해야 할 것을 명시하라. "
            "orchestrator가 재위임 여부를 판단한다."
        )
    return (
        f"[SYSTEM] {limit_reason} (비용: ${total_cost:.3f}/${budget_usd:.2f}, "
        f"라운드: {round_num}/{max_rounds}). "
        "추가 도구를 사용하지 말고, 지금까지 수행한 작업과 수집한 데이터를 있는 그대로 정리하라. "
        "보고서 형식이 아니어도 된다. 시행착오, 중간 결과, raw 데이터 모두 포함하라."
        + escalation_hint
    )


def build_budget_warning(total_cost: float, budget_usd: float) -> str:
    """Build the 80% budget warning message."""
    return (
        f"[SYSTEM] 예산 80% 소진 (${total_cost:.3f}/${budget_usd:.2f}). "
        "작업을 계속하라. 한도 도달 시 시스템이 자동 종료한다."
    )


def build_round_warning(round_num: int, max_rounds: int) -> str:
    """Build the round limit warning message."""
    return (
        f"[SYSTEM] 라운드 한도 임박 ({round_num}/{max_rounds}). "
        "다음 라운드가 마지막이다. 파일 저장 등 최종 도구 호출을 지금 하라."
    )


def build_stripped_limit_message(limit_reason: str) -> str:
    """Short limit message used after stripping tool protocol."""
    return f"[SYSTEM] {limit_reason}. 지금까지 수집한 정보만으로 답변하세요."


# ── Fallback text ────────────────────────────────────────────────────

EMPTY_RESPONSE_FALLBACK = "응답을 생성하지 못했습니다."
