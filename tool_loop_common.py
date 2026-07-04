"""tool_loop_common.py — Shared utilities for claude_loop.py and openai_tool_loop.py.

Centralizes duplicated budget, progress, Redis state, and limit-message helpers.
Runtime tool execution now lives in tool_gateway.dispatcher and is re-exported
here for compatibility with older smoke tests and helper scripts.
"""

import logging

logger = logging.getLogger(__name__)

from tool_gateway.dispatcher import compact_tool_definitions


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


# ── Tool execution compatibility exports ─────────────────────────────

# Runtime tool dispatch now lives in tool_gateway.dispatcher. Re-export these
# names for older smoke tests and helper scripts that still import from this
# shared utility module.
from tool_gateway.dispatcher import (  # noqa: E402
    PARALLEL_SAFE_TOOLS,
    execute_tool,
    execute_tools_batch,
)

# ── System messages (forced final response, warnings) ────────────────

def build_limit_message(
    limit_reason: str, total_cost: float, budget_usd: float,
    round_num: int, max_rounds: int, was_still_working: bool,
    finalization_tools: list[str] | None = None,
) -> str:
    """Build the [SYSTEM] message injected when budget/round limit is reached.

    If finalization_tools is provided, instruct the agent to call those tools
    to persist its work (they remain available in the forced-final call).
    """
    escalation_hint = ""
    if was_still_working:
        escalation_hint = (
            " 미완료 작업이 있으면 수행한 것, 못한 것, 다음에 해야 할 것을 명시하라. "
            "orchestrator가 재위임 여부를 판단한다."
        )
    if finalization_tools:
        names = ", ".join(finalization_tools)
        return (
            f"[SYSTEM] {limit_reason} (비용: ${total_cost:.3f}/${budget_usd:.2f}, "
            f"라운드: {round_num}/{max_rounds}). "
            f"다른 도구는 사용할 수 없다. 그러나 마감 도구({names})는 "
            "아직 호출할 수 있다. 지금까지 준비한 결과를 반드시 해당 도구로 저장한 뒤, "
            "수행한 작업과 수집한 데이터를 있는 그대로 정리하라."
            + escalation_hint
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
