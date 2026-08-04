"""DeepSeek 장애 시 2차 공급자로 넘기는 턴 단위 페일오버.

DeepSeek은 Anthropic 호환 엔드포인트(claude_loop)를, GPT-5.6은 OpenAI
프로토콜(openai_tool_loop)을 쓴다. 두 루프는 메시지·툴 표현이 서로 달라서
진행 중인 루프를 중간에 옮길 수 없다 — kimi→deepseek 폴백이 openai_tool_loop
안에서만 움직이는 것과 다른 점이다.

그래서 여기서는 **턴 전체를 다시 돌린다**. 두 루프 모두 같은 정규 history를
입력으로 받으므로, 실패한 부분 실행의 툴 결과만 버리고 처음부터 다시 시작한다.
이미 실행된 툴의 부수효과는 되돌리지 않는다 (되돌릴 수 없다) — 재실행되는
쪽이 같은 툴을 다시 부를 수 있다는 뜻이다. 쓰기 툴이 멱등하지 않은 경로에는
쓰지 말 것.

claude_loop는 transient 오류를 이미 3회 재시도한다. 여기까지 예외가 올라왔다면
그 3회가 전부 실패한 것이다.
"""

from __future__ import annotations

import logging

# 같은 판정을 두 벌 유지하면 어긋난다 — 두 루프가 공유하는 술어를 그대로
# 재사용한다 (429/5xx/타임아웃/커넥션 계열).
from tool_loop_common import emit_progress, is_transient_provider_error

logger = logging.getLogger(__name__)


async def resolve_kimi_fallback_options(runtime_kind: str, deepseek_client) -> dict:
    """Kimi 콘텐츠필터 폴백용 kimi_openai_tool_options kwargs를 해상.

    폴백 모델은 DeepSeek high 티어. deepseek_client가 없으면 폴백 없이
    Kimi 단독 옵션을 돌려준다. bot/web_chat/a2a에 복붙돼 있던 4줄 통합.
    """
    from runtime_profile import resolve_runtime_profile
    from llm.provider_registry import kimi_openai_tool_options

    fallback_model = None
    if deepseek_client:
        profile = await resolve_runtime_profile(
            runtime_kind, provider_override="deepseek", tier_override="high",
        )
        fallback_model = profile.model_id
    return kimi_openai_tool_options(
        fallback_client=deepseek_client,
        fallback_model=fallback_model,
    )


async def resolve_deepseek_failover_model(runtime_kind: str, openai_client) -> str | None:
    """DeepSeek 장애 시 턴을 다시 돌릴 2차 모델 (OpenAI medium 티어 = Terra).

    openai 클라이언트가 없거나 해상 실패면 None — 페일오버 없이 원래 오류가
    그대로 올라간다.
    """
    if not openai_client:
        return None
    from runtime_profile import resolve_runtime_profile
    try:
        profile = await resolve_runtime_profile(
            runtime_kind, provider_override="openai", tier_override="medium",
        )
        return profile.model_id
    except Exception as exc:
        logger.warning("DeepSeek failover model unresolved; failover disabled: %s", exc)
        return None


async def run_with_provider_failover(
    primary,
    fallback,
    *,
    primary_label: str,
    fallback_label: str,
    budget_tracker: dict | None = None,
    on_progress=None,
):
    """primary()를 돌리고, 공급자 장애면 fallback()으로 턴을 다시 돈다.

    primary/fallback은 코루틴을 반환하는 **thunk**다. fallback 코루틴은 실제로
    필요할 때만 만들어진다.

    fallback이 None이거나 오류가 공급자 장애 계열이 아니면 그대로 올려보낸다 —
    설정 오류·예산 초과·툴 실패까지 2차 공급자로 넘겨서 돈을 두 번 쓰면 안 된다.
    """
    try:
        return await primary()
    except Exception as err:
        if fallback is None or not is_transient_provider_error(err):
            raise

        # 실패한 1차 실행이 이미 쓴 비용. budget_tracker는 라운드마다
        # .update()로 덮어써지는 out-param이라, 그대로 두면 2차 실행 결과가
        # 1차 비용을 지워버린다.
        carried_cost = 0.0
        if budget_tracker is not None:
            try:
                carried_cost = float(budget_tracker.get("total_cost") or 0.0)
            except (TypeError, ValueError):
                carried_cost = 0.0

        logger.warning(
            "Provider failover: %s unavailable (%s: %s); retrying whole turn on %s"
            " (discarding partial run, carrying $%.4f already spent)",
            primary_label, type(err).__name__, err, fallback_label, carried_cost,
        )
        await emit_progress(
            on_progress, "provider_failover",
            f"{primary_label} 응답 없음 — {fallback_label}로 턴을 다시 시작합니다.",
        )

        result = await fallback()

        if budget_tracker is not None and carried_cost:
            try:
                budget_tracker["total_cost"] = (
                    float(budget_tracker.get("total_cost") or 0.0) + carried_cost
                )
            except (TypeError, ValueError):
                pass
        return result
