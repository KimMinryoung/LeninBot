"""공급자 페일오버(llm/provider_failover.py) 스모크.

네트워크를 타지 않는다 — thunk와 예외를 직접 넣어 분기와 예산 이월만 확인한다.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from llm.provider_failover import run_with_provider_failover  # noqa: E402


class APIConnectionError(Exception):
    """클래스명 토큰으로 transient 판정되는 케이스 (커넥션 계열)."""


class ServerError(Exception):
    """status_code로 transient 판정되는 케이스."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _thunk(result=None, raises=None, calls=None, name=""):
    async def _run():
        if calls is not None:
            calls.append(name)
        if raises is not None:
            raise raises
        return result

    return _run


async def case(label, *, primary_raises=None, fallback=True, tracker=None,
               primary_cost=None, fallback_cost=None):
    """한 케이스를 돌리고 (결과, 호출된 thunk 목록, 예외)를 돌려준다."""
    calls = []

    async def _primary():
        calls.append("primary")
        if primary_cost is not None and tracker is not None:
            tracker["total_cost"] = primary_cost
        if primary_raises is not None:
            raise primary_raises
        return "primary-result"

    async def _fallback():
        calls.append("fallback")
        if fallback_cost is not None and tracker is not None:
            tracker["total_cost"] = fallback_cost
        return "fallback-result"

    try:
        out = await run_with_provider_failover(
            _primary,
            _fallback if fallback else None,
            primary_label="deepseek",
            fallback_label="gpt-5.6-terra",
            budget_tracker=tracker,
        )
        err = None
    except Exception as e:
        out, err = None, e
    print(f"[{label}]")
    print(f"  calls   = {calls}")
    print(f"  result  = {out!r}")
    print(f"  raised  = {type(err).__name__ if err else None}")
    if tracker is not None:
        print(f"  tracker = {tracker}")
    return calls, out, err, tracker


async def main() -> int:
    failures = []

    def check(label, cond, detail):
        print(f"  {'OK  ' if cond else 'FAIL'} {detail}")
        if not cond:
            failures.append(f"{label}: {detail}")

    calls, out, err, _ = await case("1. 정상 — 폴백 안 탐")
    check("1", calls == ["primary"], "primary만 호출")
    check("1", out == "primary-result", "1차 결과 반환")

    calls, out, err, _ = await case(
        "2. 커넥션 오류 — 폴백 탐", primary_raises=APIConnectionError("boom"))
    check("2", calls == ["primary", "fallback"], "primary→fallback 순서로 호출")
    check("2", out == "fallback-result", "폴백 결과 반환")

    calls, out, err, _ = await case(
        "3. 503 — 폴백 탐", primary_raises=ServerError(503))
    check("3", calls == ["primary", "fallback"], "status_code로도 판정됨")

    calls, out, err, _ = await case(
        "4. 400 — 폴백 안 탐 (공급자 장애 아님)", primary_raises=ServerError(400))
    check("4", calls == ["primary"], "폴백 호출 안 함")
    check("4", isinstance(err, ServerError), "원래 예외 그대로 전파")

    calls, out, err, _ = await case(
        "5. 설정 오류 — 폴백 안 탐", primary_raises=ValueError("bad config"))
    check("5", calls == ["primary"], "폴백 호출 안 함")
    check("5", isinstance(err, ValueError), "ValueError 그대로 전파")

    calls, out, err, _ = await case(
        "6. 폴백 없음 — 전파", primary_raises=APIConnectionError("boom"), fallback=False)
    check("6", calls == ["primary"], "폴백 없으면 그대로 전파")
    check("6", isinstance(err, APIConnectionError), "원래 예외 유지")

    tracker = {"total_cost": 0.0}
    calls, out, err, tracker = await case(
        "7. 예산 이월 — 1차 $0.50 쓰고 실패, 2차 $0.30",
        primary_raises=APIConnectionError("boom"), tracker=tracker,
        primary_cost=0.50, fallback_cost=0.30)
    check("7", abs(tracker["total_cost"] - 0.80) < 1e-9,
          f"1차+2차 합산 $0.80 (실제 ${tracker['total_cost']:.4f})")

    print("=" * 60)
    if failures:
        print(f"FAILED {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — 페일오버 분기 + 예산 이월 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
