"""A failover retry gets the turn budget minus what the failed attempt spent."""
import asyncio
import unittest
from unittest.mock import patch

from llm import provider_failover


class FailoverBudgetTests(unittest.TestCase):
    def run_failover(self, *, spent, budget):
        tracker = {"total_cost": 0.0}
        seen = {}

        async def primary():
            tracker["total_cost"] = spent
            raise RuntimeError("upstream down")

        async def fallback(budget_usd=None):
            seen["budget"] = budget_usd
            tracker["total_cost"] = 0.01
            return "ok"

        with patch.object(provider_failover, "is_transient_provider_error", return_value=True):
            result = asyncio.run(provider_failover.run_with_provider_failover(
                primary, fallback, primary_label="p", fallback_label="f",
                budget_tracker=tracker, budget_usd=budget,
            ))
        return result, seen, tracker

    def test_retry_budget_excludes_spent_cost(self):
        result, seen, tracker = self.run_failover(spent=0.3, budget=1.0)
        self.assertEqual(result, "ok")
        self.assertAlmostEqual(seen["budget"], 0.7)
        self.assertAlmostEqual(tracker["total_cost"], 0.31)

    def test_retry_keeps_a_floor_when_budget_is_nearly_spent(self):
        _, seen, _ = self.run_failover(spent=0.95, budget=1.0)
        self.assertAlmostEqual(seen["budget"], provider_failover.FALLBACK_BUDGET_FLOOR_RATIO)

    def test_without_budget_fallback_is_called_bare(self):
        calls = []

        async def primary():
            raise RuntimeError("down")

        async def fallback():
            calls.append("bare")
            return "ok"

        with patch.object(provider_failover, "is_transient_provider_error", return_value=True):
            asyncio.run(provider_failover.run_with_provider_failover(
                primary, fallback, primary_label="p", fallback_label="f"))
        self.assertEqual(calls, ["bare"])


if __name__ == "__main__":
    unittest.main()
