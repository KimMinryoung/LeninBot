"""DeepSeek email-announced billing cutover and UTC calendar boundaries."""

import unittest
from datetime import datetime, timedelta, timezone

from llm.provider_registry import (
    anthropic_pricing_table,
    deepseek_price_triple,
    openai_compatible_pricing,
)


class DeepSeekPricingTests(unittest.TestCase):
    models = ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-flash")

    def test_cutover_preserves_historical_prices(self):
        before = datetime(2026, 9, 10, 3, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(deepseek_price_triple(self.models[0], before), (0.44, 1.32, 0.014667))
        self.assertEqual(deepseek_price_triple(self.models[1], before), (1.32, 3.96, 0.044))
        for model in (self.models[0], self.models[2]):
            self.assertEqual(
                deepseek_price_triple(model, before + timedelta(seconds=1)),
                (0.15, 0.6, 0.003),
            )

    def test_weekday_peak_boundaries_and_weekends(self):
        for day in range(18, 25):  # Friday through Thursday, including weekend
            for hour, minute, in_window in (
                (0, 59, False), (1, 0, True), (3, 59, True), (4, 0, False),
                (5, 59, False), (6, 0, True), (9, 59, True), (10, 0, False),
            ):
                now = datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)
                peak = now.weekday() < 5 and in_window
                expected = (0.3, 1.2, 0.006) if peak else (0.15, 0.6, 0.003)
                for model in self.models:
                    with self.subTest(model=model, now=now):
                        self.assertEqual(deepseek_price_triple(model, now), expected)

    def test_calendar_uses_utc_not_callers_timezone(self):
        # Thursday locally, Friday's first peak window in UTC.
        now = datetime(2026, 9, 17, 18, tzinfo=timezone(timedelta(hours=-8)))
        for model in self.models:
            self.assertEqual(deepseek_price_triple(model, now), (0.3, 1.2, 0.006))
        # Monday locally but still Sunday UTC: never a peak.
        now = datetime(2026, 9, 21, 2, tzinfo=timezone(timedelta(hours=20)))
        for model in self.models:
            self.assertEqual(deepseek_price_triple(model, now), (0.15, 0.6, 0.003))

    def test_old_weekends_and_flat_rates_are_unchanged(self):
        weekend = datetime(2026, 9, 6, 2, tzinfo=timezone.utc)
        self.assertEqual(deepseek_price_triple(self.models[1], weekend), (1.32, 3.96, 0.044))
        flat = datetime(2026, 8, 15, 2, tzinfo=timezone.utc)
        self.assertEqual(deepseek_price_triple(self.models[0], flat), (0.14, 0.28, 0.0028))
        self.assertEqual(deepseek_price_triple(self.models[1], flat), (0.435, 0.87, 0.003625))

    def test_protocol_pricing_tables_share_new_rates(self):
        now = datetime(2026, 9, 18, 2, tzinfo=timezone.utc)
        for model in self.models:
            anthropic = anthropic_pricing_table(now)[model]
            compatible = openai_compatible_pricing(model, now=now)
            self.assertEqual(anthropic["input"], 0.3 / 1_000_000)
            self.assertEqual(anthropic["cache_creation"], anthropic["input"])
            self.assertEqual(anthropic["cache_read"], 0.006 / 1_000_000)
            self.assertEqual(anthropic["output"], 1.2 / 1_000_000)
            self.assertEqual(compatible["input"], anthropic["input"])
            self.assertEqual(compatible["cached_input"], anthropic["cache_read"])
            self.assertEqual(compatible["output"], anthropic["output"])

    def test_pro_keeps_old_price_until_retirement(self):
        for now, expected in (
            (datetime(2026, 9, 10, 4, tzinfo=timezone.utc), (0.66, 1.98, 0.022)),
            (datetime(2026, 9, 12, 2, tzinfo=timezone.utc), (0.66, 1.98, 0.022)),
            (datetime(2026, 9, 14, 3, 59, 59, tzinfo=timezone.utc), (1.32, 3.96, 0.044)),
            (datetime(2026, 9, 14, 4, tzinfo=timezone.utc), (0.15, 0.6, 0.003)),
        ):
            self.assertEqual(deepseek_price_triple("deepseek-v4-pro", now), expected)

    def test_all_deepseek_tiers_and_legacy_selection_use_flash(self):
        from llm.provider_registry import DEEPSEEK_MODEL_MAP, TIER_MODEL_KEYS
        for alias in TIER_MODEL_KEYS["deepseek"].values():
            self.assertEqual(DEEPSEEK_MODEL_MAP[alias], "deepseek-flash")
        self.assertEqual(DEEPSEEK_MODEL_MAP["deepseek_pro"], "deepseek-flash")
