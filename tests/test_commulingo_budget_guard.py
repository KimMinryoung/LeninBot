import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from scripts import commulingo_budget_guard as guard


class BudgetGuardTests(unittest.TestCase):
    def test_disabled_cap_never_reads_journal(self):
        for value in (None, '0', '-1'):
            env = {} if value is None else {'COMMULINGO_DAILY_CAP_USD': value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True), patch.object(guard.subprocess, 'run') as journal:
                self.assertEqual(guard.main(), 0)
                journal.assert_not_called()

    def test_explicit_positive_cap_still_enforced(self):
        with patch.dict(os.environ, {'COMMULINGO_DAILY_CAP_USD': '2.5'}), patch.object(guard, 'LANE_UNITS', ['leninbot-commulingo-enrich']):
            for spend, expected in [('2.49', 0), ('2.50', 1), ('6.933e-05', 0)]:
                with self.subTest(spend=spend), patch.object(guard.subprocess, 'run', return_value=SimpleNamespace(stdout='  "cost_usd": ' + spend)):
                    self.assertEqual(guard.main(), expected)
