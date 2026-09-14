import unittest
from scripts.kg_backfill_summaries import choose_facts


class SummaryBackfillTests(unittest.TestCase):
    def test_trust_then_recency_and_deduplication(self):
        rows = [
            {'fact': 'old', 'ep_names': ['[T:anchor]old'], 'created_at': '2020'},
            {'fact': 'new', 'ep_names': ['[T:anchor]new'], 'created_at': '2026'},
            {'fact': 'new', 'ep_names': ['[T:single]copy'], 'created_at': '2027'},
            {'fact': 'unknown', 'ep_names': [], 'created_at': None},
        ]
        self.assertEqual([f['fact'] for f in choose_facts(rows, 3)], ['new', 'old', 'unknown'])
        self.assertEqual([f['fact'] for f in choose_facts(rows, 1)], ['new'])
