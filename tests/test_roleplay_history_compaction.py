"""state_history rows are stored compact and read back with a full ``before``."""
import unittest

from roleplay.memory import HISTORY_OMITTED_STATE_KEYS, compact_audit, expand_audit


class CompactionTests(unittest.TestCase):
    def test_round_trip_restores_before_without_bookkeeping(self):
        before = {"hunger": 10, "clock": {"time": "08:00"}, "pain": None, "dropped": 1,
                  "event_timestamps": list(range(500)), "recent_events": ["a"] * 100}
        after = {"hunger": 25, "clock": {"time": "09:00"}, "pain": None, "added": 2,
                 "event_timestamps": list(range(501)), "recent_events": ["a"] * 101}
        stored = compact_audit({"revision": 3, "before": before, "after": after, "reason": "r"})
        self.assertNotIn("before", stored)
        self.assertTrue(set(HISTORY_OMITTED_STATE_KEYS).isdisjoint(stored["after"]))
        self.assertEqual(stored["before_changed"], {"hunger": 10, "clock": {"time": "08:00"}, "dropped": 1})
        restored = expand_audit(stored)["before"]
        self.assertEqual(restored, {k: v for k, v in before.items() if k not in HISTORY_OMITTED_STATE_KEYS})

    def test_stage_keeps_only_changed_tables(self):
        stage = {"baseline": {"notes": [("k", "old")], "people": [("p", "{}")]},
                 "records": {"notes": [("k", "new")], "people": [("p", "{}")]},
                 "state": {"big": "x" * 1000}, "people": [{"id": "p"}]}
        stored = compact_audit({"before": {}, "after": {}, "staged_records": stage})
        self.assertEqual(stored["staged_changes"], {"notes": {"baseline": [("k", "old")], "records": [("k", "new")]}})

    def test_legacy_rows_pass_through(self):
        legacy = {"before": {"hunger": 1}, "after": {"hunger": 2}}
        self.assertEqual(expand_audit(legacy), legacy)


if __name__ == "__main__":
    unittest.main()
