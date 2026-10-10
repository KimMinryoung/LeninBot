import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "skills" / "kg-maintenance" / "scripts"))

import restore_kg  # noqa: E402
import restore_kg_drill  # noqa: E402


class ExpectedCountsTest(unittest.TestCase):
    def test_parallel_mentions_count_once(self):
        mentions = [
            {"episode_uuid": "ep1", "entity_uuid": "a"},
            {"episode_uuid": "ep1", "entity_uuid": "a"},
            {"episode_uuid": "ep1", "entity_uuid": "b"},
        ]
        self.assertEqual(
            restore_kg.expected_counts([{"uuid": "a"}, {"uuid": "b"}], [{"uuid": "r1"}], mentions),
            {"entities": 2, "relates_to": 1, "mentions": 2},
        )


class FetchLatestTest(unittest.TestCase):
    def _fetch(self, keys, today="2026-10-11", max_age_days=2):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("_r2_backup.r2_list_keys", return_value=keys), \
                mock.patch("_r2_backup.r2_get") as get, \
                mock.patch.object(restore_kg_drill, "datetime", wraps=restore_kg_drill.datetime) as dt:
            dt.now.return_value = restore_kg_drill.datetime.fromisoformat(today + "T12:00:00+09:00")
            path = restore_kg_drill._fetch_latest_from_r2(Path(tmp), max_age_days)
            get.assert_called_once()
            return path.name

    def test_picks_newest_including_manual_same_day_snapshot(self):
        keys = [
            "kg-backup-2026-10-10.tar.gz",
            "kg-backup-2026-10-11.tar.gz",
            "kg-backup-2026-10-11_093000.tar.gz",
            "kg-backup-notes.txt",
        ]
        self.assertEqual(self._fetch(keys), "kg-backup-2026-10-11_093000.tar.gz")

    def test_stale_backup_fails(self):
        with self.assertRaises(restore_kg_drill.DrillError):
            self._fetch(["kg-backup-2026-10-01.tar.gz"])


if __name__ == "__main__":
    unittest.main()
