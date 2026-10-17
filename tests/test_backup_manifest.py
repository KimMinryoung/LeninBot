import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
# These scripts put scripts/ first on sys.path, where scripts/security_gateway.py
# would shadow the security_gateway package for every later test module in the
# same unittest run. Restore the path afterwards; keep scripts/ only at the end
# for the scripts' function-level imports.
_saved_path = list(sys.path)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _r2_backup  # noqa: E402
import r2_retention  # noqa: E402
import restore_db  # noqa: E402
import restore_kg_drill  # noqa: E402

sys.path[:] = _saved_path + [str(ROOT / "scripts")]


class ShrunkTablesTest(unittest.TestCase):
    def test_large_table_halved_or_gone_is_reported(self):
        previous = {"people": 4000, "terms": 3000, "events": 2000, "tiny": 10}
        current = {"people": 1999, "terms": 2900, "tiny": 0}
        self.assertEqual(
            _r2_backup.shrunk_tables(previous, current),
            ["events: 2000 -> missing", "people: 4000 -> 1999"],
        )

    def test_small_tables_and_growth_are_quiet(self):
        self.assertEqual(_r2_backup.shrunk_tables({"a": 999, "b": 5000}, {"a": 0, "b": 9000}), [])


class PublishManifestTest(unittest.TestCase):
    def test_compares_with_previous_day_not_same_day_rerun(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(_r2_backup, "r2_put") as put:
            backup_dir = Path(tmp)
            (backup_dir / "main-db-backup-2026-10-16.manifest.json").write_text(
                json.dumps({"rows_after": {"people": 4000}})
            )
            (backup_dir / "main-db-backup-2026-10-17.manifest.json").write_text(
                json.dumps({"rows_after": {"people": 10}})
            )
            found = _r2_backup.publish_manifest(
                "bucket", backup_dir, "main-db-backup", "main-db-backup-2026-10-17.dump",
                "leninbot", {"people": 1000}, {"people": 1001},
            )
            self.assertEqual(found, ["people: 4000 -> 1000"])
            put.assert_called_once()
            self.assertEqual(put.call_args.args[1], "main-db-backup-2026-10-17.manifest.json")
            saved = json.loads((backup_dir / "main-db-backup-2026-10-17.manifest.json").read_text())
            self.assertEqual(saved["rows_before"], {"people": 1000})

    def test_first_manifest_has_nothing_to_compare(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(_r2_backup, "r2_put"):
            found = _r2_backup.publish_manifest(
                "bucket", Path(tmp), "writer-db-backup", "writer-db-backup-2026-10-17.dump",
                "writer", {"t": 5000}, {"t": 5000},
            )
            self.assertEqual(found, [])


class RowCountMismatchTest(unittest.TestCase):
    manifest = {"rows_before": {"a": 100, "b": 50}, "rows_after": {"a": 120, "b": 50}}

    def test_within_bracket_passes(self):
        self.assertEqual(restore_db.row_count_mismatches(self.manifest, {"a": 115, "b": 50}), [])

    def test_short_or_missing_table_fails(self):
        problems = restore_db.row_count_mismatches(self.manifest, {"a": 80})
        self.assertEqual(problems, ["a: restored 80, backup counted 100..120", "b: missing after restore"])

    def test_table_created_between_counts_is_not_required(self):
        manifest = {"rows_before": {}, "rows_after": {"new": 3}}
        self.assertEqual(restore_db.row_count_mismatches(manifest, {}), [])


class RetentionLockTest(unittest.TestCase):
    def test_locked_delete_is_skipped_not_raised(self):
        keys = [f"main-db-backup-2026-10-{d:02d}.dump" for d in (1, 2, 16, 17)]

        def delete(_bucket, key):
            if key.endswith("10-01.dump"):
                raise RuntimeError("AccessDenied: object is locked")

        with mock.patch.object(r2_retention, "r2_list_keys", return_value=keys), \
                mock.patch.object(r2_retention, "r2_delete", side_effect=delete):
            deleted = r2_retention.prune_r2_prefix(
                "bucket", "main-db-backup", ".dump", date(2026, 10, 10)
            )
        self.assertEqual(deleted, ["main-db-backup-2026-10-02.dump"])

    def test_manifest_suffix_does_not_touch_dumps(self):
        keys = ["main-db-backup-2026-10-01.dump", "main-db-backup-2026-10-01.manifest.json",
                "main-db-backup-2026-10-16.manifest.json", "main-db-backup-2026-10-17.manifest.json"]
        with mock.patch.object(r2_retention, "r2_list_keys", return_value=keys), \
                mock.patch.object(r2_retention, "r2_delete") as delete:
            r2_retention.prune_r2_prefix(
                "bucket", "main-db-backup", _r2_backup.MANIFEST_SUFFIX, date(2026, 10, 10)
            )
        delete.assert_called_once_with("bucket", "main-db-backup-2026-10-01.manifest.json")


class DrillCleanupTest(unittest.TestCase):
    def test_trace_hides_neo4j_password(self):
        with mock.patch("subprocess.run"), mock.patch("builtins.print") as out:
            restore_kg_drill._run(["docker", "run", "-e", "NEO4J_AUTH=neo4j/s3cret", "img"])
        line = out.call_args.args[0]
        self.assertNotIn("s3cret", line)
        self.assertIn("NEO4J_AUTH=neo4j/***", line)


if __name__ == "__main__":
    unittest.main()
