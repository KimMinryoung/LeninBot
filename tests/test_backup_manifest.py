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


class DumpDigestTest(unittest.TestCase):
    def test_header_timestamp_is_ignored_but_data_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = b"PGDMP" + bytes(59) + b"table data" * 100
            a, b, c = (Path(tmp) / n for n in "abc")
            a.write_bytes(body)
            b.write_bytes(body[:14] + b"\x01\x02\x03\x04" + body[18:])
            c.write_bytes(body[:-1] + b"X")
            self.assertEqual(_r2_backup.dump_digest(str(a)), _r2_backup.dump_digest(str(b)))
            self.assertNotEqual(_r2_backup.dump_digest(str(a)), _r2_backup.dump_digest(str(c)))


class UnchangedArchiveTest(unittest.TestCase):
    def _check(self, previous, on_r2):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(_r2_backup, "r2_list_keys", return_value=on_r2):
            (Path(tmp) / "legacy-game-db-backup-2026-10-18.manifest.json").write_text(json.dumps(previous))
            return _r2_backup.unchanged_archive(
                "bucket", Path(tmp), "legacy-game-db-backup",
                "legacy-game-db-backup-2026-10-19.dump", "abc",
            )

    def test_same_digest_and_dump_on_r2_reuses_it(self):
        prev = {"content_sha256": "abc", "archive": "legacy-game-db-backup-2026-10-01.dump"}
        self.assertEqual(self._check(prev, ["legacy-game-db-backup-2026-10-01.dump"]),
                         "legacy-game-db-backup-2026-10-01.dump")

    def test_changed_or_missing_dump_uploads(self):
        prev = {"content_sha256": "abc", "archive": "legacy-game-db-backup-2026-10-01.dump"}
        self.assertIsNone(self._check(prev, []))
        self.assertIsNone(self._check({**prev, "content_sha256": "old"}, [prev["archive"]]))
        self.assertIsNone(self._check({"archive": prev["archive"]}, [prev["archive"]]))


class RetentionProtectTest(unittest.TestCase):
    def test_referenced_old_dump_survives(self):
        keys = [f"writer-db-backup-2026-10-{d:02d}.dump" for d in (1, 2, 3)]
        with mock.patch.object(r2_retention, "r2_list_keys", return_value=keys), \
                mock.patch.object(r2_retention, "r2_delete") as delete:
            deleted = r2_retention.prune_r2_prefix(
                "bucket", "writer-db-backup", ".dump", date(2026, 10, 20),
                protect=frozenset({"writer-db-backup-2026-10-03.dump"}),
            )
        # Only one key would be kept, below min_keep: nothing is deleted.
        self.assertEqual(deleted, [])
        delete.assert_not_called()


class FetchFollowsManifestTest(unittest.TestCase):
    def _fetch(self, keys, manifests):
        def get(_bucket, key, path):
            Path(path).write_text(json.dumps(manifests[key]) if key in manifests else "dump")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("_r2_backup.r2_list_keys", return_value=keys), \
                mock.patch("_r2_backup.r2_get", side_effect=get), \
                mock.patch.object(restore_db, "datetime", wraps=restore_db.datetime) as dt:
            dt.now.return_value = restore_db.datetime.fromisoformat("2026-10-20T12:00:00+09:00")
            dt.strptime = restore_db.datetime.strptime
            path = restore_db._fetch_latest_from_r2(restore_db.SPECS["writer"], Path(tmp), 2)
            manifest = json.loads(restore_db._manifest_path(path).read_text())
            return path.name, manifest

    def test_unchanged_day_points_at_earlier_dump(self):
        keys = ["writer-db-backup-2026-10-15.dump", "writer-db-backup-2026-10-15.manifest.json",
                "writer-db-backup-2026-10-20.manifest.json"]
        manifests = {
            "writer-db-backup-2026-10-15.manifest.json": {"archive": "writer-db-backup-2026-10-15.dump"},
            "writer-db-backup-2026-10-20.manifest.json": {"archive": "writer-db-backup-2026-10-15.dump",
                                                          "rows_after": {"t": 1}},
        }
        name, manifest = self._fetch(keys, manifests)
        self.assertEqual(name, "writer-db-backup-2026-10-15.dump")
        self.assertEqual(manifest["rows_after"], {"t": 1})

    def test_dangling_pointer_fails(self):
        keys = ["writer-db-backup-2026-10-19.dump", "writer-db-backup-2026-10-20.manifest.json"]
        manifests = {"writer-db-backup-2026-10-20.manifest.json": {"archive": "writer-db-backup-2026-10-01.dump"}}
        with self.assertRaises(restore_db.RestoreError):
            self._fetch(keys, manifests)

    def test_newer_dump_without_manifest_wins(self):
        keys = ["writer-db-backup-2026-10-19.manifest.json", "writer-db-backup-2026-10-19.dump",
                "writer-db-backup-2026-10-20.dump"]
        manifests = {"writer-db-backup-2026-10-19.manifest.json": {"archive": "writer-db-backup-2026-10-19.dump"}}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("_r2_backup.r2_list_keys", return_value=keys), \
                mock.patch("_r2_backup.r2_get", side_effect=lambda b, k, p: Path(p).write_text("x")):
            path = restore_db._fetch_latest_from_r2(restore_db.SPECS["writer"], Path(tmp), None)
            self.assertEqual(path.name, "writer-db-backup-2026-10-20.dump")


class DrillCleanupTest(unittest.TestCase):
    def test_trace_hides_neo4j_password(self):
        with mock.patch("subprocess.run"), mock.patch("builtins.print") as out:
            restore_kg_drill._run(["docker", "run", "-e", "NEO4J_AUTH=neo4j/s3cret", "img"])
        line = out.call_args.args[0]
        self.assertNotIn("s3cret", line)
        self.assertIn("NEO4J_AUTH=neo4j/***", line)


if __name__ == "__main__":
    unittest.main()
