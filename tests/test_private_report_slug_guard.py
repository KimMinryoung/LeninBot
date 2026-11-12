"""save_private must not overwrite (and unpublish) a public research document."""
import unittest
from unittest.mock import patch

from publishing import private_reports


class SlugGuardTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(private_reports, "ensure_private_reports_table")
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self):
        return private_reports.save_private_report_sync(title="T", slug="foo", markdown_body="body")

    def test_public_slug_is_refused_without_writing(self):
        with patch.object(private_reports.research_store, "get_document", return_value={"status": "public"}), \
             patch.object(private_reports.research_store, "upsert_document") as upsert:
            with self.assertRaisesRegex(ValueError, "public research document"):
                self.save()
        upsert.assert_not_called()

    def test_existing_private_slug_is_updated(self):
        row = {"id": 1, "slug": "foo", "title": "T", "markdown": "body", "status": "private"}
        with patch.object(private_reports.research_store, "get_document", return_value={"status": "private"}), \
             patch.object(private_reports.research_store, "upsert_document", return_value=(row, True)) as upsert, \
             patch.object(private_reports, "_private_row_to_compat", side_effect=lambda r: r):
            self.assertEqual(self.save()["slug"], "foo")
        upsert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
