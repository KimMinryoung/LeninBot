"""Healthcheck failures must survive large successful diagnostic payloads."""
import sys
import unittest
from unittest.mock import patch

from scripts import check_kg_integrity as check


class HealthcheckTests(unittest.TestCase):
    def status(self):
        healthy = {"complete": True, "lag_hours": 2, "phase": "finished"}
        return {"ok": True, "smoke_search": {"ok": True, "preview": "x" * 5000},
                "semantic_search": {"ok": True},
                "metrics": {"sync": {"commulingo": dict(healthy), "documents": dict(healthy)}}}

    def test_document_failure_visible_despite_successful_search(self):
        status = self.status()
        status["metrics"]["sync"]["documents"].update(
            complete=False, failed=1, remaining=1,
            error="archival:iwma-general-rules-1871: rejected 1 invalid fact(s)")
        health = check.healthcheck_status(status, metrics_enabled=True)
        self.assertFalse(health["ok"])
        alert = check.format_healthcheck_alert(health)
        self.assertIn("sync.documents", alert)
        self.assertIn("iwma-general-rules-1871", alert)
        self.assertNotIn("preview", alert)
        self.assertLessEqual(len(alert), 3500)

    def test_healthy_and_pending_coverage_do_not_alert(self):
        status = self.status()
        status["metrics"]["coverage"] = {"documents_changed": 1}
        self.assertTrue(check.healthcheck_status(status, metrics_enabled=True)["ok"])

    def test_missing_sync_and_metrics_error(self):
        status = self.status()
        status["metrics"]["sync"] = {"error": "database unavailable"}
        health = check.healthcheck_status(status, metrics_enabled=True)
        self.assertEqual(len(health["failures"]), 3)
        self.assertIn("database unavailable", check.format_healthcheck_alert(health))

    def test_main_exit_and_notification_use_aggregate_status(self):
        status = self.status()
        status["smoke_search"]["ok"] = False
        with patch.object(sys, "argv", ["check", "--smoke-query", "test", "--notify"]), \
             patch.object(check, "check_kg_integrity", return_value={"ok": True}), \
             patch.object(check, "_run_smoke_search", return_value=status["smoke_search"]), \
             patch.object(check, "_notify_telegram") as notify, patch("builtins.print"):
            self.assertEqual(check.main(), 1)
        self.assertIn("smoke_search", notify.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
