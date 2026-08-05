"""Static guards for the production LLM proxy boundary."""

import unittest
from pathlib import Path

from scripts.migrate_secrets_to_credstore import SERVICE_CREDS, _LLM_PROVIDER_KEYS


ROOT = Path(__file__).resolve().parent.parent


class TestCredentialBoundary(unittest.TestCase):
    def test_generated_consumer_dropins_never_include_provider_keys(self):
        for service, credentials in SERVICE_CREDS.items():
            if service == "leninbot-llm-proxy":
                continue
            leaked = credentials & _LLM_PROVIDER_KEYS
            self.assertFalse(leaked, f"{service} leaks provider keys: {sorted(leaked)}")

    def test_static_provider_keys_exist_only_on_proxy(self):
        for unit in (ROOT / "systemd").glob("*.service"):
            active_lines = [
                line for line in unit.read_text().splitlines()
                if line.startswith("LoadCredentialEncrypted=")
            ]
            provider_lines = [
                line for line in active_lines
                if any(key.lower() in line for key in _LLM_PROVIDER_KEYS)
            ]
            if unit.name == "leninbot-llm-proxy.service":
                self.assertTrue(provider_lines)
            else:
                self.assertFalse(provider_lines, f"provider key in {unit.name}")


class TestProxyOrdering(unittest.TestCase):
    CONSUMERS = {
        "leninbot-a2a-api.service", "leninbot-api.service",
        "leninbot-autonomous.service", "leninbot-browser.service",
        "leninbot-commulingo-enrich.service",
        "leninbot-commulingo-maintainer.service",
        "leninbot-commulingo-new.service", "leninbot-commulingo-terms.service",
        "leninbot-event-backfill.service", "leninbot-experience.service",
        "leninbot-kg-integrity.service", "leninbot-razvedchik.service",
        "leninbot-roleplay.service", "leninbot-telegram.service",
        "novel-writer-api.service",
    }

    def test_consumers_order_after_and_want_proxy(self):
        for name in self.CONSUMERS:
            text = (ROOT / "systemd" / name).read_text()
            self.assertIn("Wants=", text, name)
            wants = " ".join(
                line for line in text.splitlines() if line.startswith("Wants=")
            )
            after = " ".join(
                line for line in text.splitlines() if line.startswith("After=")
            )
            self.assertIn("leninbot-llm-proxy.service", wants, name)
            self.assertIn("leninbot-llm-proxy.service", after, name)


class TestRazvedchikMigration(unittest.TestCase):
    def test_legacy_client_is_deleted_and_unreferenced(self):
        self.assertFalse((ROOT / "agents/razvedchik/cloud_llm.py").exists())
        for path in (ROOT / "agents/razvedchik").glob("*.py"):
            self.assertNotIn("agents.razvedchik.cloud_llm", path.read_text())


if __name__ == "__main__":
    unittest.main()
