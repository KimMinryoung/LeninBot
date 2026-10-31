"""scripts/affected_services.py: which services a deploy must restart."""

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import affected_services as aff  # noqa: E402


class RulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = aff.repo_modules()

    def run_rules(self, *changes):
        return aff.affected(list(changes), self.modules)

    def test_docs_tests_and_deploy_scripts_restart_nothing(self):
        services, _ = self.run_rules(
            ("M", "dev_docs/README.md"), ("M", "tests/test_x.py"), ("M", "AGENTS.md"),
            ("M", "migrations/README.md"), ("M", "scripts/svc"), ("M", "deploy.sh"),
            ("M", "config.json.example"),
        )
        self.assertEqual(services, set())

    def test_unattributable_changes_restart_all(self):
        for change in [("M", "requirements.txt"), ("M", "requirements.lock.txt"),
                       ("M", "systemd/leninbot-api.service"), ("M", "config/agent_runtime.json"),
                       ("M", "skills/x/SKILL.md"), ("D", "llm/client.py"),
                       ("R100", "llm/client.py")]:
            with self.subTest(change=change):
                services, reasons = self.run_rules(change)
                self.assertIsNone(services)
                self.assertTrue(reasons)

    def test_module_change_restarts_importers_only(self):
        services, _ = self.run_rules(("M", "llm_proxy/app.py"))
        self.assertEqual(services, {"llm-proxy"})
        services, _ = self.run_rules(("M", "services/embedding_server.py"))
        self.assertEqual(services, {"embedding"})

    def test_shared_core_reaches_the_large_services(self):
        services, _ = self.run_rules(("M", "tool_gateway/dispatcher.py"))
        self.assertTrue({"api", "telegram", "worker"} <= services)
        self.assertNotIn("llm-proxy", services)
        self.assertNotIn("embedding", services)

    def test_standalone_script_restarts_nothing(self):
        services, reasons = self.run_rules(("M", "scripts/kg_weekly_report.py"))
        self.assertEqual(services, set())
        self.assertIn("not imported by any long-running service", reasons[-1])

    def test_relative_and_function_level_imports_are_followed(self):
        closure = aff.import_closure("telegram.bot", self.modules)
        # telegram.bot imports chat_runtime, which lazily imports tool_evidence
        # inside chat_with_tools.
        self.assertIn("telegram.chat_runtime", closure)
        self.assertIn("telegram.tool_evidence", closure)


class EntrypointTest(unittest.TestCase):
    def test_entrypoints_match_systemd_units(self):
        unit_for = {"novel-writer": "novel-writer-api.service"}
        for service, module in aff.SERVICE_ENTRYPOINTS.items():
            unit = unit_for.get(service, f"leninbot-{service}.service")
            path = os.path.join(ROOT, "systemd", unit)
            with self.subTest(service=service):
                self.assertTrue(os.path.exists(path), unit)
                with open(path, encoding="utf-8") as fh:
                    exec_start = next(l for l in fh if l.startswith("ExecStart="))
                self.assertRegex(exec_start, rf"(-m |uvicorn ){re.escape(module)}(:app)?\b")
                self.assertIn(module, aff.repo_modules())


if __name__ == "__main__":
    unittest.main()
