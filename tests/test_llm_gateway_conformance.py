"""Static guards for the production LLM proxy boundary."""

import ast
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


class TestClientConstructionBoundary(unittest.TestCase):
    """No provider SDK client may be built outside the two places that wrap them.

    The proxy is the enforcement point and it identifies callers by the
    x-llm-caller header, which only the wrappers in llm/instrumented_clients.py
    send. A module that builds its own client reaches the proxy anonymously and
    misses whatever defaults the wrapper supplies — on 2026-08-09 that was
    DeepSeek's thinking mode, which is on unless the request says otherwise and
    spends the reply's token budget on reasoning. Two maintenance scripts and
    browser/worker.py had built their own.

    llm/ owns the wrappers and the gateway's own sync clients; bot_config owns
    the shared instances. Everything else asks one of them.
    """

    CONSTRUCTORS = {"AsyncAnthropic", "Anthropic", "AsyncOpenAI", "OpenAI"}
    ALLOWED = ("bot_config.py", "llm/", "tests/")
    # temp_dev is scratch space, not shipped code.
    SKIP_DIRS = ("venv", "__pycache__", ".git", "node_modules", "temp_dev")
    # An exception has to say so on the line, and say why. A smoke script whose
    # whole job is to reproduce a provider's own configuration is a real one.
    OPT_OUT = "llm-client-ok:"
    WRAPPERS = {"AuditedAsyncAnthropic", "AuditedAsyncOpenAI"}

    @staticmethod
    def _called_name(node):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _sources(self):
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if any(part in self.SKIP_DIRS for part in path.parts):
                continue
            if relative.startswith(self.ALLOWED):
                continue
            yield relative, path

    def test_no_provider_client_is_built_outside_the_gateway(self):
        offenders = []
        for relative, path in self._sources():
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            # A raw constructor handed straight to a wrapper is the correct
            # shape, so those nodes are collected first and excused.
            wrapped = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and self._called_name(node) in self.WRAPPERS:
                    for argument in node.args:
                        if isinstance(argument, ast.Call):
                            wrapped.add(id(argument))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or id(node) in wrapped:
                    continue
                if self._called_name(node) not in self.CONSTRUCTORS:
                    continue
                line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                if self.OPT_OUT in line:
                    continue
                offenders.append(f"{relative}:{node.lineno}: {line.strip()}")
        self.assertFalse(
            offenders,
            "provider SDK clients built outside bot_config.py and llm/:\n"
            + "\n".join(offenders)
            + "\n\nImport the shared client from bot_config, or wrap the new one with "
            "AuditedAsyncAnthropic/AuditedAsyncOpenAI.",
        )

    def test_bot_config_exports_only_wrapped_clients(self):
        tree = ast.parse((ROOT / "bot_config.py").read_text(encoding="utf-8"))
        wrappers = {"AuditedAsyncAnthropic", "AuditedAsyncOpenAI"}
        raw = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(name.endswith("_client") or name == "_claude" for name in names):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and value.value is None:
                continue  # the lazy `= None` placeholders
            called = getattr(getattr(value, "func", None), "id", None)
            if called not in wrappers:
                raw.append(f"line {node.lineno}: {', '.join(names)} = {called or type(value).__name__}")
        self.assertFalse(
            raw,
            "bot_config exports a client that is not wrapped:\n" + "\n".join(raw),
        )


if __name__ == "__main__":
    unittest.main()
