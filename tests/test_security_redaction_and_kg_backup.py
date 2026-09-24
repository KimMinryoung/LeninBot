"""Regression checks for persisted tool secrets and pre-mutation KG backups."""

import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from security_gateway.audit import redact_args
from security_gateway.redaction import redact_log_text, tool_input_summary
from mcp_gateway import tools as mcp_tools
from mcp_gateway import policy as mcp_policy
from mcp_gateway import server as mcp_server
from tool_gateway import dispatcher


class RedactionTests(unittest.TestCase):
    def test_nested_secret_never_enters_audit_or_progress(self):
        data = {"request": [{"body": {"api_key": "canary-123", "text": "ok"}}]}
        for rendered in (redact_args(data), tool_input_summary(data), redact_log_text(json.dumps(data))):
            self.assertNotIn("canary-123", rendered)
            self.assertIn("«redacted»", rendered)
            self.assertIn("ok", rendered)

    def test_labelled_secret_in_plain_result(self):
        self.assertNotIn("canary-123", redact_log_text("token=canary-123 succeeded"))
        self.assertNotIn("canary-123", redact_log_text("Authorization: Bearer canary-123"))

    def test_live_tool_progress_is_redacted(self):
        events = []

        async def progress(event, detail):
            events.append(detail)

        async def execute(*args, **kwargs):
            return '{"password":"canary-456"}', False

        with patch.object(dispatcher, "execute_tool", execute):
            asyncio.run(dispatcher.execute_tools_batch(
                [("1", "test_tool", {"nested": {"api_key": "canary-123"}})], {},
                on_progress=progress, round_num=1,
            ))
        self.assertNotIn("canary-123", " ".join(events))
        self.assertNotIn("canary-456", " ".join(events))


class KgBackupTests(unittest.TestCase):
    def test_operator_profile_requires_repository_owner_uid(self):
        with patch.object(mcp_policy.os, "geteuid", return_value=-1):
            self.assertEqual(mcp_policy.normalize_profile("operator"), "inspect")

    def test_invalid_backup_prevents_mutation(self):
        calls = []

        def command(argv, timeout):
            calls.append(argv)
            return "backup completed but artifact absent"

        async def inline(func, *args):
            return func(*args)

        with patch.object(mcp_tools, "_run_project_command", command), \
             patch.object(mcp_tools.asyncio, "to_thread", inline):
            result = asyncio.run(mcp_tools.kg_maintenance_run(
                "cleanup_orphans", execute=True, confirm="APPLY_KG_MAINTENANCE",
            ))
        self.assertIn("mutation refused", result)
        self.assertEqual(len(calls), 1)

    def test_verifier_requires_fresh_parseable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup_dir = root / "data" / "kg_backups"
            backup_dir.mkdir(parents=True)
            stamp = "20260924_000000_000001"
            for name, rows in (("entities", [{"uuid": "a"}]), ("edges", []), ("mentions", [])):
                (backup_dir / f"{name}_{stamp}.json").write_text(json.dumps(rows))
            with patch.object(mcp_tools, "ROOT", root):
                result = mcp_tools._verify_kg_backup(f"백업 완료 (timestamp: {stamp})", 0)
                self.assertIn("entities=1", result)
                (backup_dir / f"edges_{stamp}.json").write_text("broken")
                with self.assertRaises(json.JSONDecodeError):
                    mcp_tools._verify_kg_backup(f"백업 완료 (timestamp: {stamp})", 0)


class McpAuditTests(unittest.TestCase):
    def test_allowed_call_passes_authorization_and_audit(self):
        audit_module = importlib.import_module("security_gateway.audit")
        gateway_module = importlib.import_module("security_gateway.gateway")
        decision = gateway_module.Decision(True, "allow", "read", "", "enforce", "none")
        recorded = []

        async def handler(**kwargs):
            return "healthy"

        with patch.object(mcp_server, "build_handlers", return_value={"gateway_status": handler}), \
             patch.object(gateway_module, "authorize", return_value=decision) as authorize, \
             patch.object(audit_module, "audit", side_effect=lambda *a, **kw: recorded.append((a, kw))):
            result = asyncio.run(mcp_server._call_tool("gateway_status", {}, "inspect"))
        self.assertFalse(result["isError"])
        authorize.assert_called_once()
        self.assertEqual(recorded[0][1]["result_status"], "ok")
        self.assertEqual(recorded[0][0][0].interface, "mcp")

    def test_unexposed_call_is_denied_and_audited(self):
        audit_module = importlib.import_module("security_gateway.audit")
        recorded = []
        with patch.object(mcp_server, "build_handlers", return_value={}), \
             patch.object(audit_module, "audit", side_effect=lambda *a, **kw: recorded.append((a, kw))):
            result = asyncio.run(mcp_server._call_tool("kg_maintenance_run", {}, "inspect"))
        self.assertTrue(result["isError"])
        self.assertEqual(recorded[0][0][3].label, "deny")
        self.assertEqual(recorded[0][1]["result_status"], "denied")
