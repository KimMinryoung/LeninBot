"""MCP operator boundaries and diagnostic failure semantics."""
import os
os.environ["LENINBOT_TOOL_AUDIT_DB"] = "0"
os.environ["LENINBOT_EXECUTION_KIND"] = "test"

import asyncio
import json
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from mcp_gateway import tools
from mcp_gateway.server import _call_tool
from tool_gateway.results import ToolFailure, ToolRejection


class FailureTests(unittest.TestCase):
    def test_command_failure(self):
        with patch.object(tools.subprocess, 'run', return_value=MagicMock(returncode=2, stdout='bad', stderr='')):
            self.assertIsInstance(tools._run_project_command(['dummy'], 10), ToolFailure)

    def test_maintenance_preserves_failure_and_blocks_mutation(self):
        with patch.object(tools, '_run_project_command', return_value=ToolFailure('failed')) as run:
            result = asyncio.run(tools.kg_maintenance_run('cleanup_orphans'))
            self.assertIsInstance(result, ToolFailure)
            result = asyncio.run(tools.kg_maintenance_run('cleanup_orphans', execute=True, confirm='APPLY_KG_MAINTENANCE'))
            self.assertIsInstance(result, ToolFailure)
            self.assertEqual(run.call_count, 2)
        with patch.object(tools, '_run_project_command', return_value='backup'), patch.object(tools, '_verify_kg_backup', side_effect=ValueError('stale')):
            self.assertIsInstance(asyncio.run(tools.kg_maintenance_run('cleanup_orphans', execute=True, confirm='APPLY_KG_MAINTENANCE')), ToolFailure)
        with self.assertRaises(ToolRejection):
            asyncio.run(tools.kg_maintenance_run('cleanup_orphans', execute=True))

    def test_audit_and_iserror(self):
        with patch('mcp_gateway.credentials.bootstrap_kg_credentials', return_value={'status': 'configured'}), patch('security_gateway.audit.audit') as audit, patch.object(tools, '_run_project_command', side_effect=subprocess.TimeoutExpired('check', 10)):
            result = asyncio.run(_call_tool('kg_integrity_check', {}, 'inspect'))
            self.assertTrue(result['isError'])
            self.assertEqual(audit.call_args.kwargs['result_status'], 'error')
            result = asyncio.run(_call_tool('kg_integrity_check', {'profile': 'operator'}, 'inspect'))
            self.assertTrue(result['isError'])
            self.assertEqual(audit.call_args.kwargs['result_status'], 'rejected')

    def test_status(self):
        self.assertFalse(json.loads(asyncio.run(tools.gateway_status()))['write_tools_exposed'])
        self.assertIn('kg_maintenance_run', json.loads(asyncio.run(tools.gateway_status(profile='operator')))['mutable_tools'])


class CredentialTests(unittest.TestCase):
    def test_priority_missing_owner_and_no_other_secrets(self):
        import tempfile
        from pathlib import Path
        from mcp_gateway import credentials as c
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch('dotenv.load_dotenv'), patch.object(c, 'SERVICE_CREDENTIALS', Path(tmp)), patch.object(c.os, 'geteuid', return_value=c.ROOT.stat().st_uid):
            self.assertEqual(c.bootstrap_kg_credentials()['status'], 'credential_missing')
            for unit, password in [('leninbot-api.service', 'api-secret'), ('leninbot-telegram.service', 'telegram-secret')]:
                directory = Path(tmp) / unit
                directory.mkdir()
                (directory / 'neo4j_password').write_text(password)
                (directory / 'other_secret').write_text('never-read')
            self.assertEqual(c.bootstrap_kg_credentials()['source'], 'leninbot-api.service')
            self.assertEqual(os.environ['NEO4J_PASSWORD'], 'api-secret')
            self.assertNotIn('OTHER_SECRET', os.environ)
            os.environ['NEO4J_PASSWORD'] = 'explicit'
            self.assertEqual(c.bootstrap_kg_credentials()['source'], 'environment')
            self.assertEqual(os.environ.pop('NEO4J_PASSWORD'), 'explicit')
            with patch.object(c.os, 'geteuid', return_value=-1):
                self.assertEqual(c.bootstrap_kg_credentials()['status'], 'permission_denied')
            os.environ['CREDENTIALS_DIRECTORY'] = str(Path(tmp) / 'missing')
            self.assertEqual(c.bootstrap_kg_credentials()['status'], 'credential_missing')
            self.assertNotIn('NEO4J_PASSWORD', os.environ)
            with patch.object(Path, 'read_text', side_effect=PermissionError):
                self.assertEqual(c.bootstrap_kg_credentials()['status'], 'permission_denied')


class HealthLogTests(unittest.TestCase):
    def test_health_partial_preserves_outage(self):
        from ops.diagnostics import health_snapshot
        with patch('ops.diagnostics.systemd_properties', return_value={'ActiveState': 'failed'}), patch('ops.diagnostics.dependency_probe', side_effect=TimeoutError('offline')):
            data = asyncio.run(health_snapshot(['api']))
        self.assertTrue(data['partial'])
        self.assertEqual(data['results']['api']['systemd']['status'], 'unhealthy')
        self.assertEqual(data['results']['api']['probe']['collection_status'], 'error')

    def test_logs_mask_and_empty_and_failure(self):
        from ops.logs import fetch_server_logs
        from mcp_gateway.diagnostic_tools import get_service_logs
        with patch('subprocess.run', return_value=MagicMock(returncode=0, stderr='', stdout='2026 SECRET_KEY=hidden NEO4J_PASSWORD=hidden2 Authorization: Bearer hidden3')):
            text = str(fetch_server_logs('api'))
            self.assertNotIn('hidden', text)
        with patch('ops.logs.fetch_server_logs', return_value=[]):
            self.assertFalse(json.loads(asyncio.run(get_service_logs('api', profile='operator')))['partial'])
        with patch('ops.logs.fetch_server_logs', return_value=[{'error': 'permission denied'}]):
            self.assertIsInstance(asyncio.run(get_service_logs('api', profile='operator')), ToolFailure)
        for args in ({'service': 'evil'}, {'service': 'api', 'hours_back': 169}, {'service': 'api', 'limit': 1001}):
            with self.assertRaises(ToolRejection):
                asyncio.run(get_service_logs(**args, profile='operator'))

    def test_operator_boundary(self):
        from mcp_gateway.diagnostic_tools import service_health_snapshot
        with self.assertRaises(ToolRejection):
            asyncio.run(service_health_snapshot())
        self.assertNotIn('get_service_logs', tools.build_handlers('inspect'))


class ConfigTests(unittest.TestCase):
    def test_shared_snapshot_allowlist_and_unknown_live_state(self):
        from ops.model_runtime import build_snapshot
        import bot_config
        with patch.dict(bot_config._config, {'unexpected_secret': 'do-not-return'}):
            data = build_snapshot()
        self.assertNotIn('unexpected_secret', data['runtime_config'])
        self.assertEqual(data['running_services']['status'], 'unknown')
        self.assertIn('budget_usd', data['surfaces']['telegram_chat'])
        from scripts.model_runtime_audit import build_snapshot as cli_snapshot
        self.assertIs(cli_snapshot, build_snapshot)


class PipelineTests(unittest.TestCase):
    def test_freshness_shared_with_runners(self):
        from translation_runtime.freshness import source_hash_sql, missing_translation_sql
        from scripts import translate_db_content as cli
        self.assertIs(cli._source_hash_sql, source_hash_sql)
        self.assertIs(cli._missing_translation_sql, missing_translation_sql)
        from scripts import translate_research_documents as research
        with patch.object(research.research_store, 'ensure_research_table'), patch.object(research, 'db_query', return_value=[]) as query:
            research._select_rows(limit=0, max_chars=0, force=False)
            self.assertIn('markdown_en_source_sha256 IS DISTINCT FROM content_sha256', query.call_args.args[0])

    def test_pending_changed_and_current_cooldown(self):
        from ops.pipeline_status import translation_queue
        rows = [{'id': 1, 'missing': True, 'source_changed': True, 'fingerprint': 'new'},
                {'id': 2, 'missing': False, 'source_changed': True, 'fingerprint': 'old'}]
        with patch('ops.pipeline_status._query', return_value=rows), patch('translation_runtime.batch_state.BatchState.deferred', side_effect=lambda key, sha: sha == 'old'):
            data = translation_queue('research', 1)
        self.assertEqual((data['pending'], data['missing'], data['source_changed'], data['deferred'], data['eligible']), (2, 1, 2, 1, 1))
        self.assertEqual(len(data['sample']), 1)


class BudgetTests(unittest.TestCase):
    def test_day_boundary_and_no_invented_remaining(self):
        from ops.usage_report import llm_today, budget_remaining
        self.assertIsNone(budget_remaining(None, 1))
        self.assertIsNone(budget_remaining(10, None))
        self.assertEqual(budget_remaining(10, 3), 7)
        policy = {'daily_budget_usd': 10, 'daily_budget_per_provider': {}}
        with patch('llm.gateway.load_policy', return_value=policy), patch('ops.usage_report.http_json', return_value={'spend': {'deepseek': 2}}), patch('ops.usage_report.utc_day', side_effect=['2026-10-05', '2026-10-06']):
            with self.assertRaises(RuntimeError):
                llm_today()

    def test_local_failure_never_zero_and_official_retained(self):
        from ops.usage_report import snapshot
        with patch('ops.usage_report.fetch_official', return_value={'status': 'ok', 'balances': [{'available': 42, 'currency': 'USD'}]}), patch('ops.usage_report.read_local_spend', return_value=({}, 'db unavailable')), patch('ops.usage_report.llm_today', side_effect=TimeoutError):
            data = asyncio.run(snapshot(7, 'deepseek'))
        self.assertTrue(data['partial'])
        self.assertNotIn('data', data['results']['local_estimates'])
        self.assertEqual(data['results']['official']['deepseek']['data']['balances'][0]['available'], 42)


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_all_five_absent_from_runtime_and_denied_outside_mcp_operator(self):
        from runtime_tools.registry import TOOLS
        from security_gateway.gateway import authorize
        from security_gateway.context import CallerContext
        from mcp_gateway.diagnostic_tools import DIAGNOSTIC_HANDLERS
        names = set(DIAGNOSTIC_HANDLERS)
        self.assertEqual(len(names), 5)
        self.assertFalse(names & {t['name'] for t in TOOLS})
        for name in names:
            for interface, agent, owner in [('webchat', None, False), ('telegram', 'orchestrator', True), ('mcp', 'inspect', True)]:
                self.assertTrue(authorize(CallerContext(interface=interface, agent_name=agent, is_owner=owner), name).denied)

    def test_observation_timeout_is_bounded(self):
        from ops.diagnostics import observe
        import time
        result = asyncio.run(observe(lambda: time.sleep(.02), timeout=.001))
        self.assertEqual(result['collection_status'], 'error')
        self.assertIn('TimeoutError', result['reason'])

    def test_credential_missing_and_secret_failure_masked(self):
        with patch('mcp_gateway.credentials.bootstrap_kg_credentials', return_value={'status': 'credential_missing', 'source': None}), patch.object(tools, '_run_project_command') as command:
            result = asyncio.run(tools.kg_integrity_check())
        self.assertIsInstance(result, ToolFailure)
        command.assert_not_called()
        with patch.dict(os.environ, {'NEO4J_PASSWORD': 'never-expose-this'}), patch.object(tools.subprocess, 'run', return_value=MagicMock(returncode=1, stdout='failure never-expose-this', stderr='')):
            self.assertNotIn('never-expose-this', tools._run_project_command(['dummy'], 10))

    def test_sql_guard_is_rejection_and_connect_failure_is_error(self):
        with patch.object(tools.subprocess, 'run', return_value=MagicMock(returncode=2, stdout='', stderr='query-db: multiple statements are not allowed')):
            with self.assertRaises(ToolRejection):
                asyncio.run(tools.readonly_query_db('SELECT 1; SELECT 2'))
        with patch.object(tools.subprocess, 'run', return_value=MagicMock(returncode=2, stdout='', stderr='psql: connection failed')):
            self.assertIsInstance(asyncio.run(tools.readonly_query_db('SELECT 1')), ToolFailure)

    def test_http_outage_is_observed_not_collection_failure(self):
        import httpx
        from ops.diagnostics import dependency_probe
        response = httpx.Response(503, request=httpx.Request('GET', 'http://localhost/health'))
        with patch('ops.diagnostics.http_json', side_effect=httpx.HTTPStatusError('not ready', request=response.request, response=response)):
            result = dependency_probe('embedding')
        self.assertEqual(result['status'], 'unhealthy')
        self.assertNotIn('collection_status', result)


class CompletenessTests(unittest.TestCase):
    def test_logs_output_bound_and_schema_ranges(self):
        from mcp_gateway.diagnostic_tools import get_service_logs
        with patch('ops.logs.fetch_server_logs', return_value=[{'raw': 'x' * 5000}] * 1000):
            data = json.loads(asyncio.run(get_service_logs('api', limit=1000, profile='operator')))
        self.assertTrue(data['results']['journal']['data']['truncated'])
        self.assertLess(len(json.dumps(data)), 62000)
        with patch('security_gateway.audit.audit') as audit:
            for name, args in [('usage_and_budget_report', {'days': 31}), ('pipeline_status', {'limit': 0}),
                               ('get_service_logs', {'service': '../passwd'}), ('service_health_snapshot', {'services': ['unlisted']})]:
                self.assertTrue(asyncio.run(_call_tool(name, args, 'operator'))['isError'])
                self.assertEqual(audit.call_args.kwargs['result_status'], 'rejected')

    def test_web_today_separate_and_global_under_filter(self):
        from ops.usage_report import web_usage
        response = {'since_utc': '2026-10-05', 'daily_budget_usd': 10, 'accounted_usd': 3,
                    'rows': [{'provider': 'tavily', 'accounted_usd': 2}, {'provider': 'brave', 'accounted_usd': 1}]}
        with patch('web_gateway.client.usage', return_value=response), patch('ops.usage_report.utc_day', return_value='2026-10-05'):
            today = web_usage(1, today=True, provider='brave')
            self.assertEqual(today['remaining_usd'], 7)
            period = web_usage(7, provider='brave')
            self.assertEqual(period['accounted_usd'], 1)

    def test_cli_failed_local_read_is_null(self):
        from ops.llm_balances import collect
        with patch('ops.llm_balances.read_local_spend', return_value=({}, 'unavailable')), patch('ops.llm_balances.fetch_official', return_value={'status': 'unsupported'}):
            result = collect('http://localhost', 7)
        self.assertIsNone(next(row for row in result['providers'] if row['provider'] == 'deepseek')['local_audit'])


class InvalidArgumentTests(unittest.TestCase):
    def test_nonobject_arguments_are_audited_rejections(self):
        from mcp_gateway.server import handle
        with patch('security_gateway.audit.audit') as audit:
            response = asyncio.run(handle({'id': 1, 'method': 'tools/call', 'params': {'name': 'gateway_status', 'arguments': []}}, 'inspect'))
        self.assertTrue(response['result']['isError'])
        self.assertEqual(audit.call_args.kwargs['result_status'], 'rejected')

    def test_legacy_local_ranges_reject(self):
        with self.assertRaises(ToolRejection):
            asyncio.run(tools.search_dev_docs('MCP', limit=-1))
        with self.assertRaises(ToolRejection):
            asyncio.run(tools.get_task_status(1, field='invalid'))


class ConfigFilterTests(unittest.TestCase):
    def test_surface_filters_related_config_and_agents(self):
        from mcp_gateway.diagnostic_tools import get_effective_runtime_config
        data = json.loads(asyncio.run(get_effective_runtime_config(surface='webchat', profile='operator')))['results']['config']['data']
        self.assertEqual(set(data['surfaces']), {'webchat'})
        self.assertEqual(data['agents'], [])
        self.assertNotIn('task_budget', data['runtime_config'])
