"""Operator-only diagnostic adapters. Not registered on any runtime surface."""
from __future__ import annotations

import json

from mcp_gateway.audit_tools import _integer, _operator
from ops.diagnostics import HEALTH_SERVICES, SERVICE_UNITS, health_snapshot, observe, report
from tool_gateway.results import ToolFailure, ToolRejection


def _result(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return ToolFailure(text) if value.get('partial') else text


async def service_health_snapshot(services=None, *, profile='inspect'):
    _operator(profile)
    if services is not None and (not isinstance(services, list) or not services or any(s not in HEALTH_SERVICES for s in services)):
        raise ToolRejection('services must be a nonempty list of allowed service names')
    return _result(await health_snapshot(services or HEALTH_SERVICES))


async def get_service_logs(service, hours_back=1, level=None, query='', limit=200, *, profile='inspect'):
    _operator(profile)
    _integer(hours_back, 'hours_back', 1, 168)
    _integer(limit, 'limit', 1, 1000)
    if service not in SERVICE_UNITS:
        raise ToolRejection('unknown service')
    if level not in (None, 'debug', 'info', 'notice', 'warning', 'error', 'critical'):
        raise ToolRejection('unknown log level')
    if not isinstance(query, str) or len(query) > 500:
        raise ToolRejection('query must be a string of at most 500 characters')
    from ops.logs import fetch_server_logs
    def collect():
        rows = fetch_server_logs(service, hours_back, query, limit, level=level)
        if rows and rows[0].get('error'):
            raise RuntimeError(rows[0]['error'])
        output, size = [], 0
        for row in rows:
            size += len(json.dumps(row, ensure_ascii=False))
            if size > 60000:
                break
            output.append(row)
        return {'rows': output, 'truncated': len(output) < len(rows),
                'filter_scope': 'level is maximum journal priority; query searches latest limit rows',
                'limit_reached': len(rows) == limit}
    return _result(report({'journal': await observe(collect, timeout=20)}, service=service))


async def get_effective_runtime_config(surface=None, agent=None, *, profile='inspect'):
    _operator(profile)
    if surface is not None and surface not in ('telegram_chat', 'delegated_task', 'autonomous', 'webchat', 'writer'):
        raise ToolRejection('unknown surface')
    from ops.model_runtime import build_snapshot
    from agents import agent_names
    if agent is not None and agent not in agent_names():
        raise ToolRejection('unknown agent')
    def collect():
        data = build_snapshot()
        if surface:
            data['surfaces'] = {k: v for k, v in data['surfaces'].items() if k == surface}
            if surface != 'writer':
                data['writer_call_policies'] = []
            if surface != 'delegated_task':
                data['agents'] = [row for row in data['agents'] if surface == 'autonomous' and row['agent'] == 'autonomous_project']
            keys = {
                'telegram_chat': {'provider', 'chat_model', 'chat_budget', 'max_rounds_chat'},
                'delegated_task': {'provider', 'task_provider', 'task_model', 'task_budget', 'max_rounds_task'},
                'autonomous': {'provider', 'autonomous_provider', 'autonomous_model', 'task_budget', 'max_rounds_task'},
                'webchat': {'webchat_provider', 'webchat_model', 'webchat_budget'},
                'writer': set(),
            }[surface]
            data['runtime_config'] = {k: v for k, v in data['runtime_config'].items() if k in keys}
        if agent:
            data['agents'] = [row for row in data['agents'] if row['agent'] == agent]
        return data
    return _result(report({'config': await observe(collect)}))


def schema(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}


def integer(low, high):
    return {'type': 'integer', 'minimum': low, 'maximum': high}


DIAGNOSTIC_TOOLS = [
    {'name': 'service_health_snapshot', 'description': 'Operator-only read-only dependency health. Collection failures differ from observed outages.',
     'input_schema': schema({'services': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string', 'enum': list(HEALTH_SERVICES)}}})},
    {'name': 'get_service_logs', 'description': 'Operator-only masked journal tail. No service actions. Query filters the latest limit rows.',
     'input_schema': schema({'service': {'type': 'string', 'enum': list(SERVICE_UNITS)}, 'hours_back': integer(1, 168),
                            'level': {'type': 'string', 'enum': ['debug', 'info', 'notice', 'warning', 'error', 'critical']},
                            'query': {'type': 'string', 'maxLength': 500}, 'limit': integer(1, 1000)}, ['service'])},
]
DIAGNOSTIC_HANDLERS = {name: globals()[name] for name in ('service_health_snapshot', 'get_service_logs')}

DIAGNOSTIC_TOOLS.append({'name': 'get_effective_runtime_config',
    'description': 'Operator-only allowlisted model, provider and execution limits. Live service application is unknown unless attested.',
    'input_schema': schema({'surface': {'type': 'string', 'enum': ['telegram_chat', 'delegated_task', 'autonomous', 'webchat', 'writer']},
                            'agent': {'type': 'string'}})})
DIAGNOSTIC_HANDLERS['get_effective_runtime_config'] = get_effective_runtime_config


async def pipeline_status(pipeline='all', limit=20, *, profile='inspect'):
    _operator(profile)
    _integer(limit, 'limit', 1, 100)
    from ops.pipeline_status import PIPELINES, snapshot
    if pipeline not in ('all', *PIPELINES):
        raise ToolRejection('unknown pipeline')
    return _result(await snapshot(pipeline, limit))


DIAGNOSTIC_TOOLS.append({'name': 'pipeline_status', 'description': 'Operator-only queues, stale leases, translation freshness and KG sync; excludes archival batches and manual static-page translation.',
    'input_schema': schema({'pipeline': {'type': 'string', 'enum': ['all', 'commulingo', 'translation', 'kg_sync']}, 'limit': integer(1, 100)})})
DIAGNOSTIC_HANDLERS['pipeline_status'] = pipeline_status


async def usage_and_budget_report(days=7, provider=None, *, profile='inspect'):
    _operator(profile)
    _integer(days, 'days', 1, 30)
    from ops.usage_report import BILLING_PROVIDERS, snapshot
    if provider is not None and provider not in BILLING_PROVIDERS:
        raise ToolRejection('unknown provider')
    return _result(await snapshot(days, provider))


DIAGNOSTIC_TOOLS.append({'name': 'usage_and_budget_report', 'description': 'Operator-only official billing, local estimates, and separate UTC today budgets; failures remain unknown.',
    'input_schema': schema({'days': integer(1, 30), 'provider': {'type': 'string', 'enum': ['deepseek', 'kimi', 'openai', 'claude', 'gemini', 'local', 'tavily', 'brave']}})})
DIAGNOSTIC_HANDLERS['usage_and_budget_report'] = usage_and_budget_report
