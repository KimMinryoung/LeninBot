"""Bounded, read-only operational observations. No alerts or state writes."""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from datetime import datetime, timezone

from security_gateway.redaction import redact_log_text

SERVICE_UNITS = {
    'api': 'leninbot-api.service', 'telegram': 'leninbot-telegram.service',
    'roleplay': 'leninbot-roleplay.service', 'writer': 'novel-writer-api.service',
    'email': 'leninbot-email-api.service', 'a2a': 'leninbot-a2a-api.service',
    'embedding': 'leninbot-embedding.service', 'llm_proxy': 'leninbot-llm-proxy.service',
    'web_gateway': 'leninbot-web-gateway.service', 'neo4j': 'leninbot-neo4j.service',
    'nginx': 'nginx.service', 'commulingo': 'leninbot-commulingo-pipeline.service',
    'translation': 'research-document-translation.service', 'kg_sync': 'leninbot-kg-sync.service',
}
DAEMONS = tuple(k for k in SERVICE_UNITS if k not in {'commulingo', 'translation', 'kg_sync'})
HEALTH_SERVICES = (*DAEMONS, 'postgresql', 'redis')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_text(value):
    text = str(value)
    # Exact locally known values also cover unlabelled exception messages.
    for key, secret in os.environ.items():
        if len(secret) >= 6 and re.search(r'PASSWORD|SECRET|TOKEN|API_KEY|CREDENTIAL', key):
            text = text.replace(secret, '«redacted»')
    text = re.sub(r'(?i)([a-z][a-z0-9+.-]*://[^\s/:]+:)[^\s/@]+@', r'\1«redacted»@', text)
    text = re.sub(
        r'''(?i)([\w.-]*(?:password|secret|api[_-]?key|token)[\w.-]*["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;}]+)''',
        r'\1«redacted»', text,
    )
    return redact_log_text(text)


def sanitize(value):
    """Preserve diagnostic field names (including token limits); mask text only."""
    if isinstance(value, dict):
        return {k: ('«redacted»' if re.search(r'(?i)password|secret|api[_-]?key|authorization|cookie|^token$|access[_-]?token', str(k)) else sanitize(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return safe_text(value) if isinstance(value, str) else value


def systemd_properties(unit, *, timer=False):
    fields = ['LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus',
              'ExecMainStartTimestamp', 'ExecMainExitTimestamp']
    if timer:
        fields += ['NextElapseUSecRealtime', 'NextElapseUSecMonotonic', 'LastTriggerUSec']
    proc = subprocess.run(['systemctl', 'show', unit, '--no-pager',
                           '--property=' + ','.join(fields)], capture_output=True, text=True, timeout=5)
    if proc.returncode:
        raise RuntimeError(safe_text(proc.stderr or 'systemctl failed'))
    values = dict(line.split('=', 1) for line in proc.stdout.splitlines() if '=' in line)
    if not values or values.get('LoadState') == 'not-found':
        raise RuntimeError('unit not installed')
    return values


def http_json(url):
    import httpx
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def dependency_probe(service):
    if service == 'postgresql':
        from ops.readonly_query import query as _query
        _query('SELECT 1 AS ok', ())
    elif service == 'redis':
        import redis
        with redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
                                  socket_connect_timeout=3, socket_timeout=3) as client:
            client.ping()
    elif service == 'neo4j':
        from mcp_gateway.credentials import bootstrap_kg_credentials
        state = bootstrap_kg_credentials()
        if state['status'] != 'configured':
            return {'status': 'unknown', 'collection_status': 'error', 'reason': state['status']}
        from neo4j import GraphDatabase, Query
        from secrets_loader import get_secret
        with GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                auth=(os.getenv('NEO4J_USER', 'neo4j'), get_secret('NEO4J_PASSWORD')),
                connection_timeout=3, connection_acquisition_timeout=5,
                max_transaction_retry_time=0) as driver:
            with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
                session.run(Query('RETURN 1 AS ok', timeout=3)).consume()
    else:
        urls = {'api': os.getenv('API_HEALTH_URL', 'http://172.17.0.1:8000/health'),
                'embedding': os.getenv('EMBEDDING_SERVER_URL', 'http://127.0.0.1:8100').rstrip('/') + '/health',
                'llm_proxy': 'http://127.0.0.1:8110/health', 'web_gateway': 'http://127.0.0.1:8111/health'}
        import httpx
        try:
            value = http_json(urls[service])
        except httpx.HTTPStatusError as exc:
            return {'status': 'unhealthy', 'reason': f'health endpoint HTTP {exc.response.status_code}'}
        return {'status': 'healthy' if value.get('status') in ('ok', 'healthy', 'ready') else 'unhealthy',
                'detail': value}
    return {'status': 'healthy'}


async def observe(collector, *, timeout=15):
    started = time.monotonic()
    try:
        value = await asyncio.wait_for(asyncio.to_thread(collector), timeout)
        return {'collection_status': 'ok', 'observed_at': utc_now(),
                'duration_ms': round((time.monotonic() - started) * 1000), 'data': sanitize(value)}
    except Exception as exc:
        return {'collection_status': 'error', 'observed_at': utc_now(),
                'duration_ms': round((time.monotonic() - started) * 1000),
                'reason': safe_text(f'{type(exc).__name__}: {exc}')}


def has_collection_error(value):
    if isinstance(value, dict):
        return value.get('collection_status') == 'error' or any(has_collection_error(v) for v in value.values())
    if isinstance(value, list):
        return any(has_collection_error(v) for v in value)
    return False


def report(parts, **extra):
    return {'observed_at': utc_now(), 'partial': has_collection_error(parts), **extra, 'results': parts}


async def health_snapshot(services):
    async def one(service):
        parts = {}
        if service in SERVICE_UNITS:
            parts['systemd'] = await observe(lambda: systemd_properties(SERVICE_UNITS[service]))
            if parts['systemd']['collection_status'] == 'ok':
                state = parts['systemd']['data']['ActiveState']
                parts['systemd']['status'] = 'healthy' if state == 'active' else 'unhealthy'
        if service in {'postgresql', 'redis', 'neo4j', 'api', 'embedding', 'llm_proxy', 'web_gateway'}:
            parts['probe'] = await observe(lambda: dependency_probe(service))
        return service, parts
    return report(dict(await asyncio.gather(*(one(s) for s in services))))
