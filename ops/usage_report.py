"""Official billing, local estimates and today's budget are distinct observations."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ops.diagnostics import http_json, observe, report
from ops.llm_balances import PROVIDERS, ESTIMATED_PROVIDERS, fetch_official, read_local_spend

BILLING_PROVIDERS = (*PROVIDERS, 'tavily', 'brave')
PROXY = 'http://127.0.0.1:8110'


def budget_remaining(cap, spent):
    return None if cap is None or spent is None else max(0.0, round(float(cap) - float(spent), 8))


def utc_day():
    return datetime.now(timezone.utc).date().isoformat()


def llm_today():
    from llm.gateway import load_policy
    day = utc_day()
    policy = load_policy()
    spend = http_json(PROXY + '/audit/spend/today')['spend']
    if day != utc_day():
        raise RuntimeError('UTC day changed during observation; retry for a consistent budget day')
    total = sum(float(v) for v in spend.values())
    cap = policy['daily_budget_usd']
    return {'day_utc': day, 'accounted_usd': total, 'spend_by_provider': spend,
            'daily_budget_usd': cap, 'remaining_usd': budget_remaining(cap, total),
            'provider_budgets': {p: {'daily_budget_usd': c, 'accounted_usd': spend.get(p, 0.0),
                                    'remaining_usd': budget_remaining(c, spend.get(p, 0.0))}
                                 for p, c in policy['daily_budget_per_provider'].items()},
            'budget_source': 'llm.gateway.load_policy: defaults < config/llm_gateway.defaults.json < config/llm_gateway.local.json',
            'running_service_policy': 'unknown', 'budget_scope': 'all providers, never provider-filtered',
            'spend_source': 'running LLM proxy /audit/spend/today (budget accounting, not invoice)',
            'remaining_basis': 'calculated using current-process policy; null when uncapped or unknown'}


def web_usage(days, *, today=False, provider=None):
    from web_gateway.client import usage
    day = utc_day()
    data = usage(days=days, by='service')
    if data.get('error'):
        raise RuntimeError(data.get('message', 'web usage unavailable'))
    if today and (day != utc_day() or data.get('since_utc') != day):
        raise RuntimeError('web usage day does not match current UTC budget day')
    if today:
        return {'day_utc': day, 'daily_budget_usd': data['daily_budget_usd'],
                'accounted_usd': data['accounted_usd'],
                'remaining_usd': budget_remaining(data['daily_budget_usd'], data['accounted_usd']),
                'budget_scope': 'all web providers including outstanding reservations',
                'source': 'running web gateway /usage?days=1; local estimate, not invoice'}
    if provider:
        data['rows'] = [r for r in data['rows'] if r.get('provider') == provider]
        data['accounted_usd'] = sum(r['accounted_usd'] for r in data['rows'])
    data['window'] = 'UTC calendar days including today'
    return data


async def snapshot(days, provider):
    selected = [p for p in PROVIDERS if not provider or p == provider]
    async def official(p):
        row = await observe(lambda: fetch_official(PROXY, p, days), timeout=25)
        if row.get('data', {}).get('status') not in ('ok', 'unsupported'):
            row['collection_status'] = 'error'
            row['reason'] = row.get('data', {}).get('status', row.get('reason', 'unavailable'))
        if row.get('data', {}).get('has_more'):
            row['collection_status'] = 'error'
            row['reason'] = 'provider returned incomplete cost pagination (has_more)'
        return p, row
    parts = {}
    if selected:
        parts['official'] = dict(await asyncio.gather(*(official(p) for p in selected)))
        if set(selected) & ESTIMATED_PROVIDERS:
            def local():
                rows, error = read_local_spend(days)
                if error:
                    raise RuntimeError(error)
                return {'window': f'rolling {days} days', 'estimate_only': True,
                        'providers': {p: rows.get(p, {'calls': 0, 'spend_usd': 0.0})
                                      for p in selected if p in ESTIMATED_PROVIDERS}}
            parts['local_estimates'] = await observe(local, timeout=40)
        parts['llm_today'] = await observe(llm_today)
    if not provider or provider in ('brave', 'tavily'):
        period, today = await asyncio.gather(observe(lambda: web_usage(days, provider=provider)),
                                            observe(lambda: web_usage(1, today=True)))
        parts['web_period'], parts['web_today'] = period, today
    return report(parts, days=days, provider=provider,
                  note='Official costs/balances retain provider units and windows. Never infer official balances from local spend.')
