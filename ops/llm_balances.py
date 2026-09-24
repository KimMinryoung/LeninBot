"""Show official balances/costs, with estimates only for providers without cost reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx

ROOT = Path(__file__).resolve().parents[1]
ESTIMATED_PROVIDERS = {"deepseek", "kimi", "gemini", "local"}
PROVIDERS = ("deepseek", "kimi", "openai", "claude", "gemini", "local")
LABELS = {
    "deepseek": "DeepSeek",
    "kimi": "Kimi",
    "openai": "OpenAI",
    "claude": "Claude",
    "gemini": "Gemini",
    "local": "Local",
}

# The default instrumented SDK callers historically duplicated the loop's
# authoritative billed row.  New requests suppress that duplicate through the
# audit-owner context, but excluding these wrapper identities keeps a 30-day
# estimate correct across the migration boundary.  Feature-specific direct SDK
# callers (for example kg_graphiti) remain included.
_WRAPPER_CALLERS = (
    "anthropic_client",
    "deepseek_anthropic_direct",
    "deepseek_client",
    "kimi_anthropic_client",
    "kimi_client",
    "openai_client",
)


def local_spend_sql(days: int) -> str:
    quoted = ", ".join("'" + item + "'" for item in _WRAPPER_CALLERS)
    sql = f"""
SELECT CASE
         WHEN provider IN ('anthropic', 'claude') THEN 'claude'
         WHEN provider IN ('moonshot', 'kimi') THEN 'kimi'
         ELSE COALESCE(provider, 'unknown')
       END AS provider,
       COUNT(*) AS calls,
       ROUND(COALESCE(SUM(cost_usd), 0)::numeric, 8) AS spend_usd
  FROM llm_audit_log
 WHERE ts >= now() - make_interval(days => {int(days)})
   AND provider NOT IN ('claude', 'anthropic', 'openai')
   AND status = 'ok'
   AND cost_usd IS NOT NULL
   AND surface <> 'proxy'
   AND NOT (surface = 'external_sdk' AND caller IN ({quoted}))
 GROUP BY 1
 ORDER BY 1
""".strip()
    # scripts/query-db validates the first word with a line-oriented awk
    # expression, so pass one statement on one line.
    return " ".join(sql.split())


def parse_query_db_tsv(stdout: str) -> dict[str, dict]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    header_index = next(
        (i for i, line in enumerate(lines) if line.split("\t") == ["provider", "calls", "spend_usd"]),
        None,
    )
    if header_index is None:
        raise ValueError("query-db output did not contain the expected header")
    rows: dict[str, dict] = {}
    for line in lines[header_index + 1:]:
        if line.startswith("(") or "\t" not in line:
            continue
        provider, calls, spend = line.split("\t", 2)
        rows[provider] = {"calls": int(calls), "spend_usd": float(spend)}
    return rows


def read_local_spend(days: int) -> tuple[dict[str, dict], str | None]:
    sql = local_spend_sql(days)
    # Service callers (including Telegram) already have DB credentials and can
    # use the shared pool without spawning psql.  The CLI fallback remains
    # useful for operator shells, where scripts/query-db resolves credentials.
    try:
        from db import query

        rows = query(sql)
        return {
            str(row["provider"]): {
                "calls": int(row["calls"]),
                "spend_usd": float(row["spend_usd"]),
            }
            for row in rows
        }, None
    except Exception:
        pass

    try:
        result = subprocess.run(
            [str(ROOT / "scripts/query-db"), sql],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {}, e.__class__.__name__
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return {}, detail[-1][:160] if detail else f"query-db exit {result.returncode}"
    try:
        return parse_query_db_tsv(result.stdout), None
    except (TypeError, ValueError) as e:
        return {}, str(e)


def fetch_official(proxy_base: str, provider: str, days: int) -> dict:
    if provider not in {"deepseek", "kimi", "openai", "claude"}:
        return {"provider": provider, "status": "unsupported"}
    url = f"{proxy_base.rstrip('/')}/billing/{provider}?days={days}"
    try:
        # Ignore HTTP(S)_PROXY for the loopback gateway.  urllib inherited the
        # host proxy settings and made a healthy local service look unavailable.
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.get(url)
        payload = response.json()
        if response.status_code >= 400:
            return {
                "provider": provider,
                "status": payload.get("status", "http_error") if isinstance(payload, dict) else "http_error",
                "error": payload.get("error", f"HTTP {response.status_code}") if isinstance(payload, dict) else f"HTTP {response.status_code}",
            }
        return payload if isinstance(payload, dict) else {
            "provider": provider, "status": "invalid_response",
        }
    except (httpx.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as e:
        return {"provider": provider, "status": "unavailable", "error": e.__class__.__name__}


def _money(items: list[dict], key: str) -> str:
    if not items:
        return "-"
    return ", ".join(
        f"{item.get('currency', 'USD')} {float(item.get(key) or 0):,.4f}"
        for item in items
    )


def official_summary(result: dict, days: int) -> str:
    status = result.get("status")
    if status == "ok" and result.get("kind") == "balance":
        return "공식 잔액: " + _money(result.get("balances", []), "available")
    if status == "ok" and result.get("kind") == "cost":
        suffix = " (일부 결과)" if result.get("has_more") else ""
        return f"공식 사용 비용 ({days}일): " + _money(result.get("costs", []), "amount") + suffix
    if status == "credential_missing":
        return f"admin key missing ({result.get('required_credential', '?')})"
    if status == "unsupported":
        return "no official adapter"
    return f"{status or 'unavailable'}"


def collect(proxy_base: str, days: int) -> dict:
    local, local_error = read_local_spend(days)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(
            lambda provider: fetch_official(proxy_base, provider, days),
            PROVIDERS,
        )
        official = dict(zip(PROVIDERS, results))
    return {
        "window_days": days,
        "proxy": proxy_base,
        "local_audit_error": local_error,
        "providers": [
            {
                "provider": provider,
                "official": official[provider],
                **({"local_audit": local.get(provider, {"calls": 0, "spend_usd": 0.0})}
                   if provider in ESTIMATED_PROVIDERS else {}),
            }
            for provider in PROVIDERS
        ],
    }


def estimated_summary(row: dict, report: dict) -> str | None:
    if row["provider"] not in ESTIMATED_PROVIDERS:
        return None
    if report.get("local_audit_error") or "local_audit" not in row:
        return "사용 비용 추정: 조회 실패"
    local = row["local_audit"]
    return (f"사용 비용 추정 ({report['window_days']}일, 봇 기록): "
            f"${float(local['spend_usd']):,.4f} / {int(local['calls']):,}회")


def format_telegram_report(report: dict) -> str:
    """Render a compact, parse-mode-free balance report for Telegram."""
    days = int(report["window_days"])
    lines = [
        f"💳 LLM 잔액·비용 (최근 {days}일)",
        "잔액은 남은 금액, 사용 비용은 조회 기간에 쓴 금액입니다.",
    ]
    for row in report["providers"]:
        summary = official_summary(row["official"], days)
        lines.extend([
            "",
            f"{LABELS[row['provider']]} — {summary}",
        ])
        estimate = estimated_summary(row, report)
        if estimate:
            lines.append("  " + estimate)
        if row["provider"] == "openai":
            lines.extend([
                "  크레딧 잔액: 대시보드에서 확인 (현재 자동 조회 미지원)",
                "  https://platform.openai.com/settings/organization/billing/overview",
            ])
    return "\n".join(lines)
