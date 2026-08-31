#!/usr/bin/env python3
"""Show provider balances/official costs beside deduplicated local LLM spend.

DeepSeek and Kimi expose prepaid balance APIs.  OpenAI and Claude expose
organization cost reports that require optional admin credentials held only by
the local LLM proxy.  Providers without such an API still get the local
llm_audit_log estimate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
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
        return "balance " + _money(result.get("balances", []), "available")
    if status == "ok" and result.get("kind") == "cost":
        suffix = " (partial)" if result.get("has_more") else ""
        return f"cost {days}d " + _money(result.get("costs", []), "amount") + suffix
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
                "local_audit": local.get(provider, {"calls": 0, "spend_usd": 0.0}),
            }
            for provider in PROVIDERS
        ],
    }


def format_telegram_report(report: dict) -> str:
    """Render a compact, parse-mode-free balance report for Telegram."""
    days = int(report["window_days"])
    lines = [
        f"💳 LLM 잔액·비용 (최근 {days}일)",
        "공식 조회 / 중복 제거 로컬 감사 추정액",
    ]
    for row in report["providers"]:
        local = row["local_audit"]
        summary = official_summary(row["official"], days)
        lines.extend([
            "",
            f"{LABELS[row['provider']]} — {summary}",
            f"  로컬: ${float(local['spend_usd']):,.4f} / {int(local['calls']):,}회",
        ])
    if report.get("local_audit_error"):
        lines.extend(["", f"⚠️ 로컬 감사 조회 실패: {report['local_audit_error']}"])
    return "\n".join(lines)


def print_table(report: dict) -> None:
    days = report["window_days"]
    print(f"{'PROVIDER':<10} {'OFFICIAL':<45} {'LOCAL ' + str(days) + 'D':>14} {'CALLS':>9}")
    print(f"{'-' * 10} {'-' * 45} {'-' * 14} {'-' * 9}")
    for row in report["providers"]:
        local = row["local_audit"]
        print(
            f"{LABELS[row['provider']]:<10} "
            f"{official_summary(row['official'], days):<45} "
            f"${local['spend_usd']:>12,.4f} {local['calls']:>9,}"
        )
    if report.get("local_audit_error"):
        print(f"\nwarning: local audit unavailable: {report['local_audit_error']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--proxy", default="http://127.0.0.1:8110")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = collect(args.proxy, args.days)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
