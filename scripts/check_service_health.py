#!/usr/bin/env python3
"""Alert when an enabled long-running leninbot service is down or the API stops answering.

On 2026-09-23 a failed start of the Docker DB stack cancelled the start jobs
of leninbot-api and leninbot-a2a-api. systemd never retried them, and web chat
was down for two days: the off-VM watchdog only sees the frontend, which kept
returning 200. This check covers that gap from inside the VM.

A problem is reported once it persists across two consecutive runs (so a
deploy restart does not alert), and again when it clears. State lives in
data/service_health_state.json. Exit 0 when healthy, 1 otherwise. Scheduled by
leninbot-service-health.timer.
"""

import json
import os
import subprocess
import sys
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _notify import notify_telegram as _notify_telegram  # noqa: E402

STATE_PATH = ROOT / "data" / "service_health_state.json"
API_HEALTH_URL = os.environ.get("API_HEALTH_URL", "http://172.17.0.1:8000/health")


def _enabled_services() -> list[str]:
    out = subprocess.run(
        ["systemctl", "list-unit-files", "leninbot-*.service", "--state=enabled",
         "--no-legend", "--no-pager"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    return sorted(line.split()[0] for line in out.splitlines() if line.strip())


def _inactive(units: list[str]) -> dict[str, str]:
    if not units:
        return {}
    states = subprocess.run(
        ["systemctl", "is-active", *units], capture_output=True, text=True, timeout=30,
    ).stdout.split()
    return {unit: f"{unit} is {state}" for unit, state in zip(units, states) if state != "active"}


def _api_problem() -> str | None:
    try:
        with urllib.request.urlopen(API_HEALTH_URL, timeout=10) as resp:
            if resp.status == 200:
                return None
            return f"API {API_HEALTH_URL} returned {resp.status}"
    except Exception as e:
        return f"API {API_HEALTH_URL} unreachable ({e})"


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"pending": [], "alerted": []}


def main() -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notify", action="store_true",
                        help="send a Telegram message when a problem starts or clears")
    args = parser.parse_args()

    # Keyed by unit name (or "api") so a changing state or error text still
    # counts as the same problem across runs.
    problems = _inactive(_enabled_services())
    api = _api_problem()
    if api:
        problems["api"] = api

    state = _load_state()
    # Report only what was also seen on the previous run.
    confirmed = sorted(k for k in problems if k in state["pending"] or k in state["alerted"])
    new = [k for k in confirmed if k not in state["alerted"]]
    cleared = [k for k in state["alerted"] if k not in problems]

    if args.notify and new:
        _notify_telegram("🔴 leninbot 서비스 이상\n\n" + "\n".join(f"• {problems[k]}" for k in new))
    if args.notify and cleared:
        _notify_telegram("🟢 leninbot 서비스 복구\n\n" + "\n".join(f"• {k}" for k in cleared))

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"pending": sorted(problems), "alerted": confirmed}, indent=2))

    for message in problems.values():
        print(f"PROBLEM: {message}")
    print("SERVICES UNHEALTHY" if problems else "SERVICES OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
