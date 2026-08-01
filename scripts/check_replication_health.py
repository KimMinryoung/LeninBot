#!/usr/bin/env python3
"""Check the leninbot-standby streaming replica from the primary's side.

Everything here is read from the primary: pg_stat_replication carries the LSNs
the standby itself reported, so a single vantage point is enough and the check
does not need credentials for the standby host.

Four things can go wrong, in rising order of how bad they are:

  1. lag in bytes  — the standby is behind but keeping up
  2. lag in time   — the standby stopped applying
  3. no connection — the walreceiver is gone; the slot now pins WAL on the
     primary and `max_slot_wal_keep_size` is the only thing bounding it
  4. slot lost     — retention was exceeded and the slot was invalidated. The
     standby can no longer catch up and must be re-seeded from a new base
     backup. The primary is safe; this is the failure the cap is designed to
     produce instead of a full disk.

Exit 0 when healthy, 1 otherwise. Scheduled by
leninbot-replication-health.timer.
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONTAINER = "leninbot-pg"
SLOT = "standby_hel1"
EXPECTED_CLIENT = "100.124.58.85"


def _notify_telegram(message: str) -> bool:
    """Send `message` to the configured Telegram chat (stale-secrets pattern)."""
    import os
    import urllib.parse
    import urllib.request

    try:
        from secrets_loader import get_secret
    except Exception as exc:
        print(f"WARNING: cannot import secrets_loader ({exc}); skipping notify", file=sys.stderr)
        return False
    token = get_secret("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping notify",
              file=sys.stderr)
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        print(f"WARNING: telegram notify failed: {exc}", file=sys.stderr)
        return False


def _query(sql: str) -> list[list[str]]:
    """Run SQL on the primary through the container's own psql."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-tAF", "\x1f", "-c", sql],
        capture_output=True, check=True, timeout=30,
    )
    return [line.split("\x1f") for line in result.stdout.decode().splitlines() if line]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag-bytes", type=int, default=128 * 1024 * 1024,
                        help="replay lag in bytes above which the replica is unhealthy")
    parser.add_argument("--max-lag-seconds", type=int, default=300,
                        help="replay lag in seconds above which the replica is unhealthy")
    parser.add_argument("--notify", action="store_true",
                        help="Notify Telegram on failure/degradation.")
    args = parser.parse_args()

    problems: list[str] = []
    lines: list[str] = []

    # --- slot: checked first, because a lost slot explains a missing connection
    slots = _query(
        "SELECT slot_name, active, coalesce(wal_status,'?'), "
        "coalesce(pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)),'0') "
        f"FROM pg_replication_slots WHERE slot_name = '{SLOT}';"
    )
    if not slots:
        problems.append(f"replication slot '{SLOT}' does not exist")
    else:
        name, active, wal_status, retained = slots[0]
        lines.append(f"slot {name}: active={active} wal_status={wal_status} retained_wal={retained}")
        if wal_status in ("lost", "unreserved"):
            problems.append(
                f"slot '{name}' wal_status={wal_status} — retention exceeded, the standby "
                "cannot catch up and needs a fresh pg_basebackup"
            )
        if active != "t":
            problems.append(f"slot '{name}' is inactive — the standby is not connected")

    # --- connection and lag
    rows = _query(
        "SELECT coalesce(host(client_addr),'?'), state, sync_state, "
        "coalesce(pg_wal_lsn_diff(sent_lsn, replay_lsn)::bigint::text,'-1'), "
        "coalesce(extract(epoch from replay_lag)::numeric(10,1)::text,'0') "
        "FROM pg_stat_replication;"
    )
    if not rows:
        problems.append("no walreceiver connected (pg_stat_replication is empty)")
    else:
        for client, state, sync_state, lag_bytes_s, lag_secs_s in rows:
            lag_bytes, lag_secs = int(lag_bytes_s), float(lag_secs_s)
            lines.append(
                f"replica {client}: state={state} sync={sync_state} "
                f"replay_lag={lag_bytes}B / {lag_secs}s"
            )
            if client != EXPECTED_CLIENT:
                problems.append(f"unexpected replica client_addr {client} (expected {EXPECTED_CLIENT})")
            if state != "streaming":
                problems.append(f"replica {client} state={state}, expected streaming")
            if lag_bytes > args.max_lag_bytes:
                problems.append(
                    f"replica {client} replay lag {lag_bytes}B exceeds {args.max_lag_bytes}B"
                )
            if lag_secs > args.max_lag_seconds:
                problems.append(
                    f"replica {client} replay lag {lag_secs}s exceeds {args.max_lag_seconds}s"
                )

    for line in lines:
        print(line)

    if problems:
        print("\nREPLICATION UNHEALTHY:")
        for problem in problems:
            print(f"  - {problem}")
        if args.notify:
            _notify_telegram(
                "🔴 Postgres 복제 이상 (leninbot-standby)\n\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\n"
                + "\n".join(lines)
            )
        return 1

    print("\nREPLICATION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
