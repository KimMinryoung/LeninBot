#!/usr/bin/env python3
"""Explicit schema migration runner for LeninBot-owned tables.

The project still has legacy startup paths that call idempotent schema setup
functions. This runner gives operators a single command to apply those DDL
blocks before service startup, which makes it possible to remove runtime DDL
from services incrementally.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _telegram_core() -> None:
    from telegram.schema import ensure_telegram_tables

    ensure_telegram_tables()


def _telegram_summaries() -> None:
    from telegram.schema import ensure_summary_tables

    ensure_summary_tables({})


def _roleplay_tables() -> None:
    from telegram.schema import ensure_roleplay_tables

    ensure_roleplay_tables()


def _research_documents() -> None:
    from research_store import ensure_research_table

    ensure_research_table()


def _publication_records() -> None:
    from publication_records import ensure_publish_record_table

    ensure_publish_record_table()


def _site_publishing() -> None:
    from site_publishing import _ensure_hub_table, _ensure_static_page_table

    _ensure_hub_table()
    _ensure_static_page_table()


def _experiential_memory() -> None:
    from experience_writer import _ensure_table

    _ensure_table()


def _autonomous_projects() -> None:
    from autonomous_project import _ensure_tables

    _ensure_tables()


def _x402_ledger() -> None:
    from crypto_wallet.x402_ledger import ensure_x402_ledger_table

    ensure_x402_ledger_table()


MIGRATIONS: list[tuple[str, Callable[[], None]]] = [
    ("telegram-core", _telegram_core),
    ("telegram-summaries", _telegram_summaries),
    ("roleplay-tables", _roleplay_tables),
    ("research-documents", _research_documents),
    ("publication-records", _publication_records),
    ("site-publishing", _site_publishing),
    ("experiential-memory", _experiential_memory),
    ("autonomous-projects", _autonomous_projects),
    ("x402-ledger", _x402_ledger),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply LeninBot schema migrations.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available migrations without applying them.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=[name for name, _ in MIGRATIONS],
        help="Apply only the named migration. Can be provided more than once.",
    )
    args = parser.parse_args()

    selected = args.only or [name for name, _ in MIGRATIONS]
    if args.list:
        for name, _ in MIGRATIONS:
            marker = "*" if name in selected else "-"
            print(f"{marker} {name}")
        return 0

    for name, fn in MIGRATIONS:
        if name not in selected:
            continue
        print(f"applying {name}")
        fn()
    print("schema migrations ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
