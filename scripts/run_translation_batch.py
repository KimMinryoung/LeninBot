#!/usr/bin/env python3
"""Run both scheduled translation jobs and expose any failure to systemd."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    failed = False
    for script, args in (
        ('translate_research_documents.py', ['--limit', '0', '--max-chars', '0']),
        ('translate_db_content.py', ['--kind', 'all', '--limit', '0']),
    ):
        try:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args], cwd=ROOT)
            failed |= result.returncode != 0
        except OSError as exc:
            print(f'failed to start {script}: {exc}', file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
