#!/usr/bin/env python3
"""Offline checks for the retained CommuLingo helpers and current editor.

Default: shared author/storage contracts, source handling, decisions and draft
repair. --extended also exercises the editor, independent review, publication
binding and citation gate. Neither mode opens production storage or providers.
"""
import argparse
import faulthandler
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
os.environ['LENINBOT_LLM_AUDIT_DB'] = '0'

SMOKE = (
    'test_commulingo_write_contract',
    'test_commulingo_editor.SourceAndIssueTests',
    'test_commulingo_editor_decisions',
    'test_commulingo_draft_repair',
    'test_commulingo_classify',
)
EXTENDED = (
    'test_commulingo_editor.EditorTests',
    'test_commulingo_editor.ReviewAndPublishTests',
    'test_commulingo_citation_gate',
    'test_commulingo_review_policy',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--real-threads', action='store_true',
                        help='exercise the real asyncio executor; requires working cross-thread event-loop wakeups')
    parser.add_argument('--timeout', type=int, default=60, help='whole-suite deadline in seconds, with a traceback on expiry')
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    os.environ['COMMULINGO_TEST_REAL_THREADS'] = '1' if args.real_threads else '0'
    from commulingo_test_support import no_external_io
    faulthandler.dump_traceback_later(args.timeout, exit=True)
    try:
        with no_external_io():
            # Schemas read the term category catalog at import. Use the built-in
            # fallback during collection even if the caller inherited credentials.
            with patch('db.query', return_value=[]):
                suite = unittest.defaultTestLoader.loadTestsFromNames(SMOKE + (EXTENDED if args.extended else ()))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    raise SystemExit(main())
