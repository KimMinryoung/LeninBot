"""Regression checks for staged gap writes; no LLM or production credentials."""
import ast
from pathlib import Path
import re
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/commulingo_gap_worker.py'


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'hold_pending_review']
    ns = {'re': re, 'db_query_one': Mock(), 'close_gap': Mock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class GapReview(unittest.TestCase):
    def test_pending_links_gap_without_marking_created(self):
        ns = load_functions()
        ns['db_query_one'].return_value = {'id': 9567, 'target_id': 'gjergj-kokoshi'}
        result = ns['hold_pending_review']({'id': 1845}, 'OK — pending: Logged as edit #9567. Pending review; no content changed.')
        self.assertEqual(result, {'status': 'pending_review', 'suggestionId': 9567})
        ns['close_gap'].assert_called_once_with(1845, 'pending', 'gjergj-kokoshi', 'awaiting review #9567')

    def test_approved_result_is_not_held(self):
        ns = load_functions()
        self.assertIsNone(ns['hold_pending_review']({'id': 1}, 'OK — approved: Logged as edit #1.'))
        ns['db_query_one'].assert_not_called()
        ns['close_gap'].assert_not_called()

    def test_review_completed_during_write_returns_to_normal_lookup(self):
        ns = load_functions()
        ns['db_query_one'].return_value = None
        self.assertIsNone(ns['hold_pending_review']({'id': 1}, 'OK — pending: Logged as edit #1.'))
        ns['close_gap'].assert_not_called()

    def test_malformed_pending_does_not_silently_requeue(self):
        ns = load_functions()
        with self.assertRaises(RuntimeError):
            ns['hold_pending_review']({'id': 1}, 'OK — pending: missing id')
        ns['close_gap'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
