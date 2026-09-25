"""Checked-in configs must not pin host paths; tokens resolve per checkout."""
import unittest

from ops.paths import FRONTEND_DIR, PROJECT_ROOT, expand_path_tokens
from translation_runtime.archival import SPEC_DIR


class ExpandPathTokensTest(unittest.TestCase):
    def test_leading_tokens_resolve_recursively(self):
        spec = {"output": "${FRONTEND_DIR}/data/x.html",
                "documents": [{"source": {"path": "${PROJECT_ROOT}/docs/a.html"}}],
                "note": "mentions ${FRONTEND_DIR} mid-text", "n": 3}
        self.assertEqual(expand_path_tokens(spec), {
            "output": f"{FRONTEND_DIR}/data/x.html",
            "documents": [{"source": {"path": f"{PROJECT_ROOT}/docs/a.html"}}],
            "note": "mentions ${FRONTEND_DIR} mid-text", "n": 3})

    def test_token_must_be_a_whole_path_segment(self):
        self.assertEqual(expand_path_tokens("${FRONTEND_DIR}x/y"), "${FRONTEND_DIR}x/y")


class ArchivalSpecPathsTest(unittest.TestCase):
    def test_specs_use_tokens_not_host_paths(self):
        offenders = [p.name for p in sorted(SPEC_DIR.glob("*.json"))
                     if '"/home/' in p.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [], "write ${FRONTEND_DIR}/… or ${PROJECT_ROOT}/…")


if __name__ == "__main__":
    unittest.main()
