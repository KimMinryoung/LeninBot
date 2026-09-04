"""Hermetic tests for the abbreviation normalizer used on episode bodies and
structured fact names (graph_memory.graphiti_patches.normalize_entity_names_in_text).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_memory import graphiti_patches as gp  # noqa: E402


class NameNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gp._build_name_normalization_regex()

    def test_uppercase_abbreviations_expand(self):
        self.assertEqual(gp.normalize_entity_names_in_text("The US and the UN met."), "The United States and the United Nations met.")
        self.assertEqual(gp.normalize_entity_names_in_text("U.S. tariffs"), "United States tariffs")
        self.assertEqual(gp.normalize_entity_names_in_text("WHO data"), "World Health Organization data")

    def test_lowercase_words_are_not_abbreviations(self):
        for text in ("Kim Jong Un", "telling an employee who asked", "let us go", "Trump nominated Warsh, who"):
            self.assertEqual(gp.normalize_entity_names_in_text(text), text)

    def test_long_keys_stay_case_insensitive(self):
        self.assertEqual(gp.normalize_entity_names_in_text("bichon said"), "관리자 비숑 said")
        self.assertEqual(gp.normalize_entity_names_in_text("비숑은 왔다"), "관리자 비숑은 왔다")


if __name__ == "__main__":
    unittest.main()
