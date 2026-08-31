"""generic_html 표 블록의 번역 어휘 — 글자가 든 칸은 문자 체계와 무관하게 낸다."""
import unittest

from runtime_tools.archival_translation.sources import generic_html


class TableVocabulary(unittest.TestCase):
    HTML = """<div id="d"><p>Deficits:</p>
    <table><tr><th>Country</th><th>$ millions</th></tr>
    <tr><td>England</td><td>1,250</td></tr>
    <tr><td>Франция</td><td>800</td></tr></table></div>"""

    def test_latin_and_cyrillic_cells_become_vocab_numbers_do_not(self):
        blocks = generic_html(self.HTML, selector="div#d")
        table = next(b for b in blocks if b["tag"] == "table")
        self.assertEqual(table["lines"], ["Country", "$ millions", "England", "Франция"])
        self.assertEqual(len(table["rows"]), 3)


if __name__ == "__main__":
    unittest.main()


class LooseTextFallback(unittest.TestCase):
    def test_raw_line_wraps_are_joined_but_blank_lines_and_br_split(self):
        html = ("<div id='d'>First line of a\nwrapped paragraph.\n\n"
                "Second paragraph<br>with a hard break.</div>")
        blocks = generic_html(html, selector="div#d")
        self.assertEqual([b["lines"] for b in blocks],
                         [["First line of a wrapped paragraph."],
                          ["Second paragraph", "with a hard break."]])
