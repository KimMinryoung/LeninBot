"""셀렉터 범위 범용 HTML 어댑터."""
import unittest

from runtime_tools.archival_translation.sources import generic_html, parse

PAGE = """<html><body><div id="nav"><ul><li>Главная</li><li>Поиск</li></ul></div>
<div id="content"><h2>Приказ № 1</h2><p>Первый абзац приказа.</p>
<div class="share">Поделиться</div>
<blockquote><p>Цитата внутри.</p></blockquote>
<ul><li>пункт а) один</li><li>пункт б) два</li></ul>
<table><tr><td>Область</td><td>Число</td></tr><tr><td>Москва</td><td>100</td></tr></table>
<p>Подпись<br/>И. СТАЛИН</p></div>
<div id="footer"><p>© сайт</p></div></body></html>"""

LOOSE = """<html><body><td class="text">ПОСТАНОВЛЕНИЕ<br><br>Первый абзац текста документа, довольно длинный,
чтобы перевесить.<br><br>Второй абзац текста документа, тоже довольно длинный для теста.<br><br>
Подпись<br>Керенский</td></body></html>"""


class GenericHtml(unittest.TestCase):
    def test_scoped_leaf_blocks_in_order_with_drop(self):
        blocks = generic_html(PAGE, selector="#content", drop=[".share"])
        self.assertEqual([b["tag"] for b in blocks], ["h3", "p", "p", "li", "li", "table", "p"])
        self.assertEqual(blocks[0]["lines"], ["Приказ № 1"])
        self.assertEqual(blocks[2]["lines"], ["Цитата внутри."])
        self.assertEqual(blocks[5]["rows"], [["Область", "Число"], ["Москва", "100"]])
        self.assertEqual(blocks[5]["lines"], ["Область", "Число", "Москва"])  # 숫자는 모델에 안 간다
        self.assertEqual(blocks[6]["lines"], ["Подпись", "И. СТАЛИН"])
        text = " ".join(ln for b in blocks for ln in b["lines"])
        self.assertNotIn("Поделиться", text)
        self.assertNotIn("Главная", text)
        self.assertNotIn("© сайт", text)

    def test_loose_br_text_is_split_on_blank_lines(self):
        blocks = generic_html(LOOSE, selector="td.text")
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0]["lines"], ["ПОСТАНОВЛЕНИЕ"])
        self.assertEqual(blocks[3]["lines"], ["Подпись", "Керенский"])

    def test_raw_newlines_inside_a_paragraph_are_not_line_breaks(self):
        page = '<div id="c"><p>Первая строка абзаца,\n    вторая строка того же абзаца.</p><p>Подпись<br>Ленин</p></div>'
        blocks = generic_html(page, selector="#c")
        self.assertEqual(blocks[0]["lines"], ["Первая строка абзаца, вторая строка того же абзаца."])
        self.assertEqual(blocks[1]["lines"], ["Подпись", "Ленин"])

    def test_nth_match_selects_the_second_cell(self):
        page = '<table><tr><td class="c">Меню</td><td class="c"><p>Текст документа.</p></td></tr></table>'
        self.assertEqual(generic_html(page, selector="td.c", nth=1)[0]["lines"], ["Текст документа."])
        self.assertEqual(parse({"format": "html", "selector": "td.c", "nth": 1}, page)[0]["lines"], ["Текст документа."])
        with self.assertRaises(ValueError):
            generic_html(page, selector="td.c", nth=2)

    def test_missing_selector_is_an_error(self):
        with self.assertRaises(ValueError):
            generic_html(PAGE, selector="#nope")
        with self.assertRaises(KeyError):
            parse({"format": "html"}, PAGE)

    def test_parse_dispatches_named_adapters_unchanged(self):
        # wikisource 등 기존 어댑터는 selector 없이 그대로 동작한다 (빈 페이지 → 빈 목록)
        self.assertEqual(parse({"format": "wikisource"}, "<html></html>"), [])
        self.assertEqual(parse({"format": "html", "selector": "#content"}, PAGE)[0]["lines"], ["Приказ № 1"])


class DecodePage(unittest.TestCase):
    def test_declared_cp1251_and_fallbacks(self):
        from runtime_tools.archival_translation.core import _decode_page
        ru = "Постановление"
        cp = ('<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1251"></head>'
              '<body>' + ru + '</body></html>').encode("cp1251")
        self.assertIn(ru, _decode_page(cp))
        self.assertIn(ru, _decode_page(("<html><body>" + ru + "</body></html>").encode("utf-8")))
        # 선언 없는 cp1251도 UTF-8 실패 뒤 복구된다
        self.assertIn(ru, _decode_page(("<html><body>" + ru + "</body></html>").encode("cp1251")))


if __name__ == "__main__":
    unittest.main()
