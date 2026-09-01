"""저본 페이지 디코딩 — 선언된 charset을 따른다 (cp1251 외 라틴 인코딩 포함)."""
import unittest

from runtime_tools.archival_translation.core import _decode_page


class DecodePage(unittest.TestCase):
    def test_declared_cp1252_keeps_umlauts(self):
        html = '<meta charset="windows-1252"><p>Führer Weisung für Übung</p>'.encode("cp1252")
        self.assertIn("Führer Weisung für Übung", _decode_page(html))

    def test_declared_iso_8859_1(self):
        html = '<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"><p>Grundgesetz für die Bundesrepublik</p>'.encode("latin-1")
        self.assertIn("für", _decode_page(html))

    def test_declared_cp1251_still_works(self):
        html = '<meta charset="windows-1251"><p>Приказ</p>'.encode("cp1251")
        self.assertIn("Приказ", _decode_page(html))

    def test_undeclared_utf8(self):
        self.assertIn("Приказ", _decode_page("<p>Приказ</p>".encode("utf-8")))

    def test_undeclared_falls_back_to_cp1251(self):
        self.assertIn("Приказ", _decode_page("<p>Приказ</p>".encode("cp1251")))

    def test_declared_latin1_but_utf8_bytes_is_read_as_utf8(self):
        # marxists.org: charset=iso-8859-1 선언, 실제는 UTF-8 (2026-09-01 회귀)
        html = '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"><p>Истекший год был годом великого перелома</p>'.encode("utf-8")
        self.assertIn("великого перелома", _decode_page(html))

    def test_unknown_declared_charset_ignored(self):
        self.assertIn("ok", _decode_page(b'<meta charset="x-nonsense"><p>ok</p>'))


if __name__ == "__main__":
    unittest.main()
