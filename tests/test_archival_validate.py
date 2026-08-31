"""사료 번역 검증기 — 응답 끊김 검사와 캐시 재심사."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_tools.archival_translation.core import (
    Cache, Options, RUSSIAN, Stats, _cached_blocks, _translate_chunk, validate,
)

_SRC = ("Смысл плана Дауэса состоит в том, что Германия должна выплатить Антанте "
        "около 130 миллиардов золотых марок в разные сроки. В этой части план стоит на глиняных ногах.")
_FULL = ["도스 플랜의 의미는 독일이 앙탕트에 약 1300억 금 마르크를 여러 시기에 걸쳐 지불해야 한다는 데 있습니다. "
         "이 부분에서 그 계획은 진흙 발로 서 있습니다."]
_CUT = ["도스 플랜의 의미는 독일이 앙탕트에 약 1300억 금 마르크를 여러 시기에 걸쳐 지불해야 한다는 데 있습니다. "
        "그것은 독일을 평정하기 위해 만들어졌"]


def _chunk(*blocks):
    return [(10 + i, {"tag": "p", "lines": [text]}) for i, text in enumerate(blocks)]


class Truncation(unittest.TestCase):
    def test_cut_last_block_is_flagged(self):
        problems = validate(_chunk(_SRC), {10: _CUT}, RUSSIAN)
        self.assertEqual(len(problems), 1)
        self.assertIn("응답이 도중에 끊김", problems[0])
        self.assertIn("만들어졌", problems[0])

    def test_complete_last_block_passes(self):
        self.assertEqual(validate(_chunk(_SRC), {10: _FULL}, RUSSIAN), [])
        # 닫는 괄호·따옴표·쪽 번호로 끝나는 번역도 문장이 끝난 것이다
        for tail in ["(박수.)", "말했다. [c.358]", "“인터내셔널”", "다음과 같다:"]:
            self.assertEqual(validate(_chunk(_SRC), {10: [_FULL[0] + " " + tail]}, RUSSIAN), [])

    def test_only_the_last_block_of_the_chunk_is_checked(self):
        # 앞 블록에서 끊기면 뒤 마커가 빠져 그쪽 검사가 잡는다. 앞 블록의 줄이
        # 문장부호 없이 끝나는 것(목록 항목, 제목)은 끊김이 아니다.
        chunk = _chunk(_SRC, _SRC)
        got = {10: ["나) 그들 중 덜 활동적인 자들로서 5년에서 10년의 기간으로 수감되어야 할 제2범주"],
               11: _FULL}
        self.assertEqual(validate(chunk, got, RUSSIAN), [])

    def test_source_without_terminal_punctuation_is_exempt(self):
        src = _SRC.rstrip(".")
        self.assertEqual(validate(_chunk(src), {10: _CUT}, RUSSIAN), [])

    def test_short_source_is_exempt(self):
        self.assertEqual(validate(_chunk("Приказ народного комиссара."), {10: ["인민위원의 명령"]}, RUSSIAN), [])

    def test_trailing_empty_block_does_not_hide_the_check(self):
        chunk = _chunk(_SRC) + [(11, {"tag": "table", "lines": []})]
        self.assertEqual(len(validate(chunk, {10: _CUT}, RUSSIAN)), 1)


class CacheRevalidation(unittest.TestCase):
    def test_stale_cached_chunk_is_retranslated_and_good_one_reused(self):
        chunk = _chunk(_SRC)
        with tempfile.TemporaryDirectory() as d:
            cache = Cache(Path(d) / "c.jsonl")
            cache.put("k", {10: _CUT}, {"attempt": 1})
            blocks, problems = _cached_blocks(cache, "k", chunk, RUSSIAN)
            self.assertIsNone(blocks)
            self.assertIn("끊김", problems[0])
            self.assertEqual(_cached_blocks(cache, "missing", chunk, RUSSIAN), (None, []))

            events, stats = [], Stats()
            with mock.patch("llm.call_registry.generate_sync",
                            return_value="[[10|p]]\n" + _FULL[0]) as gen, \
                    mock.patch("time.sleep", lambda *_: None):
                got = _translate_chunk(chunk, [], cache, Options(), stats, events.append,
                                       RUSSIAN, prepared=("prompt", "k"))
            self.assertEqual(got, {10: _FULL})
            self.assertEqual(gen.call_count, 1)
            self.assertEqual([e["event"] for e in events], ["cacheInvalid"])
            self.assertEqual((stats.revalidated, stats.translated, stats.cached), (1, 1, 0))
            # 새 레코드가 같은 키를 덮어써서 다음 실행은 캐시 적중
            self.assertEqual(_cached_blocks(cache, "k", chunk, RUSSIAN), ({10: _FULL}, []))
            reloaded = Cache(Path(d) / "c.jsonl")
            self.assertEqual(_cached_blocks(reloaded, "k", chunk, RUSSIAN), ({10: _FULL}, []))
            with mock.patch("llm.call_registry.generate_sync") as gen:
                got = _translate_chunk(chunk, [], reloaded, Options(), Stats(), events.append,
                                       RUSSIAN, prepared=("prompt", "k"))
            self.assertEqual(got, {10: _FULL})
            gen.assert_not_called()



class StrayCyrillicSkipsEditorial(unittest.TestCase):
    def test_bibliography_original_title_is_not_a_hole(self):
        from runtime_tools.archival_translation.core import stray_cyrillic
        html = ('<article><h1>제목</h1><aside class="doc-editorial"><p class="doc-editorial-label">엮은이 주</p>'
                '<ul><li>원제: Приказ Народного комиссара обороны СССР № 227</li></ul></aside>'
                '<p>본문에 남은 Ставка 한 낱말.</p><p>병기는 괜찮다(Жуков).</p></article>')
        self.assertEqual(stray_cyrillic(html), ["Ставка"])


class GlossaryPatternBoundary(unittest.TestCase):
    def test_every_alternative_is_bounded(self):
        from runtime_tools.archival_translation.core import _pattern, _variants
        pat = _pattern(_variants("Горбач"))
        self.assertIsNone(pat.search("Горбачев"))      # 전치격 'Горбаче'가 안에서 걸리면 안 된다
        self.assertIsNone(pat.search("М. Горбачевым"))
        self.assertEqual(pat.search("о Горбаче").group(0), "Горбаче")
        self.assertEqual(pat.search("Горбача").group(0), "Горбача")
        adj = _pattern(_variants("Октябрьский"))
        self.assertEqual(adj.search("Октябрьского").group(0), "Октябрьского")
        self.assertIsNone(adj.search("Октябрьскими"))  # 'Октябрьским'+'и': 뒤에 키릴이 붙으면 경계가 아니다


if __name__ == "__main__":
    unittest.main()
