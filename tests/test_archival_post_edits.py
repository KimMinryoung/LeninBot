"""postEdits 치환 뒤 조사 교정."""
import unittest

from runtime_tools.archival_translation.core import apply_post_edits, assemble


def _edit(text, **edits):
    return apply_post_edits([text], {"postEdits": edits})[0]


class ParticleRepair(unittest.TestCase):
    def test_vowel_to_coda_word_fixes_following_particle(self):
        e = dict(소비에트_소유즈="소비에트 연방")
        edits = {"소비에트 소유즈": "소비에트 연방"}
        cases = {
            "즉 소비에트 소유즈와 자본주의 세계": "즉 소비에트 연방과 자본주의 세계",
            "옛 러시아가 소비에트 소유즈로 바뀐 뒤": "옛 러시아가 소비에트 연방으로 바뀐 뒤",
            "즉 소비에트 소유즈가 서고": "즉 소비에트 연방이 서고",
            "소비에트 소유즈는 서방의": "소비에트 연방은 서방의",
            "소비에트 소유즈를 향해": "소비에트 연방을 향해",
            "소비에트 소유즈였다.": "소비에트 연방이었다.",
            "소비에트 소유즈라는 이름": "소비에트 연방이라는 이름",
            "소비에트 소유즈 안에서": "소비에트 연방 안에서",  # 조사 아님
            "소비에트 소유즈의 힘": "소비에트 연방의 힘",     # 받침 무관 조사
            "소비에트 소유즈에서": "소비에트 연방에서",
        }
        for src, want in cases.items():
            self.assertEqual(apply_post_edits([src], {"postEdits": edits})[0], want, src)

    def test_rieul_coda_keeps_ro(self):
        edits = {"소련": "서울"}
        self.assertEqual(_edit("소련로 간다", **edits), "서울로 간다")
        self.assertEqual(_edit("소련으로 간다", **edits), "서울로 간다")
        self.assertEqual(_edit("소련가 있다", **edits), "서울이 있다")

    def test_coda_to_vowel_word_fixes_particle_but_not_ida_forms(self):
        edits = {"라린": "루리예"}
        self.assertEqual(_edit("라린이 말했다", **edits), "루리예가 말했다")
        self.assertEqual(_edit("라린은 웃었다", **edits), "루리예는 웃었다")
        self.assertEqual(_edit("라린을 보고", **edits), "루리예를 보고")
        self.assertEqual(_edit("라린과 함께", **edits), "루리예와 함께")
        self.assertEqual(_edit("라린으로부터", **edits), "루리예로부터")
        # 모음 뒤 '이다/이라/이었'은 문어에서 허용 — 건드리지 않는다
        self.assertEqual(_edit("라린이다.", **edits), "루리예이다.")
        self.assertEqual(_edit("라린이라는", **edits), "루리예이라는")

    def test_correct_particles_and_non_hangul_untouched(self):
        self.assertEqual(_edit("라린이 왔다", 루리예="라린"), "라린이 왔다")
        self.assertEqual(_edit("루리예가 왔다", 루리예="라린"), "라린이 왔다")
        self.assertEqual(_edit("루리예 동지가", 루리예="라린"), "라린 동지가")
        # dst가 한글로 끝나지 않으면 조사 판정 없음
        self.assertEqual(_edit("XIV 대회가", XIV="XIV."), "XIV. 대회가")
        self.assertEqual(_edit("[c.300] 뒤", **{" [c.300]": ""}), "[c.300] 뒤")

    def test_sequential_edits_keep_dict_order(self):
        edits = {"루리예)의": "라린의", "루리예)": "라린", "루리예": "라린"}
        self.assertEqual(_edit("(루리예)가 자리에서", **edits), "(라린이 자리에서")
        self.assertEqual(_edit("루리예)의 계산", **edits), "라린의 계산")
        self.assertEqual(_edit("루리예 동지는", **edits), "라린 동지는")

    def test_assembler_uses_the_same_repair(self):
        spec = {"title": "t", "postEdits": {"소비에트 소유즈": "소비에트 연방"}}
        docs = [{"id": "d", "titleKo": "d", "heading": False, "offset": 0,
                 "blocks": [{"tag": "p", "lines": ["소비에트 소유즈와 세계"]}]}]
        html = assemble(spec, docs, {0: ["소비에트 소유즈와 세계"]})
        self.assertIn("소비에트 연방과 세계", html)


if __name__ == "__main__":
    unittest.main()
