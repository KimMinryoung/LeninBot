"""용어표 충돌 감지 — 남성 성의 생격 표면과 여성형 항목이 같이 주입되는 쌍."""
import unittest

from runtime_tools.archival_translation.core import glossary_collision_pairs


class GlossaryCollisionPairs(unittest.TestCase):
    def test_detects_masc_genitive_vs_fem_entry(self):
        gloss = [{"ru": "Ежов", "ko": "예조프"}, {"ru": "Ежова", "ko": "예조바"}]
        self.assertEqual(glossary_collision_pairs(gloss), [("Ежова", "Ежов")])

    def test_same_ko_alias_is_not_a_collision(self):
        # 의도적 생격 별칭(XXVII съезд/съезда → 같은 한국어)은 충돌이 아니다.
        gloss = [{"ru": "XXVII съезд", "ko": "제27차 당대회"},
                 {"ru": "XXVII съезда", "ko": "제27차 당대회"}]
        self.assertEqual(glossary_collision_pairs(gloss), [])

    def test_no_pair_without_both_entries(self):
        gloss = [{"ru": "Ежов", "ko": "예조프"}, {"ru": "Рыжова", "ko": "리조바"}]
        self.assertEqual(glossary_collision_pairs(gloss), [])

    def test_multiple_pairs_sorted(self):
        gloss = [{"ru": x, "ko": "표기" + x} for x in
                 ["Рыжов", "Рыжова", "Ежова", "Ежов", "Иванов"]]
        self.assertEqual(glossary_collision_pairs(gloss),
                         [("Ежова", "Ежов"), ("Рыжова", "Рыжов")])

    def test_excluded_entry_removes_pair(self):
        # plan()은 glossary.exclude 필터 뒤에 검사한다 — exclude가 경고를 끈다.
        gloss = [{"ru": "Ежов", "ko": "예조프"}]
        self.assertEqual(glossary_collision_pairs(gloss), [])


if __name__ == "__main__":
    unittest.main()
