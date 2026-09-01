"""라틴 문자 저본: 인물사전 성(姓) 단독 항목은 전체 이름·이니셜이 문서에 있을 때만 주입."""
import unittest

from runtime_tools.archival_translation.core import _pattern, anchor_latin_people


def _entry(fam_en, ko, full_en, given_en):
    return {"ru": fam_en, "ko": ko, "pattern": _pattern([fam_en], True),
            "person": {"full_en": full_en, "given_en": given_en, "family_en": fam_en}}


HESSEN = _entry("Hessen", "게센", "Boris Hessen", "Boris")
CLAY = _entry("Clay", "클레이", "Lucius D. Clay", "Lucius D.")
BENJAMIN = _entry("Benjamin", "벤야민", "Walter Benjamin", "Walter")
EXTRA = {"ru": "Senat", "ko": "시정부", "pattern": _pattern(["Senat"], True)}


class LatinAnchor(unittest.TestCase):
    def test_surname_only_in_text_is_dropped_and_reported(self):
        kept, dropped = anchor_latin_people([HESSEN], "Die Länder Bayern, Hessen und Niedersachsen.")
        self.assertEqual(kept, [])
        self.assertEqual([g["ru"] for g in dropped], ["Hessen"])

    def test_full_name_anchors(self):
        kept, dropped = anchor_latin_people([HESSEN], "Boris Hessen spoke in London; Hessen argued...")
        self.assertEqual([g["ru"] for g in kept], ["Hessen"]); self.assertEqual(dropped, [])

    def test_initial_anchors(self):
        kept, _ = anchor_latin_people([CLAY], "From Clay to Rusk. L. Clay signed.")
        self.assertEqual([g["ru"] for g in kept], ["Clay"])

    def test_given_without_middle_anchors(self):
        kept, _ = anchor_latin_people([CLAY], "General Lucius Clay arrived in Berlin.")
        self.assertEqual([g["ru"] for g in kept], ["Clay"])

    def test_other_person_same_surname_does_not_anchor(self):
        # 힐데 베냐민이 있는 문서에 발터 벤야민 항목이 주입되면 안 된다
        kept, dropped = anchor_latin_people([BENJAMIN], "Genossin Hilde Benjamin, Minister der Justiz")
        self.assertEqual(kept, []); self.assertEqual([g["ru"] for g in dropped], ["Benjamin"])

    def test_absent_surname_is_silently_dropped(self):
        kept, dropped = anchor_latin_people([HESSEN], "Nothing about him here.")
        self.assertEqual(kept, []); self.assertEqual(dropped, [])

    def test_extra_and_terms_pass_through(self):
        kept, dropped = anchor_latin_people([EXTRA], "Der Senat von Berlin")
        self.assertEqual(kept, [EXTRA]); self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
