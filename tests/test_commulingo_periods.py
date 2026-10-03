import unittest

from commulingo.periods import PERIOD_COLUMNS, career_period_text, format_period, period_columns

# Same cases as frontend scripts/smoke-commulingo-structured-periods.js.
SHOWN = [
    ({"start": [1917], "end": [1918]}, "1917–1918", "1917–1918"),
    ({"start": [1918, 7], "end": [1918, 9]}, "1918.07–09", "1918.07–09"),
    ({"start": [1918, 9], "end": [1919, 7]}, "1918.09–1919.07", "1918.09–1919.07"),
    ({"start": [1965, 3, 18], "end": [1965, 3, 19]}, "1965.03.18–19", "1965.03.18–19"),
    ({"start": [1938, 10, 25]}, "1938.10.25", "1938.10.25"),
    ({"start": [1920], "startQual": "decade", "end": [1930], "endQual": "decade"}, "1920년대–1930년대", "1920s–1930s"),
    ({"start": [1920], "startQual": "late"}, "1920년대 후반", "late 1920s"),
    ({"start": [1990], "startQual": "circa", "end": [1991]}, "1990년경–1991", "c. 1990–1991"),
    ({"start": [1989], "startQual": "after"}, "1989 이후", "after 1989"),
    ({"end": [1968], "endQual": "until"}, "–1968", "–1968"),
    ({"start": [2017], "ongoing": True}, "2017–현재", "2017–present"),
    ({"ongoing": True}, "현재", "present"),
    ({"start": [1945], "endQual": "open"}, "1945–", "1945–"),
    ({"start": [1979], "endQual": "unknown"}, "1979–?", "1979–?"),
    ({"start": [1956], "startQual": "summer"}, "1956 여름", "summer 1956"),
    ({"start": [1963], "label": {"ko": "1963 또는 1964", "en": "1963 or 1964"}}, "1963 또는 1964", "1963 or 1964"),
]
BAD = ["1917–1918", None, {}, {"start": 1917}, {"start": [1917, 13]}, {"start": [1917, 2, 30]},
       {"start": [1920], "end": [1910]}, {"start": [1923], "startQual": "decade"},
       {"start": [1917], "startQual": "until"}, {"start": [1917], "end": [1918], "endQual": "open"},
       {"start": [1917], "ongoing": True, "end": [1918]}, {"start": [1917], "label": {"ko": "1917"}},
       {"start": [1917], "extra": 1}]


class PeriodTest(unittest.TestCase):
    def test_format_matches_frontend(self):
        for period, ko, en in SHOWN:
            row = period_columns(period)
            self.assertEqual(set(row), set(PERIOD_COLUMNS))
            self.assertEqual(format_period(row, "ko"), ko)
            self.assertEqual(format_period(row, "en"), en)

    def test_rejects_strings_and_contradictions(self):
        for bad in BAD:
            with self.assertRaises(ValueError, msg=repr(bad)):
                period_columns(bad)

    def test_career_text_from_store_or_patch(self):
        self.assertEqual(career_period_text({"y": {"ko": "1917–1918", "en": "1917–1918"}}), "1917–1918")
        self.assertEqual(career_period_text({"period": {"start": [2017], "ongoing": True}}, "en"), "2017–present")


if __name__ == "__main__":
    unittest.main()
