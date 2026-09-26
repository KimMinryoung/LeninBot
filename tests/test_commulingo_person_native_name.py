import unittest

from commulingo.people import _check_person_native_names


class NativeNameRequiredTests(unittest.TestCase):
    """An empty native name has no script to check, so it must be rejected on its own."""

    def test_create_without_native_name_is_rejected(self):
        for value in (None, "", "   "):
            patch = {"citizenship": {"code": "finland"}}
            if value is not None:
                patch["cyrillic"] = value
            error, _ = _check_person_native_names(None, "create", "susanne-dahlgren", patch)
            self.assertIn("cyrillic is required", error or "")

    def test_create_with_native_name_passes(self):
        error, _ = _check_person_native_names(
            None, "create", "susanne-dahlgren", {"cyrillic": "Susanne Dahlgren", "citizenship": {"code": "finland"}})
        self.assertIsNone(error)

    def test_hangul_with_hanja_gloss_passes(self):
        error, _ = _check_person_native_names(
            None, "create", "choe-chang-ik", {"cyrillic": "최창익 (崔昌益)", "citizenship": {"code": "north-korea"}})
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
