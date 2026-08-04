"""Pin restart_service's import-preflight entry-point map to reality.

The old map pointed at flat telegram_bot.py / browser_worker.py modules that
stopped existing when the bots became packages; the isfile guard then silently
skipped import validation for those services. This test fails the moment an
entry in the map stops resolving to an actual file, so the safety net can't
rot silently again.

Run from repo root:  venv/bin/python -m unittest discover tests -v
"""
import os
import unittest

from runtime_tools.registry import RESTART_PREFLIGHT_ENTRY_POINTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRestartPreflightEntryPoints(unittest.TestCase):
    def test_covers_all_restartable_services(self):
        self.assertEqual(set(RESTART_PREFLIGHT_ENTRY_POINTS), {"telegram", "api", "browser"})

    def test_every_entry_module_file_exists(self):
        for service, module in RESTART_PREFLIGHT_ENTRY_POINTS.items():
            path = os.path.join(ROOT, module.replace(".", os.sep) + ".py")
            self.assertTrue(
                os.path.isfile(path),
                f"{service}: entry module {module} does not resolve to a file "
                f"({path}) — the restart preflight would silently skip it",
            )


if __name__ == "__main__":
    unittest.main()
