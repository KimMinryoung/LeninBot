"""Sandbox checks on written Python files never run the file's code."""
import os
import stat
import tempfile
import unittest

from self_runtime import self_modification_core as core


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.marker = os.path.join(self.tmp.name, "ran")

    def write(self, name, text):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_module_code_is_not_executed(self):
        path = self.write("payload.py", f"open({self.marker!r}, 'w').write('x')\n")
        # A neighbouring test file must not run either without approval.
        self.write("test_payload.py", f"open({self.marker!r}, 'w').write('t')\n")
        result = core.run_sandbox_tests(path)
        self.assertEqual(result.status, "pass")
        self.assertFalse(os.path.exists(self.marker))

    def test_undefined_name_fails_statically(self):
        path = self.write("broken.py", "def f():\n    return missing_name\n")
        result = core.run_sandbox_tests(path)
        self.assertEqual(result.status, "fail")

    def test_patch_keeps_mode_and_allows_non_python(self):
        path = self.write("run.sh", "#!/bin/sh\necho old\n")
        os.chmod(path, 0o755)
        patch = core.generate_line_patch("#!/bin/sh\necho old\n", "#!/bin/sh\necho new — {not python\n")
        self.assertTrue(core.apply_line_patch_safe(path, patch))
        self.assertTrue(os.stat(path).st_mode & stat.S_IXUSR)
        with open(path, encoding="utf-8") as f:
            self.assertIn("echo new", f.read())


if __name__ == "__main__":
    unittest.main()
