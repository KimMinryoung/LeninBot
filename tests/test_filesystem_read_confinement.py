"""Read tools stay inside the project/frontend/nginx roots and skip secrets."""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from runtime_tools import filesystem


class ReadConfinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.project = os.path.join(self.root, "project")
        self.outside = os.path.join(self.root, "outside")
        os.makedirs(os.path.join(self.project, "data"))
        os.makedirs(self.outside)
        with open(os.path.join(self.project, "data", "note.txt"), "w") as f:
            f.write("hello\n")
        with open(os.path.join(self.project, ".env"), "w") as f:
            f.write("SECRET=1\n")
        with open(os.path.join(self.outside, "id_rsa"), "w") as f:
            f.write("private\n")
        os.symlink(os.path.join(self.outside, "id_rsa"), os.path.join(self.project, "data", "link"))
        for target, value in (("_project_root", lambda: self.project), ("_extra_read_roots", lambda: ())):
            patcher = patch.object(filesystem, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def read(self, path):
        return str(asyncio.run(filesystem._exec_read_file(path)))

    def test_project_file_is_readable(self):
        self.assertIn("hello", self.read("data/note.txt"))

    def test_outside_absolute_path_is_denied(self):
        self.assertIn("Read denied", self.read(os.path.join(self.outside, "id_rsa")))

    def test_symlink_escape_is_denied(self):
        self.assertIn("Read denied", self.read("data/link"))

    def test_env_file_is_denied(self):
        self.assertIn("secret files", self.read(".env"))

    def test_device_file_is_never_opened(self):
        # /dev is outside every root; even a root-relative device path must not block.
        self.assertIn("Read denied", self.read("/dev/zero"))

    def test_list_directory_outside_is_denied(self):
        out = str(asyncio.run(filesystem._exec_list_directory(self.outside)))
        self.assertIn("Read denied", out)


if __name__ == "__main__":
    unittest.main()
