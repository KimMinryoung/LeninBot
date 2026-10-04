import os
import unittest
from unittest import mock

from publishing import cloudflare_purge
from publishing.cloudflare_purge import with_english_paths


class WithEnglishPathsTest(unittest.TestCase):
    def test_adds_english_copies_of_language_specific_paths(self):
        self.assertEqual(with_english_paths(["/post/3", "/", "/rss.xml"]), [
            "/post/3", "/en/post/3", "/", "/en/", "/rss.xml", "/en/rss.xml",
        ])

    def test_leaves_single_language_and_english_paths_alone(self):
        self.assertEqual(with_english_paths(["/sitemap.xml", "/en/posts", "https://x/y"]), [
            "/sitemap.xml", "/en/posts", "https://x/y",
        ])


class PurgeSwitchTest(unittest.TestCase):
    def test_off_by_default_skips_the_script(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(cloudflare_purge.subprocess, "run") as run:
            os.environ.pop("LENINBOT_CLOUDFLARE_PURGE", None)
            result = cloudflare_purge.purge_paths(["/post/3"], "post:3")
        run.assert_not_called()
        self.assertEqual(result["skipped"], True)
        self.assertEqual(result["urls"], ["/post/3", "/en/post/3"])

    def test_on_runs_the_script(self):
        done = mock.Mock(returncode=0, stdout="ok", stderr="")
        with mock.patch.dict(os.environ, {"LENINBOT_CLOUDFLARE_PURGE": "1"}), \
                mock.patch.object(cloudflare_purge.os.path, "isfile", return_value=True), \
                mock.patch.object(cloudflare_purge.subprocess, "run", return_value=done) as run:
            result = cloudflare_purge.purge_paths(["/post/3"], "post:3")
        run.assert_called_once()
        self.assertTrue(result["ok"])
        self.assertNotIn("skipped", result)


if __name__ == "__main__":
    unittest.main()
