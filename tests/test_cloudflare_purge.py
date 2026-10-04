import unittest

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


if __name__ == "__main__":
    unittest.main()
