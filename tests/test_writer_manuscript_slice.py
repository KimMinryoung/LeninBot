"""read_manuscript offsets: an end-only read returns the 5000 chars before end."""
import unittest
from unittest.mock import patch

from writer import store


class SliceTests(unittest.TestCase):
    def read(self, start, end, total=100_000):
        calls = []

        def fake(sql, params):
            calls.append(params)
            return {"total": total} if "length(body)" in sql else {"piece": "x"}

        with patch.object(store, "db_query_one", side_effect=fake):
            header = store.read_manuscript_slice(1, start, end).splitlines()[0]
        return header, calls

    def test_end_only_reads_the_chars_before_end(self):
        header, calls = self.read(None, 3000)
        self.assertIn("chars 0–3000", header)
        self.assertEqual(calls[1][:2], (1, 3000))

    def test_no_offsets_reads_the_tail(self):
        header, _ = self.read(None, None)
        self.assertIn("chars 95000–100000", header)

    def test_start_only_reads_5000(self):
        header, _ = self.read(10_000, None)
        self.assertIn("chars 10000–15000", header)


if __name__ == "__main__":
    unittest.main()
