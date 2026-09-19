"""SourcePages must not grow a URL's snapshot when a retry re-seeds stored merges.

2026-09-19: job 2523 re-seeded every attempt's merged snapshot onto the next
(23 MB → 95 MB → 453 MB per attempt); the sixth attempt was OOM-killed at
9 GB and the host locked up once before that. These tests replay that shape.
"""
import unittest
from datetime import datetime, timedelta, timezone

from commulingo_pipeline import evidence
from commulingo_pipeline.evidence import MAX_SNAPSHOT_CHARS, SourcePages, snapshot

URL = "https://en.wikipedia.org/wiki/Alexei_Kosygin"
T0 = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def _attempt(sources, pages, start):
    """One research attempt: seed from stored sources, then fetch pages; store
    everything the stage would store (each page and each new merge)."""
    sp = SourcePages()
    for merged in sp.seed(sources):
        sources[merged["id"]] = merged
    for i, body in enumerate(pages):
        page = snapshot(URL, body, now=start + timedelta(seconds=i))
        sources[page["id"]] = page
        merged, span, created = sp.absorb(URL, body)
        if created:
            merged["fetched_at"] = start + timedelta(seconds=i, microseconds=1)
            sources[merged["id"]] = merged
    return sp, sources


class SourcePagesRetryTests(unittest.TestCase):
    def test_reseeding_stored_merges_does_not_grow(self):
        chunks = [f"chunk {i} " + ("x" * 1000) for i in range(3)]
        sources = {}
        sp, sources = _attempt(sources, chunks, T0)
        size_after_first = len(sp.current[URL]["body"])
        self.assertEqual(size_after_first, sum(len(c) for c in chunks) + 2)
        for n in range(1, 6):
            sp, sources = _attempt(sources, chunks, T0 + timedelta(minutes=n))
            self.assertEqual(len(sp.current[URL]["body"]), size_after_first,
                             f"snapshot grew on retry {n}")
        # Stored snapshots: 3 pages + 2 intermediate merges, never more.
        self.assertEqual(len(sources), 5)

    def test_seed_returns_nothing_when_merge_already_stored(self):
        sources = {}
        sp, sources = _attempt(sources, ["a" * 50, "b" * 50], T0)
        self.assertEqual(SourcePages().seed(sources), [])

    def test_seed_adopts_largest_stored_merge_as_prefix_chain(self):
        a, b, c = "a" * 50, "b" * 50, "c" * 50
        chain = [snapshot(URL, a, now=T0), snapshot(URL, a + "\n" + b, now=T0 + timedelta(seconds=1)),
                 snapshot(URL, a + "\n" + b + "\n" + c, now=T0 + timedelta(seconds=2))]
        sources = {s["id"]: s for s in chain}
        sp = SourcePages()
        self.assertEqual(sp.seed(sources), [])
        self.assertEqual(sp.current[URL]["body"], a + "\n" + b + "\n" + c)
        # A page already inside the merged text is located, not appended.
        merged, span, created = sp.absorb(URL, b)
        self.assertFalse(created)
        self.assertEqual(merged["body"][span[0]:span[1]], b)

    def test_new_page_still_appends(self):
        sp = SourcePages()
        sp.absorb(URL, "first page")
        merged, span, created = sp.absorb(URL, "second page")
        self.assertTrue(created)
        self.assertEqual(merged["body"], "first page\nsecond page")
        self.assertEqual(merged["body"][span[0]:span[1]], "second page")

    def test_oversize_stored_snapshot_is_skipped(self):
        big = snapshot(URL, "z" * (MAX_SNAPSHOT_CHARS + 1), now=T0)
        small = snapshot(URL, "small page", now=T0 + timedelta(seconds=1))
        sp = SourcePages()
        with self.assertLogs(evidence.logger, level="WARNING"):
            sp.seed({big["id"]: big, small["id"]: small})
        self.assertEqual(sp.current[URL]["body"], "small page")

    def test_merge_beyond_cap_restarts_from_new_page(self):
        sp = SourcePages()
        sp.absorb(URL, "y" * (MAX_SNAPSHOT_CHARS - 10))
        with self.assertLogs(evidence.logger, level="WARNING"):
            merged, span, created = sp.absorb(URL, "fresh page text")
        self.assertTrue(created)
        self.assertEqual(merged["body"], "fresh page text")
        self.assertEqual(span, (0, len("fresh page text")))


if __name__ == "__main__":
    unittest.main()
