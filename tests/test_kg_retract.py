"""Append-only fact retraction for agents (kg_runtime.writes.retract_kg_fact)."""
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from kg_runtime import writes


class _Rec(dict):
    pass


class _Result:
    def __init__(self, rows):
        self._rows = [_Rec(r) for r in rows]

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _Session:
    def __init__(self, candidates):
        self.candidates = candidates
        self.retracted = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        if cypher is writes.CYPHER_RETRACT_CANDIDATES:
            rows = [r for r in self.candidates if r["uuid"].startswith(params["edge_id"])]
            return _Result(rows)
        if cypher is writes.CYPHER_RETRACT:
            self.retracted.append(params)
            return _Result([{"uuid": params["uuid"]}])
        raise AssertionError(cypher)


def _edge(uuid, sync_key=None):
    return {"uuid": uuid, "subject": "국민의힘", "object": "DMZ 지뢰 폭발 사건",
            "fact": "국민의힘은 결의안을 발의했다.", "sync_key": sync_key, "valid_at": None}


class RetractTests(unittest.TestCase):
    def _call(self, candidates, **kwargs):
        session = _Session(candidates)

        @contextmanager
        def driver():
            yield type("D", (), {"session": lambda self, **kw: session})(), "neo4j"

        args = {"subject_name": "국민의힘", "predicate": "Involvement", "object_name": "DMZ 지뢰 폭발 사건",
                "reason": "원출처상 발의 주체는 더불어민주당이다 (뉴스핌)", "agent": "analyst"}
        args.update(kwargs)
        with patch("kg_runtime.search._get_neo4j_sync_driver", driver):
            return writes.retract_kg_fact(**args), session

    def test_single_match_is_expired_with_reason(self):
        result, session = self._call([_edge("7c557d67-aaaa")])
        self.assertEqual(result["status"], "retracted")
        self.assertEqual(session.retracted[0]["agent"], "analyst")
        self.assertIn("더불어민주당", session.retracted[0]["reason"])

    def test_ambiguous_lists_candidates_then_edge_id_selects(self):
        edges = [_edge("7c557d67-aaaa"), _edge("b2b188cc-bbbb")]
        result, session = self._call(edges)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual([c["edge_id"] for c in result["candidates"]], ["7c557d67", "b2b188cc"])
        self.assertFalse(session.retracted)
        result, session = self._call(edges, edge_id="b2b188cc")
        self.assertEqual(result["status"], "retracted")
        self.assertEqual(session.retracted[0]["uuid"], "b2b188cc-bbbb")

    def test_source_owned_fact_and_thin_reason_are_refused(self):
        result, session = self._call([_edge("c0ffee00-cccc", sync_key="commulingo:term_person:x")])
        self.assertEqual(result["status"], "refused")
        self.assertFalse(session.retracted)
        result, _ = self._call([_edge("7c557d67-aaaa")], reason="wrong")
        self.assertEqual(result["status"], "refused")

    def test_no_match(self):
        result, _ = self._call([])
        self.assertEqual(result["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
