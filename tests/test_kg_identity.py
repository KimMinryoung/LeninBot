"""Hermetic tests for kg_runtime.identity (no Neo4j).

Covers alias-key normalization, identity-prop building, the deterministic
resolution order (external id → alias/name, same label required), the
in-process AliasIndex matcher, and the sync merge runner's Cypher sequence.
"""

import asyncio
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kg_runtime import identity  # noqa: E402


class _Rec(dict):
    """neo4j Record stand-in: dict with item access."""


class _Result:
    def __init__(self, rows):
        self._rows = [_Rec(r) for r in rows]

    def single(self):
        return self._rows[0] if self._rows else None

    def consume(self):
        return None

    def __iter__(self):
        return iter(self._rows)


class FakeSyncSession:
    """Routes each Cypher constant to canned rows and records the call order."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        for key, rows in self.routes.items():
            if cypher is key:
                return _Result(rows(params) if callable(rows) else rows)
        return _Result([])


class _AsyncResult:
    def __init__(self, rows):
        self._rows = [_Rec(r) for r in rows]

    async def single(self):
        return self._rows[0] if self._rows else None

    async def consume(self):
        return None

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class FakeAsyncSession(FakeSyncSession):
    async def run(self, cypher, **params):
        self.calls.append((cypher, params))
        for key, rows in self.routes.items():
            if cypher is key:
                return _AsyncResult(rows(params) if callable(rows) else rows)
        return _AsyncResult([])


class NormalizeTests(unittest.TestCase):
    def test_basic_normalization(self):
        self.assertEqual(identity.normalize_alias_key("  Donald  Trump "), "donald trump")
        self.assertEqual(identity.normalize_alias_key("Jean-Paul Sartre"), "jean paul sartre")
        self.assertEqual(identity.normalize_alias_key("U.S.A"), "united states")  # NAME_NORMALIZATION: "usa"
        self.assertEqual(identity.normalize_alias_key("Ｓａｍｓｕｎｇ"), "samsung")  # NFKC full-width

    def test_name_normalization_map_is_final_step(self):
        self.assertEqual(identity.normalize_alias_key("US"), "united states")
        self.assertEqual(identity.normalize_alias_key("United States"), "united states")
        self.assertEqual(identity.normalize_alias_key("ROK"), "south korea")

    def test_korean_kept(self):
        self.assertEqual(identity.normalize_alias_key("전국민주노동조합총연맹 (민주노총)"), "전국민주노동조합총연맹 민주노총")

    def test_empty_and_none(self):
        self.assertEqual(identity.normalize_alias_key(None), "")
        self.assertEqual(identity.normalize_alias_key("  "), "")

    def test_alias_keys_for_dedupes_and_flattens(self):
        keys = identity.alias_keys_for("Lenin", ["V. I. Lenin", "lenin", None], "레닌")
        self.assertEqual(keys, ["lenin", "v i lenin", "레닌"])


class WeakAliasTests(unittest.TestCase):
    def test_surname_is_weak(self):
        self.assertTrue(identity.is_weak_alias("카스트로", "피델 카스트로"))
        self.assertTrue(identity.is_weak_alias("Khrushchev", "니키타 흐루쇼프"))
        self.assertTrue(identity.is_weak_alias("Хрущёв", "니키타 흐루쇼프"))
        self.assertTrue(identity.is_weak_alias("레닌", "블라디미르 레닌"))
        self.assertFalse(identity.is_weak_alias("Nikita Khrushchev", "니키타 흐루쇼프"))
        self.assertFalse(identity.is_weak_alias("Никита Хрущёв", "니키타 흐루쇼프"))
        self.assertFalse(identity.is_weak_alias("Joseph Stalin", "Stalin"))
        self.assertTrue(identity.is_weak_alias("스탈린", "Stalin"))  # short single token
        self.assertFalse(identity.is_weak_alias("", "x"))

    def test_split_alias_keys(self):
        strong, weak = identity.split_alias_keys("피델 카스트로", ["Fidel Castro", "카스트로", "Castro"], None, "Fidel Castro")
        self.assertEqual(strong, ["피델 카스트로", "fidel castro"])
        self.assertEqual(weak, ["카스트로", "castro"])

    def test_lookup_ignores_incoming_weak_aliases(self):
        names, keys = identity._lookup_params("라울 카스트로", ["Raúl Castro", "카스트로", "Castro"])
        self.assertIn("Raúl Castro", names)
        self.assertNotIn("카스트로", names)
        self.assertNotIn("castro", keys)

    def test_namespace_conflict_blocks_namesake_merge(self):
        rows = [{"uuid": "k1", "name": "블라디미르 코마로프", "labels": ["Entity", "Person"], "rels": 3,
                 "same_label": True, "external_ids": ["commulingo:person:vladimir-komarov"]}]
        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_KEY: rows})
        hit = identity.resolve_entity_sync(
            session, name="블라디미르 코마로프", entity_type="Person",
            external_id="commulingo:person:vladimir-komarov-cosmonaut", aliases=["Vladimir Komarov"],
        )
        self.assertFalse(hit.found)
        # same id (re-run) or no id → reuse
        hit = identity.resolve_entity_sync(session, name="블라디미르 코마로프", entity_type="Person",
                                           external_id="commulingo:person:vladimir-komarov")
        self.assertEqual(hit.uuid, "k1")
        hit = identity.resolve_entity_sync(session, name="블라디미르 코마로프", entity_type="Person")
        self.assertEqual(hit.uuid, "k1")

    def test_exclude_uuid(self):
        rows = [{"uuid": "self", "name": "Stalin", "labels": ["Entity", "Person"], "rels": 100, "same_label": True, "external_ids": []},
                {"uuid": "twin", "name": "Joseph Stalin", "labels": ["Entity", "Person"], "rels": 50, "same_label": True,
                 "external_ids": ["commulingo:person:stalin"]}]
        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_KEY: rows})
        self.assertEqual(identity.resolve_entity_sync(session, name="Stalin", entity_type="Person").uuid, "self")
        self.assertEqual(identity.resolve_entity_sync(session, name="Stalin", entity_type="Person", exclude_uuid="self").uuid, "twin")

    def test_weak_key_resolution_requires_uniqueness(self):
        two = [{"uuid": "fidel", "name": "피델 카스트로", "labels": ["Entity", "Person"]},
               {"uuid": "raul", "name": "라울 카스트로", "labels": ["Entity", "Person"]}]
        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_WEAK_KEY: two})
        hit = identity.resolve_entity_sync(session, name="카스트로", entity_type="Person")
        self.assertFalse(hit.found)
        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_WEAK_KEY: two[:1]})
        hit = identity.resolve_entity_sync(session, name="카스트로", entity_type="Person")
        self.assertEqual((hit.uuid, hit.method), ("fidel", "weak_alias"))


class IdentityPropsTests(unittest.TestCase):
    def test_build_identity_props(self):
        props = identity.build_identity_props(
            "니키타 흐루쇼프",
            aliases=["Nikita Khrushchev", "Никита Хрущёв", "니키타 흐루쇼프"],
            external_ids=["commulingo:person:nikita-khrushchev", "commulingo:person:nikita-khrushchev"],
            name_ko="니키타 흐루쇼프",
            name_en="Nikita Khrushchev",
        )
        self.assertEqual(props["external_ids"], ["commulingo:person:nikita-khrushchev"])
        self.assertEqual(props["aliases"], ["Nikita Khrushchev", "Никита Хрущёв"])
        self.assertIn("nikita khrushchev", props["alias_keys"])
        self.assertIn("니키타 흐루쇼프", props["alias_keys"])
        self.assertEqual(props["weak_keys"], [])
        props2 = identity.build_identity_props("니키타 흐루쇼프", aliases=["흐루쇼프", "Khrushchev"])
        self.assertEqual(props2["weak_keys"], ["흐루쇼프", "khrushchev"])
        self.assertNotIn("흐루쇼프", props2["alias_keys"])
        self.assertEqual(props["alias_text"], "Nikita Khrushchev / Никита Хрущёв")
        self.assertEqual(props["name_en"], "Nikita Khrushchev")


class ResolveSyncTests(unittest.TestCase):
    def test_external_id_wins_regardless_of_label(self):
        session = FakeSyncSession({
            identity.CYPHER_RESOLVE_BY_EXTERNAL_ID: [
                {"uuid": "u-ext", "name": "X", "labels": ["Entity", "Organization"], "rels": 3},
            ],
            identity.CYPHER_RESOLVE_BY_KEY: [
                {"uuid": "u-key", "name": "X", "labels": ["Entity", "Person"], "rels": 9, "same_label": True},
            ],
        })
        hit = identity.resolve_entity_sync(
            session, name="X", entity_type="Person", external_id="commulingo:person:x",
        )
        self.assertEqual((hit.uuid, hit.method), ("u-ext", "external_id"))
        self.assertEqual(len(session.calls), 1)

    def test_alias_match_same_label(self):
        seen = {}

        def rows(params):
            seen.update(params)
            return [{"uuid": "u1", "name": "Nikita Khrushchev", "labels": ["Entity", "Person"],
                     "rels": 4, "same_label": True}]

        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_KEY: rows})
        hit = identity.resolve_entity_sync(
            session, name="니키타 흐루쇼프", entity_type="Person", aliases=["Nikita Khrushchev"],
        )
        self.assertEqual((hit.uuid, hit.method), ("u1", "alias"))
        self.assertIn("nikita khrushchev", seen["keys"])
        self.assertIn("니키타 흐루쇼프", seen["names"])
        self.assertEqual(seen["etype"], "Person")

    def test_label_conflict_is_not_reused(self):
        session = FakeSyncSession({
            identity.CYPHER_RESOLVE_BY_KEY: [
                {"uuid": "u-org", "name": "Libya", "labels": ["Entity", "Organization"],
                 "rels": 4, "same_label": False},
            ],
        })
        hit = identity.resolve_entity_sync(session, name="Libya", entity_type="Concept")
        self.assertIsNone(hit.uuid)
        self.assertEqual(hit.method, "label_conflict")
        # 2026-09-04: a country typed Location by the extractor is the same
        # node as the Organization one (no more 미국/소련 [Location] twins).
        hit = identity.resolve_entity_sync(session, name="Libya", entity_type="Location")
        self.assertEqual((hit.uuid, hit.method), ("u-org", "alias"))

    def test_no_match(self):
        session = FakeSyncSession({})
        hit = identity.resolve_entity_sync(session, name="Nobody", entity_type="Person")
        self.assertFalse(hit.found)
        self.assertEqual(hit.method, "none")


class ResolveAsyncTests(unittest.TestCase):
    def test_async_alias_then_embedding_gate(self):
        session = FakeAsyncSession({})

        class Embedder:
            called = 0

            async def create(self, input_data):
                Embedder.called += 1
                return [0.1, 0.2]

        old = os.environ.get("KG_RESOLVE_EMBEDDING_NN")
        try:
            os.environ["KG_RESOLVE_EMBEDDING_NN"] = "0"
            hit = asyncio.run(identity.resolve_entity_async(
                session, name="Someone", entity_type="Person", embedder=Embedder(),
            ))
            self.assertFalse(hit.found)
            self.assertEqual(Embedder.called, 0)

            os.environ["KG_RESOLVE_EMBEDDING_NN"] = "1"
            session2 = FakeAsyncSession({
                identity.CYPHER_RESOLVE_BY_EMBEDDING: [
                    {"uuid": "u-nn", "name": "Some One", "labels": ["Entity", "Person"], "score": 0.95},
                ],
            })
            hit = asyncio.run(identity.resolve_entity_async(
                session2, name="Someone", entity_type="Person", embedder=Embedder(),
            ))
            self.assertEqual((hit.uuid, hit.method), ("u-nn", "embedding"))
            self.assertEqual(Embedder.called, 1)
        finally:
            if old is None:
                os.environ.pop("KG_RESOLVE_EMBEDDING_NN", None)
            else:
                os.environ["KG_RESOLVE_EMBEDDING_NN"] = old


class UpsertAndMergeTests(unittest.TestCase):
    def test_upsert_params(self):
        params = identity._upsert_params(
            "u1", ["a:1", "a:1"], ["Alias", ""], "이름", "Name", "sum",
        )
        self.assertEqual(params["external_ids"], ["a:1"])
        self.assertEqual(params["aliases"], ["Alias", "이름", "Name"])
        self.assertEqual(params["alias_keys"], ["alias", "name"])   # unknown name: short tokens are weak
        self.assertEqual(params["weak_keys"], ["이름"])
        self.assertEqual(params["summary"], "sum")

    def test_merge_sync_runs_full_sequence_and_skips_self(self):
        session = FakeSyncSession({
            identity.CYPHER_MERGE_OUT: [{"cnt": 2}],
            identity.CYPHER_MERGE_IN: [{"cnt": 1}],
            identity.CYPHER_MERGE_MENTIONS: [{"cnt": 3}],
            identity.CYPHER_MERGE_IDENTITY: [{"uuid": "canon"}],
            identity.CYPHER_MERGE_DELETE: [{"cnt": 1}],
        })
        stats = identity.merge_entity_nodes_sync(session, "canon", ["dup1", "canon", None])
        self.assertEqual(stats["edges_moved"], 3)
        self.assertEqual(stats["mentions_moved"], 3)
        self.assertEqual(stats["merged"], ["dup1"])
        order = [c for c, _ in session.calls]
        self.assertEqual(order, [
            identity.CYPHER_MERGE_OUT, identity.CYPHER_MERGE_IN, identity.CYPHER_MERGE_MENTIONS,
            identity.CYPHER_MERGE_IDENTITY, identity.CYPHER_MERGE_DELETE,
        ])
        self.assertEqual(session.calls[-1][1], {"canon_uuid": "canon", "dup_uuid": "dup1"})

    def test_post_episode_merge_folds_cross_group_duplicate(self):
        class Node:
            def __init__(self, uuid, name, labels):
                self.uuid, self.name, self.labels = uuid, name, labels

        def by_name(params):
            if "Donald Trump" not in params["names"]:
                return []
            return [
                {"uuid": "new", "name": "Donald Trump", "labels": ["Entity", "Person"], "rels": 0, "same_label": True},
                {"uuid": "old", "name": "Donald Trump", "labels": ["Entity", "Person"], "rels": 12, "same_label": True},
            ]

        session = FakeAsyncSession({
            identity.CYPHER_RESOLVE_BY_KEY: by_name,
            identity.CYPHER_MERGE_OUT: [{"cnt": 0}],
            identity.CYPHER_MERGE_IN: [{"cnt": 1}],
            identity.CYPHER_MERGE_MENTIONS: [{"cnt": 1}],
        })
        merged = asyncio.run(identity.post_episode_merge(
            session, [Node("new", "Donald Trump", ["Entity", "Person"]), Node("solo", "Unique Name", ["Entity", "Asset"])],
        ))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_uuid"], "old")
        self.assertEqual(merged[0]["merged"], ["new"])


class AliasIndexTests(unittest.TestCase):
    def setUp(self):
        self.idx = identity.AliasIndex()
        self.idx.load_rows([
            {"uuid": "u-kctu", "name": "전국민주노동조합총연맹 (민주노총)", "labels": ["Entity", "Organization"],
             "keys": ["민주노총", "kctu"]},
            {"uuid": "u-trump", "name": "Donald Trump", "labels": ["Entity", "Person"], "keys": ["trump"]},
            {"uuid": "u-us", "name": "United States", "labels": ["Entity", "Organization"], "keys": []},
            {"uuid": "u-short", "name": "AI", "labels": ["Entity", "Concept"], "keys": []},
        ])

    def test_weak_keys_indexed_only_when_unique(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "fidel", "name": "피델 카스트로", "labels": ["Entity", "Person"], "keys": [], "weak_keys": ["카스트로"]},
            {"uuid": "raul", "name": "라울 카스트로", "labels": ["Entity", "Person"], "keys": [], "weak_keys": ["카스트로"]},
            {"uuid": "lenin", "name": "블라디미르 레닌", "labels": ["Entity", "Person"], "keys": [], "weak_keys": ["레닌"]},
        ])
        self.assertEqual([h.uuid for h in idx.match("레닌이 말했다")], ["lenin"])
        self.assertEqual([h.uuid for h in idx.match("카스트로가 말했다")], [])
        self.assertEqual([h.uuid for h in idx.match("라울 카스트로가 말했다")], ["raul"])

    def test_korean_substring_match_with_particle(self):
        hits = self.idx.match("민주노총이 파업을 선언했다")
        self.assertEqual([h.uuid for h in hits], ["u-kctu"])
        self.assertEqual(hits[0].key, "민주노총")

    def test_latin_token_match_and_short_keys_ignored(self):
        hits = self.idx.match("What did Trump say about AI in the United States?")
        uuids = [h.uuid for h in hits]
        self.assertIn("u-trump", uuids)
        self.assertIn("u-us", uuids)      # multi-word key matched as phrase
        self.assertNotIn("u-short", uuids)  # "ai" is under the Latin minimum

    def test_longest_key_wins_over_contained_key(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "u-a", "name": "민주노총", "labels": ["Entity", "Organization"], "keys": []},
            {"uuid": "u-b", "name": "민주노총 공공운수노조", "labels": ["Entity", "Organization"], "keys": []},
        ])
        hits = idx.match("민주노총 공공운수노조 성명")
        self.assertEqual([h.uuid for h in hits], ["u-b"])

    def test_no_text(self):
        self.assertEqual(self.idx.match(""), [])

    def test_korean_key_must_start_a_word(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "riga", "name": "리가", "labels": ["Entity", "Location"], "keys": ["riga"]},
            {"uuid": "ussr", "name": "소련", "labels": ["Entity", "Location"], "keys": []},
        ])
        # 2026-09-03: "우리가" / "리델리가" recalled Riga into worldbuilding chats.
        self.assertEqual(idx.match("날것의 자연은 우리가 자원을 모으지 못하게 한다"), [])
        self.assertEqual(idx.match("리델리가 연방의 중앙정계에 진출하는 것은"), [])
        self.assertEqual([h.uuid for h in idx.match("리가에서 발트의 길이 시작됐다")], ["riga"])
        self.assertEqual([h.uuid for h in idx.match("1936년 소련의 헌법")], ["ussr"])
        self.assertEqual([h.uuid for h in idx.match("(소련) 헌법")], ["ussr"])
        self.assertEqual([h.uuid for h in idx.match("소련에서는 소련이라는 소련께서는")], ["ussr"])
        self.assertEqual(idx.match("스탈린주의 소련군"), [])   # compound words are other concepts

    def test_weak_key_needs_particle_or_word_end(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "manifesto", "name": "공산당 선언", "labels": ["Entity", "Concept"], "keys": [], "weak_keys": ["선언"]},
            {"uuid": "krupskaya", "name": "나데즈다 크룹스카야", "labels": ["Entity", "Person"], "keys": [], "weak_keys": ["크룹스카야"]},
        ])
        self.assertEqual(idx.match("민주노총이 파업을 선언했다"), [])
        # 2026-09-04: 선언 alone fired on every declaration in the corpus — it is a
        # generic noun now and never indexed, even as a weak key.
        self.assertEqual(idx.match("선언을 읽었다"), [])
        self.assertEqual([h.uuid for h in idx.match("크룹스카야를 읽었다")], ["krupskaya"])
        self.assertEqual(idx.match("크룹스카야주의자"), [])

    def test_generic_names_never_indexed(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "state", "name": "국가", "labels": ["Entity", "Concept"], "keys": ["국가 권력"]},
            {"uuid": "org", "name": "Organization", "labels": ["Entity", "Organization"], "keys": []},
            {"uuid": "glob", "name": "Global", "labels": ["Entity", "Location"], "keys": ["세계"]},
            {"uuid": "marx", "name": "카를 마르크스", "labels": ["Entity", "Person"], "keys": [], "weak_keys": ["개인"]},
        ])
        self.assertEqual(idx.match("국가의 공권력이 개인에게 폭력을 휘두른다"), [])
        self.assertEqual(idx.match("the Organization went Global 세계"), [])
        self.assertTrue(identity.is_generic_entity_name("경찰"))
        self.assertTrue(identity.is_generic_entity_name("Organization"))
        self.assertFalse(identity.is_generic_entity_name("대한민국 경찰청"))


class UntrustedSourceTests(unittest.TestCase):
    """2026-09-04: an LLM/agent side (no external id) resolves by name only and
    may not give aliases that belong to someone else or are common nouns."""

    def test_lookup_ignores_aliases_when_untrusted(self):
        names, keys = identity._lookup_params("소련 정부", ["미국", "Soviet government"], trusted=False)
        self.assertEqual(names, ["소련 정부"])
        self.assertNotIn("미국", keys)
        names, keys = identity._lookup_params("소련 정부", ["Soviet government"], trusted=True)
        self.assertIn("soviet government", keys)

    def test_resolve_sync_untrusted_does_not_use_alias_keys(self):
        seen = {}

        def by_key(params):
            seen.update(params)
            return []
        session = FakeSyncSession({identity.CYPHER_RESOLVE_BY_KEY: by_key})
        identity.resolve_entity_sync(session, name="소련", entity_type="Organization",
                                     aliases=["United States"], trusted=False)
        self.assertEqual(seen["names"], ["소련"])
        self.assertEqual(seen["keys"], ["소련"])

    def test_filter_untrusted_aliases(self):
        out = identity.filter_untrusted_aliases(
            "삼성바이오로직스 상생노조",
            ["노조", "회사", "삼바 노조", "Samsung Biologics Union", "정권", "SBL", "Samsung Bio", "삼성바이오로직스 상생노조"],
            owned_keys={"samsung bio"},
        )
        self.assertEqual(out, ["삼바 노조", "Samsung Biologics Union"])

    def test_location_and_organization_are_compatible(self):
        rows = [{"uuid": "us", "name": "United States", "labels": ["Entity", "Organization"], "same_label": False}]
        hit = identity._pick_key_hit(rows, "Location", "미국")
        self.assertEqual(hit.uuid, "us")
        rows = [{"uuid": "p", "name": "Stalin", "labels": ["Entity", "Person"], "same_label": False}]
        self.assertEqual(identity._pick_key_hit(rows, "Location", "Stalin").method, "label_conflict")

    def test_generic_names_cover_alias_pollution_words(self):
        for word in ("정권", "노조", "회사", "선언", "음모", "여당", "대통령", "http"):
            self.assertTrue(identity.is_generic_entity_name(word), word)

    def test_alias_index_broad_false_skips_category_words(self):
        idx = identity.AliasIndex()
        idx.load_rows([
            {"uuid": "u1", "name": "사회주의", "labels": ["Entity", "Concept"], "keys": ["사회주의"], "weak_keys": []},
            {"uuid": "u2", "name": "전환", "labels": ["Entity", "Organization"], "keys": ["전환"], "weak_keys": []},
            {"uuid": "u3", "name": "국가보안법", "labels": ["Entity", "Policy"], "keys": ["국가보안법"], "weak_keys": []},
        ])
        text = "사회주의 에너지 전환 정책과 국가보안법 폐지"
        self.assertEqual({h.name for h in idx.match(text)}, {"사회주의", "전환", "국가보안법"})
        self.assertEqual({h.name for h in idx.match(text, broad=False)}, {"국가보안법"})

    def test_generic_alias_is_never_indexed(self):
        idx = identity.AliasIndex()
        idx.load_rows([{"uuid": "u1", "name": "폴란드통일노동자당", "labels": ["Entity", "Organization"],
                        "keys": ["폴란드통일노동자당", "정권"], "weak_keys": []}])
        self.assertEqual(idx.match("정권 교체가 필요하다"), [])


if __name__ == "__main__":
    unittest.main()
