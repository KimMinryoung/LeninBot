"""Hermetic tests for telegram.curate (/curate command, write-boundary guard, outcome DM)."""

import json
import types
import unittest
from unittest.mock import AsyncMock, patch

from telegram import curate
from tool_gateway.results import ToolFailure


URL = "https://www.example.com/articles/2026/strike?utm_source=tg&id=7#top"
KEY = "https://example.com/articles/2026/strike?id=7"


def _good_args(**overrides):
    args = {
        "title": "조선소 파업이 드러낸 하청 구조의 균열",
        "source_url": URL,
        "source_title": "The shipyard strike and the subcontracting wall",
        "source_author": "홍길동",
        "source_publication": "참세상",
        "source_published_at": "2026-08-30",
        "selection_rationale": (
            "이 글은 조선소 하청 노동자 파업을 임금 인상 요구가 아니라 원청과 하청 사이의 책임 회피 구조를 겨눈 투쟁으로 읽는다. "
            "현장 노동자들의 증언과 하청 업체별 계약 조건을 표로 정리해 다른 보도에서 보기 어려운 구체성을 갖췄다. "
            "이론적으로는 노동 과정의 분절화가 조직화 전략에 미치는 영향을 차분하게 짚는다. "
            "슬로건 대신 현실의 제약을 인정한 채 다음 단계를 논한다는 점에서 기준에 부합한다."
        ),
        "context": (
            "한국 조선업은 2020년대 들어 수주 회복과 인력 부족을 동시에 겪으며 다단계 하청 구조를 더 깊게 굳혔다. "
            "이 글의 저자는 지난 몇 해 동안 조선소 노동 문제를 꾸준히 추적해 온 현장 기록자이다. "
            "글은 2022년 옥포 파업 이후 이어진 손해배상 소송과 교섭 구조 논쟁을 배경으로 삼는다. "
            "읽기 전에 원청 교섭 의무를 둘러싼 노조법 개정 논의가 아직 마무리되지 않았다는 점을 알아 두면 좋다. "
            "저자는 결론에서 산업별 교섭의 가능성을 열어 두되 그 조건을 낙관하지 않는다."
        ),
        "tags": ["노동운동", "조선업", "하청구조"],
        "slug": "shipyard-strike-subcontracting-2026",
    }
    args.update(overrides)
    return args


class NormalizeUrlTests(unittest.TestCase):
    def test_strips_tracking_fragment_www_and_trailing_slash(self):
        self.assertEqual(curate.normalize_source_url(URL), KEY)
        self.assertEqual(curate.normalize_source_url("https://example.com/articles/2026/strike/?id=7"), KEY)
        self.assertEqual(curate.normalize_source_url("HTTPS://Example.COM:443/articles/2026/strike?id=7&fbclid=abc"), KEY)

    def test_root_path_and_non_default_port_kept(self):
        self.assertEqual(curate.normalize_source_url("https://example.com"), "https://example.com/")
        self.assertEqual(curate.normalize_source_url("https://example.com/"), "https://example.com/")
        self.assertEqual(curate.normalize_source_url("http://example.com:8080/x/"), "http://example.com:8080/x")

    def test_non_tracking_params_preserved_and_sorted(self):
        self.assertEqual(
            curate.normalize_source_url("https://a.org/p?z=1&a=2&utm_medium=m"),
            "https://a.org/p?a=2&z=1",
        )


class ParseArgsTests(unittest.TestCase):
    def test_url_and_note(self):
        url, note = curate.parse_curate_args("/curate https://a.org/p 이 글은 현장 증언이 좋다")
        self.assertEqual(url, "https://a.org/p")
        self.assertEqual(note, "이 글은 현장 증언이 좋다")

    def test_note_before_url_and_bot_suffix(self):
        url, note = curate.parse_curate_args("/curate@leninbot 메모 앞  https://a.org/p")
        self.assertEqual(url, "https://a.org/p")
        self.assertEqual(note, "메모 앞")

    def test_url_from_reply_and_entities(self):
        url, note = curate.parse_curate_args("/curate 메모", reply_text="봐라 https://b.org/x 좋다")
        self.assertEqual((url, note), ("https://b.org/x", "메모"))
        url, _ = curate.parse_curate_args("/curate", reply_text=None, entity_urls=["https://c.org/y"])
        self.assertEqual(url, "https://c.org/y")

    def test_no_url(self):
        self.assertEqual(curate.parse_curate_args("/curate"), (None, ""))
        self.assertEqual(curate.parse_curate_args(None), (None, ""))


class DedupeTests(unittest.TestCase):
    def test_matches_normalized_equivalent(self):
        seen = {}

        def query_fn(sql, params):
            seen["params"] = params
            return [
                {"slug": "other", "title": "x", "source_url": "https://example.com/articles/2026/other"},
                {"slug": "strike", "title": "y", "source_url": "https://example.com/articles/2026/strike/?id=7&utm_campaign=z"},
            ]

        row = curate.find_existing_curation(query_fn, URL)
        self.assertEqual(row["slug"], "strike")
        self.assertIn("example.com", seen["params"][0])

    def test_no_match(self):
        self.assertIsNone(curate.find_existing_curation(lambda *_: [], URL))
        self.assertIsNone(
            curate.find_existing_curation(
                lambda *_: [{"slug": "s", "title": "t", "source_url": "https://example.com/articles/2026/strike?id=8"}],
                URL,
            )
        )


class ValidateArgsTests(unittest.TestCase):
    def _reject(self, needle, **overrides):
        reason = curate.validate_curation_args(_good_args(**overrides), expected_url=URL)
        self.assertIsNotNone(reason, f"expected rejection for {overrides}")
        self.assertIn(needle, reason)

    def test_good_payload_passes(self):
        self.assertIsNone(curate.validate_curation_args(_good_args(), expected_url=URL))

    def test_source_url_must_match_commission(self):
        self._reject("source_url", source_url="https://example.com/articles/2026/other")
        # Tracking-parameter and www differences are tolerated.
        self.assertIsNone(
            curate.validate_curation_args(_good_args(source_url="https://example.com/articles/2026/strike?id=7"), expected_url=URL)
        )

    def test_slug_rules(self):
        self._reject("slug", slug="")
        self._reject("slug", slug="한글-slug")
        self._reject("slug", slug="a" * 81)
        self._reject("slug", slug="-leading")

    def test_prose_rules(self):
        good = _good_args()
        self._reject("줄바꿈", selection_rationale=good["selection_rationale"].replace(". ", ".\n", 1))
        self._reject("마크다운", context="## " + good["context"])
        self._reject("마크다운", context=good["context"] + " **강조**")
        self._reject("em dash", context=good["context"].replace("굳혔다.", "굳혔다 — 그리고 넓혔다.", 1))
        self._reject("최소", selection_rationale="짧다.")
        self._reject("최대", context=good["context"] * 3)
        self._reject("한국어", selection_rationale="This piece is an excellent account of the strike. " * 6)
        self._reject("북한", context=good["context"].replace("한국 조선업", "북한 조선업"))

    def test_title_and_tags_and_date(self):
        self._reject("메타 접두어", title="큐레이션 #3: 조선소 파업")
        self._reject("메타 접두어", title="왜 이 글이 지금 중요한가: 조선소")
        self._reject("title", title="가" * 61)
        self._reject("tags", tags=["노동운동"])
        self._reject("tags", tags=["a", "b", "c", "d", "e", "f"])
        self._reject("tag", tags=["labour", "노동"])
        self._reject("YYYY-MM-DD", source_published_at="2026.08.30")
        self.assertIsNone(curate.validate_curation_args(_good_args(source_published_at=""), expected_url=URL))


class GuardedHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.task = {"id": 5, "metadata": json.dumps({"source_url": URL})}

    async def test_validation_failure_short_circuits_inner(self):
        inner = AsyncMock(return_value="Published hub curation #1")
        guarded = curate.make_guarded_publish_handler(inner, self.task)
        result = await guarded(**_good_args(slug=""))
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("slug", result)
        inner.assert_not_awaited()

    async def test_plain_error_string_becomes_failure(self):
        inner = AsyncMock(return_value="Error: slug must match ^[a-z0-9]...")
        guarded = curate.make_guarded_publish_handler(inner, self.task)
        result = await guarded(**_good_args())
        self.assertIsInstance(result, ToolFailure)
        inner.assert_awaited_once()

    async def test_success_passes_through_unchanged(self):
        inner = AsyncMock(return_value="Published hub curation #7\nSlug: shipyard-strike-subcontracting-2026")
        guarded = curate.make_guarded_publish_handler(inner, self.task)
        result = await guarded(**_good_args())
        self.assertNotIsInstance(result, ToolFailure)
        self.assertTrue(result.startswith("Published hub curation #7"))


def _message(text, user_id=42, reply=None):
    return types.SimpleNamespace(
        text=text,
        from_user=types.SimpleNamespace(id=user_id),
        entities=[],
        caption_entities=[],
        reply_to_message=reply,
        answer=AsyncMock(),
    )


class CommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ctx = {"is_allowed": lambda uid: uid == 42}

    async def test_non_owner_ignored(self):
        msg = _message("/curate https://a.org/p", user_id=1)
        await curate.cmd_curate(msg, self.ctx)
        msg.answer.assert_not_awaited()

    async def test_usage_when_no_url(self):
        msg = _message("/curate")
        with patch.object(curate, "_query") as q:
            await curate.cmd_curate(msg, self.ctx)
        q.assert_not_called()
        self.assertIn("사용법", msg.answer.await_args.args[0])

    async def test_unsafe_url_rejected(self):
        msg = _message("/curate http://127.0.0.1/admin")

        def boom(url):
            raise curate.UnsafeUrlError("private address")

        with patch.object(curate, "_validate_public_http_url", boom), patch.object(curate, "_query") as q:
            await curate.cmd_curate(msg, self.ctx)
        q.assert_not_called()
        self.assertIn("private address", msg.answer.await_args.args[0])

    async def test_duplicate_reports_existing_slug(self):
        msg = _message(f"/curate {URL}")
        existing = [{"slug": "strike", "title": "기존 글", "source_url": "https://example.com/articles/2026/strike?id=7"}]
        with patch.object(curate, "_validate_public_http_url", lambda u: u), \
             patch.object(curate, "_query", return_value=existing) as q:
            await curate.cmd_curate(msg, self.ctx)
        self.assertEqual(q.call_count, 1)  # lookup only, no INSERT
        self.assertIn("/hub/strike", msg.answer.await_args.args[0])

    async def test_active_task_reported(self):
        msg = _message(f"/curate {URL}")
        with patch.object(curate, "_validate_public_http_url", lambda u: u), \
             patch.object(curate, "_query", return_value=[]) as q, \
             patch.object(curate, "_query_one", return_value={"id": 9, "status": "processing"}):
            await curate.cmd_curate(msg, self.ctx)
        self.assertEqual(q.call_count, 1)
        self.assertIn("#9", msg.answer.await_args.args[0])

    async def test_success_enqueues_hub_curator_task(self):
        msg = _message(f"/curate {URL} 현장 증언이 좋다")
        calls = []

        def fake_query(sql, params=None):
            calls.append((sql, params))
            if sql.startswith("INSERT"):
                return [{"id": 77}]
            return []

        with patch.object(curate, "_validate_public_http_url", lambda u: u), \
             patch.object(curate, "_query", side_effect=fake_query), \
             patch.object(curate, "_query_one", return_value=None):
            await curate.cmd_curate(msg, self.ctx)

        insert = [c for c in calls if c[0].startswith("INSERT")]
        self.assertEqual(len(insert), 1)
        user_id, content, agent_type, metadata = insert[0][1]
        self.assertEqual((user_id, agent_type), (42, "hub_curator"))
        self.assertIn(URL, content)
        self.assertIn("현장 증언이 좋다", content)
        meta = json.loads(metadata)
        self.assertEqual(meta["source_url"], URL)
        self.assertEqual(meta["source_url_key"], KEY)
        self.assertEqual(meta["origin"], "command")
        self.assertIn("#77", msg.answer.await_args.args[0])


class OutcomeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = types.SimpleNamespace(send_message=AsyncMock())
        self.task = {"id": 5, "user_id": 42, "metadata": {"source_url": URL}}
        self.events = []

    def _save(self, uid, kind, content):
        self.events.append((uid, kind, content))

    async def test_success_dm_from_db_row(self):
        row = [{"slug": "strike", "title": "발행된 글", "source_url": KEY}]
        with patch.object(curate, "_query", return_value=row):
            await curate.report_curation_outcome(
                self.bot, self.task, {"status": "done", "report": "무시되는 텍스트"}, chat_id=42, save_system_event=self._save
            )
        text = self.bot.send_message.await_args.kwargs["text"]
        self.assertIn("✅", text)
        self.assertIn("발행된 글", text)
        self.assertIn("/hub/strike", text)
        self.assertIn("edit_content", text)
        self.assertEqual(self.events[0][1], "hub_curation")

    async def test_failure_dm_uses_error_then_interrupt_then_report(self):
        with patch.object(curate, "_query", return_value=[]):
            await curate.report_curation_outcome(self.bot, self.task, {"status": "failed", "error": "boom"}, chat_id=42)
            self.assertIn("boom", self.bot.send_message.await_args.kwargs["text"])
            await curate.report_curation_outcome(self.bot, self.task, {"status": "done", "was_interrupted": True}, chat_id=42)
            self.assertIn("예산", self.bot.send_message.await_args.kwargs["text"])
            await curate.report_curation_outcome(self.bot, self.task, {"status": "done", "report": "페이월 때문에 본문을 읽지 못했다."}, chat_id=42)
            text = self.bot.send_message.await_args.kwargs["text"]
        self.assertIn("❌", text)
        self.assertIn("페이월", text)


if __name__ == "__main__":
    unittest.main()
