"""Roleplay failure notices must not become conversation history."""
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telegram import roleplay_bot as bot


class TestReplyPersistence(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, reply):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=1), text="질문", message_id=2,
            chat=SimpleNamespace(id=1), answer=AsyncMock(),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
        )
        progress = SimpleNamespace(flush=AsyncMock())
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        @contextmanager
        def fake_stage(uid):
            yield {"people":[]}
        authorization={"user_text":"질문","labels":{"mode":"discussion","transition":"current","span":"brief"}}

        with patch.object(bot.asyncio, "to_thread", side_effect=inline_thread), \
             patch.object(bot, "adjudicate_turn", return_value={"status": "unchanged", "reply":reply}), \
             patch.object(bot.roleplay_turn, "committed_reply", return_value=None), \
             patch.object(bot.roleplay_turn, "authorize", return_value=authorization), \
             patch.object(bot.roleplay_turn, "staged_memory", side_effect=fake_stage), \
             patch.object(bot.roleplay_turn, "prepare", return_value={}), \
             patch.object(bot, "save_message") as save, \
             patch.object(bot, "load_history", return_value=[{"role": "user", "content": "질문"}]), \
             patch.object(bot, "load_notes", return_value=[{"key": "관계", "content": "친구"}]), \
             patch.object(bot, "load_state", return_value={"hunger": 25}), \
             patch.object(bot, "get_preference", return_value="off"), \
             patch.object(bot, "people_context", return_value={"index": [{"person_id": "ivan"}], "present": []}), \
             patch.object(bot, "build_system_prompt", return_value="sys"), \
             patch.object(bot, "_make_progress_callback", return_value=progress), \
             patch.object(bot, "chat_with_tools", new_callable=AsyncMock, return_value=reply) as chat:
            await bot.handle_message(message)
        return save, message, chat

    async def test_failure_not_saved(self):
        for reply in (bot.EMPTY_RESPONSE_FALLBACK, "   "):
            with self.subTest(reply=reply):
                save, message, chat = await self.run_turn(reply)
                save.assert_called_once_with(1, "user", "질문")
                message.answer.assert_awaited_once()
                self.assertTrue(chat.call_args.kwargs["continue_on_length"])
                self.assertEqual(chat.call_args.kwargs["max_length_continuations"], 1)

    async def test_success_saved_and_sent(self):
        save, message, chat = await self.run_turn("완성된 답변")
        payload = chat.call_args.args[0][-1]["_runtime_events"][0]["payload"]
        self.assertEqual(payload["private_notes"][0]["content"], "친구")
        self.assertEqual(payload["character_state"]["acting_cues"]["허기"], "허기가 느껴짐")
        self.assertNotIn("hunger", payload["character_state"])
        self.assertIn("direction", payload["scene_direction"])
        self.assertNotIn("automatic_adjudication", payload)
        self.assertNotIn("time_policy", payload)
        self.assertNotIn("recent_events", payload["character_state"])
        self.assertNotIn("event_timestamps", payload["character_state"])
        self.assertEqual(payload["people"]["index"][0]["person_id"], "ivan")
        self.assertEqual(save.call_count, 2)
        save.assert_called_with(1, "assistant", "완성된 답변")
        message.answer.assert_awaited_once_with("완성된 답변")


    async def test_private_progress_hidden_and_status_visible(self):
        telegram = SimpleNamespace(send_message=AsyncMock())
        progress = bot._make_progress_callback(telegram, 1)
        await progress("tool_call", '[1] 🔧 roleplay_state({"hunger": 50})')
        await progress("tool_result", '  ✓ roleplay_state: {"hunger": 50}')
        await progress("tool_call", '[1] 🔧 roleplay_person({"name": "이반"})')
        await progress("tool_result", '  ✓ roleplay_person: {"name": "이반"}')
        await progress.flush()
        telegram.send_message.assert_not_awaited()
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), answer=AsyncMock())
        from runtime_tools.roleplay_memory import STATE_DEFAULTS
        with patch.object(bot.asyncio, "to_thread", new=AsyncMock(return_value={**STATE_DEFAULTS, "hunger": 50})):
            await bot.cmd_status(message)
        self.assertIn("허기: 50", message.answer.call_args.args[0])
        self.assertIn("⏱ 시간 계산 근거", message.answer.call_args.args[0])
        self.assertIn("세부 부상:", message.answer.call_args.args[0])
        self.assertIn("다음 시도:", message.answer.call_args.args[0])
        self.assertNotIn("/status 상세", message.answer.call_args.args[0])
        self.assertIn("의지: 미설정", message.answer.call_args.args[0])
        with patch.object(bot.asyncio, "to_thread", new=AsyncMock(return_value={**STATE_DEFAULTS, "resolve": 42.26, "humiliation": 70})):
            await bot.cmd_status(message)
        self.assertIn("의지: 42.3 · 명료함: 미설정 · 굴욕: 70", message.answer.call_args.args[0])


class HistoryWindowTests(unittest.TestCase):
    def test_window_start_moves_in_steps(self):
        offsets = [bot.history_window_offset(n, cap=40, step=20) for n in range(0, 121)]
        self.assertEqual(offsets[:41], [0] * 41)
        self.assertEqual(offsets[41], 0)
        self.assertEqual(offsets[59], 0)
        self.assertEqual(offsets[60], 20)
        self.assertEqual(offsets[79], 20)
        self.assertEqual(offsets[80], 40)
        self.assertEqual(offsets[120], 80)
        # The window never drops below the cap and grows by at most step-1.
        for n, offset in enumerate(offsets):
            self.assertLessEqual(offset, max(0, n - 40))
            self.assertLess(n - offset, 60)



class UnsettledDirectiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsettled_user_message_leaves_roleplay_context(self):
        from contextlib import contextmanager
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), text="의사 재방문", message_id=3,
                                  chat=SimpleNamespace(id=1), answer=AsyncMock(), bot=SimpleNamespace(send_chat_action=AsyncMock()))
        async def inline_thread(func, *args, **kwargs):
            return func(*args, **kwargs)
        @contextmanager
        def fake_stage(uid):
            yield {"people": []}
        authorization = {"user_text": "의사 재방문", "labels": {"mode": "scene", "transition": "current", "span": "brief"}}
        with patch.object(bot.asyncio, "to_thread", side_effect=inline_thread), \
             patch.object(bot.roleplay_turn, "committed_reply", return_value=None), \
             patch.object(bot.roleplay_turn, "authorize", return_value=authorization), \
             patch.object(bot.roleplay_turn, "staged_memory", side_effect=fake_stage), \
             patch.object(bot.roleplay_turn, "prepare", side_effect=ValueError("초안이 사용자 지시의 사건 경계를 넘음")), \
             patch.object(bot, "save_message", return_value=997) as save, \
             patch.object(bot, "exclude_history_message") as exclude, \
             patch.object(bot, "load_history", return_value=[]), patch.object(bot, "load_notes", return_value=[]), \
             patch.object(bot, "load_state", return_value={"hunger": 25}), \
             patch.object(bot, "people_context", return_value={"index": [], "present": []}), \
             patch.object(bot, "build_system_prompt", return_value="sys"), \
             patch.object(bot, "_make_progress_callback", return_value=SimpleNamespace(flush=AsyncMock())), \
             patch.object(bot, "chat_with_tools", new_callable=AsyncMock, return_value="초안"):
            await bot.handle_message(message)
        save.assert_called_once_with(1, "user", "의사 재방문")
        exclude.assert_called_once()
        self.assertEqual(exclude.call_args.args[:2], (1, 997))
        final = message.answer.call_args.args[0]
        self.assertTrue(message.answer.await_args_list[-2].args[0].startswith("【미확정 초안"))
        self.assertIn("막힌 부분", final)
        self.assertIn("초안 1 거절: 초안이 사용자 지시의 사건 경계를 넘음", final)
        self.assertIn("초안 2 거절", final)
