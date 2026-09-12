"""Roleplay failure notices must not become conversation history."""
import unittest
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
        async def inline_thread(func, *args):
            return func(*args)

        with patch.object(bot.asyncio, "to_thread", side_effect=inline_thread), \
             patch.object(bot, "save_message") as save, \
             patch.object(bot, "load_history", return_value=[{"role": "user", "content": "질문"}]), \
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
        save, message, _ = await self.run_turn("완성된 답변")
        self.assertEqual(save.call_count, 2)
        save.assert_called_with(1, "assistant", "완성된 답변")
        message.answer.assert_awaited_once_with("완성된 답변")
