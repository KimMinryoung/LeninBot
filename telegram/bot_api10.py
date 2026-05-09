"""Small raw Telegram Bot API 10.0 shim for methods not yet exposed by aiogram."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiohttp


class TelegramBotApiError(RuntimeError):
    """Raised when Telegram returns a non-OK response or invalid payload."""


@dataclass(slots=True)
class TelegramBotApi10Client:
    token: str
    base_url: str = "https://api.telegram.org"
    timeout_seconds: float = 15.0

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        if not self.token:
            raise TelegramBotApiError("Telegram bot token is not configured")
        url = f"{self.base_url.rstrip('/')}/bot{self.token}/{method}"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload or {}) as response:
                text = await response.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    raise TelegramBotApiError(
                        f"{method} returned invalid JSON with HTTP {response.status}"
                    ) from e

        if not isinstance(data, dict):
            raise TelegramBotApiError(f"{method} returned a non-object response")
        if not data.get("ok"):
            description = str(data.get("description") or "Telegram API request failed")
            error_code = data.get("error_code")
            suffix = f" (error_code={error_code})" if error_code is not None else ""
            raise TelegramBotApiError(f"{method}: {description}{suffix}")
        return data.get("result")

    async def answer_guest_query(self, guest_query_id: str, result: dict[str, Any]) -> Any:
        return await self.call(
            "answerGuestQuery",
            {
                "guest_query_id": guest_query_id,
                "result": result,
            },
        )

    async def answer_guest_query_text(
        self,
        guest_query_id: str,
        text: str,
        *,
        title: str = "Cyber-Lenin",
        parse_mode: str | None = None,
    ) -> Any:
        message_content: dict[str, Any] = {"message_text": text}
        if parse_mode:
            message_content["parse_mode"] = parse_mode
        return await self.answer_guest_query(
            guest_query_id,
            {
                "type": "article",
                "id": guest_query_id[:64] or "guest",
                "title": title[:64] or "Cyber-Lenin",
                "input_message_content": message_content,
            },
        )

    async def delete_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        *,
        user_id: int | None = None,
        actor_chat_id: int | None = None,
    ) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id}
        if user_id is not None:
            payload["user_id"] = user_id
        if actor_chat_id is not None:
            payload["actor_chat_id"] = actor_chat_id
        return bool(await self.call("deleteMessageReaction", payload))

    async def delete_all_message_reactions(
        self,
        chat_id: int | str,
        *,
        user_id: int | None = None,
        actor_chat_id: int | None = None,
    ) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id}
        if user_id is not None:
            payload["user_id"] = user_id
        if actor_chat_id is not None:
            payload["actor_chat_id"] = actor_chat_id
        return bool(await self.call("deleteAllMessageReactions", payload))

    async def get_managed_bot_access_settings(self, user_id: int) -> dict[str, Any]:
        result = await self.call("getManagedBotAccessSettings", {"user_id": user_id})
        return result if isinstance(result, dict) else {}

    async def set_managed_bot_access_settings(
        self,
        user_id: int,
        is_access_restricted: bool,
        added_user_ids: list[int] | None = None,
    ) -> bool:
        payload: dict[str, Any] = {
            "user_id": user_id,
            "is_access_restricted": is_access_restricted,
        }
        if added_user_ids is not None:
            payload["added_user_ids"] = added_user_ids[:10]
        return bool(await self.call("setManagedBotAccessSettings", payload))
