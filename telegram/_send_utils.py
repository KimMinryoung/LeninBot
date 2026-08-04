"""Shared Telegram send helpers used by bot.py and roleplay_bot.py.

No aiogram import — the bot object is duck-typed (anything with an async
send_message(chat_id=..., text=..., parse_mode=...)), so this module stays
importable everywhere.
"""
import logging

logger = logging.getLogger(__name__)


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks respecting Telegram's 4096 char limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


def make_progress_callback(
    get_bot,
    chat_id: int,
    *,
    events: tuple[str, ...] = ("thinking", "tool_call", "tool_result", "budget"),
):
    """on_progress callback that streams tool-loop progress via Telegram.

    Buffers events per round and flushes one code-block message per round so
    the final answer stays clean prose. Exposes ``.flush`` for the trailing
    round after the loop returns.

    get_bot: zero-arg callable returning the bot (or None → flush no-ops);
    called at flush time so a bot created after the callback still works.
    events: which event types to stream — roleplay passes only
    ("tool_call", "tool_result") because its in-character prose already
    arrives in the final reply and budget is mechanics noise.
    """
    _buf: list[str] = []
    _current_round = [0]

    async def _flush():
        bot = get_bot()
        if not _buf or not bot:
            return
        text = "\n".join(_buf)
        _buf.clear()
        try:
            for chunk in split_message(f"```\n{text}\n```"):
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
        except Exception as e:
            logger.debug("Progress message send failed: %s", e)

    async def _on_progress(event: str, detail: str):
        # Extract round number from detail prefix "[N] ..."
        round_num = 0
        if detail.startswith("["):
            try:
                round_num = int(detail[1:detail.index("]")])
            except (ValueError, IndexError):
                pass

        # New round started → flush previous round's buffer
        if round_num > _current_round[0] and _current_round[0] > 0:
            await _flush()
        if round_num > 0:
            _current_round[0] = round_num

        if event not in events:
            return
        if event == "thinking":
            _buf.append(f"💭 {detail}")
        elif event == "budget":
            _buf.append(f"💰 {detail}")
        else:
            _buf.append(detail)

    # Expose flush for final cleanup
    _on_progress.flush = _flush
    return _on_progress
