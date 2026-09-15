"""Send the prepared text itself; record only acknowledged Telegram deliveries."""
import asyncio
import weakref

from mail_runtime import store

_locks = weakref.WeakValueDictionary()


async def deliver(bot, task_id, chat_id, items, persist):
    for item in items:
        key = (str(chat_id), item['mail_id'])
        lock = _locks.setdefault(key, asyncio.Lock())
        async with lock:
            if await asyncio.to_thread(store.is_delivered, item['mail_id'], chat_id):
                continue
            parsed = item['parsed']
            text = (f"{parsed['subject'][:200]}\n"
                    f"{parsed['from'][:200]} · {parsed['date'][:100]}\n\n"
                    f"{item['summary']}\n\n"
                    f"메일 #{item['mail_id']} · {item['folder']} UID {item['uid']}")
            # A timeout has unknown delivery outcome: no receipt, safe to retry.
            message = await bot.send_message(chat_id=chat_id, text=text, parse_mode=None)
            await asyncio.to_thread(store.mark_sent, task_id, item['mail_id'], chat_id, message.message_id)
            await persist(chat_id, text)
