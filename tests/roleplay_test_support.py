"""Helpers shared by roleplay tests."""
from contextlib import asynccontextmanager


def async_stage(sync_stage):
    """Adapt a fake sync ``staged_memory`` to the bot's ``staged_memory_async`` seam."""
    @asynccontextmanager
    async def staged(uid):
        with sync_stage(uid) as stage:
            yield stage
    return staged
