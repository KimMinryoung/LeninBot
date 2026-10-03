"""Browser lifetime regressions: overlapping fetches, cancellation and restart."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from content_fetch import browser_pool as pool


class BrowserIdleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.browser = AsyncMock()
        self.driver = AsyncMock()
        self.context = object()
        self.patches = [
            patch.object(pool, "_apw_browser", self.browser),
            patch.object(pool, "_apw_instance", self.driver),
            patch.object(pool, "_apw_context", self.context),
            patch.object(pool, "_apw_init_lock", asyncio.Lock()),
            patch.object(pool, "_apw_active_fetches", 0),
            patch.object(pool, "_apw_idle_handle", None),
            patch.object(pool, "_APW_IDLE_SECONDS", 300),
            patch.object(pool, "_save_apw_cookies", AsyncMock()),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        if pool._apw_idle_handle is not None:
            pool._apw_idle_handle.cancel()
        for p in reversed(self.patches):
            p.stop()

    async def test_overlapping_fetches_keep_browser_until_last_completion(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fetch(context, url, max_chars):
            if url == "slow":
                entered.set()
                await release.wait()
            return url

        with patch.object(pool, "_get_apw_context", AsyncMock(return_value=self.context)), \
             patch.object(pool, "_fetch_apw_page", side_effect=fetch):
            slow = asyncio.create_task(pool._playwright_fetch_async("slow"))
            await entered.wait()
            self.assertEqual(await pool._playwright_fetch_async("fast"), "fast")
            self.assertIsNone(pool._apw_idle_handle)
            await pool._close_idle_browser()
            self.browser.close.assert_not_awaited()
            release.set()
            await slow
        self.assertEqual(pool._apw_active_fetches, 0)
        self.assertIsNotNone(pool._apw_idle_handle)
        await pool._close_idle_browser()
        pool._save_apw_cookies.assert_awaited_once()
        self.browser.close.assert_awaited_once()
        self.driver.stop.assert_awaited_once()
        self.assertIsNone(pool._apw_context)

    async def test_cancellation_releases_lease_and_next_fetch_cancels_timer(self):
        entered = asyncio.Event()

        async def fetch(*args):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(pool, "_get_apw_context", AsyncMock(return_value=self.context)), \
             patch.object(pool, "_fetch_apw_page", side_effect=fetch):
            task = asyncio.create_task(pool._playwright_fetch_async("cancel"))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(pool._apw_active_fetches, 0)
        timer = pool._apw_idle_handle
        with patch.object(pool, "_get_apw_context", AsyncMock(return_value=self.context)), \
             patch.object(pool, "_fetch_apw_page", AsyncMock(return_value="ok")):
            self.assertEqual(await pool._playwright_fetch_async("next"), "ok")
        self.assertTrue(timer.cancelled())

    async def test_fetch_arriving_during_close_waits_and_recreates_context(self):
        closing = asyncio.Event()
        release = asyncio.Event()

        async def close():
            closing.set()
            await release.wait()
        self.browser.close.side_effect = close
        cleanup = asyncio.create_task(pool._close_idle_browser())
        await closing.wait()
        replacement = object()

        async def acquire():
            async with pool._apw_init_lock:
                self.assertIsNone(pool._apw_context)
                pool._apw_context = replacement
                pool._apw_browser = AsyncMock()
                return replacement

        with patch.object(pool, "_get_apw_context", side_effect=acquire), \
             patch.object(pool, "_fetch_apw_page", AsyncMock(return_value="restored")) as fetch:
            task = asyncio.create_task(pool._playwright_fetch_async("new"))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            await cleanup
            self.assertEqual(await task, "restored")
            fetch.assert_awaited_once_with(replacement, "new", 10000)

    async def test_browser_close_failure_still_stops_driver_and_resets_pool(self):
        self.browser.close.side_effect = RuntimeError("disconnected")
        await pool._close_idle_browser()
        self.driver.stop.assert_awaited_once()
        self.assertIsNone(pool._apw_browser)
        self.assertIsNone(pool._apw_instance)

    async def test_initialization_failure_does_not_leak_active_lease(self):
        with patch.object(pool, "_get_apw_context", AsyncMock(side_effect=RuntimeError("launch"))):
            with self.assertRaises(RuntimeError):
                await pool._playwright_fetch_async("failure")
        self.assertEqual(pool._apw_active_fetches, 0)

    async def test_zero_timeout_disables_idle_cleanup(self):
        with patch.object(pool, "_APW_IDLE_SECONDS", 0), \
             patch.object(pool, "_get_apw_context", AsyncMock(return_value=self.context)), \
             patch.object(pool, "_fetch_apw_page", AsyncMock(return_value="ok")):
            await pool._playwright_fetch_async("keep")
        self.assertIsNone(pool._apw_idle_handle)
