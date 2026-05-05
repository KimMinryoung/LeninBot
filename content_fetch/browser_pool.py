"""Dedicated Playwright browser pool for URL fetching."""

import asyncio
import json as _json
import logging
import os
import re as _re
import threading
from pathlib import Path
from typing import Optional as _Optional

logger = logging.getLogger(__name__)

_JS_PLACEHOLDER_RE = _re.compile(
    r"(enable\s+javascript|javascript\s+(is\s+)?(required|needed|must)|"
    r"this\s+app\s+works\s+best\s+with|turn\s+on\s+javascript|"
    r"activate\s+javascript|자바스크립트를?\s*(활성화|켜|필요)|"
    r"loading\.{2,}|please\s+wait)", _re.I
)


def _is_low_quality(text: str) -> bool:
    """Check if extracted text looks like a JS placeholder or empty shell."""
    if not text or len(text) < 80:
        return True
    if _JS_PLACEHOLDER_RE.search(text[:500]):
        return True
    # Mostly whitespace / very few real words
    words = text.split()
    if len(words) < 15:
        return True
    return False


#
# Architecture:
#   * One Chromium browser + one persistent context, owned by an asyncio
#     event loop running on a dedicated daemon thread (`_apw_loop`).
#   * All Playwright calls go through that loop, so transport access is
#     serialized to one thread (avoids the thread-affinity bugs of the
#     sync API) while many pages can still be in flight concurrently.
#   * Sync callers go through `_playwright_fetch` (a thin blocking wrapper).
#   * Async callers go through `_playwright_fetch_async` via `_pw_submit`,
#     which schedules the coroutine on `_apw_loop` and returns a future
#     awaitable from the caller's own loop.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PW_COOKIE_PATH = PROJECT_ROOT / ".pw_cookies.json"
_PW_COOKIE_PATH = str(PW_COOKIE_PATH)

# Sites whose article body lives in an iframe (Naver cafe / blog).
_IFRAME_SITES = ("cafe.naver.com", "blog.naver.com", "m.blog.naver.com")

# Loop-thread plumbing (touched only via `_apw_loop_lock`)
_apw_loop: _Optional[asyncio.AbstractEventLoop] = None
_apw_thread: _Optional[threading.Thread] = None
_apw_loop_lock = threading.Lock()

# State pinned to `_apw_loop` — only mutate from coroutines running on it.
_apw_instance = None
_apw_browser = None
_apw_context = None
_apw_init_lock: _Optional[asyncio.Lock] = None
_apw_cookies_lock: _Optional[asyncio.Lock] = None


def _ensure_pw_loop() -> asyncio.AbstractEventLoop:
    """Lazily start the dedicated Playwright event-loop thread."""
    global _apw_loop, _apw_thread
    with _apw_loop_lock:
        if _apw_loop is not None and _apw_thread is not None and _apw_thread.is_alive():
            return _apw_loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, name="playwright-loop", daemon=True)
        thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("Playwright event loop failed to start")
        _apw_loop = loop
        _apw_thread = thread
        logger.info("[Playwright] Dedicated event loop thread started")
        return loop


def _pw_submit(coro):
    """Submit a coroutine to the Playwright loop. Returns concurrent.futures.Future.

    Sync callers can call `.result(timeout=...)` to block; async callers in
    other event loops can do `await asyncio.wrap_future(fut)`.
    """
    loop = _ensure_pw_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


async def _get_apw_context():
    """Lazy singleton: persistent Chromium async context with cookie reuse.
    Must be awaited from the dedicated Playwright loop.
    """
    global _apw_instance, _apw_browser, _apw_context, _apw_init_lock
    if _apw_init_lock is None:
        _apw_init_lock = asyncio.Lock()
    async with _apw_init_lock:
        if _apw_context is not None:
            try:
                _ = _apw_context.pages  # liveness probe
                return _apw_context
            except Exception:
                _apw_context = None

        if _apw_browser is not None:
            try:
                _ = _apw_browser.contexts
            except Exception:
                _apw_browser = None
                if _apw_instance is not None:
                    try:
                        await _apw_instance.stop()
                    except Exception:
                        pass
                    _apw_instance = None

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return None

        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                locale="ko-KR",
            )
            if os.path.exists(_PW_COOKIE_PATH):
                try:
                    with open(_PW_COOKIE_PATH, "r", encoding="utf-8") as f:
                        cookies = _json.load(f)
                    await context.add_cookies(cookies)
                    logger.info("[Playwright] Restored %d cookies", len(cookies))
                except Exception:
                    pass
            _apw_instance = pw
            _apw_browser = browser
            _apw_context = context
            logger.info("[Playwright] Async browser context started")
            return _apw_context
        except Exception:
            try:
                await pw.stop()
            except Exception:
                pass
            raise


async def _save_apw_cookies():
    """Persist current cookies to disk. Runs on the Playwright loop."""
    global _apw_cookies_lock
    if _apw_context is None:
        return
    if _apw_cookies_lock is None:
        _apw_cookies_lock = asyncio.Lock()
    async with _apw_cookies_lock:
        try:
            cookies = await _apw_context.cookies()
            with open(_PW_COOKIE_PATH, "w", encoding="utf-8") as f:
                _json.dump(cookies, f, ensure_ascii=False)
        except Exception:
            pass


async def _playwright_fetch_async(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Async Playwright fetch. Many invocations on the same loop run concurrently."""
    context = await _get_apw_context()
    if context is None:
        return None

    page = None
    try:
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            logger.warning("[URL] Playwright goto error (%s): %s", url[:60], e)
            return None
        # Give JS-rendered pages a moment to hydrate
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

        # Naver Cafe/Blog: content is inside an iframe
        target = page
        if any(site in url for site in _IFRAME_SITES):
            await page.wait_for_timeout(3000)
            for frame in page.frames:
                if any(p in frame.url for p in ("/ca-fe/cafes/", "ArticleRead", "PostView.naver")):
                    target = frame
                    break

        # YouTube: extract video title + description
        if "youtube.com" in url or "youtu.be" in url:
            text = await page.evaluate("""() => {
                const title = document.querySelector('h1.ytd-watch-metadata yt-formatted-string, #title h1')?.innerText || '';
                const desc = document.querySelector('#description-inline-expander, #description')?.innerText || '';
                const chapters = [...document.querySelectorAll('ytd-macro-markers-list-item-renderer')]
                    .map(el => el.innerText.trim()).join('\\n');
                return [title, desc, chapters].filter(Boolean).join('\\n\\n');
            }""")
        else:
            _js_extract = """() => {
                ['nav','header','footer','aside','.sidebar','.menu',
                 '.advertisement','.ad','#comments','.comment','script','style']
                .forEach(s => document.querySelectorAll(s).forEach(el => el.remove()));
                const containers = ['article','main','[role="main"]',
                    '.post-content','.article-content','.entry-content',
                    '.se-main-container','.ContentRenderer','#content','.content',
                    '[data-testid="tweetText"]', '.tweet-text'];
                for (const sel of containers) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.trim().length > 100) return el.innerText.trim();
                }
                return document.body ? document.body.innerText.trim() : '';
            }"""
            text = await target.evaluate(_js_extract)
            # Loading 감지 시 최대 10초간 재시도 (2초 간격)
            import time as _time
            _deadline = _time.time() + 10
            while _is_low_quality(text) and _time.time() < _deadline:
                await page.wait_for_timeout(2000)
                text = await target.evaluate(_js_extract)

        if text and len(text) > 50:
            logger.info("[URL] Playwright 성공 (%s): %d chars", url[:60], len(text))
            await _save_apw_cookies()
            return text[:max_chars]
    except Exception as e:
        logger.warning("[URL] Playwright page error (%s): %s", url[:60], e)
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception as e:
                logger.warning("[URL] Playwright page.close failed (%s): %s", url[:60], e)

    return None


def _playwright_fetch(url: str, max_chars: int = 10000) -> _Optional[str]:
    """Sync wrapper: schedules the async fetch on the dedicated Playwright loop
    and blocks for its result. Safe to call from any thread.
    """
    try:
        fut = _pw_submit(_playwright_fetch_async(url, max_chars))
        return fut.result(timeout=90)
    except Exception as e:
        logger.warning("[URL] Playwright fetch wrapper error (%s): %s", url[:60], e)
        return None


