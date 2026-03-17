"""Playwright-based web crawler with persistent browser context (cookie/session reuse)."""

import json
import logging
import os

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_COOKIE_PATH = os.path.join(_DATA_DIR, "browser_cookies.json")
_context = None
_browser = None
_playwright = None


async def _get_context():
    """Lazy-init persistent Chromium browser context with cookie storage."""
    global _playwright, _browser, _context
    if _context is not None:
        return _context

    from playwright.async_api import async_playwright

    os.makedirs(_DATA_DIR, exist_ok=True)
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=True)
    _context = await _browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 800},
        locale="ko-KR",
    )

    # Restore cookies if they exist
    if os.path.exists(_COOKIE_PATH):
        try:
            with open(_COOKIE_PATH, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            await _context.add_cookies(cookies)
            logger.info("Restored %d cookies from %s", len(cookies), _COOKIE_PATH)
        except Exception as e:
            logger.warning("Failed to restore cookies: %s", e)

    return _context


async def _save_cookies():
    """Persist current cookies to disk."""
    if _context is None:
        return
    try:
        cookies = await _context.cookies()
        with open(_COOKIE_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to save cookies: %s", e)


async def crawl(url: str, wait_for: str | None = None, extract_links: bool = False) -> str:
    """Crawl a URL and return extracted text content.

    Args:
        url: Target URL.
        wait_for: CSS selector to wait for before extraction.
        extract_links: If True, also extract all links from the page.

    Returns:
        Extracted text content (and optionally links).
    """
    from local_agent.local_db import query as db_query, execute as db_execute

    # Check cache first
    cached = db_query("SELECT content, crawled_at FROM crawl_cache WHERE url = ?", (url,))
    if cached:
        logger.info("Cache hit for %s (crawled at %s)", url, cached[0]["crawled_at"])
        return f"[Cached: {cached[0]['crawled_at']}]\n{cached[0]['content']}"

    ctx = await _get_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        if wait_for:
            await page.wait_for_selector(wait_for, timeout=10000)

        # Extract page title
        title = await page.title()

        # Determine which frame to extract from.
        # Sites like Naver Cafe load article content inside an iframe.
        target_frame = page
        if "cafe.naver.com" in url:
            import asyncio as _aio
            await _aio.sleep(2)  # wait for iframe to load
            for frame in page.frames:
                if "/ca-fe/cafes/" in frame.url or "ArticleRead" in frame.url:
                    target_frame = frame
                    logger.info("Naver Cafe: using iframe %s", frame.url[:120])
                    break

        # Extract main text content
        text = await target_frame.evaluate("""() => {
            // Remove noise elements
            const selectors = ['nav', 'header', 'footer', 'aside', '.sidebar', '.menu',
                             '.advertisement', '.ad', '#comments', '.comment', 'script', 'style'];
            selectors.forEach(s => {
                document.querySelectorAll(s).forEach(el => el.remove());
            });

            // Try main content containers first
            const containers = ['article', 'main', '[role="main"]',
                              '.post-content', '.article-content', '.entry-content',
                              '.se-main-container', '.ContentRenderer',
                              '#content', '.content'];
            for (const sel of containers) {
                const el = document.querySelector(sel);
                if (el && el.innerText.trim().length > 100) {
                    return el.innerText.trim();
                }
            }
            return document.body.innerText.trim();
        }""")

        # Truncate to reasonable size
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars, total: {len(text)}]"

        result_parts = [f"Title: {title}", f"URL: {url}", "", text]

        # Extract links if requested
        if extract_links:
            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim().substring(0, 100), href: a.href}))
                    .filter(l => l.href.startsWith('http') && l.text.length > 0)
                    .slice(0, 50);
            }""")
            if links:
                result_parts.append("\n--- Links ---")
                for link in links:
                    result_parts.append(f"  [{link['text']}] {link['href']}")

        result = "\n".join(result_parts)

        # Cache the result
        try:
            db_execute(
                "INSERT OR REPLACE INTO crawl_cache (url, content, title) VALUES (?, ?, ?)",
                (url, result, title),
            )
        except Exception as e:
            logger.warning("Failed to cache crawl result: %s", e)

        # Save cookies after each crawl (login sessions, etc.)
        await _save_cookies()

        return result
    except Exception as e:
        return f"Crawl failed for {url}: {e}"
    finally:
        await page.close()


async def close():
    """Shutdown browser and save cookies."""
    global _context, _browser, _playwright
    if _context:
        await _save_cookies()
        await _context.close()
        _context = None
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None
