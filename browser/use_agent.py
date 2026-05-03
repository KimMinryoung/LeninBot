"""browser-use integration — AI-driven browser automation for complex web tasks.

Uses browser-use (Playwright + LLM) for tasks that require multi-step
interaction: filling forms, navigating multi-page flows, extracting data
from dynamic sites, etc.

Simple page fetches should still use shared.fetch_url_text (faster, cheaper).
"""

import asyncio
import json
import logging
import os

from browser_use import Agent, Browser

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_COOKIE_PATH = os.path.join(_DATA_DIR, "browser_use_cookies.json")

# Default limits
_DEFAULT_MAX_STEPS = 20
_DEFAULT_BROWSER_USE_PROVIDER = "google"
_DEFAULT_BROWSER_USE_MODEL = "gemini-2.5-flash"


def _resolve_provider_and_model() -> tuple[str, str]:
    """Resolve the multimodal LLM used inside browser-use.

    The outer browser worker may still use Claude for LeninBot tool orchestration,
    but browser-use's screen-reading/clicking loop should default to a cheaper
    multimodal model. Override with BROWSER_USE_PROVIDER/BROWSER_USE_MODEL.
    """
    provider = (os.getenv("BROWSER_USE_PROVIDER") or _DEFAULT_BROWSER_USE_PROVIDER).strip().lower()
    model = (os.getenv("BROWSER_USE_MODEL") or _DEFAULT_BROWSER_USE_MODEL).strip()
    return provider, model


def _build_llm(model: str | None = None):
    """Build browser-use LLM based on current runtime provider config.

    Supports both Anthropic (ChatAnthropic) and OpenAI (ChatOpenAI).
    """
    provider, default_model = _resolve_provider_and_model()
    model = model or default_model

    from secrets_loader import get_secret

    if provider == "google":
        from browser_use.llm.google.chat import ChatGoogle

        llm = ChatGoogle(
            model=model,
            api_key=get_secret("GEMINI_API_KEY", "") or "",
            thinking_budget=0,
            max_output_tokens=4096,
        )
        logger.info("browser-use LLM: Google %s", model)
        return llm

    if provider == "openai":
        from browser_use.llm.openai.chat import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=get_secret("OPENAI_API_KEY", "") or "",
            timeout=120,
            max_completion_tokens=4096,
        )
        logger.info("browser-use LLM: OpenAI %s", model)
        return llm

    if provider == "deepseek":
        from browser_use.llm.deepseek.chat import ChatDeepSeek

        llm = ChatDeepSeek(
            model=model,
            api_key=get_secret("DEEPSEEK_API_KEY", "") or "",
            timeout=120,
        )
        logger.info("browser-use LLM: DeepSeek %s", model)
        return llm

    # Default: Anthropic
    from browser_use.llm.anthropic.chat import ChatAnthropic

    llm = ChatAnthropic(
        model=model,
        api_key=get_secret("ANTHROPIC_API_KEY", "") or "",
        max_tokens=8192,
        timeout=120,
    )
    logger.info("browser-use LLM: Anthropic %s", model)
    return llm


def _find_chromium() -> str | None:
    """Find the Playwright-managed Chromium binary."""
    import glob as _glob
    candidates = _glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")
    )
    if candidates:
        candidates.sort(reverse=True)  # newest first
        return candidates[0]
    return None


def _build_browser() -> Browser:
    """Build a browser-use Browser with settings matching our existing crawler."""
    storage_state = None
    if os.path.exists(_COOKIE_PATH):
        try:
            with open(_COOKIE_PATH, "r", encoding="utf-8") as f:
                storage_state = json.load(f)
        except Exception as e:
            logger.warning("Failed to load cookies for browser-use: %s", e)

    chromium_path = _find_chromium()
    if chromium_path:
        logger.info("Using Playwright Chromium: %s", chromium_path)

    return Browser(
        headless=True,
        disable_security=True,
        executable_path=chromium_path,
        chromium_sandbox=False,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        storage_state=storage_state,
    )


async def _save_cookies_from_session(browser_session):
    """Persist cookies from the browser-use session back to our cookie file."""
    try:
        ctx = getattr(browser_session, "playwright_context", None)
        if ctx is None:
            ctx = getattr(browser_session, "context", None)
        if ctx is None:
            return
        raw = ctx.cookies()
        # Handle both sync (CDP) and async (Playwright) cookies() return
        if asyncio.iscoroutine(raw) or asyncio.isfuture(raw):
            cookies = await raw
        else:
            cookies = raw
        os.makedirs(_DATA_DIR, exist_ok=True)
        state = {"cookies": cookies, "origins": []}
        with open(_COOKIE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d cookies from browser-use session", len(cookies))
    except Exception as e:
        logger.warning("Failed to save cookies from browser-use: %s", e)


async def browse(
    task: str,
    *,
    max_steps: int = _DEFAULT_MAX_STEPS,
    model: str | None = None,
    start_url: str | None = None,
) -> dict:
    """Run a browser-use agent to complete a web task.

    Args:
        task: Natural language description of the task.
        max_steps: Maximum number of browser interaction steps.
        model: LLM model override (default: claude-sonnet-4-6).
        start_url: Optional URL to open before starting the task.

    Returns:
        dict with keys: success, result, steps, urls, errors, duration_seconds
    """
    llm = _build_llm(model)
    browser = _build_browser()

    initial_actions = None
    if start_url:
        initial_actions = [{"navigate": {"url": start_url}}]

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        initial_actions=initial_actions,
        use_vision=True,
        generate_gif=False,
        max_failures=3,
    )

    try:
        history = await agent.run(max_steps=max_steps)

        # Try to save cookies
        if agent.browser_session:
            await _save_cookies_from_session(agent.browser_session)

        return {
            "success": history.is_done() and not history.has_errors(),
            "result": history.final_result() or "",
            "extracted_content": history.extracted_content(),
            "steps": history.number_of_steps(),
            "urls": history.urls(),
            "errors": history.errors(),
            "duration_seconds": round(history.total_duration_seconds(), 1),
        }
    except Exception as e:
        logger.error("browser-use agent failed: %s", e, exc_info=True)
        return {
            "success": False,
            "result": f"Agent error: {e}",
            "extracted_content": [],
            "steps": 0,
            "urls": [],
            "errors": [str(e)],
            "duration_seconds": 0,
        }
    finally:
        try:
            raw = browser.close()
            if asyncio.iscoroutine(raw) or asyncio.isfuture(raw):
                await raw
        except Exception:
            pass
