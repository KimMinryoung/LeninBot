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
from browser_use.llm.anthropic.chat import ChatAnthropic

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_COOKIE_PATH = os.path.join(_DATA_DIR, "browser_use_cookies.json")

# Default limits
_DEFAULT_MAX_STEPS = 20
_DEFAULT_BROWSER_USE_PROVIDER = "deepseek"
_DEFAULT_BROWSER_USE_MODEL = "deepseek-v4-flash"


class _DeepSeekAnthropicBrowserChat(ChatAnthropic):
    """ChatAnthropic wrapper that sends DeepSeek thinking controls per request."""

    def _get_client_params_for_invoke(self):
        from bot_config import _get_deepseek_thinking_params

        params = super()._get_client_params_for_invoke()
        params.update(_get_deepseek_thinking_params())
        return params


def _resolve_provider_and_model() -> tuple[str, str]:
    """Resolve the multimodal LLM used inside browser-use.

    browser-use supports multiple multimodal providers. Browser automation is
    high-volume, so the default must not use Claude API.
    Override with BROWSER_USE_PROVIDER/BROWSER_USE_MODEL for controlled tests.
    """
    provider = (os.getenv("BROWSER_USE_PROVIDER") or _DEFAULT_BROWSER_USE_PROVIDER).strip().lower()
    model = (os.getenv("BROWSER_USE_MODEL") or _DEFAULT_BROWSER_USE_MODEL).strip()
    return provider, model


def _build_llm(model: str | None = None):
    """Build browser-use LLM based on current runtime provider config.

    Browser automation must not use Claude API by default or fallback.
    """
    provider, default_model = _resolve_provider_and_model()
    model = model or default_model
    if str(model).lower().startswith("claude") or str(model).lower() in {"opus", "sonnet", "haiku"}:
        logger.warning("browser-use forbids Claude model override %r; using deepseek-v4-flash", model)
        model = "deepseek-v4-flash"

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
        llm = _DeepSeekAnthropicBrowserChat(
            model=model,
            api_key=get_secret("DEEPSEEK_API_KEY", "") or "",
            base_url=os.getenv(
                "DEEPSEEK_ANTHROPIC_BASE_URL",
                "https://api.deepseek.com/anthropic",
            ).rstrip("/"),
            timeout=120,
        )
        logger.info("browser-use LLM: DeepSeek Anthropic-compatible %s", model)
        return llm

    if provider == "claude" or provider == "anthropic":
        logger.warning("browser-use forbids Claude provider; using DeepSeek instead")

    llm = _DeepSeekAnthropicBrowserChat(
        model="deepseek-v4-flash",
        api_key=get_secret("DEEPSEEK_API_KEY", "") or "",
        base_url=os.getenv(
            "DEEPSEEK_ANTHROPIC_BASE_URL",
            "https://api.deepseek.com/anthropic",
        ).rstrip("/"),
        timeout=120,
    )
    logger.info("browser-use LLM: DeepSeek Anthropic-compatible deepseek-v4-flash")
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
        model: LLM model override (default: deepseek-v4-flash).
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
