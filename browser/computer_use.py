"""OpenAI native computer-use loop over an isolated Playwright browser.

The model requests screen actions through the Responses ``computer`` tool.
This module executes them and returns fresh screenshots automatically. The
The browser-use agent is available through explicit ``mode=agent``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright

from llm.provider_registry import current_text_model

logger = logging.getLogger(__name__)

_WIDTH = 1280
_HEIGHT = 800
_MAX_CALLS = 20
_MAX_ACTIONS_PER_CALL = 12
_MAX_RUN_SECONDS = 180
_URL_IN_TASK = re.compile(r"https?://[^\s<>)\]}]+")
_DEFAULT_START_URL = "https://www.google.com/"
_KEY_NAMES = {
    "CTRL": "Control", "CONTROL": "Control", "ALT": "Alt",
    "SHIFT": "Shift", "META": "Meta", "CMD": "Meta",
    "ENTER": "Enter", "RETURN": "Enter", "ESC": "Escape",
    "ESCAPE": "Escape", "BACKSPACE": "Backspace", "DELETE": "Delete",
    "TAB": "Tab", "SPACE": "Space", "ARROWUP": "ArrowUp",
    "ARROWDOWN": "ArrowDown", "ARROWLEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight", "HOME": "Home", "END": "End",
    "PAGEUP": "PageUp", "PAGEDOWN": "PageDown",
}


def _start_url(task: str, start_url: str | None) -> str:
    url = (start_url if start_url is not None else
           (match.group(0).rstrip(".,;") if (match := _URL_IN_TASK.search(task)) else _DEFAULT_START_URL))
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("computer mode needs an http(s) start_url or a URL in task")
    return url


def _point(action: dict, x_key: str = "x", y_key: str = "y") -> tuple[float, float]:
    x, y = float(action[x_key]), float(action[y_key])
    if not (0 <= x < _WIDTH and 0 <= y < _HEIGHT):
        raise ValueError(f"computer action point outside {_WIDTH}x{_HEIGHT} viewport: {x},{y}")
    return x, y


def _key(key: str) -> str:
    key = str(key)
    return _KEY_NAMES.get(key.upper(), key if len(key) == 1 else key.title())


@asynccontextmanager
async def _mouse_modifiers(page: Page, action: dict):
    keys = action.get("keys") or []
    if not isinstance(keys, list):
        raise ValueError("mouse modifiers must be a list")
    mapped = [_key(key) for key in keys]
    if any(key not in {"Control", "Alt", "Shift", "Meta"} for key in mapped):
        raise ValueError("mouse action has an unsupported modifier")
    try:
        for key in mapped:
            await page.keyboard.down(key)
        yield
    finally:
        for key in reversed(mapped):
            await page.keyboard.up(key)


async def _execute_action(page: Page, action: dict) -> None:
    """Execute one validated computer action against the visible page."""
    kind = action.get("type")
    if kind == "screenshot":
        return
    if kind == "wait":
        await asyncio.sleep(min(float(action.get("ms", 1000)) / 1000, 5))
        return
    if kind == "keypress":
        keys = action.get("keys") or []
        if not isinstance(keys, list) or not keys:
            raise ValueError("keypress needs keys")
        await page.keyboard.press("+".join(_key(key) for key in keys))
        return
    if kind == "type":
        value = action.get("text")
        if not isinstance(value, str) or len(value) > 10000:
            raise ValueError("type needs text of at most 10000 characters")
        await page.keyboard.insert_text(value)
        return
    if kind == "drag":
        path = action.get("path") or []
        if not isinstance(path, list) or not 2 <= len(path) <= 50:
            raise ValueError("drag needs 2..50 path points")
        points = [_point(point) for point in path]
        async with _mouse_modifiers(page, action):
            await page.mouse.move(*points[0])
            await page.mouse.down()
            try:
                for point in points[1:]:
                    await page.mouse.move(*point)
            finally:
                await page.mouse.up()
        return
    if kind not in {"click", "double_click", "move", "scroll"}:
        raise ValueError(f"unsupported computer action: {kind}")
    x, y = _point(action)
    async with _mouse_modifiers(page, action):
        if kind == "click":
            button = action.get("button", "left")
            if button not in {"left", "right", "middle"}:
                raise ValueError(f"unsupported mouse button: {button}")
            await page.mouse.click(x, y, button=button)
        elif kind == "double_click":
            await page.mouse.dblclick(x, y)
        elif kind == "move":
            await page.mouse.move(x, y)
        else:
            await page.mouse.move(x, y)
            await page.mouse.wheel(float(action.get("scroll_x", 0)), float(action.get("scroll_y", 0)))


def _screenshot_output(call_id: str, image: bytes) -> dict:
    return {
        "type": "computer_call_output", "call_id": call_id,
        "output": {
            "type": "computer_screenshot",
            "image_url": "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
            "detail": "original",
        },
    }


def _final_result(response) -> tuple[bool, str]:
    answer = str(response.output_text or "").strip()
    first, _, remainder = answer.partition("\n")
    success = first.strip().upper() == "STATUS: COMPLETED"
    if first.strip().upper().startswith("STATUS:"):
        answer = remainder.strip()
    return success, answer or "The model stopped without a result."


async def browse_with_computer(
    task: str, *, start_url: str | None = None,
    max_steps: int = 20, model: str = "tier:low",
) -> dict:
    """Run one native computer-use task in a fresh, headless Chromium context."""
    started = time.monotonic()
    selected = current_text_model("openai", model)
    if selected not in {"gpt-6-luna", "gpt-6-sol"}:
        raise ValueError("computer mode supports OpenAI tier:low (Luna) or tier:high (Sol)")
    url = _start_url(task, start_url)
    calls = 0
    visited: list[str] = []
    errors: list[str] = []
    from bot_config import _openai_client
    if _openai_client is None:
        raise RuntimeError("OpenAI gateway client is unavailable")

    async with asyncio.timeout(_MAX_RUN_SECONDS):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": _WIDTH, "height": _HEIGHT},
                    accept_downloads=False,
                )
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except Exception as exc:
                    logger.warning("computer initial navigation to %s: %s", url, exc)
                visited.append(page.url)
                image = await page.screenshot(type="png")
                instructions = (
                    "Use the computer tool to complete this browser task. The browser viewport is "
                    f"{_WIDTH}x{_HEIGHT}; a screenshot of its current state is attached. "
                    "When finished, begin your final answer with exactly STATUS: COMPLETED "
                    "or STATUS: INCOMPLETE, then report the result and evidence. "
                    "Do not claim a result that the screen did not verify.\n\nTask: " + task
                )
                response = await _openai_client.responses.create(
                    model=selected, tools=[{"type": "computer"}],
                    input=[{"role": "user", "content": [
                        {"type": "input_text", "text": instructions},
                        {"type": "input_image", "image_url": "data:image/png;base64,"
                         + base64.b64encode(image).decode("ascii"), "detail": "original"},
                    ]}],
                    reasoning={"effort": "low"}, max_output_tokens=1024,
                    extra_headers={"x-llm-caller": "browser_native_computer"},
                )
                limit = min(max(1, int(max_steps)), _MAX_CALLS)
                while True:
                    calls += 1
                    computer_calls = [item for item in response.output if item.type == "computer_call"]
                    if not computer_calls:
                        success, answer = _final_result(response)
                        return {
                            "success": success and response.status == "completed",
                            "result": answer, "extracted_content": [], "steps": calls,
                            "urls": visited, "errors": errors,
                            "duration_seconds": round(time.monotonic() - started, 1),
                            "provider": "openai", "model": selected, "use_vision": True,
                            "mode": "computer",
                        }
                    if calls > limit:
                        break
                    outputs = []
                    for call in computer_calls:
                        # OpenAI SDK 2.16 preserves this newer tool's actions in
                        # model_dump(), but its typed class has no .actions attr.
                        actions = call.model_dump(exclude_none=True).get("actions", [])
                        if not actions or len(actions) > _MAX_ACTIONS_PER_CALL:
                            raise ValueError("computer call has an invalid action count")
                        for action in actions:
                            # A click may open a new tab; continue on the newest live page.
                            page = next((p for p in reversed(context.pages) if not p.is_closed()), page)
                            await _execute_action(page, action)
                        page = next((p for p in reversed(context.pages) if not p.is_closed()), page)
                        if page.url and page.url not in visited:
                            visited.append(page.url)
                        outputs.append(_screenshot_output(call.call_id, await page.screenshot(type="png")))
                    response = await _openai_client.responses.create(
                        model=selected, tools=[{"type": "computer"}],
                        previous_response_id=response.id, input=outputs,
                        reasoning={"effort": "low"}, max_output_tokens=1024,
                        extra_headers={"x-llm-caller": "browser_native_computer"},
                    )
                return {
                    "success": False, "result": "Computer-use step limit reached.",
                    "extracted_content": [], "steps": calls, "urls": visited,
                    "errors": ["step limit reached"],
                    "duration_seconds": round(time.monotonic() - started, 1),
                    "provider": "openai", "model": selected, "use_vision": True,
                    "mode": "computer",
                }
            finally:
                await browser.close()
