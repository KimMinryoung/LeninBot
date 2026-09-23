"""Computer-use browser action and selection contracts (no paid API calls)."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from browser.computer_use import _execute_action, _final_result, _start_url


class ComputerUseContracts(unittest.IsolatedAsyncioTestCase):
    async def test_mouse_and_keyboard_actions_execute_on_page(self):
        mouse = SimpleNamespace(click=AsyncMock(), move=AsyncMock(), wheel=AsyncMock())
        keyboard = SimpleNamespace(down=AsyncMock(), up=AsyncMock(), press=AsyncMock(),
                                   insert_text=AsyncMock())
        page = SimpleNamespace(mouse=mouse, keyboard=keyboard)
        await _execute_action(page, {"type": "click", "x": 100, "y": 50,
                                     "button": "left", "keys": ["SHIFT"]})
        keyboard.down.assert_awaited_once_with("Shift")
        mouse.click.assert_awaited_once_with(100.0, 50.0, button="left")
        keyboard.up.assert_awaited_once_with("Shift")
        await _execute_action(page, {"type": "scroll", "x": 10, "y": 20,
                                     "scroll_x": 0, "scroll_y": 430})
        mouse.wheel.assert_awaited_once_with(0.0, 430.0)
        await _execute_action(page, {"type": "keypress", "keys": ["CTRL", "A"]})
        keyboard.press.assert_awaited_once_with("Control+A")

    async def test_out_of_bounds_action_is_refused(self):
        page = SimpleNamespace(mouse=SimpleNamespace(click=AsyncMock()),
                               keyboard=SimpleNamespace())
        with self.assertRaises(ValueError):
            await _execute_action(page, {"type": "click", "x": 9000, "y": 20})
        page.mouse.click.assert_not_awaited()

    async def test_browse_dispatches_to_native_mode(self):
        from browser.use_agent import browse
        expected = {"success": True, "mode": "computer"}
        with patch("browser.computer_use.browse_with_computer",
                   new=AsyncMock(return_value=expected)) as native:
            result = await browse("Read page", mode="computer", start_url="https://example.org",
                                  model="tier:low", max_steps=3)
        self.assertIs(result, expected)
        native.assert_awaited_once_with("Read page", max_steps=3,
                                        start_url="https://example.org", model="tier:low")

    async def test_browse_defaults_to_luna_computer_mode(self):
        from browser.use_agent import browse
        with patch("browser.computer_use.browse_with_computer",
                   new=AsyncMock(return_value={"success": True})) as native:
            await browse("Find information")
        native.assert_awaited_once_with("Find information", max_steps=20,
                                        start_url=None, model="tier:low")

    async def test_registered_tool_passes_computer_mode(self):
        from runtime_tools.media import _exec_browse_web
        result = {
            "success": True, "result": "Page read", "steps": 2,
            "urls": ["https://example.org"], "errors": [],
            "duration_seconds": 4.2, "provider": "openai", "model": "gpt-6-luna",
            "use_vision": True, "extracted_content": [], "mode": "computer",
        }
        with patch("browser.use_agent.browse", new=AsyncMock(return_value=result)) as browse:
            output = await _exec_browse_web("Read page", start_url="https://example.org",
                                            max_steps=3, mode="computer", model="tier:low")
        browse.assert_awaited_once_with("Read page", max_steps=3,
                                        start_url="https://example.org",
                                        mode="computer", model="tier:low")
        self.assertIn("[OK]", output)
        self.assertIn("Mode: native computer use", output)

    async def test_registered_tool_defaults_to_computer_mode(self):
        from runtime_tools.media import _exec_browse_web
        result = {"success": True, "result": "Done", "steps": 1,
                  "urls": [], "errors": [], "duration_seconds": 0.1,
                  "provider": "openai", "model": "gpt-6-luna",
                  "use_vision": True, "extracted_content": [], "mode": "computer"}
        with patch("browser.use_agent.browse", new=AsyncMock(return_value=result)) as browse:
            await _exec_browse_web("Find information")
        browse.assert_awaited_once_with("Find information", max_steps=20,
                                        start_url=None, mode="computer", model=None)

    def test_url_and_completion_must_be_explicit(self):
        self.assertEqual(_start_url("Visit https://example.org/ now", None),
                         "https://example.org/")
        self.assertEqual(_start_url("Read the page", None), "https://www.google.com/")
        with self.assertRaises(ValueError):
            _start_url("Read the page", "not-a-url")
        self.assertEqual(_final_result(SimpleNamespace(output_text="STATUS: COMPLETED\nDone")),
                         (True, "Done"))
        self.assertEqual(_final_result(SimpleNamespace(output_text="STATUS: INCOMPLETE\nBlocked")),
                         (False, "Blocked"))


if __name__ == "__main__":
    unittest.main()
