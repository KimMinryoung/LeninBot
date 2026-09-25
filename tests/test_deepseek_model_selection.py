"""Selection compatibility across runtime entry points and scheduled jobs."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llm.provider_registry import resolve_deepseek_model


class DeepSeekSelectionTests(unittest.TestCase):
    def test_runtime_entry_points_resolve_old_ids_to_current_model(self):
        from bot_config import _resolve_deepseek_model
        from browser.worker import _normalize_browser_model
        from browser.use_agent import _normalize_model
        for resolve in (resolve_deepseek_model, _resolve_deepseek_model,
                        _normalize_browser_model, lambda key: _normalize_model(key, "deepseek")):
            for key in ("high", "medium", "low", "deepseek_pro", "deepseek_flash"):
                with self.subTest(resolve=resolve, key=key):
                    self.assertEqual(resolve(key), "deepseek-flash")
            self.assertEqual(resolve("deepseek-v4-pro"), "deepseek-flash")
            self.assertEqual(resolve("deepseek-future"), "deepseek-future")

    def test_writer_saved_pro_choice_resolves_and_catalog_is_unique(self):
        from writer import models
        with patch.object(models, "get_writer_setting", return_value="deepseek_pro"):
            self.assertEqual(models.get_selected_model_choice(), "deepseek_flash")
        with patch.object(models, "set_writer_setting") as save:
            self.assertEqual(models.set_selected_model_choice("deepseek_pro"), "deepseek_flash")
            save.assert_called_once_with("model_choice", "deepseek_flash")
        with patch("bot_config._deepseek_anthropic_client", object()):
            self.assertEqual(models.resolve_writer_model("deepseek_pro")[1], "deepseek-flash")
        with patch.object(models, "_deepseek_available", return_value=True), patch.object(models, "_kimi_available", return_value=False):
            entries = [m for m in models.list_writer_models() if m["provider"] == "deepseek"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["key"], "deepseek_flash")
        with self.assertRaises(ValueError):
            models.set_selected_model_choice("unknown")

    def test_scheduled_agent_and_oneshot_defaults_use_flash(self):
        from agents.commulingo_curator import COMMULINGO_CURATOR
        self.assertEqual(COMMULINGO_CURATOR.model, "deepseek_flash")
        root = Path(__file__).resolve().parents[1]
        for filename in ("agent_runtime.json", "llm_call_sites.json"):
            path = root / "config" / filename
            if not path.exists():  # agent_runtime.json is local; a fresh clone has the example
                path = path.with_name(filename + ".example")
            entries = json.loads(path.read_text())
            for name, spec in entries.items():
                if spec.get("provider") in ("deepseek", "deepseek_anthropic"):
                    with self.subTest(filename=filename, name=name):
                        if filename == "llm_call_sites.json":
                            from llm.call_registry import resolve as resolve_call_site
                            model = resolve_call_site(name).model
                        else:
                            model = spec.get("model")
                        self.assertEqual(resolve_deepseek_model(model), "deepseek-flash")


class BrowserUseResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsuccessful_done_is_not_reported_as_success(self):
        from browser.use_agent import _run_browser_use_agent

        history = SimpleNamespace(
            is_done=lambda: True, is_successful=lambda: False,
            has_errors=lambda: False, final_result=lambda: "incomplete",
            extracted_content=lambda: [], number_of_steps=lambda: 1,
            urls=lambda: [], errors=lambda: [], total_duration_seconds=lambda: 1.0,
        )
        agent = SimpleNamespace(run=AsyncMock(return_value=history), browser_session=None)
        browser = MagicMock()
        with patch("browser.use_agent._build_llm", return_value=object()), \
             patch("browser.use_agent._build_browser", return_value=browser), \
             patch("browser.use_agent.Agent", return_value=agent):
            result = await _run_browser_use_agent(
                "inspect", provider="google", model="gemini-3.8-flash",
                use_vision=True, max_steps=1, start_url=None,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["result"], "incomplete")
