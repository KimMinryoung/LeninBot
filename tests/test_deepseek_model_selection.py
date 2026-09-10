"""Selection compatibility across runtime entry points and scheduled jobs."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from llm.provider_registry import resolve_deepseek_model


class DeepSeekSelectionTests(unittest.TestCase):
    def test_runtime_entry_points_share_alias_and_explicit_id_behavior(self):
        from bot_config import _resolve_deepseek_model
        from browser.worker import _normalize_browser_model
        from browser.use_agent import _normalize_model
        for resolve in (resolve_deepseek_model, _resolve_deepseek_model,
                        _normalize_browser_model, lambda key: _normalize_model(key, "deepseek")):
            for key in ("high", "medium", "low", "deepseek_pro", "deepseek_flash"):
                with self.subTest(resolve=resolve, key=key):
                    self.assertEqual(resolve(key), "deepseek-flash")
            self.assertEqual(resolve("deepseek-v4-pro"), "deepseek-v4-pro")
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
        from agents.commulingo_event_curator import COMMULINGO_EVENT_CURATOR
        for spec in (COMMULINGO_CURATOR, COMMULINGO_EVENT_CURATOR):
            self.assertEqual(spec.model, "deepseek_flash")
        root = Path(__file__).resolve().parents[1]
        for filename in ("agent_runtime.json", "llm_call_sites.json"):
            entries = json.loads((root / "config" / filename).read_text())
            for name, spec in entries.items():
                if spec.get("provider") in ("deepseek", "deepseek_anthropic"):
                    with self.subTest(filename=filename, name=name):
                        self.assertEqual(resolve_deepseek_model(spec.get("model")), "deepseek-flash")
