"""The chat/task model getters, resolve_runtime_profile and
get_current_model_selection resolve the same model, including legacy aliases."""
import unittest
from unittest.mock import AsyncMock, patch

import bot_config
from llm.runtime_profile import resolve_runtime_profile

CASES = [
    # (provider, task_provider, chat_model, task_model)
    ("claude", "default", "high", "medium"),
    ("openai", "default", "high", "low"),
    ("deepseek", "default", "high", "high"),
    ("kimi", "default", "medium", "high"),
    ("claude", "deepseek", "high", "high"),
    ("openai", "deepseek", "high", "low"),
    # Legacy aliases saved as the tier, including under a different provider.
    ("claude", "default", "gpt56", "gpt56terra"),
    ("claude", "default", "deepseek_pro", "gpt56luna"),
    ("openai", "default", "gpt56luna", "deepseek_pro"),
    ("deepseek", "claude", "deepseek_pro", "opus"),
]


class ModelResolutionPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_task_match_runtime_profile(self):
        by_alias = AsyncMock(side_effect=lambda alias: f"claude:{alias}")
        for provider, task_provider, chat_model, task_model in CASES:
            config = dict(bot_config._config, provider=provider, task_provider=task_provider,
                          chat_model=chat_model, task_model=task_model)
            with self.subTest(config=(provider, task_provider, chat_model, task_model)), \
                 patch.dict(bot_config._config, config, clear=True), \
                 patch.object(bot_config, "_get_model_by_alias", by_alias):
                self.assertEqual(await bot_config._get_model(),
                                 (await resolve_runtime_profile("chat")).model_id)
                self.assertEqual(await bot_config._get_model_task(),
                                 (await resolve_runtime_profile("task")).model_id)
                for kind in ("chat", "task"):
                    profile = await resolve_runtime_profile(kind)
                    if profile.model_id.startswith("claude:"):
                        continue  # the selection view reads the alias cache instead
                    self.assertEqual(bot_config.get_current_model_selection(kind)["model_id"],
                                     profile.model_id)

    async def test_legacy_alias_under_other_provider_keeps_its_family(self):
        config = dict(bot_config._config, provider="claude", chat_model="gpt56", task_provider="default")
        with patch.dict(bot_config._config, config, clear=True):
            self.assertEqual(await bot_config._get_model(), bot_config._resolve_openai_model("gpt56"))


if __name__ == "__main__":
    unittest.main()
