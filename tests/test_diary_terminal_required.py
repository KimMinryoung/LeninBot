"""Scheduled diary runs must ask the loop to require save_diary.

Run from repo root:  venv/bin/python -m unittest tests.test_diary_terminal_required -v
"""
import asyncio
import unittest

from telegram.tasks import _run_task_llm


def _run(**overrides):
    seen = {}

    async def chat_fn(messages, **kwargs):
        seen.update(kwargs)
        return "report"

    async def model_fn():
        return "m"

    kwargs = dict(
        task_id=1, task_user_id=1, agent_name="diary", content="c",
        chat_with_tools_fn=chat_fn, get_model_fn=model_fn,
        task_system_prompt="s", max_tokens_task=10, max_input_tokens_task=10,
        max_output_continuations=0, thinking_policy="tool_loop",
        thinking_budget_tokens=0, budget_usd=1.0, extra_tools=None,
        extra_handlers=None, finalization_tools=["save_diary"],
        terminal_tools=["save_diary"],
    )
    kwargs.update(overrides)
    asyncio.run(_run_task_llm(**kwargs))
    return seen


class RunTaskLlmTerminalRequiredTests(unittest.TestCase):
    def test_passes_terminal_required_when_set(self):
        self.assertIs(_run(terminal_required=True).get("terminal_required"), True)

    def test_omits_kwarg_for_chat_fns_without_it(self):
        # Moon and Codex chat fns reject an unknown terminal_required kwarg.
        self.assertNotIn("terminal_required", _run())


if __name__ == "__main__":
    unittest.main()
