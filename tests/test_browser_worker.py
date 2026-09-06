import contextvars
import types
import unittest
from unittest.mock import patch

from browser import worker
from tool_gateway.security import get_caller


class _FakeSpec:
    name = "browser"
    model = "medium"
    max_rounds = 3
    budget_usd = 0.25

    def effective_provider(self, default):
        return default

    def filter_tools(self, tools, handlers):
        return list(tools), dict(handlers)

    def render_prompt(self, provider):
        return "browser prompt"


class _FakePolicy:
    max_output_tokens = 1024
    max_input_tokens = 4096
    max_output_continuations = 0
    max_rounds = 3
    budget_usd = 0.25
    thinking_policy = "tool_loop"
    thinking_budget_tokens = 0


class BrowserWorkerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_browser_task_accepts_task_provenance_kwargs(self):
        observed = {}

        async def fake_deepseek_chat(messages, **kwargs):
            observed["caller"] = get_caller()
            return "completed"

        async def fake_process_task(bot, task, **kwargs):
            observed["response"] = await kwargs["chat_with_tools_fn"](
                [{"role": "user", "content": "test"}],
                task_id=task["id"],
                user_id=str(task["user_id"]),
                agent_name="browser",
                runtime_kind="task",
                scope_type="telegram_task",
                scope_id=str(task["id"]),
            )

        fake_telegram_bot = types.ModuleType("telegram.bot")
        fake_telegram_bot.current_task_ctx = contextvars.ContextVar("current_task_ctx")
        fake_telegram_bot._build_runtime_prelude = lambda *args, **kwargs: "runtime"
        fake_telegram_bot._join_context_blocks = lambda *parts: "\n".join(parts)
        fake_telegram_bot._merge_runtime_context_into_last_user = lambda messages, context: messages

        with (
            patch.dict("sys.modules", {"telegram.bot": fake_telegram_bot}),
            patch("agents.get_agent", return_value=_FakeSpec()),
            patch("tool_gateway.inference.resolve_agent_inference_policy", return_value=_FakePolicy()),
            patch("self_runtime.tools.build_task_context_tools", return_value=([], {})),
            patch("runtime_tools.registry.MISSION_TOOL", {}),
            patch("runtime_tools.registry.build_mission_handler", return_value=None),
            patch("telegram.tasks.process_task", side_effect=fake_process_task),
            patch("db.query_one", return_value={"status": "done", "result": "completed"}),
            patch.object(worker, "BROWSER_PROVIDER_OVERRIDE", "deepseek"),
            patch.object(worker, "_init_tools", return_value=([], {})),
            patch.object(worker, "_init_provider_client", return_value=object()),
            patch("llm.claude_loop.chat_with_tools", side_effect=fake_deepseek_chat),
        ):
            result = await worker.execute_browser_task(
                {"id": 1312, "user_id": 7, "agent_type": "browser", "content": "test"}
            )

        self.assertEqual(result["status"], "done")
        self.assertEqual(observed["response"], "completed")
        self.assertEqual(observed["caller"].interface, "agent")
        self.assertEqual(observed["caller"].agent_name, "browser")
        self.assertEqual(observed["caller"].user_id, "7")
        self.assertEqual(observed["caller"].task_id, "1312")
        self.assertEqual(observed["caller"].scope_type, "telegram_task")
        self.assertEqual(observed["caller"].scope_id, "1312")


if __name__ == "__main__":
    unittest.main()
