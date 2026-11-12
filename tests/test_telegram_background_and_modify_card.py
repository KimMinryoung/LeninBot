"""Background tasks are held and their failures logged; /modify cards are HTML-escaped."""
import asyncio
import unittest

from telegram import background


class SpawnTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_is_held_until_done_and_failure_is_logged(self):
        gate = asyncio.Event()

        async def boom():
            await gate.wait()
            raise RuntimeError("nope")

        task = background.spawn(boom(), name="boom")
        self.assertIn(task, background._tasks)
        with self.assertLogs(background.logger, level="ERROR") as logs:
            gate.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self.assertNotIn(task, background._tasks)
        self.assertIn("background task boom failed", logs.output[0])


class ModifyCardTests(unittest.TestCase):
    def test_markdown_and_html_metacharacters_are_escaped(self):
        from telegram.commands import _modify_summary_html
        card = _modify_summary_html("a_b*.py", "fix *bold* <tag> & `x`", 1, 2, "+```\n-<b>")
        self.assertIn("<code>a_b*.py</code>", card)
        self.assertIn("fix *bold* &lt;tag&gt; &amp; `x`", card)
        self.assertIn("<pre>+```\n-&lt;b&gt;</pre>", card)


if __name__ == "__main__":
    unittest.main()
