"""Output truncation recovery through the actual Anthropic/DeepSeek loop."""
import unittest
from unittest.mock import patch
from test_claude_loop_rounds import FakeClient, _response, _text_block, _tool_use_block
from llm.claude_loop import chat_with_tools

DECIDE = 'commulingo_review_decision'
TOOLS = [{'name': DECIDE, 'description': 'record review', 'input_schema': {'type': 'object', 'properties': {}}}]


class ReviewContinuation(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, responses):
        client = FakeClient(responses)
        executed = []
        async def execute(batch, handlers, **kwargs):
            executed.extend(batch)
            return [(tid, name, value, 'review recorded', False) for tid, name, value in batch]
        tracker = {}
        with patch('llm.claude_loop.execute_tools_batch', side_effect=execute), patch('llm.agent_loop.record_llm_call'):
            result = await chat_with_tools(
                [{'role': 'user', 'content': 'Review the existing proposal and fetched evidence.'}],
                client=client, model='deepseek-v4-pro', tools=TOOLS, tool_handlers={},
                system_prompt='Submit the review decision.', budget_usd=.20,
                max_rounds=1, max_tokens=8000, continue_on_length=True,
                max_length_continuations=2, terminal_tools=[DECIDE],
                finalization_tools=[DECIDE], budget_tracker=tracker)
        return client, executed, tracker, result

    async def test_two_truncations_preserve_context_and_finish_decision_once(self):
        client, executed, tracker, _ = await self.run_loop([
            _response([_text_block('First partial review')], stop_reason='max_tokens'),
            _response([_text_block('Second partial review')], stop_reason='max_tokens'),
            _response([_tool_use_block('decision', DECIDE)], stop_reason='tool_use'),
        ])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0][1], DECIDE)
        self.assertEqual(tracker['rounds_used'], 3)
        messages = str(client.calls[-1]['messages'])
        for text in ('First partial review', 'Second partial review', 'fetched evidence', 'Continue exactly'):
            self.assertIn(text, messages)

    async def test_empty_truncation_can_recover_to_complete_tool_call(self):
        client, executed, _, _ = await self.run_loop([
            _response([], stop_reason='max_tokens'),
            _response([_tool_use_block('decision', DECIDE)], stop_reason='tool_use'),
        ])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(executed), 1)
        self.assertIn('before any usable content', str(client.calls[-1]['messages']))

    async def test_continuation_limit_stops_without_executing_partial_decision(self):
        client, executed, _, _ = await self.run_loop([
            _response([_text_block('Still incomplete')], stop_reason='max_tokens')
            for _ in range(3)
        ])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(executed, [])


if __name__ == '__main__':
    unittest.main()
