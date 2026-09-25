"""DeepSeek tool loops use the OpenAI-compatible endpoint by default."""
import os
import unittest
from unittest.mock import patch

import bot_config
from agents.commulingo_curator import COMMULINGO_CURATOR
from tool_gateway.inference import resolve_agent_inference_policy


class DeepSeekToolLoopBindingTests(unittest.TestCase):
    def binding(self, **env):
        policy = resolve_agent_inference_policy(COMMULINGO_CURATOR)
        with patch.dict(os.environ, env), \
             patch.object(bot_config, '_deepseek_client', object()), \
             patch.object(bot_config, '_deepseek_anthropic_client', object()):
            return bot_config.resolve_agent_tool_loop(COMMULINGO_CURATOR, policy)

    def test_openai_compatible_endpoint_is_the_default(self):
        binding = self.binding()
        self.assertEqual(binding.chat.__module__, 'llm.openai_tool_loop')
        self.assertEqual(binding.reasoning['extra_body'], {'thinking': {'type': 'disabled'}})
        self.assertEqual(binding.reasoning['sdk_max_token_param'], 'max_tokens')

    def test_anthropic_endpoint_remains_available(self):
        binding = self.binding(DEEPSEEK_TOOL_LOOP_PROTOCOL='anthropic')
        self.assertEqual(binding.chat.__module__, 'llm.claude_loop')
        self.assertEqual(binding.reasoning, {'thinking': {'type': 'disabled'}})

    def test_thinking_effort_maps_to_reasoning_effort(self):
        options = bot_config._deepseek_openai_loop_options(
            {'thinking': {'type': 'enabled'}, 'output_config': {'effort': 'max'}})
        self.assertEqual(options['extra_body'], {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'max'})
        self.assertTrue(options['preserve_reasoning_content'])


if __name__ == '__main__':
    unittest.main()
