import unittest
from unittest.mock import MagicMock, patch

from llm import call_registry as cr

SCHEMA = {"type": "object", "properties": {"facts": {"type": "array", "items": {
    "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": ["string", "null"]},
                                     "c": {"type": "array", "items": {"type": "string"}}},
    "required": ["a"]}}}, "required": ["facts"]}


class ClosedSchemaTests(unittest.TestCase):
    def test_claude_schema_is_closed_and_keeps_optional_fields(self):
        item = cr._closed_schema(SCHEMA)["properties"]["facts"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(item["required"], ["a"])

    def test_openai_strict_schema_requires_all_and_nullables_optionals(self):
        item = cr._closed_schema(SCHEMA, all_required=True)["properties"]["facts"]["items"]
        self.assertEqual(item["required"], ["a", "b", "c"])
        self.assertEqual(item["properties"]["b"]["type"], ["string", "null"])
        self.assertEqual(item["properties"]["c"]["type"], ["array", "null"])
        self.assertEqual(item["properties"]["a"]["type"], "string")


def _profile(provider, model, **extra):
    return cr.CallSiteProfile(feature="t", provider=provider, model=model, max_tokens=100,
                              extra={"response_schema": SCHEMA, **extra})


class ExecutorSchemaTests(unittest.TestCase):
    def test_claude_sends_thinking_and_merged_output_config(self):
        client = MagicMock()
        client.messages.create.return_value = MagicMock(content=[], stop_reason="end_turn", usage=None)
        with patch("anthropic.Anthropic", return_value=client), \
             patch.object(cr, "resolve_provider_connection", return_value=MagicMock(base_url=None)):
            cr._generate_claude(_profile("claude", "claude-haiku-5-5", thinking={"type": "disabled"},
                                         output_config={"effort": "low"}), "p", None)
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["output_config"]["effort"], "low")
        self.assertEqual(kwargs["output_config"]["format"]["type"], "json_schema")

    def test_only_openai_gets_json_schema_response_format(self):
        for provider, expected in (("openai", "json_schema"), ("deepseek", "json_object")):
            client = MagicMock()
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="{}"), finish_reason="stop")], usage=None)
            profile = cr.CallSiteProfile(feature="t", provider=provider, model="m", max_tokens=100,
                                         json_mode=True, extra={"response_schema": SCHEMA})
            with patch("openai.OpenAI", return_value=client), \
                 patch.object(cr, "resolve_provider_connection", return_value=MagicMock(base_url=None, api_key="k")):
                try:
                    cr._generate_openai_compat(profile, "p", None)
                except Exception:
                    pass
            self.assertEqual(client.chat.completions.create.call_args.kwargs["response_format"]["type"], expected)


if __name__ == "__main__":
    unittest.main()
