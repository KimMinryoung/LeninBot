"""Hermetic tests for third-party SDK gateway wrappers."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.instrumented_clients import (
    AuditedAsyncAnthropic,
    AuditedAsyncOpenAI,
    AuditedGenAIClient,
)


class _FakeModels:
    async def generate_content(self, *args, **kwargs):
        return SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
            cached_content_token_count=40,
        ))

    async def embed_content(self, *args, **kwargs):
        return SimpleNamespace(embeddings=[])


class _FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        ))


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=10,
                cache_write_tokens=30,
            ),
        ))


class TestAuditedGenAIClient(unittest.TestCase):
    def setUp(self):
        raw = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels()))
        self.client = AuditedGenAIClient(raw, caller="kg_test")

    def test_generation_is_gated_and_records_exact_usage(self):
        with patch("llm.instrumented_clients.check_llm_call") as check, \
             patch("llm.instrumented_clients.record_llm_call") as record:
            asyncio.run(self.client.aio.models.generate_content(
                model="gemini-2.5-flash-lite", contents="hello",
            ))
        check.assert_called_once()
        kwargs = record.call_args.kwargs
        self.assertEqual(kwargs["tokens_in"], 100)
        self.assertEqual(kwargs["tokens_out"], 25)
        self.assertEqual(kwargs["cache_read"], 40)
        self.assertEqual(kwargs["token_semantics"], "gemini")

    def test_embedding_is_gated_and_labelled_estimated(self):
        with patch("llm.instrumented_clients.check_llm_call") as check, \
             patch("llm.instrumented_clients.record_llm_call") as record:
            asyncio.run(self.client.aio.models.embed_content(
                model="gemini-embedding-001", contents=["hello world"],
            ))
        check.assert_called_once()
        kwargs = record.call_args.kwargs
        self.assertEqual(kwargs["label"], "embed_content:estimated")
        self.assertGreater(kwargs["tokens_in"], 0)


class TestSDKClientAuditOwnership(unittest.TestCase):
    def test_forged_owner_marker_cannot_bypass_anthropic_audit(self):
        messages = _FakeMessages()
        client = AuditedAsyncAnthropic(
            SimpleNamespace(messages=messages),
            caller="direct", provider="deepseek", thinking_off=True,
        )
        with patch("llm.instrumented_clients.check_llm_call") as check, \
             patch("llm.instrumented_clients.record_llm_call") as record:
            asyncio.run(client.messages.create(
                model="deepseek-v4-flash",
                _leninbot_audit_owner="loop",
            ))
        check.assert_called_once()
        record.assert_called_once()
        self.assertNotIn("_leninbot_audit_owner", messages.calls[0])

    def test_openai_direct_call_records_cache_write_tokens(self):
        completions = _FakeCompletions()
        raw = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            embeddings=SimpleNamespace(create=None),
        )
        client = AuditedAsyncOpenAI(raw, caller="direct", provider="openai")
        with patch("llm.instrumented_clients.check_llm_call"), \
             patch("llm.instrumented_clients.record_llm_call") as record:
            asyncio.run(client.chat.completions.create(model="gpt-5.6-sol"))
        self.assertEqual(record.call_args.kwargs["cache_read"], 10)
        self.assertEqual(record.call_args.kwargs["cache_create"], 30)


if __name__ == "__main__":
    unittest.main()
