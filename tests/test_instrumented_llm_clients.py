"""Hermetic tests for third-party SDK gateway wrappers."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.instrumented_clients import AuditedGenAIClient


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


if __name__ == "__main__":
    unittest.main()
