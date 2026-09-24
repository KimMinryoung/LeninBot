"""Batch creation and collection preserve chunk identity and validation gates."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from runtime_tools.archival_translation import Options
from runtime_tools.archival_translation import batch
from runtime_tools.archival_translation import core as at


class FakeBatches:
    def __init__(self):
        self.created = []
        self.job = None

    def create(self, **kwargs):
        self.created.append(kwargs)
        return NS(name="batches/one")

    def get(self, **kwargs):
        return self.job

    def list(self):
        return []


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.opts = Options(cache_path=Path(self.tmp.name) / "spec.jsonl")
        self.spec = {"id": "test-spec"}
        self.lang = NS(feature="archival_document_translation_en", system_prompt="translate")
        self.profile = NS(provider="gemini", model="gemini-3.1-pro-preview",
                          temperature=0, max_tokens=1000, extra={})
        self.cache = at.Cache(self.opts.cache_path)
        self.chunks = [([(1, {"tag": "p", "lines": ["source one"]})], "prompt one", "key-one"),
                       ([(2, {"tag": "p", "lines": ["source two"]})], "prompt two", "key-two")]
        self.prepared = {"_lang": self.lang}
        self.batches = FakeBatches()
        self.client = NS(batches=self.batches)

    def _pending(self):
        return self.prepared, self.profile, self.cache, self.chunks

    def _submit(self, **kwargs):
        with patch.object(batch, "_pending", return_value=self._pending()), \
             patch.object(at, "preflight"):
            return batch.submit(self.spec, self.opts, client=self.client, **kwargs)

    def test_submit_once_with_stable_response_keys(self):
        result = self._submit()
        self.assertEqual(result["chunks"], 2)
        requests = self.batches.created[0]["src"]
        self.assertEqual([r["metadata"]["key"] for r in requests], ["key-one", "key-two"])
        self.assertEqual(requests[0]["config"].system_instruction, "translate")
        with self.assertRaises(at.SpecError):
            self._submit()
        self.assertEqual(len(self.batches.created), 1)

    def test_submit_can_limit_chunks_without_changing_collection_scope(self):
        result = self._submit(max_chunks=1)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(len(self.batches.created[0]["src"]), 1)

    def test_collect_only_valid_matching_chunks(self):
        self._submit()
        good = NS(metadata={"key": "key-one"}, error=None,
                  response=NS(text="[[1|p]]\n번역된 문장", candidates=[NS(finish_reason="STOP")]))
        bad = NS(metadata={"key": "key-two"}, error=NS(message="failed"), response=None)
        self.batches.job = NS(state=NS(name="JOB_STATE_SUCCEEDED"),
                              dest=NS(inlined_responses=[bad, good]))
        with patch.object(batch, "_pending", return_value=self._pending()), \
             patch.object(at, "validate", return_value=[]), \
             patch.object(batch, "_audit_response"):
            result = batch.collect(self.spec, self.opts, client=self.client)
        self.assertEqual(result["collected"], 1)
        self.assertEqual(result["failed"][0]["blocks"], [2])
        self.assertEqual(self.cache.get("key-one")["blocks"], {"1": ["번역된 문장"]})
        self.assertIsNone(self.cache.get("key-two"))

    def test_mismatched_response_keys_never_enter_cache(self):
        self._submit()
        wrong = NS(metadata={"key": "different"}, response=NS(text=""), error=None)
        self.batches.job = NS(state=NS(name="JOB_STATE_SUCCEEDED"),
                              dest=NS(inlined_responses=[wrong, wrong]))
        with patch.object(batch, "_pending", return_value=self._pending()):
            with self.assertRaises(at.SpecError):
                batch.collect(self.spec, self.opts, client=self.client)
        self.assertFalse(self.cache.data)

    def test_new_batch_requires_terminal_collected_job(self):
        self._submit()
        self.batches.job = NS(state=NS(name="JOB_STATE_PENDING"))
        with self.assertRaises(at.SpecError):
            self._submit(new_batch=True)
        self.batches.job.state.name = "JOB_STATE_SUCCEEDED"
        with self.assertRaises(at.SpecError):
            self._submit(new_batch=True)
        self.assertEqual(len(self.batches.created), 1)

    def test_batch_audit_keeps_cache_hit_at_standard_rate(self):
        response = NS(response=NS(usage_metadata=NS(
            prompt_token_count=1000, candidates_token_count=100,
            thoughts_token_count=0, cached_content_token_count=200)), error=None)
        manifest = {"feature": self.lang.feature, "spec_id": self.spec["id"],
                    "job_name": "batches/one"}
        with patch("llm.gateway.record_llm_call") as record:
            batch._audit_response(manifest, {"key": "key-one"}, response, self.profile)
        # (800 uncached input * $2 + 100 output * $12) / 2,
        # plus 200 cached input * $0.20 at the standard rate, per million.
        self.assertAlmostEqual(record.call_args.kwargs["cost_usd"], 0.00144)


if __name__ == "__main__":
    unittest.main()
