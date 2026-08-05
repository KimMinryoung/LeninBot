"""Hermetic guards for the CommUlingo evidence-label LLM path."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import commulingo_event_evidence_links as evidence


ROOT = Path(__file__).resolve().parent.parent


class TestEvidenceLabelGateway(unittest.TestCase):
    def setUp(self):
        self.person = {
            "id": "example-person",
            "name_ko": "예시 인물",
            "years_label": "1890–1950",
            "epithet_ko": "혁명가",
            "hits": ["He joined the Red Army in 1919."],
        }

    def test_label_batch_uses_registered_gateway_executor(self):
        response = {
            "links": [{
                "person_id": "example-person",
                "name_ko": "예시 인물",
                "relation_ko": "적군 복무",
                "relation_en": "Red Army service",
                "kind": "participant",
            }]
        }
        with patch(
            "llm.call_registry.generate_sync",
            return_value=json.dumps(response, ensure_ascii=False),
        ) as generate:
            result = evidence.label_batch(
                evidence.EVENTS["civil-war"], [self.person]
            )

        self.assertEqual(result, response["links"])
        args, kwargs = generate.call_args
        self.assertEqual(args[0], "commulingo_event_evidence_labels")
        self.assertIn("He joined the Red Army in 1919.", args[1])
        self.assertEqual(kwargs["provider"], "deepseek")
        self.assertTrue(kwargs["json_mode"])

    def test_empty_gateway_result_fails_batch_instead_of_silently_skipping(self):
        with patch("llm.call_registry.generate_sync", return_value=None):
            with self.assertRaises(RuntimeError):
                evidence.label_batch(evidence.EVENTS["civil-war"], [self.person])

    def test_script_has_no_direct_provider_client(self):
        source = (
            ROOT / "scripts" / "commulingo_event_evidence_links.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("OpenAI(", source)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("api.deepseek.com", source)


if __name__ == "__main__":
    unittest.main()
