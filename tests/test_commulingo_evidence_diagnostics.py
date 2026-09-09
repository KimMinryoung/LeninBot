"""Evidence retry diagnostics, without production DB or an LLM."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "runtime_tools/commulingo_people.py"
module = ast.parse(source.read_text())
namespace = {"_EDITORIAL_CONTRACT": json.loads(Path(
    "/home/grass/frontend/data/commulingo/person-editorial-contract.json").read_text())}
exec(compile(ast.Module(body=[node for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "_person_evidence_errors"],
    type_ignores=[]), str(source), "exec"), namespace)
diagnose = namespace["_person_evidence_errors"]


class EvidenceDiagnostics(unittest.TestCase):
    def test_reports_source_and_all_missing_fields_together(self):
        fields = {"bio": {"en": "Biography"}, "years": "1900–1980",
                  "citizenship": {}, "nationalOrigin": {}, "evidence": [{
                      "field": "bio", "claim": "Biography", "locator": "p. 2",
                      "source": "https://example.org/archive"}]}
        errors = diagnose(fields, ["https://example.org/archive — biography"])
        self.assertEqual(len(errors), 2)
        self.assertIn("evidence[0].source", errors[0])
        self.assertIn("years, citizenship, nationalOrigin", errors[1])

    def test_exact_source_and_partial_update_are_valid(self):
        citation = "https://example.org/archive — biography"
        fields = {"bio": {"ko": "소개"}, "expectedRevision": "v1-original",
                  "evidence": [{"field": "bio", "claim": "소개",
                                "source": citation, "locator": "p. 2"}]}
        before = json.dumps(fields)
        self.assertEqual(diagnose(fields, [citation]), [])
        self.assertEqual(json.dumps(fields), before)

    def test_old_career_evidence_and_blank_locator_are_specific(self):
        errors = diagnose({"evidence": [{"field": "career", "claim": "Post",
                            "source": "Archive", "locator": " "}]}, ["Archive"])
        self.assertTrue(any("evidence[0].locator" in error for error in errors))
        self.assertTrue(any("career" in error for error in errors))

    def test_section_body_and_malformed_evidence(self):
        self.assertIn("body", diagnose({"body": {}}, ["Archive"])[0])
        for evidence in (None, {}, [None], [{"field": []}]):
            self.assertTrue(diagnose({"evidence": evidence}, ["Archive"]))

    def test_optional_evidence_for_supplied_career_is_allowed(self):
        self.assertEqual(diagnose({"career": [], "evidence": [{"field": "career",
            "claim": "Post", "source": "Archive", "locator": "p. 2"}]}, ["Archive"]), [])


if __name__ == "__main__":
    unittest.main()
