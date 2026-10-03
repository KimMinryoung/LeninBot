"""Person alias guidance follows the frontend alias standard.

The frontend store (data/commulingo/person-alias-rules.js) rejects aliases that
repeat the headword or bare surname, carry labels/parentheses or use the wrong
script; every leninbot person write goes through that store, so the prompts
only need to stop asking for those aliases.
"""
import unittest

from agents.commulingo_curator import _PROMPT as CURATOR_PROMPT
from commulingo.pipeline.prompts import WRITING_RULES
from commulingo.people import _COMMULINGO_FIELD_SCHEMA


class PersonAliasGuidanceTests(unittest.TestCase):
    def test_new_card_does_not_require_aliases(self):
        requirement = CURATOR_PROMPT.split("- A new card requires", 1)[1].split("Names are stored", 1)[0]
        self.assertNotIn("aliases", requirement)
        self.assertIn("Person `aliases` are optional", CURATOR_PROMPT)

    def test_curator_states_the_alias_standard(self):
        for phrase in ("bare one-word surname", "Hangul only", "Latin script", "본명", "`cyrillic`"):
            self.assertIn(phrase, CURATOR_PROMPT)

    def test_pipeline_policy_and_schema_state_the_standard(self):
        self.assertIn("Person aliases are optional", WRITING_RULES)
        description = _COMMULINGO_FIELD_SCHEMA["properties"]["aliases"]["description"]
        self.assertIn("never the headword", description)
        self.assertIn("ko in Hangul, en in Latin script", description)


if __name__ == "__main__":
    unittest.main()
