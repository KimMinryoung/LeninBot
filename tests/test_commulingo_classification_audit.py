"""Classification audit: the operator's group rules live in the criteria; only the group is audited."""
import unittest

from llm.call_registry import Decision, DecisionResult

GROUPS = [{'id': 'thaw', 'title_en': 'Thaw', 'range_label': '1953–1985', 'blurb_en': ''},
          {'id': 'world-interwar', 'title_en': 'Non-Soviet', 'range_label': '', 'blurb_en': ''}]


def decision(group, conf=0.95):
    return DecisionResult(decision=Decision(answers={
        'group': {'choice': group, 'confidence': conf, 'probabilities': {group: conf}}},
        model='test', usage={'input_tokens': 10}))


class AuditTests(unittest.TestCase):
    def test_group_rules_are_in_criteria_and_only_the_group_is_asked(self):
        from scripts import commulingo_classification_audit as audit
        question = audit.group_question(GROUPS)
        self.assertIn('never belongs to a Soviet era group', question['criteria']['world-interwar'])
        seen = []
        def decide(feature, state, questions, label=None):
            seen.append((feature, set(questions)))
            return decision('world-interwar')
        person = {'id': 'x', 'name_ko': 'X', 'years_label': '1900–1980', 'epithet_ko': '', 'bio_ko': 'b', 'bio_en': 'b',
                  'group_id': 'thaw', 'citizenship_code': 'poland', 'career': None}
        row = audit.judge(person, GROUPS, decide)
        self.assertEqual(seen[0], ('commulingo_classification_audit', {'group'}))
        self.assertEqual(row['stored'], {'group': 'thaw'})
        self.assertEqual(row['jev']['group']['choice'], 'world-interwar')

    def test_report_skips_accepted_boundary_pairs(self):
        from scripts import commulingo_classification_audit as audit
        rows = [{'id': 'a', 'name': 'A', 'years': '', 'cost': 0.0, 'tokens': 1,
                 'stored': {'group': 'old-regime'},
                 'jev': {'group': {'choice': 'world-before-1917', 'conf': 0.99,
                                   'top': [('world-before-1917', 0.99)]}}},
                {'id': 'b', 'name': 'B', 'years': '', 'cost': 0.0, 'tokens': 1,
                 'stored': {'group': 'thaw'},
                 'jev': {'group': {'choice': 'world-interwar', 'conf': 0.97,
                                   'top': [('world-interwar', 0.97)]}}}]
        text = audit.report(rows, 0.85)
        self.assertIn('`thaw` → `world-interwar`', text)
        self.assertNotIn('`old-regime` → `world-before-1917`', text)
        self.assertIn('group: 일치 0/2, 보고 대상 불일치 1', text)


if __name__ == '__main__':
    unittest.main()
