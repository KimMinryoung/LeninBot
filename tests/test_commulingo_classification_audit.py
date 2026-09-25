"""Classification audit: operator rules live in the criteria; non-Soviet people get categories only."""
import unittest
from unittest.mock import patch

from llm.call_registry import CallSiteProfile, Decision, DecisionResult

GROUPS = [{'id': 'thaw', 'title_en': 'Thaw', 'range_label': '1953–1985', 'blurb_en': ''},
          {'id': 'international-revolutionary', 'title_en': 'Non-Soviet', 'range_label': '', 'blurb_en': ''}]
OFFICES = [{'id': 'nationalities-federal', 'title_en': 'Nationalities', 'range_label': '1917–1991'},
           {'id': 'party-leadership', 'title_en': 'Party leadership', 'range_label': '1922–1991'}]
CATS = [{'id': 'socialist-bloc-leader', 'label_en': 'Bloc leader', 'label_ko': '사회주의권 지도자'},
        {'id': 'scholar', 'label_en': 'Scholar', 'label_ko': '연구자'},
        {'id': 'ccp-security', 'label_en': 'CCP security', 'label_ko': '중공 보안·정보'}]


def decision(group, role, conf=0.95):
    return DecisionResult(decision=Decision(answers={
        'group': {'choice': group, 'confidence': conf, 'probabilities': {group: conf}},
        'role': {'choice': role, 'confidence': conf, 'probabilities': {role: conf}}}, model='test', usage={'input_tokens': 10}))


class AuditTests(unittest.TestCase):
    def test_rules_are_in_criteria_and_offices_only_for_soviet_citizens(self):
        from scripts import commulingo_classification_audit as audit
        soviet = audit.build_questions(GROUPS, OFFICES, CATS, soviet=True)
        self.assertIn('Union-republic first secretaries', soviet['role']['criteria']['nationalities-federal'])
        self.assertIn('never belongs to a Soviet era group', soviet['group']['criteria']['international-revolutionary'])
        foreign = audit.build_questions(GROUPS, OFFICES, CATS, soviet=False)
        self.assertEqual(set(foreign['role']['criteria']), {'socialist-bloc-leader', 'scholar'})
        seen = []
        def decide(feature, state, questions, label=None):
            seen.append((feature, set(questions['role']['criteria'])))
            return decision('international-revolutionary', 'socialist-bloc-leader')
        person = {'id': 'x', 'name_ko': 'X', 'years_label': '1900–1980', 'epithet_ko': '', 'bio_ko': 'b', 'bio_en': 'b',
                  'group_id': 'thaw', 'citizenship_code': 'poland', 'category_id': None, 'office_id': 'party-leadership', 'career': None}
        row = audit.judge(person, (GROUPS, OFFICES, CATS), decide)
        self.assertEqual(seen[0][0], 'commulingo_classification_audit')
        self.assertEqual(seen[0][1], {'socialist-bloc-leader', 'scholar'})
        self.assertEqual(row['jev']['group']['choice'], 'international-revolutionary')
        successor = {**person, 'citizenship_code': 'azerbaijan'}
        audit.judge(successor, (GROUPS, OFFICES, CATS), decide)
        self.assertIn('nationalities-federal', seen[1][1])

    def test_chinese_citizens_get_the_party_state_categories_and_nobody_else_does(self):
        from scripts import commulingo_classification_audit as audit
        from commulingo.classify import role_scope
        self.assertEqual(role_scope('china'), 'china')
        self.assertEqual(role_scope('soviet'), 'soviet')
        self.assertEqual(role_scope('poland'), 'other')
        china = audit.build_questions(GROUPS, OFFICES, CATS, soviet=False, scope='china')
        self.assertEqual(set(china['role']['criteria']), {'ccp-security', 'scholar'})
        self.assertIn('Kang Sheng', china['role']['criteria']['ccp-security'])
        self.assertNotIn('nationalities-federal', china['role']['criteria'])
        self.assertNotIn('ccp-security', audit.build_questions(GROUPS, OFFICES, CATS, soviet=True)['role']['criteria'])
        self.assertNotIn('ccp-security', audit.build_questions(GROUPS, OFFICES, CATS, soviet=False)['role']['criteria'])
        seen = []
        def decide(feature, state, questions, label=None):
            seen.append(set(questions['role']['criteria']))
            return decision('international-revolutionary', 'ccp-security')
        person = {'id': 'k', 'name_ko': '캉성', 'years_label': '1898–1975', 'epithet_ko': '', 'bio_ko': 'b', 'bio_en': 'b',
                  'group_id': 'thaw', 'citizenship_code': 'china', 'category_id': None, 'office_id': None, 'career': None}
        audit.judge(person, (GROUPS, OFFICES, CATS), decide)
        self.assertEqual(seen[0], {'ccp-security', 'scholar'})

    def test_undecided_citizenship_asks_the_role_three_ways(self):
        from commulingo.classify import ROLE_KEYS, person_card_questions
        q = person_card_questions({'citizenship': {'label': {'en': 'China'}}}, GROUPS, OFFICES, CATS, ['china', 'soviet'], ['china'])
        self.assertEqual(set(ROLE_KEYS.values()) - set(q), set())
        self.assertIn('ccp-security', q['role_china']['criteria'])
        self.assertNotIn('ccp-security', q['role_soviet']['criteria'])
        self.assertNotIn('ccp-security', q['role_non_soviet']['criteria'])

    def test_report_skips_accepted_boundary_pairs(self):
        from scripts import commulingo_classification_audit as audit
        rows = [{'id': 'a', 'name': 'A', 'years': '', 'cost': 0.0, 'tokens': 1,
                 'stored': {'group': 'thaw', 'role': 'nationalities-federal'},
                 'jev': {'group': {'choice': 'thaw', 'conf': 0.9, 'top': []},
                         'role': {'choice': 'party-leadership', 'conf': 0.99, 'top': [('party-leadership', 0.99)]}}},
                {'id': 'b', 'name': 'B', 'years': '', 'cost': 0.0, 'tokens': 1,
                 'stored': {'group': 'thaw', 'role': 'scholar'},
                 'jev': {'group': {'choice': 'international-revolutionary', 'conf': 0.97, 'top': [('international-revolutionary', 0.97)]},
                         'role': {'choice': 'scholar', 'conf': 0.9, 'top': []}}}]
        text = audit.report(rows, 0.85)
        self.assertIn('`thaw` → `international-revolutionary`', text)
        self.assertNotIn('`nationalities-federal` → `party-leadership`', text)
        self.assertIn('role: 일치 1/2, 보고 대상 불일치 0', text)


if __name__ == '__main__':
    unittest.main()
