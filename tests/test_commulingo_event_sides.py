"""The event-link writer and named sides (frontend migration 193).

An event may name its camps; a link to it may carry the side the person acted
for. The writer refuses a side the event does not name, a side on an event
without sides, and opponent on an event with sides.
"""
import unittest
from unittest.mock import patch as mock_patch

from commulingo import people
from commulingo.people import _validate_history_event_person
from commulingo_test_support import FakeReads

SIDES = [{'id': 'china', 'label': {'ko': '중국', 'en': 'China'}},
         {'id': 'vietnam', 'label': {'ko': '베트남', 'en': 'Vietnam'}}]


def patch(**extra):
    return {'personId': 'xu-shiyou', 'relationKind': 'executor',
            'relation': {'ko': '동부 전선 사령관', 'en': 'Eastern front commander'},
            'note': {'ko': '동부 전선을 지휘했다.', 'en': 'Commanded the eastern front.'}, **extra}


class EventSideValidationTests(unittest.TestCase):
    def check(self, sides, **extra):
        reads = FakeReads(events={'sino-vietnamese-war-1979': {'sides': sides}}, existing={'person': {'xu-shiyou'}})
        with mock_patch.object(people, '_reads', reads):
            return _validate_history_event_person(None, 'create', 'sino-vietnamese-war-1979', patch(**extra))

    def test_a_named_side_is_accepted(self):
        self.assertIsNone(self.check(SIDES, side='china'))
        self.assertIsNone(self.check(SIDES, side=None))
        self.assertIsNone(self.check(SIDES))

    def test_an_unknown_side_is_refused(self):
        self.assertIn('not one of this event', self.check(SIDES, side='japan'))
        self.assertIn('names no sides', self.check(None, side='china'))
        self.assertIn('side ids', self.check(SIDES, side='Not An Id'))

    def test_opponent_is_refused_only_where_the_event_has_sides(self):
        self.assertIn('instead of opponent', self.check(SIDES, relationKind='opponent', side='vietnam'))
        self.assertIsNone(self.check(None, relationKind='opponent'))


if __name__ == '__main__':
    unittest.main()
