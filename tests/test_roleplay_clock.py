import unittest
from roleplay.clock import interpret_clock, validate_temporal
from roleplay.dynamics import with_defaults, advance


class ClockTests(unittest.TestCase):
    def state(self):
        return with_defaults({'hunger': 20, 'fatigue': 40, 'pain': 10, 'tension': 20,
                              'conditions_initialized': True})

    def interpret(self, state, operation, **fields):
        temporal = {'relation': 'current', 'certainty': 'explicit', 'source_quote': '장면의 시간 표현',
                    'interpretation': '장면 시간 해석', 'operation': operation, **fields}
        return interpret_clock(state, temporal, advance)

    def test_midnight_year_and_leap_day(self):
        state = self.interpret(self.state(), 'anchor', date='1939-12-31', time='23:45')
        result = self.interpret(state, 'advance', elapsed_minutes=30)
        self.assertEqual(result['clock']['date'], '1940-01-01')
        self.assertEqual(result['clock']['time'], '00:15')
        self.assertEqual(result['clock']['year'], 1940)
        self.assertEqual(result['hunger'], 21.5)
        state = self.interpret(self.state(), 'anchor', date='1940-02-28', time='23:45')
        result = self.interpret(state, 'until', date='1940-02-29', time='00:15')
        self.assertEqual(result['scene_minute'], 30)
        self.assertEqual(result['clock']['date'], '1940-02-29')

    def test_past_plan_and_estimated_time(self):
        state = self.interpret(self.state(), 'anchor', date='1939-09-17', time='10:00')
        for relation in ('past', 'plan'):
            result = self.interpret(state, 'advance', elapsed_minutes=180, relation=relation)
            self.assertEqual(result['scene_minute'], 0)
            self.assertEqual(result['hunger'], state['hunger'])
            self.assertEqual(result['clock']['time'], '10:00')
            self.assertEqual(result['clock']['last_interpretation']['relation'], relation)
        result = self.interpret(state, 'advance', elapsed_minutes=20, certainty='estimated')
        self.assertEqual(result['clock']['time'], '10:20')
        self.assertEqual(result['clock']['certainty'], 'estimated')

    def test_partial_dates_and_unknown_interval(self):
        state = self.interpret(self.state(), 'anchor', year=1939, daypart='night')
        result = self.interpret(state, 'next_day', daypart='morning')
        self.assertIsNone(result['clock']['date'])
        self.assertIsNone(result['clock']['time'])
        self.assertEqual(result['clock']['daypart'], 'morning')
        self.assertEqual(result['clock']['relative_day'], 1)
        self.assertFalse(result['clock']['elapsed_complete'])
        self.assertEqual(result['clock']['unquantified_gaps'], 1)
        self.assertEqual(result['hunger'], state['hunger'])
        self.assertEqual(result['scene_minute'], 0)
        dated = self.interpret(self.state(), 'anchor', date='1939-09-30', daypart='night')
        self.assertEqual(self.interpret(dated, 'next_day', daypart='morning')['clock']['date'], '1939-10-01')
        with self.assertRaises(ValueError):
            self.interpret(result, 'until', date='1939-09-18', time='09:00')

    def test_conflicts_validation_and_correction(self):
        state = self.interpret(self.state(), 'anchor', date='1939-09-17', time='10:00')
        for operation, values in [('anchor', {'time': '09:00'}), ('until', {'date': '1939-09-17', 'time': '09:00'}),
                                  ('anchor', {'date': '1939-02-29'}), ('anchor', {'time': '25:00'}),
                                  ('advance', {'elapsed_minutes': 20, 'certainty': 'unknown'}),
                                  ('advance', {'elapsed_minutes': True})]:
            with self.assertRaises(ValueError):
                self.interpret(state, operation, **values)
        result = self.interpret(state, 'correct', time='09:00')
        self.assertEqual(result['clock']['time'], '09:00')
        self.assertEqual(result['scene_minute'], 0)
        with self.assertRaises(ValueError):
            validate_temporal({'operation': 'advance', 'elapsed_minutes': 30})
