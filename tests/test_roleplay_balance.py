import unittest
from roleplay import jev
from roleplay.dynamics import with_defaults, advance, resolve_event_delta
from roleplay.memory import STATE_DEFAULTS
from roleplay.pacing import policy_for, turn_time_scope


def initial(**values):
    return with_defaults({**STATE_DEFAULTS, 'resolve':5, 'humiliation':100, 'clarity':50,
        'hunger':40, 'pain':20, 'fatigue':40, 'tension':50, 'conditions_initialized':True,
        'activity':'rest','threat':'uncertain', **values})


def event(state, kind, scope):
    verdict={'labels':{'mode':'scene','event':kind,'elapsed':'0','intensity':'moderate'}}
    with turn_time_scope(policy_for('현재의 명시적 사건')):
        return jev.project(state,'현재의 명시적 사건',[],verdict,str(scope))[0]


class GameBalanceTests(unittest.TestCase):
    def test_recovery_sequence_exits_extremes_without_skip(self):
        state=initial()
        for i,kind in enumerate(('agency','small_success','recognition','boundary_respected')):
            state=event(state,kind,i)
        self.assertEqual(state['resolve'],27)
        self.assertEqual(state['humiliation'],76)
        self.assertEqual(state['scene_minute'],0)
        self.assertEqual(state['fatigue'],40)

    def test_repeated_reward_is_not_an_infinite_farm(self):
        state=initial()
        gains=[]
        for i in range(8):
            before=state;state=event(state,'agency',i);gains.append(round(state['resolve']-before['resolve'],4))
        self.assertEqual(gains,[6,3,1.5,0,0,0,0,0])
        self.assertEqual(state['humiliation'],89.5)
        # A genuinely later scene can earn that kind of reward again.
        state['scene_minute']=181
        self.assertEqual(event(state,'agency',9)['resolve']-state['resolve'],6)

    def test_activities_recover_with_costs_and_no_full_reset(self):
        self_care=advance(initial(activity='self_care'),60,'몸 돌봄')
        work=advance(initial(activity='focused_work'),60,'작은 과제')
        rest=advance(initial(),60,'휴식')
        for state in (self_care,work,rest):
            self.assertGreater(state['resolve'],5)
            self.assertLess(state['humiliation'],100)
            self.assertGreater(state['humiliation'],90)
            self.assertGreater(state['hunger'],40)
        self.assertGreater(self_care['resolve'],rest['resolve'])
        self.assertGreater(work['fatigue'],self_care['fatigue'])
        self.assertLess(self_care['humiliation'],work['humiliation'])
        threatened=advance(initial(activity='self_care',threat='immediate',participants=['guard']),60,'위협 아래')
        self.assertEqual(threatened['humiliation'],100)
        self.assertLess(threatened['resolve'],5)

    def test_low_resolve_softens_loss_and_high_humiliation_softens_gain(self):
        low=resolve_event_delta(initial(resolve=5),'futile_effort',2)[0]
        high=resolve_event_delta(initial(resolve=60),'futile_effort',2)[0]
        self.assertGreater(low,high)
        self.assertLess(low,0)
        lower=initial(humiliation=30);upper=initial(humiliation=90)
        self.assertGreater(event(lower,'public_submission',1)['humiliation']-30,event(upper,'public_submission',1)['humiliation']-90)

    def test_activity_time_partition_is_stable(self):
        state=initial(activity='self_care')
        whole=advance(state,60,'돌봄')
        split=advance(advance(state,30,'돌봄'),60,'돌봄')
        for key in ('resolve','humiliation','clarity','fatigue','hunger'):
            self.assertAlmostEqual(whole[key],split[key],places=4)
