"""Allowlisted narrative context for the actor; numerical state stays in the engine."""
from runtime_tools.roleplay_dynamics import isolation_stage, holdout_titles, RESOLVE_EVENT_KINDS


def actor_state_view(state):
    # revision is an opaque write token, not a character metric.
    view = {key: state.get(key) for key in ('revision', 'period', 'location', 'scene', 'body', 'mood',
            'goal', 'avoid', 'next_action', 'unresolved', 'last_event', 'participants')}
    cues = {}
    bands = {
        'hunger': ('허기', ('허기가 두드러지지 않음', '허기가 느껴짐', '먹을 것에 신경이 쓰임', '허기가 심해 집중하기 어려움')),
        'fatigue': ('피로', ('움직임에 여유가 있음', '피로가 느껴짐', '동작과 반응이 둔해짐', '몹시 지쳐 움직임과 말이 힘겨움')),
        'pain': ('통증', ('통증이 두드러지지 않음', '아픈 부위를 의식함', '통증 때문에 자세와 동작을 조심함', '심한 통증이 말과 움직임을 방해함')),
        'tension': ('긴장', ('비교적 긴장이 풀림', '상대를 경계함', '작은 자극에도 긴장함', '극도로 긴장해 반응이 굳거나 급해질 수 있음')),
        'humiliation': ('굴욕', ('굴욕감이 두드러지지 않음', '불편함이 남아 있음', '시선과 자세에 위축이 드러날 수 있음', '심한 굴욕감이 침묵과 회피에 드러날 수 있음')),
    }
    for key, (name, texts) in bands.items():
        value = state.get(key)
        cues[name] = '아직 알 수 없음; 임의로 확정하지 않음' if value is None else texts[sum(value >= edge for edge in (25, 50, 75))]
    value = state.get('resolve')
    cues['의지'] = ('아직 알 수 없음; 임의로 확정하지 않음' if value is None else
        '버티기 매우 어렵고 반응이 위축될 수 있음. 작은 선택·요청·거절의 여지는 남아 있음' if value <= 10 else
        '요구에 쉽게 응하고 말을 아끼지만 없는 혐의에는 부인할 여지가 있음' if value <= 25 else
        '조건을 붙여 일부를 인정하거나 대안을 먼저 제시할 수 있음' if value <= 40 else
        '자기 판단을 유지하며 반박하거나 요구에 조건을 붙일 여력이 있음')
    value = state.get('clarity')
    cues['명료함'] = ('아직 알 수 없음; 임의로 확정하지 않음' if value is None else
        '생각을 잇기 힘들고 시간·이름을 혼동할 수 있음' if value <= 25 else
        '설명이 꼬이거나 같은 말을 되풀이할 수 있음' if value <= 50 else '생각과 말을 비교적 또렷하게 이어갈 수 있음')
    stage = isolation_stage(state.get('isolation_minutes', 0))
    cues['고립 부담'] = stage['description'] if stage else '뚜렷한 누적 고립 증상을 단정하지 않음'
    view['acting_cues'] = cues
    if (state.get('resolve') is not None and state['resolve'] <= 25) or (state.get('humiliation') is not None and state['humiliation'] >= 75):
        view['possible_approaches'] = ['작은 선택이나 경계 존중을 요청할 수 있음', '여건이 허락하면 몸을 돌보거나 작은 과제를 제안할 수 있음', '제안은 실행·성공이 아님. 사용자 선택 전에 시간이나 효과를 진행하지 않음']
    view['activity'] = {'rest':'깨어 쉬는 중', 'light':'가벼운 움직임·대화', 'moderate':'몸을 쓰는 활동',
                        'strenuous':'격렬히 움직이는 중', 'sleep':'잠든 상태', 'restrained':'억제되거나 얼어붙어 움직이지 못하는 상태', 'self_care':'스스로 몸과 마음을 돌보는 중', 'focused_work':'작은 목적 있는 일에 집중하는 중'}.get(state.get('activity'), '미확인')
    view['threat'] = {'safe':'현재 드러난 위협 없음', 'uncertain':'안전을 확신할 수 없음', 'threatening':'위협이 존재함', 'immediate':'당장의 위협 아래 있음'}.get(state.get('threat'), '미확인')
    view['injuries'] = [{'description': injury.get('description'),
                        'severity': {1:'경미', 2:'뚜렷함', 3:'심함'}.get(injury.get('severity'), '미확인'),
                        'condition': {'stable':'지속', 'recovering':'회복 중', 'worsening':'악화 중'}.get(injury.get('trend'), '미확인'),
                        'care': '처치받음' if injury.get('treated') else '처치 여부 확인 필요'} for injury in state.get('injuries', [])]
    clock = state.get('clock') or {}
    view['clock'] = {k: clock[k] for k in ('date', 'time', 'daypart', 'certainty') if k in clock}
    view['story_events'] = [{k: event.get(k) for k in ('title', 'status', 'outcome')} for event in state.get('story_events', []) if event['status'] in {'pending', 'ready'}]
    if any(e.get('when_alone') and e['status'] == 'ready' for e in state.get('story_events', [])):
        view['story_events'].append({'title': '(신호) 혼자인 장면이면 미뤄 둔 반응을 몸과 행동으로 드러낼 수 있음. 남이 있으면 아직 아님', 'status': 'cue'})
    held, lost = holdout_titles(state, 'held'), holdout_titles(state, 'lost')
    view['holdouts'] = {'아직 지키는 것': held or ['(등록 없음 — 인물이 실제로 아직 넘기지 않은 구체적인 것을 roleplay_state update의 holdouts로 등록할 수 있음)'],
                        '이미 넘긴 것': lost or [],
                        'cue': '지키는 항목은 아직 넘기지 않은 실제 사실이다. 넘길지는 장면과 압박이 정하며, 넘기면 그 행위를 서술에 분명히 드러낸다. 잃은 항목을 되찾은 것처럼 쓰지 않는다'}
    return view


def actor_outcome_view(outcome):
    status = outcome.get('status', 'deferred')
    applied = outcome.get('applied') or {}
    if status == 'applied':
        cue = '현재 요청에서 확정된 사건의 결과까지만 연기한다.'
        if applied.get('interrupted'):
            cue = '예정 사건이 도래한 장면에서 멈춘다. 요청한 나머지 행동은 아직 일어나지 않았다.'
        elif applied.get('deferred_components'):
            cue = '확정된 사건의 즉시 반응만 연기한다. 시간 경과나 이동 완료는 확정되지 않았다.'
    elif status == 'unchanged':
        cue = '현재 상태에서 질문·회상·계획에 답한다. 새로운 사건이나 시간 경과를 실행하지 않는다.'
    else:
        cue = '이번 행동의 결과가 확정되지 않았다. 현재 장면에서 멈추고 필요한 사실만 짧게 확인한다.'
    event = applied.get('event')
    label = RESOLVE_EVENT_KINDS.get(event, (None, None))[1]
    view = {'direction': cue, **({'confirmed_event': label} if label else {})}
    if applied.get('holdouts_lost'):
        view['holdouts_lost'] = list(applied['holdouts_lost'])
    if applied.get('delayed_reaction') == 'scheduled':
        view['delayed_reaction'] = '이번 굴욕은 지금 반응으로 나오지 않고, 혼자 남는 장면에서 드러날 수 있다'
    return view
