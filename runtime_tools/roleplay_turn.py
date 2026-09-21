"""Authorize -> isolated draft -> adjudicate/project -> verify -> commit/send."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import json
import math
import sqlite3
import tempfile
from pathlib import Path

from llm.call_registry import decide_detailed, generate_detailed, resolve, resolve_provider_connection
from runtime_tools import roleplay_memory as memory, roleplay_jev as jev
from runtime_tools.roleplay_actor import actor_state_view
from runtime_tools.roleplay_review import REVIEW_RULES, screen_reply, validate_review
from runtime_tools.roleplay_clock import interpret_clock
from runtime_tools.roleplay_story import blocking_events
from runtime_tools.roleplay_pacing import policy_for, turn_time_scope, check_time_request, check_time_result


def decide(label, payload, questions, defaults=None):
    """Closed-set answers for every question. A key listed in ``defaults`` falls back to the
    conservative value when Jev stays unsure; any other unsure key becomes a PendingChoice
    so the player can settle it instead of losing the turn."""
    profile = resolve(jev.FEATURE)
    if not profile.extra.get('enabled', False):
        raise ValueError('판정기가 비활성화됨')
    result = decide_detailed(jev.FEATURE, payload, questions, profile=profile, label=label)
    if result.decision is None:
        raise ValueError('장면 검증을 완료하지 못함')
    labels, unresolved = {}, []
    threshold = float(profile.extra.get('thresholds', {}).get('accept', .75))
    def accepted(decision, key):
        value, confidence = decision.choice(key), decision.confidence(key)
        return value if value in questions[key]['criteria'] and confidence is not None and math.isfinite(confidence) and confidence >= threshold else None
    for key in questions:
        value = accepted(result.decision, key)
        if value is None: unresolved.append(key)
        else: labels[key] = value
    review = None
    candidate_decision = result.decision
    if unresolved:
        focused = ({'current_user':payload['current_user'],'accepted':labels,'current':{'clock':(payload.get('current') or {}).get('clock')},'history':payload.get('history') if 'mode' in unresolved else []} if label == 'roleplay-authorize' else payload)
        retry = decide_detailed(jev.FEATURE, focused, {key:questions[key] for key in unresolved}, profile=profile, label=label+'-review')
        if retry.decision is not None:
            review = retry.decision.answers
            candidate_decision = retry.decision
            for key in unresolved:
                value = accepted(retry.decision,key)
                if value is not None: labels[key] = value
    defaulted = {}
    for key in sorted(set(questions) - set(labels)):
        if defaults and key in defaults:
            labels[key] = defaulted[key] = defaults[key]
            continue
        probabilities = candidate_decision.probabilities(key)
        ranked = sorted(questions[key]['criteria'], key=lambda k: -float(probabilities.get(k) or 0))
        raise jev.PendingChoice(key, [(k, questions[key]['criteria'][k]) for k in ranked[:4]], '장면 검증 미확정: ' + key)
    return {'labels': labels, 'answers': result.decision.answers, 'review':review, 'model': result.decision.model, 'defaulted': defaulted}


def authorize(user_text, state, history):
    from runtime_tools.roleplay_time import authorize_time
    return authorize_time(user_text, state, history)


def policy_for_authorization(authorization):
    """The authorization LLM supplies time permission and the selected duration; see roleplay_pacing.policy_for."""
    labels = authorization['labels']
    return policy_for(authorization['user_text'], mode=labels['mode'], time_scope=labels.get('time_scope', 'none'),
                      span=labels.get('span', 'brief'), transition=labels.get('transition', 'current'),
                      explicit_minutes=authorization.get('duration_minutes'))


def direction(authorization, state=None):
    mode = authorization['labels']['mode']
    if mode != 'scene':
        return '장면을 진행하지 않고 질문·회상·계획·정정 요청에 답하는 초안을 쓴다. 새 사건이 일어났다고 서술하지 않는다.'
    text = '사용자가 지정한 한 장면의 초안을 쓴다. 다음 아침으로의 전환이 허용되면 아침 장면에서 바로 시작하고, 생략된 밤의 수면·회복·사건을 만들어 넣지 않는다. 지정한 종료점에서 멈추고 후속 사건을 붙이지 않는다. 아직 저장·확정되지 않은 초안이다.'
    if state is not None:
        clock = state.get('clock') or {}
        text += (f" 현재 기준 시계는 {clock.get('date') or '날짜 미상'} {clock.get('time') or '시각 미상'}이다."
                 " 과거 답변의 OOC 시각이나 오래된 장면 요약으로 이 시계를 덮어쓰지 않는다.")
        location = state.get('location') or '미확인'
        text += (f" 현재 위치는 \"{location}\"이며, 사용자가 이동을 지시하지 않았으면 사건은 그 자리에서 일어난다. 감방으로 돌아가는 길·계단·다른 방을 지어내지 않는다."
                 " 초안은 이 사건 하나와 그 직후의 반응까지다. 한두 문단이면 충분하고, 그 뒤의 식사·수면·다음 방문·다음 날은 쓰지 않는다.")
    stop = expected_stop(authorization, state) if state is not None else None
    if stop:
        text += f" 이 장면은 {stop['title']}({stop['minutes']}분 뒤)에서 멈춘다. 그 도래 장면까지만 쓰고 그 뒤는 쓰지 않는다."
    elif state is not None and authorization['labels'].get('transition', 'current') == 'current':
        policy = policy_for_authorization(authorization)
        if clock.get('date') and clock.get('time'):
            minutes = policy.explicit_minutes if policy.explicit_passage and policy.explicit_minutes else policy.max_minutes
            endpoint = datetime.fromisoformat(f"{clock['date']}T{clock['time']}") + timedelta(minutes=minutes)
            text += f" 이번 장면은 늦어도 {endpoint:%Y-%m-%d %H:%M} 안에 끝낸다. 그보다 먼 배식·방문·시각에 도달했다고 쓰지 않는다."
    return text


def expected_stop(authorization, state):
    """The first clock-stopping beat an explicit passage will cross: a registered routine
    item or a scheduled event. Told to the actor before drafting, so the settled stop and
    the written scene agree."""
    from runtime_tools.roleplay_dynamics import routine_occurrences
    from runtime_tools.roleplay_story import blocking_events
    policy = policy_for_authorization(authorization)
    if authorization['labels'].get('transition', 'current') != 'current' or not policy.explicit_passage or policy.calendar_skip:
        return None
    span = min(policy.max_minutes, 1440)
    candidates = [(at, f"{item['time']} {item['title']}") for at, item in routine_occurrences(state, span)]
    for event in blocking_events(state, span):
        due = max(0, event.get('due_minute', state['scene_minute']) - state['scene_minute'])
        candidates.append((due, event['title']))
    if not candidates:
        return None
    minutes, title = min(candidates, key=lambda pair: pair[0])
    return {'title': title, 'minutes': minutes}


def _records(conn, uid):
    return {table: conn.execute(f'SELECT {columns} FROM {table} WHERE user_id=? ORDER BY 1', (str(uid),)).fetchall()
            for table, columns in (('notes','key,content'),('people','person_id,payload'))}


@contextmanager
def staged_memory(uid):
    """Existing tools validate normally, but write only to a disposable snapshot."""
    with tempfile.TemporaryDirectory(prefix='roleplay-draft-') as tmp:
        path = Path(tmp)/'memory.sqlite3'
        with memory._connection() as source, sqlite3.connect(path) as dest:
            baseline = _records(source, uid)
            source.backup(dest)
        token = memory.MEMORY_OVERRIDE.set(path)
        stage = {'baseline':baseline}
        try:
            yield stage
            stage['state'] = memory.load_state(uid)
            stage['people'] = memory.load_people(uid)
            with memory._connection() as conn:
                stage['records'] = _records(conn,uid)
        finally:
            memory.MEMORY_OVERRIDE.reset(token)


def check_staged(conn, uid, stage):
    if _records(conn, uid) != stage['baseline']:
        raise ValueError('초안 작성 중 인물/메모 기록이 바뀜')


def commit_staged(conn, uid, stage):
    for table in ('notes','people'):
        if stage['records'][table] != stage['baseline'][table]:
            conn.execute(f'DELETE FROM {table} WHERE user_id=?',(str(uid),))
            conn.executemany(f'INSERT INTO {table} VALUES (?,?,?)',[(str(uid),*row) for row in stage['records'][table]])


def prepare(user_text, before, people, history, scope_id, draft, authorization, stage=None, verdict=None, scope_ok=False, final_attempt=False):
    """Gate and classify the draft, then project it. A ``verdict`` from an earlier attempt
    (the player settled a choice Jev left open) skips the scope gate and the classifier;
    ``scope_ok`` means the player confirmed the draft stays within the authorized scene."""
    mode = authorization['labels']['mode']
    if mode not in {'scene', 'discussion', 'plan', 'correction', 'reset'} or authorization['user_text'] != user_text:
        raise ValueError('현재 입력의 유효한 모드 판정 없음')
    if mode != 'scene':
        # The input interpreter already selected the operation and its arguments.
        # A scene classifier must not veto or reinterpret a non-scene operation.
        verdict = {'status': 'classified', 'labels': {'mode': mode}, 'uncertain': [],
                   'model': authorization.get('model'), 'cost_usd': 0, 'latency_ms': 0,
                   'answers': {}, 'calls': {}, 'rules_version': jev.RULES_VERSION,
                   'classification_skipped': 'non_scene', 'draft': draft, 'authorization': authorization}
        if mode == 'plan':
            from runtime_tools.roleplay_time import validate_appointment
            validate_appointment(authorization.get('appointment'), mode, authorization.get('duration_minutes'))
            verdict['labels']['plan_action'] = 'schedule'
    elif verdict is None:
        # No scope gate. The classifier's own labels (location, events, elapsed minutes
        # capped by the pacing policy) settle what the draft actually did; a second Jev
        # call to ask "did it overrun?" only ever produced rejected turns.
        gate = {'labels': {'within_scope': 'yes'}, 'skipped': True, 'player_confirmed': bool(scope_ok)}
        verdict = jev.classify(user_text,before,people,history,draft=draft,authorization=authorization)
        if verdict['status'] != 'classified':
            raise ValueError('초안 사건 판정 실패')
        verdict.update(draft=draft, authorization=authorization, scope_review=gate)
    else:
        verdict = deepcopy(verdict)
    verdict['labels']['mode'] = mode
    if any(e in {'interrogation','coerced_confession','implicating_others'} for e in jev.resolve_events(verdict['labels'])[0]) and 'activity' not in verdict['labels']:
        verdict['labels']['activity'] = 'light'
    policy = policy_for_authorization(authorization)
    state = deepcopy(before)
    with turn_time_scope(policy):
        if mode == 'scene' and authorization['labels']['transition'] != 'current':
            if blocking_events(state, 1440):
                raise ValueError('예정 사건을 지나쳐 다음 날로 건너뛸 수 없음')
            temporal={'operation':'next_day','relation':'current','certainty':'explicit','source_quote':user_text[:400],
                'interpretation':'사용자가 지정한 다음 장면으로 전환. 생략된 밤의 활동은 미상이며 수치에 적용하지 않음',
                'daypart':'morning' if authorization['labels']['transition']=='next_morning' else 'unknown'}
            checked = check_time_request(state,temporal,scope_id)
            state = interpret_clock(state,temporal,lambda *args: None)
            check_time_result(before,state,temporal,scope_id,checked)
            # A next-morning scene needs a usable clock anchor. 06:00 is an
            # explicit game estimate, not evidence that the omitted night was sleep.
            if authorization['labels']['transition'] == 'next_morning':
                previous_time = before.get('clock', {}).get('time')
                gap = None
                if previous_time:
                    hour, minute = map(int, previous_time.split(':'))
                    gap = 1440 - (hour * 60 + minute) + 360
                    state['scene_minute'] += gap
                    state['last_calculated_minute'] += gap
                    state['clock']['uncalculated_minutes'] = state['clock'].get('uncalculated_minutes', 0) + gap
                state['clock']['time'] = '06:00'
                state['clock']['certainty'] = 'estimated'
                temporal = {**temporal, 'estimated_scene_start':'06:00', 'gap_minutes':gap,
                            'gap_effects':'not_applied: omitted activity unknown'}
            verdict['transition'] = temporal
        if mode == 'scene':
            if policy.explicit_passage and authorization['labels']['transition']=='current':
                verdict['labels']['elapsed']='explicit'
            else:
                # The duration generator, not ambiguous temporal word classification,
                # decides how much time the authorized draft actually uses.
                verdict['labels']['elapsed']='brief'
                verdict['duration_limit']=180 if authorization['labels']['span']=='session' or authorization['labels'].get('time_scope')=='open_ended' else 10
                if not verdict.get('duration_estimate'):
                    verdict['duration_estimate']=jev.estimate_duration(user_text,state,verdict)
        try:
            projected, applied = jev.project(state,user_text,people,verdict,scope_id)
        except jev.PendingChoice as exc:
            # Keep everything already paid for so the player's pick completes this same draft.
            exc.verdict = verdict
            raise
    if mode == 'scene' and verdict['labels'].get('location') in {None, 'unknown'}:
        projected['location'] = '미확인 — 확정된 장면 서술 참조'
    if authorization.get('auto_settled'):
        applied = {**applied, 'auto_settled': {**authorization['auto_settled'], **(applied.get('auto_settled') or {})}}
    if applied.get('interrupted') and not (mode == 'scene' and verdict['labels'].get('elapsed') == 'explicit'):
        raise ValueError('예정 사건 도래로 초안 끝까지 실행할 수 없음. 도래 장면에서 멈춰야 함')
    # An explicit passage was told where it stops (see direction/expected_stop); the settled
    # scene ends there and the rest of the requested time is simply not spent.
    if stage is not None and mode in {'scene','discussion','plan'}:
        for key in ('goal','avoid','next_action','unresolved','body','mood','scene'):
            if stage['state'].get(key) != before.get(key):
                projected[key] = stage['state'].get(key)
        if stage['state'].get('holdouts') != before.get('holdouts'):
            # The actor may add what the character still holds; Jev's losses this turn win by id.
            settled = {h['id']: h for h in projected.get('holdouts', [])}
            projected['holdouts'] = [settled.get(h['id'], h) if h['id'] in settled and settled[h['id']]['status'] == 'lost' else h
                                     for h in stage['state'].get('holdouts', [])]
        if mode != 'scene' and (projected != before or stage['records'] != stage['baseline']):
            applied = {**applied,'no_change':False,'narrative_only':True}
    # The consistency review is advisory: its findings ride on the settlement line and
    # the player corrects what matters. Rewriting a whole draft over "someone coughed
    # upstairs" cost more than any contradiction it caught.
    try:
        review = (review_reply(draft, before, projected, stage, applied=applied) if mode == 'scene'
                  else {'approved': None, 'issues': [], 'status': 'skipped', 'reason': 'non_scene'})
    except ValueError as exc:
        # Advisory review must not discard an otherwise settled scene.
        # Keep the failure distinct from approval in the audit record.
        review = {'approved': None, 'issues': [], 'status': 'unavailable', 'reason': str(exc)[:300]}
        applied = {**applied, 'review_unavailable': True}
    if review['approved'] is False:
        applied = {**applied, 'review_issues': [issue[:160] for issue in review['issues']][:3]}
    verdict['final_review']=review
    return {'before_revision':before['revision'],'state':projected,'applied':applied,
            'verdict':verdict,'reply':draft,'stage':stage}


def review_reply(draft, before, projected, stage=None, *, applied=None):
    """Text consistency only; no numeric rules or event selection in this call.

    Uses the actor's existing provider/endpoint, and only its qualitative state
    plus records it just changed. Never export a full memory snapshot.
    """
    import bot_config
    feature = 'roleplay_scene_consistency'
    profile = resolve(feature)
    if not profile.extra.get('enabled', False):
        return {'approved': None, 'issues': [], 'status': 'skipped', 'reason': 'disabled'}
    if profile.provider != 'deepseek_anthropic' or resolve_provider_connection(profile.provider).base_url.rstrip('/') != bot_config.DEEPSEEK_ANTHROPIC_BASE_URL.rstrip('/'):
        raise ValueError('서술 검토 경로가 기존 연기 모델 경로와 다름')
    changes = {}
    if stage:
        for table in ('notes', 'people'):
            baseline = dict(stage['baseline'].get(table, []))
            changes[table] = [row for row in stage['records'].get(table, []) if baseline.get(row[0]) != row[1]]
    # Only qualitative accepted events, never the arithmetic or entire verdict.
    events = (applied or {}).get('events') or []
    payload = {'draft':draft, 'before':actor_state_view(before),
               'settled_state':actor_state_view(projected), 'new_records':changes,
               'settled_events': [jev.EVENT_LABELS.get(event, event) for event in events]}
    screen = screen_reply(payload) if profile.extra.get('jev_precheck', False) else None
    if screen and screen['clean']:
        return {'approved': True, 'issues': [], 'screen': screen, 'reviewer': 'jev'}
    result = generate_detailed(feature, json.dumps(payload,ensure_ascii=False), profile=profile,
        system=REVIEW_RULES + ' Return only JSON {"approved":true,"issues":[]} or '
        '{"approved":false,"issues":[{"explanation":"한국어 모순 설명","claims":['
        '{"source":"draft","quote":"exact substring"},{"source":"settled_state","quote":"exact substring"}]}]}. '
        'Each issue must quote one string value from settled_state and one from draft or new_records. '
        'Do not cite before or invent a quote. If you cannot quote both incompatible claims, omit the issue.',
        label=feature)
    if result.error_kind or result.truncated or not result.text:
        raise ValueError('최종 서술 검토를 완료하지 못함')
    try:
        review = validate_review(json.loads(result.text), payload)
    except (ValueError,TypeError) as exc:
        raise ValueError('최종 서술 검토 응답 형식 오류') from exc
    if screen is not None:
        review['screen'] = screen
        review['reviewer'] = 'deepseek'
    return review


def feedback_line(outcome, state=None):
    """One line telling the director what the engine believed happened. No numbers, no thresholds."""
    from runtime_tools.roleplay_dynamics import RESOLVE_EVENT_KINDS
    status = outcome.get('status')
    applied = outcome.get('applied') or {}
    if status == 'unchanged' or applied.get('no_change'):
        return '⚙ 상담·회상·계획으로 처리. 장면 시간과 상태는 그대로.'
    if status != 'applied':
        return f"⚙ 보류: {outcome.get('reason') or '확정하지 못함'}"
    if applied.get('mode') == 'reset':
        return '⚙ 새 장면으로 초기화.'
    if applied.get('mode') == 'correction':
        return '⚙ 수치 정정 반영.'
    if applied.get('scheduled'):
        return '⚙ 예정 사건 등록.'
    names = {'meal': '식사', 'snack': '간식', 'water': '물', 'treatment': '처치', 'injury': '새 부상', 'none': '뚜렷한 사건 없음',
             **{k: v[1] for k, v in RESOLVE_EVENT_KINDS.items()}}
    events = applied.get('events') if applied.get('events') is not None else ([applied['event']] if applied.get('event') not in (None, 'none') else [])
    parts = ['확정: ' + ('도래 사건까지 진행' if applied.get('interrupted') and not events
                        else (' + '.join(names.get(e, e) for e in events) or '사건 없음'))]
    if applied.get('deferred_components'):
        parts.append('시간·장면 조건 보류')
    elif applied.get('minutes') is not None:
        parts.append(f"{applied['minutes']}분")
    clock = (state or {}).get('clock') or {}
    if clock.get('time'):
        parts.append(clock['time'] + (' 추정' if clock.get('certainty') == 'estimated' else ''))
    if applied.get('interrupted'):
        parts.append(', '.join(applied.get('stopped_at') or ['예정 사건']) + '에서 멈춤')
    if applied.get('holdouts_lost'):
        parts.append('넘긴 것: ' + ', '.join(applied['holdouts_lost']))
    for request, action in (applied.get('bargains') or {}).items():
        parts.append({'paid': '값 치름', 'kept': '이행됨', 'broken': '파기됨'}[action] + ': ' + request)
    if applied.get('delayed_reaction') == 'scheduled':
        parts.append('미뤄 둔 반응 예약')
    elif applied.get('delayed_reaction') == 'released':
        parts.append('미뤄 둔 반응 해소')
    if applied.get('illness_changes'):
        from runtime_tools.roleplay_illness import KINDS, STATUSES
        parts.append('질병 갱신: ' + ', '.join(KINDS[i['kind']] + ' ' + STATUSES[i['status']] for i in applied['illness_changes']))
    if applied.get('narrative_only'):
        parts.append('기록만 갱신')
    if applied.get('review_issues'):
        parts.append('서술 검토 지적: ' + ' / '.join(applied['review_issues']))
    if applied.get('review_unavailable'):
        parts.append('보조 서술 검토 미완료')
    if applied.get('auto_settled'):
        korean = {'mode': '모드', 'event': '사건', 'intensity': '강도', 'activity': '활동', 'within_scope': '범위'}
        values = {**names, 'scene': '장면 실행', 'discussion': '질문·상담', 'plan': '예정 등록', 'mild': '스침', 'moderate': '보통', 'severe': '극심',
                  'light': '가벼운 움직임', 'rest': '휴식', 'yes': '안'}
        parts.append('애매해서 자동 처리: ' + ', '.join(f"{korean.get(k, k)}={values.get(v, v)}" for k, v in applied['auto_settled'].items()))
    return '⚙ ' + ' · '.join(parts)


def committed_reply(uid, scope_id):
    with memory._connection() as conn:
        row = conn.execute('SELECT payload FROM automatic_turns WHERE user_id=? AND scope_id=?', (str(uid),str(scope_id))).fetchone()
    return json.loads(row[0]).get('reply') if row else None
