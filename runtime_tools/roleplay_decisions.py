"""Draft-local decisions: batch ordinary estimates, protect consequential outcomes."""
from copy import deepcopy
import math


class AdjudicationUnavailable(ValueError):
    """A model did not return usable adjudication; regenerating prose cannot fix it."""


class DraftOutOfScope(ValueError):
    """The narration itself needs one bounded rewrite."""


class StateConflict(ValueError):
    """Committed state or records changed while a draft was being prepared."""


IMPORTANT_EVENTS = {'coerced_confession', 'implicating_others', 'betrayal'}


def describe_important(item):
    from runtime_tools.roleplay_jev import EVENT_LABELS
    names = {'lost': '넘김', 'paid': '값 치름', 'kept': '이행', 'broken': '파기', 'keep': '변화 없음'}
    return item['title'] + ' → ' + names.get(item['label'], EVENT_LABELS.get(item['label'], item['label']))


def important_candidates(state, people, verdict):
    from runtime_tools import roleplay_jev as jev
    labels = verdict['labels']
    questions = jev.build_questions(state, people)
    result = []
    for key, question in questions.items():
        if key != 'event_pressure' and not key.startswith(('holdout_', 'bargain_')):
            continue
        value = labels.get(key)
        if key == 'event_pressure' and value is None and labels.get('event') in IMPORTANT_EVENTS:
            value = labels['event']
        missing = value is None
        if missing:
            if key not in verdict.get('uncertain', []) and key != 'event_pressure':
                continue
            value = jev.ranked_candidates(verdict, key, list(question['criteria'].items()), 'keep' if key != 'event_pressure' else 'none')[0][0]
        consequential = (value in IMPORTANT_EVENTS if key == 'event_pressure' else value != 'keep')
        # No usable record decision is uncertainty, not evidence that nothing happened.
        if missing and key.startswith(('holdout_', 'bargain_')):
            consequential = True
        if not consequential:
            continue
        target_id = None
        title = jev.EVENT_LABELS.get(value, value)
        if key.startswith(('holdout_', 'bargain_')):
            index = int(key.split('_')[1])
            record = state['holdouts' if key.startswith('holdout_') else 'bargains'][index]
            target_id = record['id']
            title = record.get('title', record.get('request'))
        answer = (verdict.get('answers') or {}).get(key) or {}
        confidence = answer.get('confidence')
        reliable = (not missing and isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                    and math.isfinite(confidence) and confidence >= .9)
        result.append({'key': key, 'label': value, 'target_id': target_id, 'title': title,
                       'reliable': reliable, 'unresolved': missing})
    return result


def pending_important(state, people, verdict):
    evidence = (verdict.get('duration_estimate') or {}).get('important_evidence', [])
    confirmed = verdict.get('confirmed_important', [])
    pending = []
    for item in important_candidates(state, people, verdict):
        identity = {k: item[k] for k in ('key', 'label', 'target_id')}
        if identity in confirmed:
            continue
        proven = any(all(e.get(k) == v for k, v in identity.items())
                     and isinstance(e.get('quote'), str) and e['quote'].strip()
                     and e['quote'] in verdict.get('draft', '') for e in evidence if isinstance(e, dict))
        if not item['reliable'] or not proven:
            pending.append(item)
    return pending


def settle_general(state, people, verdict):
    """Fill all ordinary gaps once; consequential gaps remain for the director."""
    from runtime_tools import roleplay_jev as jev
    result = deepcopy(verdict)
    labels = result['labels']
    picks = result.setdefault('auto_settled', {})
    protected = {item['key'] for item in important_candidates(state, people, result)}
    questions = jev.build_questions(state, people)
    keys = set(result.get('uncertain', [])) | {'intensity', 'activity'}
    if 'event' not in labels:
        keys.update(jev.FAMILY_KEYS)
    for key in sorted(keys):
        if key in labels or key in protected or key not in questions:
            continue
        if key not in {*jev.FAMILY_KEYS, 'intensity', 'activity'} and not key.startswith(('holdout_', 'bargain_', 'story_')):
            continue
        default = 'moderate' if key == 'intensity' else ('light' if key == 'activity' else ('none' if key in jev.FAMILY_KEYS else 'keep'))
        value = jev.ranked_candidates(result, key, list(questions[key]['criteria'].items()), default)[0][0]
        labels[key] = picks[key] = value
    events, unresolved = jev.resolve_events(labels)
    if not unresolved:
        labels['event'] = events[0] if events else 'none'
        labels['events'] = events
    result['player_settled'] = True
    return result
