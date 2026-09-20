import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'config/commulingo_pipeline.json'


def load():
    value = json.loads(PATH.read_text())
    value.setdefault('workflow','legacy')
    if value['workflow'] not in {'legacy','editor'}:
        raise ValueError('workflow must be legacy or editor')
    if value['phase'] not in {'draft','canary','live'}:
        raise ValueError('invalid pipeline phase')
    for name in ('daily_cap_usd','stage_budget_usd'):
        if type(value[name]) not in (int,float) or not math.isfinite(value[name]) or value[name]<=0:
            raise ValueError(f'{name} must be positive and finite')
    if type(value['review_fraction']) not in (int,float) or not 0<=value['review_fraction']<=1:
        raise ValueError('review_fraction must be 0..1')
    if type(value['canary_per_group_per_day']) is not int or value['canary_per_group_per_day']<1:
        raise ValueError('canary_per_group_per_day must be a positive integer')
    for name in ('legacy_shared_budget','term_editorial_service'):
        if type(value[name]) is not bool:
            raise ValueError(f'{name} must be boolean')
    # Discovery mines public material for new entries. Off by operator decision
    # (2026-09-17): the queue was 2,630 material jobs, mostly single person cards.
    # Explicitly requested entries (curation gaps) are not affected by this switch.
    value.setdefault('discovery', True)
    if type(value['discovery']) is not bool:
        raise ValueError('discovery must be boolean')
    # A new term whose name coincides with a history event is not registered;
    # allowlisted ids may be, because the event registry only holds a larger
    # encompassing event. Existing terms are never removed by that rule.
    value.setdefault('term_event_overlap_allow', [])
    if not isinstance(value['term_event_overlap_allow'], list) or any(
            type(v) is not str for v in value['term_event_overlap_allow']):
        raise ValueError('term_event_overlap_allow must be a list of term ids')
    # Existing terms whose entry only restates a history event; the operator
    # keeps them but commissions no further enrichment (2026-09-17).
    value.setdefault('term_enrichment_exclude', [])
    if not isinstance(value['term_enrichment_exclude'], list) or any(
            type(v) is not str for v in value['term_enrichment_exclude']):
        raise ValueError('term_enrichment_exclude must be a list of term ids')
    return value


def legacy_reserve(amount, lane):
    config = load()
    if not config['legacy_shared_budget']:
        return None
    from .store import Store
    store = Store()
    token = store.reserve(amount,lane=lane,cap=config['daily_cap_usd'],
                          review_fraction=config['review_fraction'])
    return store, token
