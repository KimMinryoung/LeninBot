import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'config/commulingo_pipeline.json'


def load():
    value = json.loads(PATH.read_text())
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
