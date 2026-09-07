import sys
sys.path.insert(0, '/home/grass/leninbot')
import db
db.query = lambda *a, **k: []
db.query_one = lambda *a, **k: None
from runtime_tools.commulingo_people import _NATIONALITY_SCHEMA, _NATIONAL_ORIGIN_SCHEMA, _NATIONALITY_CODES
from scripts.commulingo_people_maintainer import NATIONALITY_CODES, CARD_STYLE_GUIDANCE
from scripts.commulingo_backfill_person_nationality import plan
assert set(NATIONALITY_CODES.split(', ')) == _NATIONALITY_CODES
for code in ('soviet', 'yugoslavia'):
    assert code in _NATIONALITY_SCHEMA['properties']['code']['enum']
    assert code not in _NATIONAL_ORIGIN_SCHEMA['properties']['code']['enum']
    for origin in ('', code):
        try: plan([{'id':'unknown-test-person','citizenship':code,'origin':origin}])
        except RuntimeError: pass
        else: raise AssertionError('Unsupported origin accepted')
assert plan([{'id':'test-person','citizenship':'yugoslavia','origin':'croatia'}])[0]['new_origin'] == 'croatia'
assert 'otherwise use `russia`' not in CARD_STYLE_GUIDANCE
print('Registration schema, prompt, code parity and backfill policy passed')
