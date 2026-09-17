"""Calendar bookkeeping for LLM-interpreted fictional time, never wall-clock time."""
from copy import deepcopy
from datetime import date, datetime, timedelta
import re

CLOCK_DEFAULTS = {
    "date": None, "year": None, "time": None, "daypart": "unknown", "relative_day": 0,
    "certainty": "unknown", "elapsed_complete": True, "unquantified_gaps": 0,
    "last_interpretation": None,
}
DAYPARTS = ["unknown", "dawn", "morning", "afternoon", "evening", "night"]


def clock_defaults(value=None):
    return {**deepcopy(CLOCK_DEFAULTS), **(value or {})}


def validate_temporal(value):
    if not isinstance(value, dict):
        raise ValueError("temporal must include relation, certainty, source_quote and interpretation")
    allowed = {"relation", "certainty", "source_quote", "interpretation", "operation", "elapsed_minutes", "date", "year", "time", "daypart"}
    if set(value) - allowed:
        raise ValueError("Unknown temporal field")
    if value.get("relation") not in {"current", "past", "plan"}:
        raise ValueError("relation must distinguish current/past/plan")
    if value.get("certainty") not in {"explicit", "estimated", "unknown"}:
        raise ValueError("certainty must be explicit/estimated/unknown")
    for key, limit in (("source_quote", 400), ("interpretation", 300)):
        if not isinstance(value.get(key), str) or not 1 <= len(value[key].strip()) <= limit:
            raise ValueError(f"Provide {key} (1–{limit} characters)")
    if value.get("operation") not in {"anchor", "correct", "advance", "until", "next_day", "reference"}:
        raise ValueError("Invalid temporal operation")
    if "date" in value:
        if not isinstance(value['date'], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value['date']):
            raise ValueError("date must be YYYY-MM-DD")
        parsed = date.fromisoformat(value['date'])
        if "year" in value and value['year'] != parsed.year:
            raise ValueError("Date and year disagree")
    if "year" in value and (type(value['year']) is not int or not 1 <= value['year'] <= 9999):
        raise ValueError("Invalid year")
    if "time" in value and (not isinstance(value['time'], str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value['time'])):
        raise ValueError("time must be HH:MM")
    if "daypart" in value and value['daypart'] not in DAYPARTS:
        raise ValueError("Invalid daypart")
    if "elapsed_minutes" in value and (type(value['elapsed_minutes']) is not int or not 1 <= value['elapsed_minutes'] <= 1440):
        raise ValueError("elapsed_minutes must be 1–1440")
    operation = value['operation']
    if operation == 'advance' and ('elapsed_minutes' not in value or value['certainty'] == 'unknown'):
        raise ValueError("advance needs explicit or estimated elapsed_minutes")
    if operation != 'advance' and 'elapsed_minutes' in value:
        raise ValueError("Only advance accepts elapsed_minutes")
    if operation == 'advance' and any(k in value for k in ('date', 'year', 'time', 'daypart')):
        raise ValueError("advance derives the calendar; do not supply a second target")
    if operation == 'next_day' and any(k in value for k in ('date', 'year', 'time')):
        raise ValueError("next_day accepts only a daypart; use advance for a quantified interval")
    if operation == 'until' and (not all(k in value for k in ('date', 'time')) or value['certainty'] == 'unknown'):
        raise ValueError('until requires a dated target time and explicit/estimated certainty')
    if operation in {'anchor', 'correct'} and not any(k in value for k in ('date', 'year', 'time', 'daypart')):
        raise ValueError("anchor needs known calendar fields")


def _daypart(time):
    hour = int(time[:2])
    return 'dawn' if hour < 6 else 'morning' if hour < 12 else 'afternoon' if hour < 18 else 'evening' if hour < 22 else 'night'


def move_minutes(clock, minutes, certainty):
    result = clock_defaults(clock)
    if result['time'] is not None:
        hour, minute = map(int, result['time'].split(':'))
        days, minute = divmod(hour * 60 + minute + minutes, 1440)
        result['time'] = f'{minute // 60:02d}:{minute % 60:02d}'
        result['relative_day'] += days
        if result['date']:
            result['date'] = (date.fromisoformat(result['date']) + timedelta(days=days)).isoformat()
            result['year'] = date.fromisoformat(result['date']).year
        result['daypart'] = _daypart(result['time'])
    else:
        # With no start time, midnight crossings are unknowable; don't retain a stale daypart/date.
        result['date'] = None
        result['daypart'] = 'unknown'
    if certainty == 'estimated':
        result['certainty'] = 'estimated'
    return result


def interpret_clock(state, temporal, advance_fn):
    """Return state after a validated interpretation. Past/plans never advance anything."""
    validate_temporal(temporal)
    result = deepcopy(state)
    clock = clock_defaults(state.get('clock'))
    operation = temporal['operation']
    if temporal['relation'] == 'current':
        if operation in {'anchor', 'correct'}:
            if operation == 'anchor':
                if 'date' in temporal and clock['year'] is not None and date.fromisoformat(temporal['date']).year != clock['year']:
                    raise ValueError('Date conflicts with the known year')
                for field in ('date', 'year', 'time'):
                    if field in temporal and clock[field] is not None and temporal[field] != clock[field]:
                        raise ValueError('Existing clock differs; use until to progress or correct for an explicit correction')
                if 'daypart' in temporal and clock['time'] is not None and temporal['daypart'] != _daypart(clock['time']):
                    raise ValueError('Daypart conflicts with known time')
            if any(k in temporal for k in ('date', 'year')):
                clock['date'] = temporal.get('date', clock['date'] if operation == 'anchor' else None)
                clock['year'] = date.fromisoformat(clock['date']).year if clock['date'] else temporal.get('year')
            if 'time' in temporal:
                clock['time'] = temporal['time']
                clock['daypart'] = _daypart(clock['time'])
            elif 'daypart' in temporal:
                clock['time'] = None
                clock['daypart'] = temporal['daypart']
            clock['certainty'] = 'estimated' if clock['certainty'] == 'estimated' and operation != 'correct' else temporal['certainty']
        elif operation in {'advance', 'until'}:
            if operation == 'until':
                if clock['date'] is None or clock['time'] is None:
                    raise ValueError('until requires a known start date and time; do not invent elapsed minutes')
                start = datetime.fromisoformat(clock['date'] + 'T' + clock['time'])
                end = datetime.fromisoformat(temporal['date'] + 'T' + temporal['time'])
                minutes = int((end - start).total_seconds() / 60)
                if minutes < 0:
                    raise ValueError('Cannot move backwards; use correct for a correction or reset for a new scene')
            else:
                minutes = temporal['elapsed_minutes']
            result = advance_fn(state, state['last_calculated_minute'] + minutes, temporal['interpretation'])
            clock = move_minutes(clock, minutes, temporal['certainty'])
        elif operation == 'next_day':
            if clock['date']:
                clock['date'] = (date.fromisoformat(clock['date']) + timedelta(days=1)).isoformat()
                clock['year'] = date.fromisoformat(clock['date']).year
            clock['relative_day'] += 1
            clock['time'] = None
            clock['daypart'] = temporal.get('daypart', 'unknown')
            clock['certainty'] = 'estimated' if clock['certainty'] == 'estimated' and operation != 'correct' else temporal['certainty']
            clock['unquantified_gaps'] += 1
            clock['elapsed_complete'] = False
    clock['last_interpretation'] = deepcopy(temporal)
    result['clock'] = clock
    return result


TEMPORAL_SCHEMA = {
    'type': 'object', 'properties': {
        'relation': {'type': 'string', 'enum': ['current', 'past', 'plan']},
        'certainty': {'type': 'string', 'enum': ['explicit', 'estimated', 'unknown']},
        'source_quote': {'type': 'string', 'minLength': 1, 'maxLength': 400},
        'interpretation': {'type': 'string', 'minLength': 1, 'maxLength': 300},
        'operation': {'type': 'string', 'enum': ['anchor', 'correct', 'advance', 'until', 'next_day', 'reference']},
        'elapsed_minutes': {'type': 'integer', 'minimum': 1, 'maximum': 1440},
        'date': {'type': 'string', 'description': 'YYYY-MM-DD; omit unknown dates'},
        'year': {'type': 'integer', 'minimum': 1, 'maximum': 9999},
        'time': {'type': 'string', 'description': 'HH:MM; omit unknown times'},
        'daypart': {'type': 'string', 'enum': DAYPARTS},
    }, 'required': ['relation', 'certainty', 'source_quote', 'interpretation', 'operation'],
    'additionalProperties': False,
}
