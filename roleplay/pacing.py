"""User-grounded, per-message fictional time bounds for the Telegram roleplay loop."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
import unicodedata

DEFAULT_TURN_MINUTES = 10
MAX_EXPLICIT_MINUTES = 7 * 1440
_POLICY = ContextVar('roleplay_turn_time_policy', default=None)
_NUMBERS = {'한': 1, '하나': 1, '두': 2, '둘': 2, '세': 3, '셋': 3, '네': 4, '넷': 4,
            '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
            '하루': 1, '이틀': 2, '사흘': 3, '나흘': 4}
_DURATION = re.compile(r'(?P<n>\d+(?:\.\d+)?|다섯|여섯|일곱|여덟|아홉|하나|둘|셋|넷|한|두|세|네|열)\s*(?P<u>시간|분|일|주)|(?P<days>하루|이틀|사흘|나흘)')
def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


@dataclass(frozen=True)
class TurnTimePolicy:
    user_text: str
    max_minutes: int
    calendar_skip: bool
    correction: bool
    explicit_passage: bool
    reset: bool = False
    explicit_minutes: int | None = None

    def view(self):
        return {'max_elapsed_minutes_this_turn': self.max_minutes,
                'explicit_calendar_skip': self.calendar_skip,
                'max_passages_this_turn': None if self.explicit_passage else 1,
                'rule': '사용자가 지시한 사건의 완료가 턴 종료점. 감방으로 보내면 도착에서 끝내며 식사·수면·다음 날을 추가하지 않음. 현재 사용자 원문의 연속 인용만 시간 근거로 허용. 한 턴 전체의 누적 한도이며 호출을 나눠도 증가하지 않음. '
                        '진행 방안 질문·계획은 실행 아님. 허용되지 않은 다음 날·회복 구간을 서술로도 완료하지 않음.'}


def duration_minutes(user_text):
    """Parse literal durations; meaning/authorization is adjudicated separately."""
    active = _POLICY.get()
    if active is not None and active.user_text == normalized(user_text) and active.explicit_minutes is not None:
        return active.explicit_minutes
    text, total = normalized(user_text), 0
    for match in _DURATION.finditer(text):
        if re.match(r'\s*(?:전|후에\s*할|뒤에\s*할)', text[match.end():]):
            continue
        if match['days']:
            total += _NUMBERS[match['days']] * 1440
        else:
            value = _NUMBERS.get(match['n'])
            if value is None:
                value = float(match['n'])
            total += value * {'분': 1, '시간': 60, '일': 1440, '주': 10080}[match['u']]
    return int(total)


SESSION_MINUTES = 180
TIME_SCOPES = ('none', 'explicit', 'open_ended', 'day_skip')


def policy_for(user_text, *, mode='scene', time_scope='auto', span='brief', transition='current', explicit_minutes=None):
    """The turn's time permission from the authorization LLM. Nothing here reads
    meaning out of keywords: the only text parsing is the number+unit duration
    ("2시간", "이틀") is retained for offline/legacy callers only. Telegram supplies
    explicit_minutes from the LLM, including zero, bypassing the text parser.

    time_scope: none / explicit (a stated duration) / open_ended (length left to the
    character) / day_skip (move to the next day). 'auto' means explicit when a number
    and unit are present, otherwise none."""
    active = _POLICY.get()
    if active is not None and active.user_text == normalized(user_text) and time_scope == 'auto' and mode == 'scene':
        return active
    text = normalized(user_text)
    if explicit_minutes is not None and (type(explicit_minutes) is not int or not 0 <= explicit_minutes <= MAX_EXPLICIT_MINUTES):
        raise ValueError('Invalid authorized duration')
    if mode != 'scene':
        return TurnTimePolicy(text, DEFAULT_TURN_MINUTES, False, mode == 'correction', False, mode == 'reset', explicit_minutes)
    total = duration_minutes(text) if explicit_minutes is None else explicit_minutes
    if time_scope == 'auto':
        time_scope = 'explicit' if total else 'none'
    if time_scope not in TIME_SCOPES:
        raise ValueError(f'time_scope must be one of {TIME_SCOPES}')
    calendar = time_scope == 'day_skip' or transition != 'current'
    budget = max(DEFAULT_TURN_MINUTES, total if time_scope == 'explicit' else 0,
                 SESSION_MINUTES if (time_scope == 'open_ended' or span == 'session') else 0)
    if calendar:
        budget += 1440  # the skipped day itself, on top of whatever the scene then spends
    return TurnTimePolicy(text, min(MAX_EXPLICIT_MINUTES, budget), calendar, False,
                          bool(time_scope == 'explicit' and total) or calendar, explicit_minutes=explicit_minutes)


@contextmanager
def turn_time_scope(policy):
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def check_time_request(state, temporal, scope_id):
    """Validate before calculation; returns policy or None for offline callers."""
    policy = _POLICY.get()
    if policy is None or temporal['relation'] != 'current':
        return None
    operation = temporal['operation']
    if operation == 'reference':
        return None
    quote = normalized(temporal['source_quote'])
    if not quote or quote not in policy.user_text:
        raise ValueError('시간 진행 거절: source_quote는 이번 사용자 메시지의 실제 연속 인용이어야 함. '
                         '자신이 만든 밤·수면·사건을 근거로 다시 시간을 진행하지 말고 현재 장면에서 답변을 끝낼 것')
    if operation == 'correct' and not policy.correction:
        raise ValueError('시간 정정 거절: 사용자가 현재 시각의 오류를 정정한 경우에만 correct 사용')
    if operation == 'next_day' and not policy.calendar_skip:
        raise ValueError('날짜 진행 거절: 사용자의 명시적인 다음 날 진행 없이 날짜를 건너뛸 수 없음')
    return policy


def check_time_result(before, after, temporal, scope_id, policy):
    """Charge only applied minutes (including event stops), inside the SQLite transaction."""
    if policy is None or temporal['operation'] not in {'advance', 'until', 'next_day'}:
        return
    spent = before.get('pacing_spent_minutes', 0) if before.get('pacing_scope_id') == scope_id else 0
    passages = before.get('pacing_passages', 0) if before.get('pacing_scope_id') == scope_id else 0
    elapsed = after['scene_minute'] - before['scene_minute']
    if temporal['operation'] == 'next_day':
        elapsed = 1440  # an unquantified calendar day still spends a day of permission
    if elapsed > 0 and passages >= 1 and not policy.explicit_passage:
        raise ValueError('추가 사건 진행 거절: 이번 사용자 지시의 사건은 이미 처리됨. '
                         '이동 뒤 식사·휴식·수면 같은 별개 사건을 이어가지 말고 그 도착 장면에서 답변을 끝낼 것')
    if spent + elapsed > policy.max_minutes:
        raise ValueError(f'시간 진행 거절: 이번 사용자 메시지는 총 {policy.max_minutes}분까지만 허용함 '
                         f'(이미 {spent}분 반영, 이번 요청 {elapsed}분). '
                         '진행 방안 질문을 시간 건너뛰기 허락으로 해석하지 말 것. '
                         '더 진행하지 말고 현재 장면의 반응·다음 선택을 답변할 것. 큰 시간 이동은 사용자가 직접 지정해야 함')
    after['pacing_scope_id'] = scope_id
    after['pacing_spent_minutes'] = spent + elapsed
    after['pacing_passages'] = passages + int(elapsed > 0)


def check_reset_request():
    policy = _POLICY.get()
    if policy and not policy.reset:
        raise ValueError('초기화 거절: 사용자가 새 장면·초기화를 명시하지 않았음(허가 판정 mode=reset 아님). reset으로 시간 제한을 우회하지 말 것')
