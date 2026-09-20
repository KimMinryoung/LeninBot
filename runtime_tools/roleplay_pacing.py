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
# A request for suggestions or a hypothetical is not permission to enact a skip.
_DISCUSSION = re.compile(r'어떻게|어떨|할까|해볼까|좋을까|나을까|가정|계획|제안|설명|상담|만약|했더라면|지나면|지난다면')
_PROGRESS = re.compile(r'넘겨|넘기|건너뛰|진행|흘렀|흘러|지났다|지났어|지났고|지나갔|기다려|기다린|기다렸|쉬어|쉬자|쉰다|쉬었다|쉬었|잤다|잤어|잤고|잠을\s*잔다|보냈|보낸다|보내자|보내라|보내줘|씻어|씻었다|정돈해|정돈한다|읽어|읽었다|정리해|정리했다|가다듬어|가다듬었다')
_DAY_SKIP = re.compile(r'다음\s*날|다음날|내일|아침까지|밤새|밤을\s*넘')
_CORRECTION = re.compile(r'정정|수정|바로잡|되돌|잘못|아니라|틀렸')


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


@dataclass(frozen=True)
class TurnTimePolicy:
    user_text: str
    max_minutes: int
    calendar_skip: bool
    correction: bool
    explicit_passage: bool

    def view(self):
        return {'max_elapsed_minutes_this_turn': self.max_minutes,
                'explicit_calendar_skip': self.calendar_skip,
                'max_passages_this_turn': None if self.explicit_passage else 1,
                'rule': '사용자가 지시한 사건의 완료가 턴 종료점. 감방으로 보내면 도착에서 끝내며 식사·수면·다음 날을 추가하지 않음. 현재 사용자 원문의 연속 인용만 시간 근거로 허용. 한 턴 전체의 누적 한도이며 호출을 나눠도 증가하지 않음. '
                        '진행 방안 질문·계획은 실행 아님. 허용되지 않은 다음 날·회복 구간을 서술로도 완료하지 않음.'}


def duration_minutes(user_text):
    """Parse literal durations; meaning/authorization is adjudicated separately."""
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


def policy_for(user_text):
    text = normalized(user_text)
    discussion = bool(_DISCUSSION.search(text))
    explicit = not discussion and bool(_PROGRESS.search(text))
    total = duration_minutes(text) if explicit else 0
    calendar = not discussion and bool(_DAY_SKIP.search(text)) and (explicit or bool(re.search(r'다음\s*날(?:이다|이\s*되|\s*아침)|아침이\s*되', text)))
    budget = min(MAX_EXPLICIT_MINUTES, max(DEFAULT_TURN_MINUTES, total, 1440 if calendar else 0))
    return TurnTimePolicy(text, budget, calendar, bool(_CORRECTION.search(text)), bool(total or calendar))


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
    if policy and not re.search(r'초기화|새\s*장면|처음부터|장면.{0,8}(?:바꿔|변경)|리셋', policy.user_text):
        raise ValueError('초기화 거절: 사용자가 새 장면·초기화를 명시하지 않았음. reset으로 시간 제한을 우회하지 말 것')
