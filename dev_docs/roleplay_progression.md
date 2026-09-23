# 예조프 장면 진행과 고립 부담

현재 Telegram 상태 판정은 [roleplay_jev.md](roleplay_jev.md)가 기준이다. JEV가 분류하고 별도 LLM이 단일 사건의 소요 분만 추정하며 코드가 수치를 계산한다. 아래 time/update의 수치·조건·사건 인자는 내부 유지보수 API이며 연기 모델에 노출하지 않는다.

현재 소유자는 `runtime_tools/roleplay_dynamics.py`(계산), `roleplay_clock.py`(달력),
`roleplay_story.py`(예정 사건), `roleplay_pacing.py`(사용자 메시지별 진행 경계), `roleplay_memory.py`(트랜잭션·도구), `telegram/roleplay_bot.py`(문맥·표시),
`identity/roleplay_persona.md`(행동 지침)다. 모든 진행은 장면 시간으로 이루어지며 현실 타이머는 없다.

## 접촉과 고립

기존에는 짧은 접촉마다 240분을 차감하고 30분 이상 접촉하면 누적을 0으로 만들었다.
배식·심문도 이를 적용해 독방 부담이 잘 쌓이지 않았다. 현재는 다음 조건을 사용한다.

| 조건 | 게임상 부담 변화 |
|---|---|
| `isolation_mode=solitary` 또는 `unknown`, 의미 있는 교류 없음 | 경과 1분당 +1분 |
| `social_contact=incidental` (배식·점검), `hostile` (위협·심문) | solitary/unknown에서는 동일하게 누적, 회복으로 취급하지 않음 |
| `social_contact=meaningful`, 현장 인물 있음 | 경과 1분당 −2분, 0까지 |
| `isolation_mode=ordinary` (일상 생활), meaningful 아님 | 경과 1분당 −1분, 0까지 |

`none`은 교류 없음, `unknown`은 미확인이다. 현장 인물이 없는 구간은 접촉 조건이 남아 있어도 none이다.
인물이 바뀐 update는 명시적인 social_contact가 없으면 none/unknown으로 되돌린다. 두 새 조건은
`changes`와 `interval_conditions` 모두 지원한다. 초기 시간 계산의 필수 조건은 기존 activity/sleep_quality/
threat/injuries 네 가지이며, 추가 조건 때문에 legacy 저장이 초기화되지 않는다.

`isolation_hours`는 **고립 누적 부담의 게임 환산값**이며 독방 체류시간의 통계가 아니다.
24/72/168시간의 기존 단계와 정신 수치 계수는 유지하지만, 증상의 고정 발현 시각으로 해석하지 않는다.
지지 대화는 누적 부담을 한 번에 지우지 않으며, 가해자의 접촉을 기다린다는 서술도 회복·신뢰·동의를 뜻하지 않는다.
기존 상태의 고립 수치는 보존한다. 과거의 참가자 목록만으로 접촉의 질을 복원하거나 정신 수치를 소급 변경하지 않는다.

자료상 근거는 다음과 같다. 이 자료가 게임 계수나 특정 인물의 반응을 검증해 주지는 않는다.

- [UNODC, Nelson Mandela Rules 이행 점검표](https://www.unodc.org/documents/justice-and-prison-reform/UNODC_Checklist_-_Nelson_Mandela_Rules.pdf):
  독방의 기준은 하루 22시간 이상 의미 있는 인간적 접촉이 없는 상태다. 단순한 사람의 출입과 구분한다.
- [유럽평의회, European Prison Rules 해설](https://search.coe.int/cm?i=09000016809c9086):
  의미 있는 교류는 순간적·부수적인 접촉을 넘어서는 상호작용이다.
- [NCCHC, Solitary Confinement (Isolation)](https://ncchc.org/position-statements/solitary-confinement-isolation-2016/):
  불안·우울·인지적 어려움·감각 과민·지각 이상 등 악화 가능성을 설명한다. 모든 사람의 동일한 진행표는 아니다.

## 시간 분할과 깨어 있는 휴식

계산은 1분 단위다. `metric_remainders`에 반올림 잔여를 보존하고 모델용 뷰에서는 제외한다.
같은 조건의 60분 한 번과 30분 두 번은 같은 결과가 나온다. 직접 수치 변경은 해당 잔여를 지운다.

`alone_rest_minutes`는 사람이 없는 rest/sleep의 연속 분이다. 60분에 도달한 뒤부터 threat를 uncertain으로
완화하며, 앞선 위협 효과는 취소하지 않는다. 사람이 출입하거나 다른 활동을 하면 연속 시간이 끊긴다. 새로운 위협으로의 명시적 변경이나 부상·긴장 상승 사건도 연속 시간을 끊는다.
미래 위협의 해석 자체는 모델의 책임이며, 코드가 장면의 실제 안전을 판단하는 것은 아니다.

`wakefulness_minutes`는 게임상 각성 누적이다. 깨어 있는 1분당 +1, 수면 1분당 `2 × sleep_quality 계수`만큼
차감한다(0까지). rest의 피로 회복 하한은 `min(80, 각성누적시간 × 2)`다. 하한보다 높으면 최대 −2/h,
하한 이하에서는 회복하지 않으며 각성 누적 16시간 이상이면 +2/h다. 통증에 따른 회복 저하는 기존대로다.
수면이 아니어도 피로가 무한히 회복되던 경로를 막으며, 실제 수면 생리의 수치 모형을 주장하지 않는다.

## 예정 사건

`roleplay_state(action="update", story_updates=[...])`는 상태·인물 변경과 같은 SQLite 트랜잭션으로 처리된다.
새 도구나 외부 스케줄러를 사용하지 않는다.

- `schedule`: id/title/source와 due_minute 또는 after_event. due_minute는 절대 누적 장면 분이다.
  after_event는 이미 등록된 선행 사건 ID다. 두 조건이 있으면 둘 다 충족해야 한다.
- `complete`: ready 사건의 id/outcome. 실제 장면에서 완료된 결과만 기록한다.
- `cancel`: id/outcome. pending/ready를 취소하고 후속 의존 사건도 취소한다.

등록 시 선행 사건이 이미 존재해야 하고 기존 ID를 변경할 수 없어 순환 의존을 만들지 못한다.
동일 등록·동일 완료/취소 재시도는 기존 결과를 유지한다. 다른 결과로 덮어쓰거나 종료된 사건을 되살리지 않는다.
활성 사건은 20개, 전체는 최대 100개다. 초과 시 다른 사건이 참조하지 않는 오래된 종료 기록부터 제거하고,
제거할 수 없으면 오류다. 모델용 뷰에는 활성 사건과 최근 종료 5건만 보낸다. reset은 모두 비운다.

advance/until은 첫 도래 사건에서 멈추며 시계·수치도 실제 진행한 분까지만 계산한다.
동시 도래 사건은 모두 ready가 된다. ready를 처리하지 않으면 다음 시간 진행도 그 시점에서 멈춘다.
`story_interrupt`에 사건 ID, 요청 시점, 중단 시점, 남은 분을 반환한다. 요청 끝의 changes/person_updates/
resolve_event는 **중단 시 적용하지 않는다**. 완료/취소를 기록하고 남은 구간을 새 event_id로 진행해야 한다.
중단 시점과 요청 끝이 같아도 끝 시점 변경은 보류해 아직 다루지 않은 사건의 결과를 선행 확정하지 않는다.

미래 계획·회상은 시간을 진행하지 않는다. 하루 안에 도래하거나 ready인 사건이 있으면 경과량 미상의 next_day를 거절한다(먼 연표 사건·혼자 남을 때 신호는 막지 않음).
명시적인 시계 정정은 기존처럼 수치를 소급 취소하지 않으며 due_minute도 장면의 상대 시간 기준을 유지한다.
source의 의미나 예정의 정당성까지 코드로 검증하지 않는다. 사용자 설정·실제 약속에 근거하고
불확실한 큰 사건은 먼저 제안하도록 페르소나 지침을 둔다.

`/status`는 활성 예정 사건 전체와 교류·고립 환경·각성 누적·고립 부담 등 상세 내용을 항상 표시한다.
`/status 상세`와 `/status detail`도 같은 출력이다. 인물의 자발적 행동은 goal/next_action과 예정 사건 문맥을 읽는 **연기 지침**이다. 사용자가 지시한 사건의 완료가 턴 종료점이다. 감방으로 보내면 도착에서 끝내며 식사·수면·다음 날을
추가하지 않는다. 작은 자발적 반응도 이 범위 안에서만 허용하고, 시도와 성공을 구별한다. 행동 연기 외에 단일 사건 시간 추정용 LLM 호출이 있다.

## 검증과 적용

- `venv/bin/python -m unittest discover -s tests -p 'test_roleplay*.py'`
- `venv/bin/python scripts/smoke_tool_allowlists.py`

회귀 검증은 시간 분할, 배식·심문이 있는 3일 독방, 지지 대화의 점진 회복, 일상적 고독과 수면,
예정 사건에서 시계 중단, 끝 시점 효과 보류, 완료 후 재개, 동시 사건, 취소 전파, 트랜잭션 롤백,
재시도·사용자 분리·reset·입력 한도를 포함한다. 실제 모델의 연기 품질은 이 단위 테스트만으로 보장하지 않는다.
Python·도구 스키마 변경은 `leninbot-roleplay.service` 재시작 후 적용된다.


## 사용자 지시의 사건 경계와 시간 근거

`telegram.roleplay_bot.handle_message`는 실제 사용자 원문에서 `TurnTimePolicy`를 만들고 ContextVar로
도구 실행까지 전달한다. 모델이 도구 인자로 허가 범위를 확장할 수 없다. runtime context에도 같은 정책을 표시한다.

기본은 **사용자가 지시한 하나의 사건을 완료하고 그 장면에서 멈춤**이다. "감방으로 보내" 뒤에는 도착 장면만
서술하며 씻기·식사·잠·다음 날은 추가하지 않는다. "어떻게 할까?"라는 전개 상담도 실행 허락이 아니다.
서술의 의미상 사건 경계는 페르소나 지침이며, 코드의 추가 방어는 다음과 같다.

- current 시간 요청의 source_quote가 이번 사용자 원문의 연속 인용인지 NFKC·공백 정규화 후 확인한다.
  모델이 자기 서술을 시간 근거로 다시 제출하면 저장 전에 거절한다.
- 명시적 기간 진행이 없으면 경과 구간은 한 턴에 최대 한 번, 총 10분을 안전 상한으로 둔다.
  10분을 자동 진행한다는 뜻이 아니며 사건이 끝나면 그보다 먼저 멈춘다. 두 번째 구간은 남은 한도와 무관하게 거절한다.
- 시간 허가의 의미(진행인지 질문인지, 명시 기간인지, 인물에게 맡긴 개방형 휴식인지, 다음 날 전환인지, 정정·초기화인지)는
  허가 LLM의 라벨(mode/time_scope/span/transition)이 정한다(`roleplay_pacing.policy_for`). 키워드 목록으로 의미를 추정하지 않는다.
  Telegram 명시 기간은 허가 LLM이 문맥에서 선택한 duration_minutes를 재사용하며 원문 기간을 합산하지 않는다. 숫자+단위 파서는 내부/legacy 호환용이다. 명시 기간은 여러 활동 구간으로 나눌 수 있다.
- 누적 분과 구간 수는 SQLite의 pacing_scope_id/pacing_spent_minutes/pacing_passages로 관리한다.
  같은 Telegram 메시지의 반복 도구 호출·continuation으로 한도를 우회하지 못한다. 사건 중단은 실제 적용 분만 사용한다.
  거절은 전체 상태·인물 트랜잭션을 롤백하고 기존 event_id 재시도는 재적용하지 않는다.
- next_day는 명시적 날짜 진행에만 허용하고 하루분을 한도에서 차감한다. correct는 사용자 정정 문구가,
  reset은 새 장면·초기화 문구가 있어야 한다. past/plan/reference는 기존처럼 시간 진행을 하지 않는다.
- 거절한 시간은 답변 서술에서도 완료된 사실로 쓰지 않도록 지침을 둔다. 자유서술 전체를 코드로 의미 검증하지는 않는다.

## 철회한 턴과 복구

`history_exclusions(user_id,message_id,reason)`는 PostgreSQL의 대화 원본 ID를 보존하면서 해당 사용자 문맥에서
철회한 턴을 제외한다. load_history의 개수 계산과 본문 조회 모두 제외 조건을 적용한다.
`turn_retractions(user_id,scope_id,reason)`는 state_history 원본을 유지하면서 history 도구에서 철회한 scope를 제외한다.
현재 상태 복구는 잘못된 턴 이전의 전체 스냅샷으로 하며 revision은 계속 증가시킨다. 인물 필드가 바뀐 경우
이전 유효 이력에서 복원하고, 별도 수정과 충돌하면 중단해야 한다. 메모·직접 인물 변경은 상태 이력에 모두
남지 않으므로 tool_audit_log에서 해당 턴의 실제 도구 호출을 먼저 확인한다. 복구 전 SQLite 백업을 보관하고
별도 repair 이력에 복구 대상과 전후 상태·인물 필드를 기록한다. 원본 대화나 감사 기록을 삭제하지 않는다.

검증: `tests/test_roleplay_pacing.py`는 실제 오류 입력의 100/640/780/630분 연속 진행 거절,
도착 이후 두 번째 사건 차단, 원문 밖 인용, 명시적 기간의 분할·누적 한도, 날짜·정정·reset 우회,
메시지별 분리, 철회 대화·상태 이력 제외를 확인한다.

새 초안 사후 정산 절차·심문 사건 3종·다음 아침 시간의 미상 구간 처리 및 현재 활성화 상태는
[초안 사후 정산](roleplay_postdraft.md)을 참조한다. 연기 모델의 기록은 최종 확정 전 임시 저장된다.

폐렴은 부상과 별도의 `illnesses` 상태로 기록한다. 현재 장면 판정·시간 효과·회복과 재발·표시 계약은
[질병 상태](roleplay_game_balance.md#질병-상태)를 따른다. 과거 사료로 현재 발병을 소급 적용하지 않는다.

ready 상태의 when_alone 신호는 실제 예정 사건과 구분하며 경과 정산의 중단 목록에도 넣지 않는다.
신호가 준비돼 있다는 이유만으로 같은 초안을 반복 폐기하지 않는다.
