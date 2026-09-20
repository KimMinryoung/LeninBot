# 예조프 자동 상태 판정

Telegram의 새 처리 절차는 사용자 허용 범위를 확인하고 초안을 생성한 다음 실제 초안의 사건을 판정한다.
정산·서술 검토·저장·전송 순서와 현재 활성화 상태는 [초안 사후 정산](roleplay_postdraft.md)을 참조한다.
운영 상태를 과거 대화에서 자동 소급 재계산하지 않는다.

| 책임 | 실행 주체 |
|---|---|
| 실행/상담/과거/계획 구분, 사건·활동·접촉·부상·인물 출입·예정 사건 판정 | Jev의 고정 선택지 |
| 명시적 기간이 없는 단일 사건의 소요 분 | 별도 LLM 원샷 `roleplay_duration_estimate` |
| 사용자가 명시한 기간 | 원문 파서 + 진행 허용 검사 |
| 수치 변화·시간 적분·달력·예정 사건 도래·저장 | Python 코드 |
| 초안 연기·목적·기분·질적 관찰 | 기존 연기 모델 |
| 정산 결과와 초안/새 기록의 서술 모순 검사 | 같은 제공자의 별도 검토 호출 |

분류는 `roleplay_scene_adjudication` 등록을 사용하며 세 호출로 나뉜다(`question_group`): `roleplay-scene`(mode/elapsed/plan_action/event/
sexual_act/intensity/활동/수면/위협/접촉/고립/장소/새 부상, 현재 상태에서 15문항), `roleplay-people`(인물별 출입, 저장 인물 수만큼),
`roleplay-records`(부상·예정 사건·holdout·bargain). 2026-09-20 이전에는 한 호출에 33문항(사건 선택지 25개, 설명 5천 자)을 얹어 항목당 확신이
옅어졌다. people/records 호출이 실패하면 그 키만 미확정으로 남고 장면 판정은 진행된다. 수락 기준은 핵심 라벨(mode/event/elapsed/
intensity/sexual_act/plan_action) 0.65, 보조 라벨 0.5다(`CORE_LABELS`, thresholds.accept/secondary; 확정 턴의 실측치는 쉬운 항목 0.80~0.92,
애매한 항목 0.57~0.59였다). 핵심 라벨이 핵심 mode/event/elapsed가
불확실하면 원칙적으로 변경을 보류한다. 단, mode/event/intensity가 확정된 즉시 사건(kindness/recognition/sexual_harassment/sexual_assault/rape/threat_to_kin/public_submission/futile_effort)은 elapsed만 불확실할 때 해당 효과를 적용하고 시간과 장면 조건은 보류한다. applied.deferred_components에 이를 표시한다. 그 외 불확실한 선택지는 기존 값을 유지하거나 초기화를 보류한다.
분류 장애를 생성 모델로 대체하지 않는다. 이동+회복 상담은 이동 한 사건만 처리한다.

시간 LLM에는 원문·실제 초안·기존 장소/장면·Jev 라벨을 전달한다. 출력은 정수 elapsed_minutes와 reason만
허용한다. 기본 한도는 0~10분(허용된 긴 세션은 최대 180분)이며 한도에 맞지 않는 사건은 억지로 축약하지 않고 보류한다.
잘못된 JSON·잘린 출력·오류·범위 밖 값도 보류한다. 명시적인 ‘한 시간 쉬어’는 이 호출 없이 60분이다.
Jev가 상담/과거/계획으로 판정하면 실제 경과를 적용하지 않는다. 생성 모델은 사건과 수치 계수를 선택하지 못한다.

이벤트 효과는 `EVENT_DELTAS`의 게임 상수와 기존 의지 사건 표를 사용한다. 식사 -35, 간식 -15의
허기는 섭취 사건에만 적용되며 음식 도착은 섭취가 아니다. 부상/처치·강도도 고정 선택지와 표를 사용한다.
의학적 예측값이 아니다. 사용자 숫자가 있는 명시적 수치 정정만 원문에서 파싱한다.

예정 사건은 명시적인 상대 기간과 등록 의도가 모두 확인될 때 등록한다. 코드가 도래 시점에서 경과를
중단하고 끝 시점의 사건 효과를 보류한다. 완료/취소는 사용자 사건에 대한 Jev 판정으로 처리한다.

분류·시간 추정은 DB 쓰기 잠금 밖에서 실행한다. 저장 직전 revision 비교와 SQLite 트랜잭션으로
경쟁 변경을 덮어쓰지 않는다. `automatic_turns`는 사용자/메시지별 처리 결과를 보존하여 재처리를 막는다.
`state_history`에는 판정 원문, 시간 추정 모델/근거, 변경 전후를 남긴다. 보류 시 기존 상태를 보존한다.

연기 모델에 공개하는 `roleplay_state` 스키마는 조회/이력과 질적 필드 변경만 허용한다. Telegram 실행
컨텍스트에서도 숫자·시간·조건·이벤트 변경을 거절한다. 내부 Python 유지보수 인터페이스는 기존
time/update/reset 계약을 유지한다. 다른 문서의 내부 도구 예시는 연기 모델의 현재 권한을 뜻하지 않는다.

검증: `venv/bin/python -m unittest discover -s tests -p 'test_roleplay*.py'`.
단일 사건 경계, 식사/배달, 명시적 기간, 예정 사건 중단, 실패 시 보존, 재처리 방지, revision 충돌,
연기 모델 권한 제한과 시간 출력 검증을 포함한다. 실제 모델 샘플 확인은 전반적인 분류 정확도 평가가 아니다.

현재 인물 대사 속 인정·허락은 현재 사건이다. 발언에 내일의 협조 조건이 있어도 현재의 인정 효과를 미래 계획으로 바꾸지 않는다. `recognition`은 능력·기여·쓸모 인정, `kindness`는 배려·양보다. 같은 발언의 인정+양보는 recognition 하나로 처리한다. 현재 회복 보상은 종류별 고정값이며 recognition은 의지 +5/굴욕 -6/긴장 -3, kindness는 의지 +3/굴욕 -3/긴장 -3이다. 반복 보상과 활동 규칙은 [roleplay_game_balance.md](roleplay_game_balance.md)를 따른다. 인정이 안전·신뢰 회복을 자동 보장하지 않는다.

mode/event/intensity가 불확실하면 사용자 원문과 직전 장면, 이미 수락한 사건 라벨만으로 Jev에 한 번 집중 재판정을 요청한다. 같은 0.75 기준을 유지하며 재판정 답변도 감사 기록에 보존한다. 그래도 사건/강도가 불확실하면 임의 가산하지 않는다.

## 지키는 것·미뤄 둔 반응·플레이어 선택

held 상태의 holdout마다 `holdout_<i>`(keep/lost), ready 상태의 `when_alone` 신호에는 완료 기준이 다른 `story_<i>` 질문이 붙는다.
효과와 근거는 [roleplay_game_balance.md](roleplay_game_balance.md). event 선택지에서 `holdout_lost`·`sexual_coercion`은 제외한다.

열린 거래마다 `bargain_<i>`(keep/paid/kept/broken), `kind=routine`·`track` 예정 사건에는 각각의 완료 기준을 가진 `story_<i>` 질문이 붙는다.

event 또는 intensity가 재판정 뒤에도 불확실하면 `project`는 `PendingChoice(key, candidates)`를 던진다. candidates는 JEV의 event 확률
상위 3개(+none) 또는 강도 3단계다. Telegram은 이를 버튼으로 플레이어에게 묻고, 고른 라벨을 같은 verdict에 넣어 동일 초안을
다시 정산한다(`roleplay_turn.prepare(verdict=…)`는 범위 검사·분류·시간 추정을 반복하지 않는다). 다른 불확실성(elapsed·activity·location 등)은
이전처럼 보류다. 플레이어 선택은 감사 기록의 verdict.labels에 남는다. 흐름은 [roleplay_postdraft.md](roleplay_postdraft.md).

## 성적 가해와 당사자 활동

구형 sexual_coercion은 과거 기록/내부 호환용이며 새로운 JEV 선택지에서 제외한다.
성적 가해는 행위 종류와 강도를 따로 판정한다. 다음은 보통 강도의 게임상 기본 변화이며 임상적 피해 순위나 법적 정의가 아니다.

| 이벤트 | 근거 | 의지 | 긴장 | 굴욕 |
|---|---|---:|---:|---:|
| sexual_harassment | 비접촉 성적 언행 | -1 | +2 | +2 |
| sexual_assault | 원치 않는 접촉·강제 탈의, 삽입 없음 | -3 | +5 | +5 |
| rape | 현재 사건에서 명시된 비동의 삽입 | -8 | +10 | +10 |

별도 sexual_act 판정과 사건 종류가 일치해야 한다. 명시적 삽입 부정과 rape가 충돌하면 거절한다.
행위가 불명확한 sexual_unspecified는 최대 피해로 추정하지 않고 보류한다. 유형과 강도는 독립이며
유형 자체를 이유로 항상 severe로 올리지 않는다. 시간만 불확실하면 확정된 사건 효과만 반영할 수 있다.
부상은 명시된 새 부위/정도가 있어야 생성하며 사건 종류만으로 상처·통증을 만들어 내지 않는다.

활동은 피해 당사자의 실제 움직임을 뜻한다. restrained는 붙잡힘·위협 아래 동결 상태로, 피로 +2/h,
움직임 통증 가산 0, 급성 통증의 휴식 회복 없음이다. 실제 격렬한 저항은 strenuous이며, 가해자의
움직임으로 이를 선택하지 않는다. 진행 중 신체적 가해를 rest/sleep 회복으로 계산하면 거절하고,
계산 구간은 immediate 위협·hostile 접촉으로 처리한다. 수면 질은 근거가 없으면 keep을 선택할 수 있다.

## 연기 에이전트의 문맥

`roleplay_actor.py`는 내부 수치를 연기용 문장으로 변환한다. Telegram의 character_state와
roleplay_state read/history/update 결과는 이 허용 목록 뷰를 사용한다. 수치·변화량·계수·계산 이력·
부상 진행 분·시간 예산·판정 신뢰도·모델 정보는 연기 에이전트에 전달하지 않는다.
revision은 서술 저장의 동시성 토큰으로, 장면 날짜/시각은 연속성 정보로 유지한다.
scene_direction은 확정된 사건 범위/중단/보류를 자연어 연기 지침으로 전달한다.
시스템 프롬프트에 점수 임계값이나 수치 초기화 안내를 두지 않는다.

원본 상태, JEV 입력, 시간 추정, 감사 기록과 사용자의 /status 수치 표시는 유지한다.
state_view는 내부 진단용 숫자 뷰이며 Telegram 연기 문맥에는 사용하지 않는다.
기존 대화 원문과 사용자가 직접 말한 숫자를 삭제하거나 바꾸지는 않는다.

규칙 버전 4에서 회복 이벤트는 강도 판정 대신 코드 고정값을 사용한다. JEV가 종류를 확정하면 intensity 미확정만으로 보류하지 않는다. 추가 이벤트·활동·감쇠 규칙은 [roleplay_game_balance.md](roleplay_game_balance.md)가 기준이다.
