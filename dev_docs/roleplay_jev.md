# 예조프 자동 상태 판정

Telegram의 새 처리 절차는 사용자 허용 범위를 확인하고 초안을 생성한 다음 실제 초안의 사건을 판정한다.
정산·서술 검토·저장·전송 순서와 현재 활성화 상태는 [초안 사후 정산](roleplay_postdraft.md)을 참조한다.
운영 상태를 과거 대화에서 자동 소급 재계산하지 않는다.

| 책임 | 실행 주체 |
|---|---|
| 사용자 실행/상담/계획·시간 허가·명시 기간·수치 정정의 대상과 최종값 해석 | 초안 전 LLM `roleplay_time_authorization` |
| 사건·활동·접촉·부상·인물 출입·예정 사건 판정 | Jev의 고정 선택지 |
| 초안 범위·시각·소요 분과 중요한 결과의 원문 근거 | 별도 LLM 원샷 `roleplay_duration_estimate` |
| 시간 허가(진행·개방형 휴식·다음 날·정정·초기화) | 초안 전 LLM의 라벨과 선택된 duration_minutes; Python은 형식·범위·일관성 검증 |
| 수치 변화·시간 적분·달력·예정 사건 도래·저장 | Python 코드 |
| 초안 연기·목적·기분·질적 관찰 | 기존 연기 모델 |
| 정산 결과와 초안/새 기록의 서술 모순 검사 | 현재 비활성. 선택적으로 Jev 선별·기존 연기 제공자 상세 검토를 켤 수 있음 |

분류는 `roleplay_scene_adjudication` 등록을 사용하며 세 호출로 나뉜다(`question_group`): `roleplay-scene`(mode/elapsed/plan_action/
사건 축 5개/sexual_act/intensity/활동/수면/위협/접촉/고립/장소/새 부상), `roleplay-people`(인물별 출입), `roleplay-records`(부상·예정 사건·
holdout·bargain). people/records 호출이 실패하면 그 키만 미확정으로 남고 장면 판정은 진행된다. 수락 기준은 핵심 라벨(mode/사건 축/elapsed/
intensity/sexual_act/plan_action) 0.5, 보조 라벨 0.5다(`CORE_LABELS`, thresholds.accept/secondary). 핵심 라벨이 재판정 뒤에도 불확실하면
`PendingChoice`가 나고 봇이 자동 처리하거나 버튼으로 묻는다([roleplay_postdraft.md](roleplay_postdraft.md)). mode/사건/강도가 확정된 즉시 사건은
elapsed만 불확실할 때 해당 효과를 적용하고 시간과 장면 조건은 보류한다(applied.deferred_components). 그 외 불확실한 보조 선택지는 기존 값을
유지한다. 분류 장애를 생성 모델로 대체하지 않는다. 이동+회복 상담은 이동 한 사건만 처리한다.

0.5는 API의 `confidence` 기준이며 최상위 선택지의 확률이나 정답률 50%를 뜻하지 않는다.
정정 가능한 게임 판정에서 불필요한 재판정을 줄이기 위한 운영값이다. 설정 누락 시 코드 기본값도 0.5이며,
그 미만의 재판정·자동 선택 표시와 시간·저장 검증은 유지한다. 실제 정확도는 별도 정답 표본으로 평가해야 한다.
해석 근거: [TypeSafe confidence 문서](https://docs.typesafe.ai/confidence).

Telegram은 scene 모드에만 이 사건 분류를 호출한다. discussion/correction/reset/plan은 초안 전 LLM의 판정을
코드가 검증·적용하며 사건 분류·소요 추정·서술 검토를 생략한다. 세 경로의 계약은
[입력 모드별 처리](roleplay_postdraft.md#입력-모드별-세-처리-경로)를 따른다.

시간 검사는 모든 scene 초안에서 호출한다. 원문·초안·저장 시계·허가·일과·예정 중단점을 전달하고
elapsed_minutes/reason/within_scope/timeline_anchor와 important_evidence를 받는다. 명시 기간은 여전히 허가된 분으로 정산하며
시간 검사는 초안이 그 범위를 넘었는지만 확인한다. 비명시 기간은 검증된 추정값을 사용한다.
초안의 마지막 명시 시각은 날짜·시각·원문 인용으로 추출하고, Python이 저장 시계와의 차이를 계산한다.
인용이 초안에 없거나 과거 시각·허용 상한·예정 중단점을 넘으면 저장하지 않는다. -1이나 상한 초과도
절단하지 않고 초안을 다시 쓴다. 명시 시각의 추출과 행위 범위의 의미 판정은 여전히 모델에 의존한다.

규칙 버전 11부터 holdout/bargain/story 미확정은 별도 roleplay-record-review로 한 번 재판정한다.
규칙 버전 12에서는 미확정 holdout/bargain을 중요한 결과 확인으로 남긴다. story 등 일반 기록은 기본 모드에서
최신 확률순으로 한 번에 선택하며 확률 부재·동률이면 keep을 우선한다. /ask 켜기에서는 일반 기록도 묻는다.
이전처럼 미확정을 조용히 keep으로 버리지 않는다. 재판정 비용·지연·답변은 calls/answers에 포함한다.

사건은 다섯 독립 축으로 판정한다(`EVENT_FAMILIES`). 물질적 축 harm(부상·구타·성적 가해)/care(처치)/intake(식사·간식·물)와
영향 축 pressure(심문·자백·연루·협박·공개 굴욕·헛수고·실패·파기)/relief(배려·인정·선택권·성취·경계 존중·지지)는 서로 독립이며, 한 축 안에서는
가장 구체적인 결과 하나만 고른다. 간수가 죽을 건네고 먹은 장면은 intake=meal과 relief=kindness가 함께 성립한다. 각 축은 none을 포함한 별도
질문(`event_<family>`)이고, 코드는 확정된 축들을 우선순위(harm > pressure > relief > care > intake)로 정렬해 `events` 목록과 대표 라벨 `event`를 만든다
(`resolve_events`). 불확실한 축이 none 쪽으로 기울면(none 확률 ≥ 0.5) none으로 확정하고, 다른 결과 쪽으로 기울면 그 축만 미확정으로 남겨 후보
(`event_candidates`: 미확정 축의 상위 결과 + 없음)를 만든다. 자동 처리나 플레이어가 하나를 고르면 그 축이 채워지고 남은 미확정 축은 none이 된다.
효과는 축마다 적용한다: 수치 변화(`EVENT_DELTAS`)·의지 사건은 사건별로, 강도는 공통, 굴욕 포화·평온 초기화·위협 전환은 어느 하나라도 해당하면 적용한다.
확정 한 줄과 연기 지침에는 "배려·양보 + 식사"처럼 모두 표시한다.

이벤트 효과는 `EVENT_DELTAS`의 게임 상수와 기존 의지 사건 표를 사용한다. 식사 -35, 간식 -15의
허기는 섭취 사건에만 적용되며 음식 도착은 섭취가 아니다. 부상/처치·강도도 고정 선택지와 표를 사용한다.
의학적 예측값이 아니다. 수치 정정은 원문 정규식으로 추출하지 않는다.
규칙 버전 8부터 초안 전 LLM이 `corrections`에 정정할 지표와 최종 절대값을 반환한다.
`허기 60이 아니라 20으로 정정해`는 `{hunger: 20}`이며, 상대적 정정에는 현재 저장 수치를 문맥으로 제공한다.
Python은 허용된 7개 지표, 유한한 숫자 0~100, correction 모드와 현재 사용자 입력의 일치만 검증한다.
boolean·문자열 숫자·범위 밖 값·다른 모드의 정정값은 거절하며, 누락·오류 시 원문 숫자 파싱으로 대체하지 않는다.
정정값과 근거는 `verdict.authorization`에 보존하고, 해당 지표의 적분 잔여값만 초기화한다.
내부 직접 `project` 호출도 해당 입력의 LLM authorization이 있어야 정정을 적용할 수 있다.

예정 사건은 명시적인 상대 기간과 등록 의도가 모두 확인될 때 등록한다. 코드가 도래 시점에서 경과를
중단하고 끝 시점의 사건 효과를 보류한다. 완료/취소는 사용자 사건에 대한 Jev 판정으로 처리한다.
Telegram의 별도 plan 입력은 입력 LLM이 반환한 appointment.title과 duration_minutes를 사용해 등록한다.
기존 내부/legacy `project` 호출의 plan_action 경로는 남아 있으나 Telegram의 plan은 Jev를 호출하지 않는다.

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

mode/사건 축/intensity가 불확실하면 사용자 원문과 직전 장면, 이미 수락한 라벨로 Jev에 한 번 집중 재판정을 요청한다.
초안 경로의 미확정 location/activity도 이 재판정에 포함하며, 기준 상태인 current의 location/activity/threat/participants/last_event를 전달한다.
같은 수락 기준을 유지하며 재판정 답변도 감사 기록에 보존한다.

## 지키는 것·미뤄 둔 반응·플레이어 선택

held 상태의 holdout마다 `holdout_<i>`(keep/lost), ready 상태의 `when_alone` 신호에는 완료 기준이 다른 `story_<i>` 질문이 붙는다.
효과와 근거는 [roleplay_game_balance.md](roleplay_game_balance.md). event 선택지에서 `holdout_lost`·`sexual_coercion`은 제외한다.

열린 거래마다 `bargain_<i>`(keep/paid/kept/broken), `kind=routine`·`track` 예정 사건에는 각각의 완료 기준을 가진 `story_<i>` 질문이 붙는다.

사건 축·강도·활동이 재판정 뒤에도 불확실하면 `project`는 `PendingChoice(key, candidates)`를 던진다. candidates는 미확정 축의 확률 상위 3개(+none),
강도 3단계, 활동 8종이다. 처리(자동 또는 버튼)는 [roleplay_postdraft.md](roleplay_postdraft.md). 같은 verdict로 다시 정산할 때
`roleplay_turn.prepare(verdict=…)`는 기준 상태·인물·입력·초안·허가가 같으면 분류·시간 추정을 반복하지 않는다.
Telegram은 prepare_result의 pending 데이터를 처리하며 중요한 결과에는 0.9 이상 신뢰도와 대상에 맞는 초안 원문 근거를 추가로 요구한다. 선택은 감사 기록의 verdict.labels에 남는다.

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

## 검토 비용 선별

현재 서술 검토와 Jev 검토 선별은 꺼져 있으며 아래는 선택적으로 다시 활성화했을 때의 동작이다.
매 턴 생성 모델로 서술 검토를 하기 전에 Jev 고정 선택지로 무모순 여부를 선별한다.
고신뢰 무모순만 생성 호출을 생략하며, 자세한 장애·설정 계약은 [초안 사후 정산](roleplay_postdraft.md)을 따른다.
장면 판정의 `cost_usd`에는 scene/people/records뿐 아니라 집중 재판정 `event_review`의 비용도 포함한다.
이는 전체 턴 비용이 아니며 시간 추정·초안 생성·서술 검토 비용과 구분한다.
`latency_ms`는 classify 시작부터 끝까지의 실제 경과 시간으로, 순차 그룹 호출·집중 재판정·내부 재시도 대기를 포함한다.
scene 호출 실패에도 경과 시간을 반환한다. 개별 HTTP 응답 지연은 `calls.<group>.latency_ms`와 `event_review.latency_ms`로 유지한다.
이 값은 시간 허가·초안 작성·시간 추정·최종 서술 검토를 포함한 전체 턴 지연은 아니다.
그룹 호출은 현재 순차 실행한다. 운영의 전체 분류 시간과 그룹별 시간을 비교한 뒤 병렬화 효과를 평가한다.

규칙 버전 7은 미확정 강도·활동 후보를 최신 유효 확률순으로 정렬한다. NaN/무한대/문자열/범위 밖 확률은 무시한다.
확률이 없거나 동률이면 명시적 정책 기본값을 우선한다: 강도 moderate, 활동 질문 light, 이미 선택을 마친 긴 구간의 남은 활동 rest.
10분 이하의 미확정 활동을 light로 처리하는 기존 정책은 유지한다. 확정 라벨이나 플레이어의 직접 선택을 확률로 덮어쓰지 않는다.
진행 중 신체적 성적 가해와 rest/sleep의 충돌은 기본값 적용 뒤, 시간 적분 전에 검사한다. 충돌 시 기존 상태를 보존하고 턴을 보류한다.
회귀 검사는 `tests/test_roleplay_jev_review_fixes.py`에 포함한다.

Jev는 자유 텍스트 대신 typed decision을 반환한다. 공식 공개 요율은 입력 $0.042/백만 토큰, 출력 무료다
([TypeSafe 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev), 2026-09-20 확인).
따라서 서술 작성은 기존 생성 모델이 맡고, 선택지로 표현 가능한 검토 선별을 Jev가 맡는다.

장소 선택지는 원문의 정확한 문자열 포함 여부로 제한하지 않는다. 등록된 장소 전체와 keep/unknown을
Jev에 보내 동의어·문맥을 판정하게 한다. 미확정 사건 축의 none 확정은 유효한 none 확률 0.5 이상일 때만
허용하며, 낮은 신뢰도의 최상위 라벨이 none이라는 이유만으로 사건을 지우지 않는다.
성적 행위와 사건의 일치는 sexual_act 라벨로 검증한다. 원문의 특정 부정 어구만으로 현재 사건 판정을 뒤집지 않는다.
현재 행위의 부정·과거 언급 구분은 Jev 질문의 책임이며 실제 모델 오분류까지 코드 검증이 보장하지는 않는다.
고정 회복/심문 효과의 강도는 해당 사건에만 적용한다. 같은 장면의 다른 축에 구타·부상이 있어도 그 강도를 moderate로 덮어쓰지 않는다.

Telegram은 `roleplay_time_authorization`의 mode/시간 결정을 초안 판정에 전달한다. 이 경로에서는 Jev의 mode/elapsed 질문을 생략한다.
초안 전 LLM은 지시 기간과 예정 지연을 문맥에서 골라 분으로 반환하며, duration_minutes=0도 명시적으로 전달한다.
따라서 과거 기간·부정·선택지·미래 언급이 원문에 함께 있어도 숫자 파서를 다시 돌려 합산하지 않는다.
초안 후 명시 기간이 없는 행동의 실제 소요 추정은 기존 `roleplay_duration_estimate`가 맡는다.

폐렴은 부상과 별도의 `illnesses` 상태로 기록한다. 현재 장면 판정·시간 효과·회복과 재발·표시 계약은
[질병 상태](roleplay_game_balance.md#질병-상태)를 따른다. 과거 사료로 현재 발병을 소급 적용하지 않는다.
