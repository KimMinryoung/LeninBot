# Jev 판정 연동

Jev는 고정 선택지·확률·점수를 반환하는 System One 판정 모델이다.
이 프로젝트에서는 단순 분류와 인용 지지 판정에 사용한다. 서술 생성·자료 조사·독립 사실 검토는 에이전트가 맡고
schema·길이·날짜 형식·원문 위치 검증은 코드가 맡는다. Jev 결과는 도구 권한이나 공개 승인 자체가 아니다.

## 실행 구조

설정 원본은 `config/llm_call_sites.json`, 실행기는 `llm/call_registry.py`다.
현재 Jev 항목은 TypeSafe 경로를 사용한다. 모델·문턱값·enabled/enforce는 레지스트리에서 확인한다.
`decide_detailed`는 판정과 오류 정보를, `decide` 계열은 Decision 또는 None을 반환한다.
choice/noul/score, confidence, 실제 모델·usage·비용을 공통 객체로 읽는다.
연결·credential·재시도·감사는 [호출 레지스트리](llm_call_registry.md#system-one-판정-호출-decide)와
[LLM gateway](llm_gateway.md)를 따른다. 호출부가 API 키를 직접 관리하지 않는다.

| 적용 지점 | 역할 | 판정 불가 시 처리 |
|---|---|---|
| `commulingo_person_classification` | 인물 그룹·관직 | editor 초안 보존 후 재시도, LLM 분류로 대체하지 않음 |
| `commulingo_person_codes` | 국가·출신·사망 유형 코드 | 누락 코드를 가진 수정안은 공개하지 않고 재시도 |
| `commulingo_term_classification` | 신규 용어 카테고리 | editor 초안 보존 후 재시도 |
| `commulingo_citation_support` | 주장과 원문 인용의 지지 관계 | unavailable 집계, 독립 검토 유지 |
| `commulingo_review_citation_support` | 검토 finding과 직접 조회한 인용의 지지 관계 | 독립 검토 유지 |
| `task_routing_decision` | 태스크 담당 에이전트 후보 | 기존 LLM 라우팅 advisor |
| `scout_kg_fact_filter` | KG에 넣을 문장의 사실성 | 기존 문장 유지 |

CommuLingo의 필요한 설명·라벨·원문 발췌 전송과 자동 실행은 지속 승인되어 있다.
정상 호출마다 재승인을 요청하지 않는다. 상세 범위는 [파이프라인 승인](commulingo_pipeline.md#운영-경로와-승인-범위)을 따른다.

## 분류 기준과 감사

`runtime_tools/commulingo_classify.py`가 criteria와 허용 선택지를 관리한다.
`commulingo_pipeline/decisions.py`는 작성 schema에서 분류 필드를 제거하고 판정 결과를 채운다.
기존 인물의 그룹·관직은 보존하며 신규 또는 누락 항목만 채운다.
수정되는 citizenship/nationalOrigin/fate의 코드는 라벨과 해당 근거를 사용한다.
분류 입력과 결과는 체크포인트에 캐시하여 동일 초안의 재시도 비용을 줄인다.

- 인물 그룹·관직은 편집 규칙인 GROUP_RULES·OFFICE_RULES를 따른다. 소련·후계국 국적 여부에 따라 관직 선택지를 제한한다.
- nationalOrigin은 민족·국가 배경이며 출생지·활동지·시민권으로 대신하지 않는다. 혼합 배경의 설명은 서술 라벨에 남긴다.
- 생존 인물의 fate는 코드로 빈 kind를 처리한다. 불확실한 사망 유형을 확정하지 않는다.
- 용어 카테고리는 정의와 본문의 주제를 우선한다. 기존 저장 분류를 정답으로 간주하지 않는다.

낮은 신뢰도의 판정도 검토 위험으로 전달하며 독립 검토자가 확인한다.
판정 장애와 낮은 신뢰도는 다르다. editor는 장애로 초안을 버리거나 작성 LLM에 분류를 떠넘기지 않는다.
직접 등록 도구도 공통 분류 함수를 사용하지만 editor의 체크포인트 복구는 editor 전용이다.

`scripts/commulingo_classification_audit.py`는 저장값과 고신뢰 판정이 다른 인물을 보고한다.
감사 자체는 쓰지 않는다. 정정은 근거를 갖춘 공통 편집 서비스의 제출·승인을 거친다.

## 인용 지지 게이트

`commulingo_pipeline/citation_gate.py`는 주장·필드·출처 URL·실제 발췌를 판정한다.
검토 게이트는 검토자가 직접 가져온 발췌와 finding을 사용한다.
두 게이트는 enforce 설정을 사용하며 임계값 이상의 unrelated/contradicts 또는 boilerplate를
해당 항목의 수정 요청으로 반환한다. 반박 근거를 나타내는 stance=disputes에서는 contradicts를 허용한다.
판정 불가 시 독립 검토를 유지하며 분류 장애와 동일하게 취급하지 않는다.

요청은 최대 12항목·state 24k자 단위로 묶고 같은 입력은 캐시한다.
캐시 키는 실제 발췌·질문·provider/model·임계값·stance를 포함한다. 같은 source ID와 offset이라도
다시 가져온 본문이 달라지면 재판정하며 모델/임계값 변경 후 이전 거절 결과를 재사용하지 않는다.
Usage에는 `citation_requests`(배치 요청 수), `citation_cache_hits`(캐시로 처리한 항목 수),
`citation_unique_items`(이번 호출에서 판정을 요청한 고유 항목 수)를 누적한다.
검토 게이트의 지표에는 `review_` 접두사가 붙는다. requests는 내부 HTTP 재시도 횟수를 포함하지 않는다.
판정 수치는 해당 claim에 붙여 근거가 재배치되어도 대응이 어긋나지 않게 한다.
P 문단 라벨·원문 offset·인용문 포함 여부는 결정적 코드로 검증한다.
Jev가 라벨이나 인용 위치를 추측하지 않는다. editor는 legacy 검색 shadow triage를 호출하지 않는다.
실제 호출 비용은 단계 비용과 `jev_calls`·`jev_cost_usd`에 기록한다.

## 태스크 라우팅

`task_routing_decision`이 담당 후보를 먼저 분류하고 accept 미만이거나 판정 불가이면
`task_routing_advisor`의 기존 LLM 경로를 사용한다. 최종 delegate 선택은 orchestrator가 한다.
이 fallback은 CommuLingo의 분류 장애 정책과 별개다.

## Scout 사실 필터

`kg_runtime/scout_ingest.py`의 `_filter_fact_lines`는 추출 문장이 외부 사건·인물에 관한 구체적 사실인지 판정한다.
절차 메모·주제명만 있는 줄을 제외하고 keep 이상인 문장만 제한된 수로 저장한다.
남는 사실이 없으면 episode를 만들지 않는다. 판정 불가이면 기존 문장을 유지한다.
Scout의 KG 그룹 분류를 Jev로 대체하는 안은 채택하지 않았으며 기존 분류 경로를 유지한다.

## 평가 근거와 한계

아래는 2026-09-19 도입 당시의 소규모 표본 평가이며 현재 전체 운영의 정확도나 비용 절감률이 아니다.

| 평가 | 관찰과 적용 근거 |
|---|---|
| 인용 지지 30쌍 | 무관한 5쌍을 모두 unrelated로 판정했으나 고신뢰 차단 대상은 3쌍이었다. 해당 표본에서 고신뢰 오차단은 없었다. |
| 라우팅 30건 | Jev 단독 28/30, 기존 advisor 26/30. 저신뢰 fallback은 유지한다. |
| Scout KG 그룹 30건 | 두 모델의 일치는 23/30이며 정답 자체가 모호한 사례가 있어 교체 근거가 부족했다. |
| Scout 사실 필터 | 평가한 사실 8건 중 7건을 유지하고 절차 줄의 오저장은 없었다. 제목만 있는 줄과 문맥 의존 문장은 한계로 남았다. |

저장 라벨과의 일치는 정확도가 아니다. 분류 기준 변경·자료 누락·기존 오분류를 따로 검토해야 한다.
외부 원문은 판정 대상 데이터이며 실행 지시나 권한 근거로 취급하지 않는다.
지속적인 사후 품질 감사 없이 테스트 통과를 역사적 사실의 정확성 보증으로 해석하지 않는다.

## 공식 문서·외부 사례 검토와 반복 평가

2026-09-21 공식 문서와 GitHub 사례를 대조했다. 현재의 고정 선택지 분류, 독립적인 질문의
일괄 호출, 코드로 판정 결합, 낮은 신뢰도 fallback은 권장 패턴과 부합한다.

- [공식 Confidence](https://docs.typesafe.ai/confidence): confidence는 선택지 분포에서 계산한 통계다.
  `confidence=.9`를 실제 정답률 90%로 해석하지 않는다. 임계값은 적용 업무의 검증 데이터로 평가한다.
- [공식 Primitives](https://docs.typesafe.ai/primitives): 질문은 독립적으로 평가된다. 질문 ID는 모델에
  전달되지 않으므로 대상 항목은 instructions에도 명시해야 한다. 기존 `fan_out`은 이를 수행한다.
- [공식 Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out): 같은 state에 필요한 질문들을
  모으면 왕복 지연을 줄일 수 있다. 현재 인용·역할극의 다중 질문 방식에 이미 적용되어 있다.
- [문서 분류 사례](https://github.com/Charlyhno-eng/jev-document-classification): 입력 크기 제한,
  짧은 문서 묶음 처리와 비용·실행시간 기록을 참고했다. 현재 인용 배치·캐시를 유지하고 사용량 지표를 보강했다.
- [jevcal](https://github.com/abhixhek/jevcal), [jev-eval-agent](https://github.com/vinilana/jev-eval-agent),
  [jev-harness](https://github.com/AntonioCoppe/jev-harness): 임계값별 처리율·정확도, 실제 행동의 오판,
  저장 응답 재평가를 참고했다. 외부 코드나 의존성을 설치하지 않았다. 해당 프로젝트의 수치를
  Leninbot의 성능으로 인용하지 않는다.

`scripts/eval_jev_citations.py`는 운영 citation 질문과 `verdict()`를 그대로 사용한다.
기본 fixture는 지지/모순/무관/부분 지지/반박 stance/접근 차단/러시아어-한국어/부정/원문 속 지시문을
담은 합성 10건이다. 품질 회귀의 출발점이며 운영 정답률이나 임계값 최적화의 충분한 근거가 아니다.

```bash
# 실제 외부 호출·과금·감사 기록 발생. 접근 가능한 기존 gateway/credential 환경에서 실행.
venv/bin/python scripts/eval_jev_citations.py --live > /tmp/jev-citations.json
# 저장한 응답으로 현재 정책과 임계값별 결과 비교: 외부 호출 없음.
venv/bin/python scripts/eval_jev_citations.py --replay /tmp/jev-citations.json
```

출력은 실제 모델, 응답·오류, 알려진 호출 비용, p50/p95 지연, 고신뢰 처리율·분류 정확도,
오차단·차단 누락, 0.70~0.95의 reject 임계값 비교를 포함한다. unavailable은 별도로 세며
성공으로 간주하지 않는다. 고신뢰 처리율은 전체 표본 대비 support confidence 문턱을 넘은 비율로,
독립 검토를 생략해도 된다는 뜻은 아니다. 지연은 성공 요청의 HTTP 지연이고 재시도 대기를 포함하지 않는다.
저장된 평가 파일은 state를 포함하므로 실제 운영 자료로 확장할 때 접근 범위를 관리한다.
재평가는 질문·모델을 바꾸지 않고 거절 임계값만 비교한다. 질문 변경은 새로운 호출 결과가 필요하다.

추가 적용 후보인 검색 재정렬과 LLM 모델 티어 선택은 자체 정답 데이터와 기존 경로 대비 평가가
마련된 뒤 검토한다. 기존 30건 라우팅 표본만으로 다른 도메인의 임계값을 정하거나 자동 대체하지 않는다.

2026-09-21 합성 예제 3건의 실제 연결 스모크는 모두 성공했다(`jev-1.13.0`, 요청별
338/346/745ms, 입력 합계 1,291토큰, 총 추정 $0.000054222). 이는 연결·형식 확인이며
평가 fixture 10건의 실측이나 운영 정확도·절감률 검증은 아니다.

검증은 `tests/test_commulingo_classify.py`, `test_commulingo_editor_decisions.py`,
`test_commulingo_citation_gate.py`, `test_commulingo_classification_audit.py`,
`test_scout_kg_fact_filter.py`를 사용한다. 모의 판정 기반 회귀 검사와 실제 모델 표본 평가는 구분한다.

## 역할극 자동 판정

`roleplay_scene_adjudication`은 Jev 고정 선택지 분류이며, `roleplay_duration_estimate`는 단일 사건 소요 분만 생성하는 별도 호출이다. 수치 계산·저장은 코드가 수행한다. 계약과 실패 정책은 [roleplay_jev.md](roleplay_jev.md)를 따른다.
