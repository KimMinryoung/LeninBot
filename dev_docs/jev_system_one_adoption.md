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

검증은 `tests/test_commulingo_classify.py`, `test_commulingo_editor_decisions.py`,
`test_commulingo_citation_gate.py`, `test_commulingo_classification_audit.py`,
`test_scout_kg_fact_filter.py`를 사용한다. 모의 판정 기반 회귀 검사와 실제 모델 표본 평가는 구분한다.
