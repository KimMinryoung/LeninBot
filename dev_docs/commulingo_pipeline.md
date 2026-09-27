# CommuLingo 영속 편집 파이프라인

인물·용어의 등록과 보강은 `commulingo/pipeline/`이 맡는다. 진입점은
`scripts/commulingo_pipeline.py`, 운영 설정은 `config/commulingo_pipeline.json`이다.
저장 도구의 계약은 [편집 서비스](commulingo_editorial.md), 판정 정책은 [Jev 연동](jev_system_one_adoption.md)을 따른다.

## 운영 경로와 승인 범위

운영 설정은 `workflow=editor`, `phase=live`다. 작성 모델은 GPT-6 Luna (`provider=openai`,
`model=gpt6luna`, `thinking_policy=disabled` → Responses `reasoning.effort=none`)이며 제출·무편집
도구는 strict로 전달한다. 기본 AgentSpec·설정 예제·운영 agent_runtime 설정을 같은 값으로 유지한다.
독립 검토 모델은 기존 DeepSeek을 유지한다. timer의 다음 새 프로세스부터 설정이 적용된다. workflow는 `editor`만 허용하며 미지정 기본값도 `editor`다.
옛 두 RPC 방식의 legacy 단계(`Research`·`Draft`·`Review`·`validate`·`submit`)는 2026-09-24에 제거했다.
`leninbot-commulingo-pipeline.timer`가 배치를 실행하고, 별도 `leninbot-commulingo-review.timer`는 비활성화되어 있다.
독립 검토는 editor 안에서 수행한다. 사건 작성·인물-사건 연결의 기존 batch는
2026-09-20 운영자 결정으로 폐기했으며 pipeline으로 이관하지 않았다.
사건 작성 전용 runner·agent는 제거했다. `leninbot-commulingo-batch.timer/service`와
events·links service는 빈 unit 파일로 남겨 systemd에서 masked로 처리한다.
이는 기존 설치나 전체 unit 복사 시 폐기한 batch가 다시 실행되는 것을 막는 배포 표식이며 실행 코드는 없다.
기존 사건과 연결 데이터는 보존한다. 인물·용어 pipeline의 gap 완료 처리는 기존처럼
`resolved_id`를 기록하며 사건 연결을 자동 생성하지 않는다.

운영자는 독립 검토를 통과한 결과의 공개 반영과 향후 자동 실행을 지속 승인했다.
분류용 설명·라벨과 판정에 필요한 원문 발췌를 기존 TypeSafe/Jev로 전송하는 것도 지속 승인했다.
정상 작업마다 사용자 재승인을 요구하지 않는다. 이 승인은 검토·revision·승인 해시·예산 검사를 우회하지 않는다.
판단 불가와 반복 실패는 근거를 보존해 내부 보류하며 사용자 승인 요청으로 전환하지 않는다.

## 과제 선정

`issues.py`와 `planner.py`는 비어 있는 값·언어와 명시적 요청만 과제로 선정한다.
editor는 상세 절이 하나도 없는 인물에 한해 첫 절의 작성 가능성을 자동 과제로 선정한다.
작성자는 기존 카드와 근거를 살펴 독립적인 사건·시기·주제가 충분히 뒷받침될 때만
한영 상세 절 하나를 제안하며, 근거가 부족하면 사유를 남기고 무편집으로 종료한다.
이미 상세 절이 있는 인물의 추가 절은 명시적 요청으로만 선정한다.
첫 절이 없는 인물은 카드 수정 후 유예 기간과 무관하게 선정하되, 섹션 주제의 완료·보류 판정은 유지한다.
분량과 절 수는 상한이며 채워야 할 할당량이 아니다. 본문이 있으나 근거 행이 없는 항목은
과제가 아니다. 근거 백필은 기존 서술의 재작성으로 이어져 2026-09-20 운영자 결정으로 중단했다.
근거는 실제로 일어나는 수정에만 요구한다.
빈 관계나 예시·구별 항목을 일률적으로 생성하지 않는다. 명시적으로 요청된 과제는 보존한다.
기존 우선순위, 주제별 승인 후 유예 기간, pending 제안과 진행 중 대상의 중복 방지를 적용한다.
자동 discovery가 꺼져 있어도 명시적인 gap 요청의 발견·등록 경로는 남는다.

작업은 kind/action/target/topic/baseline과 payload로 식별한다. DB의 활성 작업 유일성 제약과
planner의 대상 제외를 함께 사용한다. 카드와 명시적으로 요청한 상세 절은 묶음으로 처리할 수 있으나
각각의 baseline·수정안·승인은 분리한다. `bundles.py`가 남은 과제를 영속 저장한다.
`consolidate`는 시도·비용·자료가 없는 미착수 작업만 묶으며 원문을 합치는 기능이 아니다.

## 조사·작성·수정

`editor.py`는 조사와 작성, 저장 검증 오류 수정을 한 세션에서 처리한다.
DB의 research/draft 단계명은 호환용이며 둘 다 Editor를 실행한다.
검증된 수정안은 독립 검토로 넘어간다. 분류는 `decisions.py`가 Jev에 맡기며 작성 schema에서
분류 코드를 제거한다. 기존 인물 분류는 보존하고 신규·누락 분류와 변경된 라벨의 코드를 채운다.
분류 장애 때는 초안을 저장해 재시도하며 LLM 분류로 대체하지 않는다.

출처가 있는 인물 활동은 기능 → 그 기능을 대표 경력으로 기록한 근거 발췌 → 소속 순서로
Jev에 질의한다. 질문별 판정은 독립적이므로 후속 질의 state에 앞선 선택을 명시한다.
근거 발췌 질의에도 선택된 기능의 카탈로그 기준을 전달한다. `government`는 여러 정책 분야의 정부 운영·행정 조정이 대표 경력일 때만 선택한다. 고위 직함·의원·공직자 신분만으로는 부족하고, 의례적 국가원수와 의회 의장은 `legislature`(입법·국가 대표)로 간다. 교육·종교가 대표 경력이면 `education`·`religion`을 쓴다. 대표 경력은 출처·카드가 그 인물을 규정하는 경력(보통 첫 서술)이며, 알려진 명목직·입법직을 더 짧거나 덜 알려진 이전 경력으로 바꾸지 않는다. `legislature` 근거 질의는 의장직·의례적 국가원수직 발췌가 없으면 `unsupported`를 요구한다. 집권 체제의 관변 대중조직(콤소몰·관변 노조) 운영은 `organizing`이 아니다. 그 밖에는 농업·경제·산업·외교·법률·보안·군사 등 대표 전문 분야와 당·운동 지도를 우선한다. 정부·행정 근거 질의는 직함이나 전문 부처 경력만 있으면 `unsupported`를 선택하도록 요구하며, 이 경우 소속 판정과 분류 확정을 진행하지 않는다. 기존 저장 활동과 옛 역할 매핑은 자동 재분류되지 않으므로 출처에 따른 별도 검토가 필요하다.
소속 후보는 코드가 먼저 거른다: 선택된 발췌가 적은 연도(없으면 생몰년의 성년 구간)와
카탈로그 `periods`(존속 기간, 앞뒤 1년 여유)가 겹치는 항목만 제시하고, 기간이 없는 항목은 항상 제시한다.
시대 그룹도 같은 방식으로 생몰년이 겹치는 그룹만 제시한다(`classify.GROUP_ERAS`).
그룹이 `france-revolution`이면 근거 발췌 질의에 혁명과 그 전쟁(1789–1815)에서의 역할·진영을 다른 시기 경력보다 우선하라는 기준을 덧붙인다(`classify.FRENCH_REVOLUTION_BASIS`). 다른 선반은 대표 경력 기준만 쓴다.
프런트엔드 쓰기 검증은 활동 연도가 소속의 존속 기간 밖이면 거부한다.
발췌 원문도 state에 포함하며, 허용 목록 밖 응답이나 근거 없는 선택은 보류한다.
국적·거주·학술 연구 대상을 국가기관 복무로 추정하지 않는다.
일당제 현실 사회주의 체제 내의 집권당·국가기관 활동은 국가 이름과 `kind=state`인 소속 ID로 판정하며, 국가 관계는 `service`로 저장한다. 건국 전 정당 활동과 체제 반대파는 해당 정당·세력 또는 미확정 소속으로 구분한다. 정당 카탈로그의 `governingState`와 criteria가 적용 기간을 명시하며, 국가에 `membership`을 쓰는 요청은 frontend가 거부한다. 옛 역할 매핑도 국가 ID를 사용하지만 건국 전에 사망한 인물은 `legacyBeforeState`에 따라 정당에 남긴다.


2026-09-22 혼합 분류 이관의 출처·검토 방식·운영 반영 기록은 별도 프런트엔드 저장소
`/home/grass/frontend/dev_docs/commulingo-role-model-plan.md`와
`/home/grass/frontend/scripts/content/person-activities-mixed-reviewed-20260922.json`에 있다.

### 원문 캐시와 문단 라벨

`source_session.py`는 기존 PostgreSQL 원문 캐시와 job_sources를 재사용한다.
페이지를 이어 붙이거나 별도의 합본 원문을 만들지 않는다. `evidence.py`의 `Passages`는
스냅샷과 고정 문단 범위에 불변 `P1` 같은 라벨을 부여한다. 작성기는 라벨을 인용하고
코드가 원문 위치·인용문으로 변환한다. 존재하지 않는 라벨은 오류이며 주장을 몰래 삭제하지 않는다.

캐시 재사용은 최초 조회 시각과 만료를 보존한다. 체크포인트에서 만료된 자료를 제외해도
그 라벨을 새 자료에 재배정하지 않는다. 독립 검토자는 작성기의 원문 캐시 대신 직접 자료를 가져온다.

### 필드 단위 작성 API와 체크포인트

작성자는 `fields`에 실제 필드 값을, `evidence`에 필드별 원문 근거를 보낸다.
등록·편집·수정이 같은 형식이다. `changes.<field>.value` 래퍼와 필드별 `anyOf`는 제거했다.
`author_draft.py`가 이를 영속 저장·검토용 `fields/claims/issue_results`로 변환한다.
필수 값과 타입은 기존 canonical schema가 결정한다. 분류 코드·revision·원문 위치·승인 해시는 서버가 채운다.

내부 책임은 다음과 같이 나눈다.

- `author_draft.py`: 필드 계약과 과제 목록으로 공개 입력·최종 검증 스키마를 함께 생성한다.
  근거·과제 판단·메타데이터 제약은 한 번만 정의하며, 부분 입력의 누적·저장·검증을 소유한다.
  직접 쓰기용 `DraftRepair`/`prepare_write`와 JSON 포인터 수정 경로는 사용하지 않는다.
- `editor_checkpoint.py`: 구버전 문장 배열·섹션 정렬값·중첩 메모를 작업 사본에서 복구한다.
  영속 journal의 원본은 변경하지 않는다.
- `editor.py`: 위 객체를 조립하고 원문·인용·분류·RPC 검증과 단계 전환을 실행한다.

미완성 부분 저장과 최종 스키마 반려는 모두 독립된 초안 사본을 남긴다.
독립 검토·승인 해시·게시 시 revision 검증은 기존 경로를 따른다.

OpenAI 작성 세션은 `strict_input.py`에서 제출·무편집 도구의 wire schema만 strict로 변환한다.
모든 속성을 required로 보내고 선택값의 null은 생략으로 복원한다. 원래 null을 허용하는 선택 필드는
`{value: null}`이 실제 초기화이고, 바깥 null은 변경 없음이다. 필드 철회는 기존 remove_fields를 쓴다.
uniqueItems 등 wire에서 제거한 제약은 원래 입력 스키마로 다시 검증하며, 영속 계약은 바꾸지 않는다.
다른 provider와 조회 도구에는 이 변환을 적용하지 않는다.

독립 검토의 `coverage`는 sufficient와 reason을 필수로 기록한다. 제목·commission에 필요한 핵심
행위·역할·결과가 빠져 오해를 일으키거나 주제에 답하지 못하면, 원문 근거와 기존 필드 경로를 갖춘
required_corrections를 요구한다. coverage.sufficient=false인 결과는 승인할 수 없다.
분량·선택적 배경 정보는 반려 사유가 아니며 자료 목록 설명이나 일반적인 유보문으로 핵심 사실을
대체했는지도 검토한다. 기존 bio와 일치한다는 이유만으로 신규 본문의 중심 주장을 승인하지 않고
원문으로 확인한다. 작성 검증은 공개 필드의 `(P65)`·`[P65]` 같은 실제 표시된 내부 passage 표지를
정확한 필드 경로와 함께 반려하며, 근거 목록의 passage 참조는 유지한다.


| 세션 내부 도구 | 역할 |
|---|---|
| `commulingo_pipeline_submit_draft` | 값·근거·과제 판단 제출과 부분 수정. 완성된 작은 초안은 한 번에 제출한다 |
| `commulingo_pipeline_no_edit` | 사유와 모든 과제의 판단을 남기고 무편집 종료. 저장된 초안은 이력에 보존한다 |
| `commulingo_pipeline_cached_passages` | 캐시 목록과 원문을 네트워크 없이 조회. 한 번에 라벨 8개까지 보여 주고 나머지를 안내한다 |
| `commulingo_pipeline_context` | 필요한 현재 값과 편집 가능한 필드의 schema 조회. 편집 범위를 넓히지는 않는다 |

`missing:body` 과제의 완성된 제출 예시다. P 라벨은 실제 조회한 원문의 라벨이어야 한다.

```json
{
  "fields": {
    "body": {"ko": "원문으로 확인한 설명이다.", "en": "An explanation supported by the original."}
  },
  "evidence": {
    "body": [{"claim": "The original documents this explanation.", "passages": ["P1"]}]
  },
  "issues": {
    "missing:body": {"status": "resolved", "reason": "Added both languages with original evidence."}
  },
  "reason": "The original supports the commissioned explanation."
}
```

수정은 바꿀 부분만 보낸다. 예를 들어 `fields.body.en`만 보내면 한국어와 기존 근거는 유지하며,
`evidence.body`만 보내면 본문을 다시 보내지 않고 근거 목록을 교체한다.
`issues`는 과제 ID별 resolved/deferred와 사유를 명시한다. 값을 채웠다는 이유로 해결을 자동 추정하지 않는다.
`reason`·`notes`·과제 판단은 생략하면 보존한다. notes는 비공개 메모다.
배열은 해당 목록 전체를 교체하며 `evidence.<field>: []`는 그 필드의 근거를 철회한다.
근거만 제출할 때는 같은 호출 또는 기존 초안에 해당 필드 값이 있어야 한다.
`remove_fields`는 선택 필드와 근거를 초안에서 철회한다. 공개 내용은 삭제하지 않으며 필수 필드는 철회할 수 없다.
새 작업은 aliasEdits/careerEdits/sceneEdits와 전체 배열 교체를 동시에 노출하지 않는다.

긴 초안은 여러 호출로 나눌 수 있지만, 제목→한국어→영어→근거의 고정 절차는 없다.
서버가 필수 값·양 언어·사실 근거·모든 과제 판단·사유가 모였는지 판정한다.
미완성 초안은 체크포인트에 저장하고 `StageContinues`/`continued`로 남은 입력을 안내한다.
반려로 세거나 터미널 도구 호출만으로 세션을 종료하지 않는다.
용어의 변경 없는 연도·시기와 신규 null 기본값은 공개 패치 정리 규칙에 따라 근거 대기에서 제외한다.
완성되면 원문 라벨·인용 지지·canonical 제약·저장 검증을 거쳐 독립 검토로 넘어간다.

필드별 위치를 추측해서 옮기는 구조 보정은 제거했다. 제공자의 단독 `arguments` 객체/JSON 문자열
포장만 푼 뒤 같은 schema로 검증한다. 타입·알 수 없는 필드 오류는 기존 초안을 보존하며 거절한다.
길이 초과 텍스트는 초안에 저장한 뒤 정확한 초과량을 돌려줘 해당 값만 고칠 수 있다.
진단 위치는 `fields.<field>`와 `/evidence/<field>/<index>`를 사용한다.
여러 문단으로 펼쳐진 인용도 원래 근거 인덱스를 가리킨다. JSON Pointer 수정 API는 노출하지 않는다.
모든 세션 도구는 소유자·commulingo_curator 호출자 권한을 검사한다.

상세 절은 `fields.heading`, `fields.body`, `fields.startYear`와 근거를 제출한다.
`startYear`는 필수 정수이며 주제형 절은 주제의 시작 연도를 쓴다. 선택적 `startMonth`와 함께
서버가 YYYYMM `sortOrder`로 바꾸고 날짜 근거도 그 저장 필드에 연결한다. 제목 속 연도를 추론하지 않는다.
slug는 `commulingo_section_slug` low 티어 원샷 호출로 생성하고, 실패하면 영문 제목의 결정적 slug로 대체한다.
인물 ID·기존 절과의 충돌·형식을 검사하며 기존 절과 같은 제목은 거절한다.
생성값은 체크포인트와 검토·승인 해시에 포함한다. 기존 제안 수정은 원래 slug를 보존한다.

초기 문맥은 과제·현재 관련 값·주변 배경·원문 캐시·저장 초안으로 구성한다.
`work_status`는 저장 여부, 현재 오류와 다음 행동만 안내한다. 별도의 조사 허가 상태는 없다.
2026-09-27에 `commulingo_pipeline_research`와 형식 수정 중 검색 차단·사전 조회 횟수 제한을 제거했다.
작성자는 형식 오류에는 저장된 원문을 재사용하고, 사실 충돌에는 바로 필요한 자료를 조회한다.
공통 도구 권한, 실패 URL backoff, 라운드·단계 비용 상한은 계속 적용한다.
공통 `WRITING_RULES`와 실제 필드 제한을 제공하며 분량은 채울 목표가 아니다.

`editor_checkpoint`는 초안·불변 P 라벨·조회 인자·오류·분류·slug 캐시를 보존한다.
기존 `fields/claims` 형식은 그대로 읽는다. 오래된 `repair_only` 등 조사 제한 상태는 무시한다.
구형 문장 배열·필드 안 notes·sortOrder는 기존 복구 규칙으로 읽고, 구조가 잘못된 초안은 원본을
보존한 채 재제출이나 무편집 판단을 요구한다. 같은 baseline에서 복원하며 checkpoint 쓰기도 lease로 보호한다.
같은 완성 수정안과 검증 오류의 반복은 내부 보류한다. revision 충돌은 최신 상태의 조사로 돌려보낸다.

Editor는 `EDITOR_MAX_ROUNDS`=24와 단계 비용 상한을 사용한다.
출력 길이 제한에 걸리면 설정된 추가 응답 횟수 안에서 같은 도구 세션을 이어간다.
결과 없이 종료되면 최종 제출 기회를 주고, 그래도 실패하면 영속 자료를 보존해 다음 배치에서 재시도한다.

## 독립 검토와 공개 반영

`workflow.py`는 현재 문서와의 차이, 이전 수정안과의 차이 및 이전 검토를 제공한다.
검토자는 원문을 직접 가져와 핵심 변경 사실과 위험 항목을 확인한다.
검토 문맥 조회에서 현재 인물 상세 절의 `body`·`heading`·`slug`·`sortOrder` 요청은
실제 `sections` 필드 조회로 합쳐 중복 없이 반환한다. 조회 수 상한은 실제 조회 가능한 필드 수이며,
중복 필드와 없는 필드는 거절한다. 원문 근거 조회를 대신하지 않는다.
검토 입력의 변경 위치와 이전 수정안 비교 위치는 제출 규칙과 같은 `/fields/...` 경로로 제공한다.
`required_corrections`는 사실 오류의 필드 위치와 이유를 담으며 `optional_suggestions`와 구분한다.
선택 제안만으로 revise할 수 없고 내용·근거가 그대로인 거절안은 유료 재검토 전에 보류한다.
reject는 complete, escalate는 escalated로 끝난다. 두 경우와, 수정 요청이 반영되지 않은 같은 patch가
다시 와서 보류(`revise (held)`)되는 경우 모두 `note` RPC로 판정과 사유를 사전 항목에 남겨 다음 작성자가
같은 충돌을 다시 겪지 않게 한다. 노트 저장 실패는 경고만 남긴다(판정은 review artifact에 이미 있다).
Jev는 주장과 인용, 검토 finding과 인용의 지지 관계를 판정한다. 최종 서술과 사실 검토는
작성자·독립 검토자가 담당한다. 인용 게이트 장애 정책은 분류 장애 정책과 다르다.

승인은 target/action/id/fields/sources의 정규화 SHA-256에 묶이며 baseline revision도 포함한다.
승인 후 수정안이 달라지면 다시 검토한다. frontend private RPC의 `publish`는 해시와 revision을
검증하고 submit·review·원 제안 대체·메모·영수증을 한 트랜잭션으로 반영한다.
실패하면 모두 롤백하며 동일 요청 재실행에는 기존 영수증을 반환한다. 옛 두 RPC 방식으로 우회하지 않는다.
유료 작업 전 `capabilities.atomicPublish`를 확인한다.

기존 pending 제안의 revise는 원 제안과 근거를 보존하는 수정 작업으로 이어진다.
수정안이 승인되기 전에는 원 제안을 대체하지 않으며 수동 처리된 원 제안을 덮어쓰지 않는다.
발행 전에 원 제안 상태를 확인한다. 없거나 사람이 이미 승인·반려했으면 발행하지 않고 조용히 complete로 끝낸다.
이 작업의 이전 발행이 이미 원 제안을 대체했으면(반려 노트가 `Replaced by independently approved patch <hash>`)
같은 요청을 다시 보내 저장된 receipt를 받는다. 확인과 발행 사이에 상태가 바뀌면 frontend `publish`가
트랜잭션 안에서 거부한다.

## 대기열·예산·복구

`Store`와 engine은 PostgreSQL 대기열, SKIP LOCKED lease, heartbeat, fencing을 공유한다.
lease를 잃은 작업자는 결과를 저장하지 못한다. 실패는 제한된 횟수만 재시도하고 초과하면 escalated로 남긴다.
`retry`는 deferred/escalated 작업을 재개한다. editor의 완료 단계 보류도 조사 단계로 복구한다.

예산은 예약과 실제 사용 원장을 기준으로 한다. 비용 미확정 요청의 예약은 임의로 해제하지 않는다.
일일 한도·검토 몫·단계 예약액은 운영 설정을 따르며 `legacy_shared_budget=true`로 기존 lane과 공유한다.
운영 일일 한도는 UTC 날짜 기준 $4이며(2026-09-25 $2에서 상향: job당 약 $0.023, 대기 1,039건), 설정 변경은 다음 배치부터 적용한다.
실제 응답 비용이 예약액을 넘을 수 있으므로 단일 요청의 선결제 상한은 아니다.
Jev 실제 비용도 단계 비용에 합산한다. 검색·Extract 비용은 별도 [web 사용량 장부](web_research.md)에 기록한다.

`run`은 기본적으로 초안까지만, `--review`는 검토까지, `--publish`는 공개까지 실행한다.
`tick`은 phase 설정을 따라 선정·실행·복구한다. 배치는 단계 수와 시간으로 제한하며 마지막에
승인된 작업의 공개 저장을 한 단계 추가로 끝낼 수 있다. 별도 일일 반영 건수 할당량은 없다.

```bash
# 조회와 계획 미리보기
venv/bin/python scripts/commulingo_pipeline.py list
venv/bin/python scripts/commulingo_pipeline.py plan --workflow editor
venv/bin/python scripts/commulingo_pipeline.py efficiency --since=-24h
# 지정 작업을 정상 검증·검토 후 공개
venv/bin/python scripts/commulingo_pipeline.py run --workflow editor --job-id 123 --publish
```

DB 접근·credential·쓰기 가드는 [MCP gateway](mcp_gateway.md)와 [시크릿 관리](secret_management.md)를 따른다.
모듈 import와 조회는 DDL을 실행하지 않는다. 스키마 변경은 명시적 migration으로 적용한다.

## workflow 고정과 배포 경계

`routed_stages`는 editor 단계만 실행한다. `payload.workflow`가 없는 작업은 research/draft/discover 단계에서
처음 실행될 때 `editor`로 고정되며 파생 등록 작업도 이를 계승한다. `payload.workflow`가 없는 작업이
validate/review/submit/judge 단계에 있거나 `editor` 외의 값으로 고정돼 있으면 `ValueError`로 멈춘다.
옛 두 RPC 단계로 되돌아가는 경로는 없다. 제거 시점(2026-09-24) 운영 DB의 미완료 작업은 모두 `editor`로 고정돼 있었다.
`validate` 단계는 editor 흐름에서 거치지 않는다. 그 단계에 남은 초안이 있으면 저장 검증만 다시 하고,
실패는 editor로, revision 충돌은 조사로 돌려보낸다. `--workflow` 선택지는 `editor` 하나다.

frontend 변경 자산은 `deploy/commulingo-editor-frontend.patch`다. 실제 저장은 기존 Admin 함수가 소유한다.
운영 frontend/data는 호스트 마운트이므로 파일 변경 자체가 운영 반영이다.
private RPC는 호출마다 새 Node 프로세스로 모듈을 읽는다. 이 경로의 모듈 반영에는 웹 서버 재시작이 필요 없다.
pipeline이 사용하는 `scripts/commulingo_person_reviewer.py`의 검토 함수와
`commulingo_write_session.py`의 초안 수정 함수는 보존한다.
`commulingo_gap_event_links.py`는 legacy gap worker가 가져다 쓰는 연결 조회·생성·검증·저장
공통 함수만 남겼으며 독립 CLI와 배치 반복 실행은 제거했다.
`commulingo_lane_health.py`는 pipeline과 공통 검사만 감시하며 폐기된 lane의 무실행을 장애로 보고하지 않는다.

## 검증과 효율 지표

빠른 공통 계약 검사와 확장 editor 검사는 같은 unittest 사례를 사용한다.

```bash
venv/bin/python scripts/smoke_commulingo_maintainer.py
venv/bin/python scripts/smoke_commulingo_maintainer.py --extended
# 실제 executor도 확인할 때: cross-thread wakeup이 허용되는 실행 환경에서만
venv/bin/python scripts/smoke_commulingo_maintainer.py --extended --real-threads
```

기본 검사는 작성 schema와 저장 검증의 경계, 원문 캐시, 분류, 초안 수정을 다룬다.
작성자는 분류 라벨을 제공하고 runner가 코드를 채우므로 작성 schema에 category/code를
요구하지 않는다. 저장 경계의 분류 필수 검사는 별도로 유지한다.
`--extended`는 실제 editor·인용 게이트·독립 검토·공개 승인 해시 경로까지 검사한다.
Jev 응답·모델·저장소·검색은 모의 구현하며 HTTP·DB·외부 프로세스의 미처리 호출은
즉시 실패한다. 오류를 잡아서 판정 불가로 처리하더라도 검사 종료 시 실패로 보고한다.
운영 credential 없이 실행할 수 있고, import 시 분류 목록은 내장 fallback을 사용한다.
전체 제한 시간은 기본 60초이며 `--timeout`으로 조정한다. 초과하면 traceback과 실패 종료를 남긴다.

`tests/commulingo_test_support.py`의 unittest fixture는 pytest에서도 동일하게 적용된다.
기본 단위 검사는 mock 동기 의존성을 inline으로 실행한다. 제한된 sandbox에서 단순한
`asyncio.to_thread`도 executor 종료 wakeup을 받지 못하는 현상과 editor 실패를 구분하기 위한 것이다.
실제 thread 실행은 `--real-threads`로 별도 확인하며, 운영 실행 코드는 바꾸지 않는다.

`tests/test_commulingo_author_draft.py`는 타입·필드/근거 원자 교체·초안 철회·과제 판단·구 초안 복구와
작성→독립 검토→승인 해시 공개 경로를 모의 경계에서 검사한다.
`tests/test_commulingo_editor.py`, `test_commulingo_editor_decisions.py`, `test_commulingo_draft_repair.py`는
수정·복구·분류 경계를 검사한다. `test_commulingo_pipeline.py`와 `test_commulingo_pipeline_efficiency.py`는
대기열 및 비용 집계를 검사한다. `test_commulingo_editor_db.py`는
`COMMULINGO_PIPELINE_TEST_PORT`와 격리 frontend 경로 `COMMULINGO_EDITOR_FRONTEND`가 필요하며
DB명은 `commulingo_integrity_test`로 고정한다. 운영 DB로 실행하지 않는다.
모델을 모의 구현하고 실제 queue·cache·저장·영수증을 연결해 전체 전이를 검사한다.
frontend DB 검사는 동시 반영, 승인 해시 불일치, 메모 실패 롤백, 제안 대체, 재실행, stale revision을 다룬다.

`efficiency`는 원장 비용·검증률·수정 요청·무진전 보류·승인된 해결 과제와 미해결 과제를 집계한다.
Jev 호출 수와 비용은 `jev_calls`·`jev_cost_usd`로 분리한다.
해결 과제 수는 작성자가 보고하고 검토자가 승인한 값이다. 독립 사후 표본 감사의 오류율은 아직 계측하지 않는다.
첫 저장 검증률은 처음 전체 검증을 시도한 초안의 `preflight_passed`와 `preflight_failures`로 계산한다.
부분 저장 호출 수를 반려로 간주하지 않으며, 전체 검증 전 중단은 검증률 분모에 넣지 않는다.
`partial_submissions`와 OpenAI 호환 도구 루프의 `malformed_json_calls`·`malformed_json_reasons`를
시도에 보존한다. JSON 파손은 일반 응답과 강제 최종 응답 모두에서 집계하며, journal에는
원문 대신 JSON 오류 종류·인자 길이·오류 위치를 남긴다. 효율 보고서의 두 신규 카운터는 배포 후
계측분만 포함한다. 기존 통계 정의 수정과 실제 모델 실패율 감소를 구분한다.
단위 테스트 통과나 소수 성공 사례만으로 품질 향상·비용 절감률을 주장하지 않는다.

DeepSeek(Anthropic 호환 엔드포인트, `llm/claude_loop.py`)은 인자가 필요한 도구 호출에 `input: {}`를
보낼 때가 있다(2026-09-22~25 편집기 제출 약 46회). 이때 `claude_loop._log_empty_tool_input`이
`Empty tool input from provider` WARNING으로 응답 전체(최대 8000자), `stop_reason`, `output_tokens`,
응답 id를 파이프라인 journal에 남긴다. `{}`를 정당하게 받는 도구(목록 조회 등, 스키마에 `required`나
`minProperties`가 없는 도구)는 기록하지 않는다. `output_tokens`가 크면 모델이 인자를 생성했는데 서버가
버린 경우, 작으면 모델이 인자 없이 호출한 경우다.

```bash
journalctl -u leninbot-commulingo-pipeline.service --since -1d | grep 'Empty tool input from provider'
```

도구 정의는 설명을 자르지 않고 그대로 제공자에 보낸다. 2026-09-25까지는 도구 설명 360자·필드 설명
160자로 잘려, 인물 필드의 공통 길이 안내 뒤에 오는 필드별 지침과 제출 도구의 병합 규칙이 모델에게 가지 않았다.

### 무편집 대기열 정리와 실행 결과

Editor `tick`은 실행 전에 미착수 자동 보강을 최대 200건 재평가한다.
`cleanup`은 같은 평가의 미리보기이며 `cleanup --apply`가 취소를 저장한다.
현재 DB 값에 기존 결손 판정을 적용하되 묶음의 남은 모든 주제를 확인한다.
명시적 요청, gap, 검토 수정, 실행 중 항목, 시도·비용·자료·artifact 이력이 있는 작업은 제외한다.
결손이 없는 항목만 `cancelled`와 사유를 남기며 공개 데이터와 과거 이력은 보존한다.
유지한 후보도 `updated_at`을 갱신해 다음 배치가 뒤의 후보를 확인할 수 있게 한다.
적용은 관련 queue·원장·대상 테이블의 짧은 NOWAIT 잠금 아래 DB 조회만 수행한다.
동시 쓰기가 있으면 해당 tick의 정리를 건너뛰고 정상 작업을 계속한다.

캐시 조회 도구는 빈 인자 `{}` 또는 `passages: []`로 현재 사용 가능한 페이지·ID·라벨을 나열하며,
본문 조회에는 기존 `passages` 또는 `source_id` 중 하나를 받는다.
provider 호환을 위해 도구 schema 루트에는 `not`을 넣지 않으며, 두 선택자의 동시 전달은 handler가 거절한다.
라벨 없는 저장 페이지는 `source_cache.available_pages`의 `source_id`로 열어 라벨을 얻는다.
목록에서는 만료·빈 본문·크기 초과 페이지를 제외하고, 잘못된 ID 오류에는 현재 목록을 반환한다.
원문 조회 응답에도 정확한 source ID를 표시한다. ID와 라벨은 추측하거나 자동 대체하지 않는다.
원문 시각과 만료는 유지하며 할당된 라벨은 초안이 아직 없어도 체크포인트에 보존한다.
잘못된 라벨은 대체하거나 무시하지 않고 실제 라벨과 복구 경로를 안내한다.
추가 문맥 조회의 필드 수 상한은 조회 가능한 schema 필드 수이며 중복 필드는 허용하지 않는다.

단계 JSON은 기존 필드를 유지하면서 `completed_stage`와 `disposition`을 추가한다.
`published`는 승인된 submit, `no_edit`는 공개 저장 없이 종료한 judge,
`failed`는 실행 실패, `budget_wait`는 예산 대기다. 그 외는 progress/held/deferred로 구분한다.
정리 결과는 별도 journal 로그로 기록한다. 일일 health 집계는 승인 기록 수와
고유 반영 작업 수, 마지막 공개 반영 시각, 무편집 완료·취소·실패·현재 예산 대기를 분리한다.

정리 회귀 검사는 확장 smoke에 포함된다. 실제 SQL·동시 쓰기 검사는
`COMMULINGO_CLEANUP_TEST_PORT=<임시 PostgreSQL 포트> venv/bin/python -m unittest discover -s tests -p test_commulingo_cleanup_db.py`
로 실행한다. DB명은 `commulingo_integrity_test`이며 전용 임시 schema를 생성·삭제한다.
운영 DB를 사용하지 않는다.

### 검토 입력·사전 판정·실패 URL·비용 계측

독립 검토에는 수정안과 근거를 온전히 한 번 제공하고, 변경 목록에는 이전 값만 넣는다.
미변경 본문 전체는 초기 입력에서 제외하며 `commulingo_pipeline_review_context`로
필요한 현재 필드를 조회한다. 검토자는 원문을 직접 확인하고, 분류 위험·이전 필수 수정·
양 언어 동등성·승인 해시 검증은 유지한다. 상세 절은 해당 slug의 이전 절과 비교한다.
계측은 이전 구성과 새 구성의 문자 수를 함께 남긴다. 문자 감소를 비용 절감률로 간주하지 않는다.

Editor의 대상 조회와 과제 판정은 유료 단계 예약 전에 수행한다. 결손이 없으면 기존 judge로
진행하며, 명시적 요청과 검토 수정은 유료 작업으로 남는다. 조회한 대상은 같은 시도에서
재사용한다. 이 사전 판정도 heartbeat·시간 제한·lease fencing으로 보호한다.

`fetch_url`의 명시적인 실패만 작업별 `fetch_failures` artifact로 보존한다.
403·인증서·anti-bot 실패는 30분, 연결 타임아웃·DNS 실패는 10분,
429·서버 오류는 5분간 같은 URL의 재접근을 억제한다. offset이나 use_cache 변경으로
우회하지 않으며 다른 출처는 즉시 조사할 수 있다. 최대 64개 실패를 보존하고 재시작 후에도
만료 시각을 유지한다. 범위 오류나 원문 안의 실패 문구는 저장하지 않는다. 작성·검토가
공유하는 것은 접근 실패 정보뿐이며, 이를 원문이나 인용 근거로 사용하지 않는다.
체크포인트 저장은 lease로 보호하고 편집 초안 체크포인트와 분리한다.

공통 LLM 루프는 응답마다 입력·출력·캐시 토큰과 관측 비용을 누적한다. 입력 토큰은
Anthropic의 별도 캐시 토큰을 포함하도록 정규화한다. 값은 provider 전환과 예외 뒤에도
tracker에 남지만, 부분 관측 비용만으로 예약을 정산하지 않는다. 효율 집계에는 계측된
시도 수·토큰·검토 문자 수·유료 예약 전 종료·실패 URL 재시도 억제 횟수를 표시한다.

`reconcile-costs`는 미정산 원장을 조회하며 `--apply`로 확인된 정산만 복구한다.
종료된 시도의 `cost_complete=true`와 비음수 유한 `actual_cost_usd`가 함께 있을 때만
해당 예약을 정산한다. `tick`도 이 복구를 수행한다. 완료된 작업이라는 사실이나 감사 로그의
시각별 비용 합만으로 비용을 추정하지 않는다. 이전 중단 작업에 완결된 비용 기록이 없으면
예약을 보존한다. DB schema 변경은 없다.
