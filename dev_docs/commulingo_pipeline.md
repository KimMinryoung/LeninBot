# CommuLingo 영속 편집 파이프라인

인물·용어의 등록과 보강은 `commulingo/pipeline/`이 맡는다. 진입점은
`scripts/commulingo_pipeline.py`, 운영 설정은 `config/commulingo_pipeline.json`이다.
저장 도구의 계약은 [편집 서비스](commulingo_editorial.md), 판정 정책은 [Jev 연동](jev_system_one_adoption.md)을 따른다.

## 운영 경로와 승인 범위

운영 설정은 `workflow=editor`, `phase=live`다. workflow는 `editor`만 허용하며 미지정 기본값도 `editor`다.
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

출처가 있는 인물 활동은 기능 → 해당 기능의 소속 → 선택된 기능·소속을 지지하는 근거 순서로
Jev에 질의한다. 질문별 판정은 독립적이므로 후속 질의 state에 앞선 선택을 명시한다.
발췌 원문도 state에 포함하며, 허용 목록 밖 응답이나 근거 없는 선택은 보류한다.
국적·거주·학술 연구 대상을 국가기관 복무로 추정하지 않는다.
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

`author_draft.py`는 AI용 입력을 기존 저장·검토용 `fields/claims/issue_results`로 변환한다.
등록과 편집은 같은 `changes.<field> = {value, evidence}` 형식이며 등록의 필수 필드는
기존 canonical schema가 결정한다. 각 evidence는 `claim`·`passages`와 선택적 `stance`를 받는다.
분류 코드·revision·출처 위치·승인 해시는 작성자가 제공하지 않는다.

| 세션 내부 도구 | 역할 |
|---|---|
| `commulingo_pipeline_result` | `changes`, `issues`, `reason`, 선택적 `notes`로 전체 초안 제출 |
| `commulingo_pipeline_repair` | 같은 형식으로 변경한 필드만 다시 제출; 해당 필드의 값과 근거를 함께 교체 |
| `commulingo_pipeline_no_edit` | `status`, `reason`, 모든 과제의 `issues`로 무편집 판단 제출; 기존 초안은 이력에 보존 |
| `commulingo_pipeline_cached_passages` | 캐시 목록 및 원문을 네트워크 없이 조회 |
| `commulingo_pipeline_context` | 추가 현재 값과 이번 작업에서 편집 가능한 필드의 schema 조회 |
| `commulingo_pipeline_research` | 필드와 이유를 명시해 사실 조사 재개 |

예를 들어 `missing:body` 과제를 처리하는 최초 제출은 다음 형태다. P 라벨은 실제 조회한
원문에 있어야 한다. 동일한 `changes.body` 객체를 repair에 보내면 body만 고친다.

```json
{
  "changes": {
    "body": {
      "value": {"ko": "원문으로 확인한 설명이다.", "en": "An explanation supported by the original."},
      "evidence": [{"claim": "The original documents this explanation.", "passages": ["P1"]}]
    }
  },
  "issues": {
    "missing:body": {"status": "resolved", "reason": "Added both languages with original evidence."}
  },
  "reason": "The original supports the commissioned explanation."
}
```

`issues`는 과제 ID를 키로 하는 객체다. 최초 제출과 무편집 판단은 모든 과제에
resolved/deferred와 사유를 요구하며, 필드를 채웠다는 이유만으로 해결을 추정하지 않는다.
인물 상세 절 작성 API는 `changes.heading`과 `changes.body`를 받으며 slug는 작성 도구에 노출하지 않는다.
서버는 독립 검토 전에 `commulingo_section_slug` 원샷 호출(레지스트리의 low 티어)로 절 주제 slug를
생성하고 인물 ID·기존 절과의 충돌·허용 형식을 확인한다. 생성값을 검토·승인 해시·공개 요청에
포함하며 체크포인트에 보존해 같은 제목의 형식 수정에서 재호출하지 않는다.
기존 제안 수정은 원래 절 slug를 유지하고 경량 모델을 부르지 않는다. 생성 실패는 영문 제목의
결정적 slug로 대체하고(`section_slug_fallbacks` 집계) 기존 절과 같은 제목만 작성 오류로 돌려준다.
절 작성 API는 인코딩된 `sortOrder` 대신 시기 시작 `startYear`(필수, 해당 연도가 없으면 `null`)와
`startMonth`(선택)를 받는다. 서버의 `section_sort_order`가 YYYYMM(월 모름=00)으로 바꿔
검토·해시 전에 `sortOrder`로 넣는다. 작성자가 `startYear`나 `startMonth`에 붙인 근거도
저장 시 `sortOrder` 근거로 옮겨, 공개 패치에 없는 필드의 근거가 남지 않게 한다.
제목 속 연도는 해석하지 않는다(제목에 연도를 넣으라는 지시가 없고 실제로 37%만 연도를 담는다).
공유 저장소는 값이 없으면 0을 써 절이 맨 앞으로 갔다(2026-09-22~24 편집기 절 156건).
repair의 `issues`는 명시한 과제 판단만 교체한다. `notes`와 `reason`도 생략하면 보존한다.
repair의 `remove_fields`는 해당 필드와 근거를 초안에서 철회하며 저장된 공개 데이터를
삭제하지 않는다. 동일 호출에서 같은 필드를 교체하고 철회할 수 없다.
배열은 해당 목록 전체를 교체한다. 새 작업에는 aliasEdits/careerEdits/sceneEdits와
전체 목록 교체 방식을 동시에 노출하지 않으며, 구 체크포인트의 수정 필드는 복구를 위해 유지한다.

도구 schema 자체에 이번 과제·등록 필수·저장된 초안·이전 수정안에 필요한 필드 타입을 넣는다.
별도 `draft_contract`를 문맥에 중복 제공하지 않는다. 추가 값 조회는 편집 범위를 확장하지 않는다.
형식 오류는 접수 전에 거절하고 기존 초안을 보존한다. 길이 초과 텍스트는 접수하여 저장한 뒤
canonical 제한으로 거절하므로 해당 필드만 다시 제출할 수 있다. JSON Pointer 수정은 노출하지 않는다.
진단 경로는 `changes.<field>.value` 또는 `/changes/<field>/evidence/<index>`로 변환한다.
한 주장이 여러 원문 범위로 펼쳐져도 원래 필드 내 근거 인덱스를 가리킨다.
빈 호출과 잘못된 수정 형식은 내용 무진전 횟수에 넣지 않는다.
모든 도구는 세션에만 주입하며 소유자·commulingo_curator 호출자 권한을 검사한다.

최초 문맥은 과제에 필요한 필드로 좁히되 저자 notes·인물 sections·original_proposal은 포함한다.
`surrounding_context`에는 용어의 기존 정의·원어·별칭·시기, 인물의 이름·생몰년·소개·역할 중
초기 current에 없는 값을 함께 제공한다. 이는 중복·모순을 피할 읽기용 배경이며 수정 범위를 늘리지 않는다.
초기 입력과 초안 거절·조사 차단·조사 재개 응답에는 같은 `work_status` 구조를 제공한다.
완료 조건, 저장된 필드·주장 수, 실제 조사 허용 상태, 남은 제한 조회 수, 부족한 근거,
오류 종류와 다음 행동을 표시한다. 저장된 주장은 검증·승인되었다고 표현하지 않는다.
인용 거절은 캐시 원문 재확인과 필요한 사실 조사로, 참조·근거 누락은 캐시와 라벨 확인으로,
그 밖의 저장된 초안 오류는 부분 수정으로 안내한다. `submission_tool`은 수정안 제출 도구이며
당장 실행할 `next_tool`과 구분한다. 조사 제한 상태에서는 먼저 research 도구로 조사를 재개해야 한다.
오류 종류·부족한 근거·조사 상태를 체크포인트에서 복원하며, 캐시 열람은 이전 오류를 지우지 않는다.
명시적인 조사 재개도 체크포인트에 저장한다.
Editor 입력은 세 층으로 나눈다. 시스템 문맥은 공통 편집·문체 원칙과 입력 항목의 역할을
설명하고, 작업 데이터는 `issues`·관련 현재 값·원문 캐시·필드별 저장 초안을 제공한다.
`work_status`는 현재 조사 허용 상태와 다음 행동의 단일 진입점이다. 초기 입력에서 과제 범위,
오류와 부족한 근거를 중복 전달하지 않으며 도구 응답은 독립적으로 읽을 수 있게 범위를 포함한다.
전체 제출·부분 수정의 호출 방법은 해당 도구 설명이 맡고, `work_status`는 다음 수정 행동과
별도 무편집 종료 도구를 안내한다. 제거된 legacy 단계가 쓰던 discovery·전체 결과 종료 지시는
Editor 시스템 문맥에 넣지 않는다. curator 기본 프롬프트 역시 누적하지 않고 교체한다.
저장 schema, 근거 검증, 독립 검토와 승인 조건은 프롬프트 축약과 별개로 코드에서 유지한다.
공통 WRITING_RULES를 적용하고 문장 분량은 상한의 80%를 여유 목표로 제시한다.
새 API의 notes는 최상위 비공개 항목이다. 구 체크포인트의 fields.notes는 복구 시 최상위로 옮겨 중복 없이 합친다.
형식 수정 중에는 기존 근거를 사용하고 사전 항목 조회를 제한한다. 누락 근거나 사실 충돌은
해당 필드의 조사를 재개할 수 있다. 소개문·정의는 실제 저장 형식인 문자열로 제출한다.

`editor_checkpoint`는 초안·P 라벨·조회 인자·오류·분류 캐시와 수정 상태를 저장한다.
모델 응답이 출력 토큰 한도에서 끊기면 설정된 추가 응답 횟수 안에서 같은 도구 세션을
이어가며, 이때 원문 캐시와 앞선 도구 결과를 유지한다. 추가 응답도 길이 제한에
걸리면 같은 세션에서 결과 제출 도구만 허용하는 최종 호출을 수행한다. 그래도 단계
결과가 없으면 영속 자료를 남긴 채 다음 배치에서 재시도한다.
설정된 대화 횟수가 끝날 때 모델이 결과 도구 대신 설명문만 낸 경우에도 같은 최종 호출을 사용한다.
내부 artifact 버전과 fields/claims 형식은 유지한다. 같은 baseline이면 구 체크포인트도
필드별 작성 형식으로 보여 주고 정상 수정 경로로 복구한다. 구 초안의 컨테이너 자체가 잘못된
경우 원본을 보존해 보여 주고 전체 재제출 또는 무편집 종료를 안내한다. 체크포인트 쓰기도 lease로 보호한다.
같은 수정안과 오류의 반복은 내부 보류한다. revision 충돌은 최신 상태의 조사로 돌려보낸다.

## 독립 검토와 공개 반영

`workflow.py`는 현재 문서와의 차이, 이전 수정안과의 차이 및 이전 검토를 제공한다.
검토자는 원문을 직접 가져와 핵심 변경 사실과 위험 항목을 확인한다.
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
운영 일일 한도는 UTC 날짜 기준 $2이며, 설정 변경은 다음 배치부터 적용한다.
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
단위 테스트 통과나 소수 성공 사례만으로 품질 향상·비용 절감률을 주장하지 않는다.

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
