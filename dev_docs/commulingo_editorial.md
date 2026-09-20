# CommuLingo 인물 편집 서비스

새 영속 실행기와 용어 저장·검토 확장은 [파이프라인 문서](commulingo_pipeline.md)를 따른다.
기존 timer의 동작은 전환 전까지 아래와 같으며 새 설정은 기본적으로 draft/비활성 상태다.

인물/상세 절 저장의 소유자는 frontend `data/commulingo/person-editorial-service.js`다.
Python `runtime_tools/commulingo_people.py`의 등록·수정·절 저장과
`scripts/commulingo_suggestions.py`의 인물 제안 승인은
`runtime_tools/commulingo_person_service.py`를 통해 동일한 JS Admin 저장소를 호출한다.
인물/절의 Python 직접 SQL fallback은 없다. 직책/사건/용어는 기존 저장소를 사용하되
검증 전에 동일한 `commulingo-editorial-write` 트랜잭션 advisory lock을 얻는다.

## 호출 계약

고정 argv `docker exec -i leninbot-frontend node /app/scripts/commulingo-person-service.js`와
JSON stdin/stdout을 사용한다. 요청 본문이나 DB 비밀번호를 명령행에 넣지 않는다.
frontend 컨테이너의 DB 설정을 사용하며 공개 쓰기 endpoint를 추가하지 않는다.
서비스 계정에 Docker 접근 권한이 필요하다. RPC 장애 시 쓰기는 실패한다.
`COMMULINGO_FRONTEND_CONTAINER`는 격리 테스트용 컨테이너 선택에 사용할 수 있다.

get_person은 동일 DB 스냅샷의 revision, 인물, 절 요약, 근거, 보강 상태를 반환한다.
KG 참고 정보는 기존처럼 best effort로 붙인다. 기존 인물 수정은 fields.expectedRevision,
section_save는 expected_revision이 필수다. 최신 버전의 자동 재대입으로 충돌을 우회하지 않는다.
신규 인물 도구의 fields schema에는 expectedRevision·aliasEdits·careerEdits·sceneEdits를
노출하지 않는다. 신규 등록은 aliases·career·scenes로 초기 목록을 작성하고, 수정 전용
컬렉션 연산은 update 도구에서만 사용한다. 파이프라인 작성기도 같은 schema를 사용한다.

모든 쓰기에 sources가 필요하다. 사실 필드 bio/moment/years/citizenship/nationalOrigin/body에는
필드별 evidence가 필요하다: field, claim, source, locator(쪽수/절), optional excerpt/stance.
stance는 supports/disputes다. 한영 부분 수정과 aliasEdits/careerEdits/sceneEdits는 Admin과 같다.
미확정 fate는 기존 저장소와 동일하게 `kind=""`와 한영 유보 라벨로 표현한다.
작성 도구도 빈 kind를 허용하며 존재하지 않는 unknown/unconfirmed 분류를 요구하지 않는다.
정확한 검증과 분량 원본은 frontend `person-editorial-contract.json`이다.
Python도 이 파일을 읽는다(`COMMULINGO_PERSON_CONTRACT`로 테스트 파일을 지정 가능).

인물 도구의 evidence/expectedRevision/reviewFlags는 fields 안에, citations는 최상위에
둔다. 절 저장은 evidence/expected_revision을 최상위로 받는다. evidence.source_id는
S1=citations[0], S2=citations[1]처럼 해당 요청의 출처를 명시적으로 선택한다.
Python은 이를 원래 출처 문자열로 바꾸어 공통 저장소에 전달하며 두 값이 충돌하면 거절한다.
기존 형식 evidence.source는
설명을 포함한 citations 항목 전체와 정확히 일치해야 하며 URL만 복사하면 안 된다.
Python RPC 직전 진단은 contract.factFields의 누락 근거와 출처 문자열 불일치,
현재 수정에 없는 필드 근거를 한 번에 알려준다. 근거를 자동 생성·추론하거나
revision을 갱신하지 않으며 JS 저장소의 최종 검증은 계속 적용된다.
`tests/test_commulingo_evidence_diagnostics.py`는 이 진단을 DB 없이 검증한다.

lane health는 pending_review와 과거 no_edit/OK — pending 결과를 별도로 집계한다.
no_edit 상태만으로 라운드 소진을 단정하지 않는다.

## 스케줄 작업의 실행·복구

인물·gap·사건·용어는 people maintainer의 `_call_curator_stage`를 공통 실행기로 쓴다.
`scripts/commulingo_run.py`는 gateway request_id에 대응하는 run_id, 시도 합산 비용,
라운드, 480초 조사 제한과 결과를 기존 SQLite의 runs 테이블에 기록한다.
출력 이어쓰기와 새 대화 재시도는 별개이며 새 대화는 최대 한 번만 허용한다.
이어쓰기 라운드는 남은 라운드에서 예약한다. 재시도는 남은 예산만 받으며
신규 인물의 발견·생성·보강 fallback도 같은 예산을 공유한다. 이미 발행된 LLM 요청의
실제 비용은 응답 후 확정되므로 단일 요청의 예산 초과까지 막는 선결제 한도는 아니다.
공용 LLM 루프는 예외·취소에도 완료된 호출 비용을 tracker에 남긴다.

성공은 해당 handler가 반환한 제출 번호로 확인한다. 다른 실행의 편집 수 증가는
성공 근거가 아니다. 제출 결과는 즉시 기록하며 이미 쓰기가 있는 작업은 통째로 재시도하지 않는다.
정상 결과 complete/not_applicable/sources_unavailable은 실패 cooldown에 넣지 않는다.
gap의 조사 실패는 resolution의 retry: 표식으로 6시간 재선정을 미루고,
sources_unavailable: 표식은 90일 동안 미룬다. 검토 실행도 같은 ledger에 조사 비용과
최종 승인·반려·보류 상태를 기록하되 작성기의 원문 캐시는 사용하지 않는다.

`scripts/commulingo_write_session.py`는 스케줄 실행에만 기존 쓰기 도구의 인자로
draft_id와 repairs를 추가한다. 별도 저장 도구나 권한은 만들지 않는다.
repairs는 JSON pointer의 set/remove 작업이며 완성된 인자를 원래 스키마로 다시
검증한 뒤 같은 handler에 전달한다. 대상·action·revision 변경은 금지한다.
초안은 작업/대상/주제/revision 범위에 남고, 성공한 쓰기를 재생하지 않는다.
기존 인물 스냅샷은 실행기가 최초 문맥에 제공하고 expectedRevision을 그대로 바인딩한다.

후보 SQL은 bio/years/citizenship/nationalOrigin/moment의 근거 누락을 각각 확인한다.
주제별 승인 편집 시각을 모아 cooldown을 적용하며, 다른 주제를 편집했다는 이유로
현재 주제를 지연하지 않는다. CTE를 materialize하고 최근 편집을 한 번 집계하여
후보별 같은 집계의 반복 실행을 피한다. 절 수와 글자 수는 분량 상한이다.

lane health는 실행 ledger에서 첫 저장 성공, 원문 캐시 적중, 쓰기 거절, 실패 비용을
추가 표시한다. 기존 $/edit는 실패·보류를 포함한 lane 전체 비용/반영 건수로 명시한다.
ledger는 도입 후 실행만 포함하며 과거 로그에서 소급 생성하지 않는다.
같은 조회 기간에 기록된 작성·검토 실행은 제출 번호로 연결하여 검토 비용을 포함한
반영 건당 비용도 표시한다. 기간 밖의 이전 검토 비용까지 소급 포함하는 지표는 아니다.

## 검토와 보강

config direct_apply=true도 출처 충돌, 동일인 불명, 삭제 및 대규모 본문 축약은 pending으로 끝난다.
false는 모든 인물 쓰기를 pending으로 보낸다. pending은 보강기의 정상 종료이며 재시도하지 않는다.
제안 approve는 --note가 필요하고 최초 revision을 다시 검사한다. 버전 없는 과거 제안은
거절 후 최신 인물 조회와 조사에 근거해 새 제안으로 제출한다. confidence는 승인 기준이 아니다.

no_edit는 reason/status/sources를 명시하고 basics/nationality/bio/moment/events/sections 중
현재 조사 주제의 상태를 DB에 기록한다. complete/not_applicable은 180일,
sources_unavailable은 90일 후 재검토하며 새 내용·근거가 관련 주제를 다시 연다.
후보는 필수 정보, 근거 없음, 번역 누락, 국적·일화·주제 순으로 고른다.
미승인 제안이 있는 인물과 유효한 완료 주제는 제외한다. 관계 수로 중요도를 높이지 않으며
12개 절은 상한이다. 기존 데이터의 근거 소급 작성이나 역사적 진위 검증을 자동 완료로 간주하지 않는다.

## 배포·검증

frontend migration 176 → 호스트 data 코드 및 frontend 배포 → Python 코드 반영 →
이를 import하는 API/Telegram 서비스 재시작 순서다. 스케줄 보강은 다음 실행부터 새 코드를 읽는다.
전환 동안 배치 타이머를 멈추고 실행 중인 작업이 없는지 확인한다.
롤백은 이전 호스트 data 파일·frontend 배포본·Python 코드를 함께 복구한다.
추가 테이블을 삭제하지 않아 신규 근거/보강 상태를 보존한다.

테스트: tests/test_commulingo_editorial_selection.py는 LLM/운영 DB 없이 선택·완료·pending 종료를 검증한다.
tests/test_commulingo_person_rpc.py는 COMMULINGO_FRONTEND_CONTAINER=commulingo-python-rpc인
독립 DB 컨테이너에서만 실제 도구 저장·버전 충돌·제안 승인을 실행한다.
frontend test-commulingo-editorial-db.js는 근거·검토·롤백·상태 전이를 검증한다.

국적 일괄 조사 스크립트 `commulingo_backfill_nationality.py`와
`commulingo_backfill_person_nationality.py`는 기본적으로 보고서만 만든다.
추론한 국적/민족을 직접 SQL로 반영하지 않는다. 조사 후 최신 조회 버전과 출처·근거를
포함한 명시적 Admin spec을 `--apply-spec <reviewed.json>`으로 전달하면
공통 Admin upsert CLI가 배치 전체를 한 트랜잭션으로 검증·반영한다.
예전 --apply 단독은 거부된다. 실패 시 배치 앞부분의 쓰기도 롤백한다.

## 자동 검토·수정과 내부 보류

`leninbot-commulingo-review.timer`가 15분마다 `scripts/commulingo_person_reviewer.py`를 실행한다.
`agents/commulingo_reviewer.py`는 작성자 문맥을 물려받지 않는 전용 AgentSpec이다.
일반 delegate 목록에는 등록하지 않는다. 조사 도구와 실행기 내부의 `commulingo_review_decision`
만 제공하며, LLM이 인물 저장·승인 API를 직접 호출할 수 없다. 이 내부 판단 도구는
security_gateway/policy.py에 state로 등록하며 소유자와 commulingo_reviewer 호출자만
허용한다. 테스트는 handler 직호출 대신 실제 dispatcher를 거쳐 차단 재발을 검사한다. 검토 실행기는 검증된 판단을
기존 JS 공통 승인 서비스로 전달한다. Telegram 명령도 같은 서비스를 사용한다.

- 검토기는 checks[].citation_id(S1=source_refs[0])와 **passages**(자기 fetch_url/wiki_get 결과에 표시된 문단 라벨 `R…@offset`, 한 출처의 1~8개), 한국어 finding을 낸다. 실행기는 가져온 슬라이스를 표시할 때 문단마다 라벨을 붙여 두므로(`review_source` → 조사 레인과 공유하는 `evidence.Passages`) 라벨을 그 문단 원문(citation/source/quote/finding 형식)으로 바꾸는 데 복사·매칭이 없다(`resolve_review_checks`). 여러 출처의 라벨은 출처별 check로 나누고, 표시된 적 없는 라벨만 있는 check는 결정 전체를 거절하지 않고 그 check만 뺀 뒤 `dropped_checks`(번호·라벨·사유)로 저장 결정에 남긴다 — 확인된 check가 하나도 없을 때만 거절. 인용문 복사+접기 매칭(2026-09-19 하루: 9~39개 check 중 하나가 어긋나면 결정 전체가 사유 표시 없이 떨어져 하루 159건 재제출)과 행 번호 체계(그 전: 주당 87+76건)는 폐기했다. 원문은 현재 검토에서 직접 가져온 것만 인정한다. resolved_risks는 risks 식별자만 담고 설명은 reason/finding에 쓴다.
- 승인: 새로 가져온 원문에 실제로 있는 인용, 모든 제안 출처의 확인, 검토 사유의 해소,
  위키백과 밖의 근거가 필요하다. 검색 요약·작성자가 적은 인용만으로 승인하지 않는다.
- 수정 요청(`revise`): 검토기가 직접 확인한 근거로 고칠 수 있는 사실 오류·확정 과잉·한영 불일치를 특정하면 작성기로 되돌린다. 출처별 생몰연도 이설은 병기하고 옥사와 처형을 구분한다.
- 반려: 유용하게 고칠 수 없는 부적절한 제안이나 해로운 삭제일 때. 버전이 없거나 오래된 제안은
  LLM 호출 없이 반려한다. 검토 도중 버전 충돌도 반려하며 새 버전을 자동 대입하지 않는다.
- 판단 불가(`escalate`): 접근할 수 없는 자료나 해결되지 않은 동일인은 내부 보류한다. 사용자에게 판단을 요구하지 않는다.
  시스템이 인용의 원문 포함 여부를 검사해도 역사적 판단의 정확성을 보증하지는 않는다.
인용 거절은 checks의 1부터 시작하는 항목 번호와 원인을 반환한다. 길이 규칙은 없다 — 한 글자짜리 문단도 인용할 수 있고, 그 문단이 진술을 뒷받침하는지는 Jev 인용 게이트만 판단한다.
범위를 넓히도록, 원문 불일치는 해당 source_id의 표시 행 범위를 선택하도록 안내한다.

Migration 177의 `commulingo_person_review_jobs`가 작업 상태·근거·판단·오류를 보관한다.
SKIP LOCKED와 20분 lease로 중복 실행을 막는다. 실행 제한은 650초, 조사 제한은 480초다.
실패는 1시간 후 재시도하고 3회째 내부 보류한다. 저장된 판단은 장애 복구 시 재사용하며
그때도 현재 버전과 제안 상태를 확인한다. 공통 승인의 트랜잭션으로 이중 반영을 막는다.

소유자는 Telegram 개인 DM에서 다음 명령을 쓴다. 일반 사용자/그룹 채팅은 처리하지 않는다.

```
/commulingo_review list
/commulingo_review show 123
/commulingo_review approve 123 원본 자료의 생년과 직책을 대조하여 동일인 구분을 확인함
/commulingo_review reject 123 기존의 확인된 내용을 근거 없이 삭제하므로 반려함
/commulingo_review retry 123
```

show는 현재 원문·전체 변경안·출처·검토 근거를 JSON으로 첨부한다. 승인/반려는 사유가 필수다.
재검토는 실패/판단 불가 상태만 다시 대기열에 넣으며 진행 중인 작업을 덮어쓰지 않는다.
자동 검토 요청·하루 뒤 재알림은 보내지 않는다. 기존 미처리 요청도 동일하다.
`--notify-only`는 호환용으로 유지하며 `notifications_disabled`를 반환한다.
수동 조회·승인·반려·재시도 명령은 필요할 때 직접 사용할 수 있다.

기존 pending 제안에서 `revise`를 받으면 원 제안 ID당 하나의 파이프라인 수정 작업을
영속 생성한다. 원 변경안과 검토 근거를 보존하고 원 제안은 pending으로 유지한다.
수정 작업은 최신 원문으로 조사하고 새 초안을 독립 검토한다. 통과한 뒤에만 공통
저장 서비스의 고정 idempotency key로 이전 제안을 반려하고 새 제안을 제출·승인한다.
수동으로 승인·반려된 원 제안은 대체하지 않는다. 장애 후 재실행에도 같은 수정 작업을
사용하며, 완료 작업을 새로 생성하지 않는다. 수정 작업은 일반 주제 묶음에서 제외한다.
최초 수정 요청을 포함하여 최대 두 차례 자동 수정을 허용하고, 그래도 해결되지 않으면
근거·오류를 내부 보류 상태에 보존한다. 승인 기준과 예산 제한은 그대로 적용된다.

한 번에 최대 한 제안을 검토하며 기본 LLM 예산은 $0.20, 최대 12라운드다.
CommuLingo 일일 합산 예산 제한은 해제되어 있다. systemd 원본과 운영 unit은
COMMULINGO_DAILY_CAP_USD=0을 사용하며 budget guard의 기본값도 0이다.
0 이하이면 비용 집계 없이 통과한다. 양수를 명시하면 일일 제한을 다시 적용하며,
예산 소진 시 조사는 연기하며 검토 요청 알림은 보내지 않는다. 회당 LLM 예산은 유지한다. 상태 보고에 review lane을 포함하며 빈 큐는 정상 idle로 집계한다.

배포 순서: 177 적용 → Python 코드와 systemd service/timer 설치 → daemon-reload →
Telegram 재시작(명령 등록) → 검토 서비스 첫 실행 → 타이머 활성화.
frontend 런타임 코드는 바뀌지 않으며 기존 공통 승인 서비스를 사용한다.
중지/복구: review.timer를 멈추고 실행 중인 review.service가 종료됐는지 확인한다.
필요하면 이전 Python 코드로 복귀하되 큐·판단·근거 테이블은 보존한다.

## 출신 국가 코드 확장 (2026-09-07)

Frontend 국기·지도와 Python 허용 코드에 serbia, croatia, slovenia, montenegro,
bosnia-herzegovina, switzerland를 함께 등록했다. 유고슬라비아는 소속 국가로 유지하고
출신 배경은 출생지가 아닌 문헌의 민족·가계·자기인식으로 구분한다. 혼합 배경은
단일 국가 코드에 가려지지 않도록 한영 라벨과 주장 근거에 함께 기록한다.

## 사전 등록 국가 코드

세계 지도에 사전 등록된 현대 국가도 인물·사건이 없어도 등록 도구에서 선택할 수 있다. `_NATIONALITY_CODES`는 frontend `modern-country-codes.json`과 기존 역사·지역 코드에 대응한다. 신규 코드의 국기 SVG와 영역 등록은 frontend에서 관리하며 `commulingo_people_maintainer.py`도 동일한 Python 코드 집합을 가져온다. 국가 추가 시 양쪽 코드 집합의 일치를 검증한다.

## 이벤트 요청 인물의 검토 대기

gap worker는 pending 인물/절 제안이 있는 작업을 claim에서 제외한다. 기존 인물은
target_id, 대기 중인 신규 등록은 resolved_id 및 한영 이름/별칭으로 연결한다.
신규 등록 도구가 pending을 반환하면 gap을 완료로 닫거나 실패로 재조사하지 않고
제안 target_id와 번호를 기록하여 pending_review로 종료한다. 이미 생성된 대기 제안도
이름으로 차단한다. 승인 후에는 실제 카드 확인과 이벤트 연결을 진행할 수 있고,
반려 후에는 더 이상 pending 제안이 없는 작업만 다시 조사 대상이 된다.

자동 검토의 출력 길이 제한 중단은 기존 조사 문맥과 수집 원문을 유지한 채 최대 2회
이어받는다. AgentSpec의 max_output_continuations=2를 실행기가 continue_on_length와
max_length_continuations로 실제 도구 루프에 전달한다. 응답당 8,000토큰과 회당 예산,
전체 조사 시간 제한은 유지하며 유효한 검토 판단 도구가 성공하면 즉시 종료한다.

`ResearchMemory` 지침은 위임된 주장에 충분한 근거가 모이면 조사 종료 후 편집/no-edit 판단으로 진행하도록 한다. 기존 형식 오류 후 조사 재시작 차단은 유지하며 누락 근거·상충·변경 가능성은 재확인할 수 있다. 검색·Extract 비용은 runs의 LLM 비용과 별도로 [web 사용량 장부](web_research.md)에 기록된다.

근거 배열과 독립 검토 checks에는 개수 상한을 두지 않는다. 각 항목의 출처·인용·필드 지원 검증은 유지하며, 충분히 확인된 사실에 인용을 더 모으는 것은 품질 목표가 아니다.

## 인물 분류 자동 배정과 감사

**등록 시 자동 배정 (2026-09-19).** 인물 create의 `groupId`와 `role`은 작성 모델이 고르지 않는다. 도구 schema에서 두 필드는
필수가 아니고(작성 모델에게는 보이지 않음), 실행기가 초안의 이름·생몰·국적·출신·별칭·경력·bio·moment·fate 라벨과, 파이프라인에서는 조사 claim의
bio·career·moment·years 발췌(필드당 4건, 900자)를 state로 Jev(registry
`commulingo_person_classification`, `runtime_tools/commulingo_classify.py`)에 choice 판정을 받아 채운다. 관직 선택지는 소련·후계국
국적에만 제시하고 비소련 인물은 카테고리만 고른다. 작성 모델의 schema에는 이 필드들이 아예 없다(create 도구와 초안 schema에서 제거). confidence가 `thresholds.accept`(0.7)
미만이어도 채우되 draft artifact `metrics.classification`에 수치가 남아 검토 단계가 `classification_low_confidence` /
`code_low_confidence` 위험 항목으로 독립 검토자에게 확인을 요구한다(`stages.classification_risks`). 판정 불가(None)면
작성 모델에게 넘기지 않는다: 파이프라인은 초안을 버리고 작업을 미뤄 다음 tick에 재시도(3회 후 escalate), 도구는
"나중에 다시" 오류를 돌려준다.
파이프라인 초안 단계(`stages.Draft`)와 `commulingo_person_create` 도구 양쪽이 같은 함수를 쓰며, 초안 프롬프트에서는 그룹·카테고리
카탈로그가 빠진다. update에서는 기존 분류가 잠겨 있으므로 해당 없음. `enabled=false`면 예전처럼 작성 모델이 고른다.
`citizenship.code`(국가 코드)와 `fate.kind`도 실행기가 채운다(`classify_person_codes`, registry `commulingo_person_codes`,
create·update 모두, schema에서 code/kind 키 제거): 작성 모델은 라벨 문장만 쓰고, 파이프라인은 라벨 + 해당 필드의 조사 claim 발췌를, 도구 경로는 라벨만 Jev에
준다. 생존 인물(`years`가 `–`로 끝남)의 fate는 호출 없이 빈 kind. 기준선(최근 초안 60건): citizenship 50/50, fate 32/35 —
불일치 3건 중 claim 없는 초안에서 작성 모델이 natural로 적은 것을 Jev가 unconfirmed로 본 것이 포함된다. `nationalOrigin.code`도 채운다: 출신 규칙(민족·국가 배경이지 출생지·활동지·시민권이 아님, 유대계는 가족의 출신 국가이지
israel이 아님, 비러시아 민족의 소련 관리는 그 민족)을 instructions에 적자 42/49 → 48/49가 됐고, 남은 1건은 작성 모델 라벨이
출생지였던 것을 Jev가 0.49로 유보한 사례다. 이로써 등록 API의 닫힌 집합 필드는 전부 실행기가 채운다. 용어 create의 `category`(10종)도 같다: `classify_term`(registry `commulingo_term_classification`)이 term·정의·기간·본문으로 고른다. 저장된 용어 1,086건(작성 모델이 고른 값)과의 일치는 813,
conf ≥0.85에서 628/714 — 저장값 자체의 일관성이 낮아 정확도 상한이 아니라 관행 재현율이다.

**감사.** `scripts/commulingo_classification_audit.py`가 인물 전원의 groupId·role을 Jev(System One)로 재판정해 저장값과 다른
고신뢰 건을 `logs/commulingo/person_classification_audit_<날짜>.md`로 뽑는다(쓰기 없음, 인물당 ~$0.00012). criteria에는
운영자가 확정한 편집 규칙이 들어 있다: 공화국 제1서기·공화국 정부 수반은 `nationalities-federal`, 지방·주 서기와 콤소몰·중앙위
서기는 `party-secretariat-cadres`, `ideology-propaganda`는 친소련 이데올로그 관직, 비소련 국적 인물은 소련 시대 그룹에 두지
않으며, `scholar`는 이 역사를 연구한 역사가·사회과학자다. 정정은 편집 서비스 제출→승인으로 하고 승인 메모에 근거를 적는다
(2026-09-19 첫 실행: 106명 정정, `dev_docs/jev_system_one_adoption.md` 4.11.1).
