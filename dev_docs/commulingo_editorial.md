# CommuLingo 인물 편집 서비스

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

모든 쓰기에 sources가 필요하다. 사실 필드 bio/moment/years/citizenship/nationalOrigin/body에는
필드별 evidence가 필요하다: field, claim, source, locator(쪽수/절), optional excerpt/stance.
stance는 supports/disputes다. 한영 부분 수정과 aliasEdits/careerEdits/sceneEdits는 Admin과 같다.
정확한 검증과 분량 원본은 frontend `person-editorial-contract.json`이다.
Python도 이 파일을 읽는다(`COMMULINGO_PERSON_CONTRACT`로 테스트 파일을 지정 가능).

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
