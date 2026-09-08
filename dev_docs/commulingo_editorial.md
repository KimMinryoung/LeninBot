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

## 자동 검토와 소유자 처리

`leninbot-commulingo-review.timer`가 15분마다 `scripts/commulingo_person_reviewer.py`를 실행한다.
`agents/commulingo_reviewer.py`는 작성자 문맥을 물려받지 않는 전용 AgentSpec이다.
일반 delegate 목록에는 등록하지 않는다. 조사 도구와 실행기 내부의 `commulingo_review_decision`
만 제공하며, LLM이 인물 저장·승인 API를 직접 호출할 수 없다. 검토 실행기는 검증된 판단을
기존 JS 공통 승인 서비스로 전달한다. Telegram 명령도 같은 서비스를 사용한다.

- 승인: 새로 가져온 원문에 실제로 있는 인용, 모든 제안 출처의 확인, 검토 사유의 해소,
  위키백과 밖의 근거가 필요하다. 검색 요약·작성자가 적은 인용만으로 승인하지 않는다.
- 반려: 확인한 근거로 오류/해로운 삭제임이 드러났을 때. 버전이 없거나 오래된 제안은
  LLM 호출 없이 반려한다. 검토 도중 버전 충돌도 반려하며 새 버전을 자동 대입하지 않는다.
- 판단 불가: 접근할 수 없는 자료, 해결되지 않은 동일인/상충 근거는 소유자에게 전달한다.
  시스템이 인용의 원문 포함 여부를 검사해도 역사적 판단의 정확성을 보증하지는 않는다.

Migration 177의 `commulingo_person_review_jobs`가 작업 상태·근거·판단·오류를 보관한다.
SKIP LOCKED와 20분 lease로 중복 실행을 막는다. 실행 제한은 650초, 조사 제한은 480초다.
실패는 1시간 후 재시도하고 3회째 소유자에게 넘긴다. 저장된 판단은 장애 복구 시 재사용하며
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
판단 불가 알림은 ALLOWED_USER_IDS에 지정된 단일 소유자에게만 보낸다. 전송 실패는
다음 타이머 실행에서 재시도하고, 미처리 요청은 하루 뒤 재알림한다. 전송 성공 직후
프로세스가 죽으면 알림이 중복될 수 있지만 승인 내용이 중복 반영되지는 않는다.

한 번에 최대 한 제안을 검토하며 기본 LLM 예산은 $0.20, 최대 12라운드다.
기존 CommuLingo 일일 합산 예산($2.5)에 포함된다. 예산 소진 시 조사는 연기하지만
운영자 알림 확인은 계속한다. 상태 보고에 review lane을 포함하며 빈 큐는 정상 idle로 집계한다.

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
