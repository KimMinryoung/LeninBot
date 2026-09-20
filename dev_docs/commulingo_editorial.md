# CommuLingo 편집 저장 계약

자동 인물·용어 보강의 실행·복구·예산·공개 정책은 [파이프라인](commulingo_pipeline.md)을 따른다.
이 문서는 공통 저장 서비스와 직접 편집 도구의 계약을 설명한다.

## 저장 소유권과 RPC

인물·상세 절 저장은 frontend `data/commulingo/person-editorial-service.js`가 소유한다.
Python `runtime_tools/commulingo_people.py`와 제안 승인 스크립트는
`runtime_tools/commulingo_person_service.py`를 통해 같은 JS Admin 저장소를 호출한다.
직접 SQL fallback은 없다. 파이프라인의 용어·원자적 공개는 frontend
`editorial-pipeline-service.js`가 담당한다. 편집 쓰기는 공통 advisory lock으로 직렬화한다.

인물 RPC는 고정 argv `docker exec -i leninbot-frontend node /app/scripts/commulingo-person-service.js`와
JSON stdin/stdout을 사용한다. 요청 본문·DB 비밀번호는 명령행에 넣지 않는다.
frontend 컨테이너 DB 설정을 사용하며 공개 쓰기 endpoint를 추가하지 않는다.
서비스 계정에 Docker 접근 권한이 필요하고 RPC 장애 시 쓰기는 실패한다.
`COMMULINGO_FRONTEND_CONTAINER`는 격리 테스트용 컨테이너를 선택한다.

## Revision·필드·근거

get_person은 같은 DB 스냅샷의 revision·인물·절 요약·근거·보강 상태를 반환한다.
KG 참고 정보는 best effort다. 기존 인물 수정은 fields.expectedRevision,
section_save는 최상위 expected_revision이 필수다. 충돌한 revision을 자동 갱신하지 않는다.
신규 등록은 aliases·career·scenes로 초기 목록을 작성하며 aliasEdits·careerEdits·sceneEdits는 수정 전용이다.

모든 쓰기에 sources가 필요하고 사실 필드에는 필드별 evidence가 필요하다.
정확한 사실 필드·분량·검증 규칙은 frontend `person-editorial-contract.json`이 원본이다.
Python도 같은 파일을 읽으며 테스트에서는 `COMMULINGO_PERSON_CONTRACT`로 바꿀 수 있다.
미확정 fate는 빈 kind와 한영 유보 라벨을 사용한다.

인물 도구의 evidence·expectedRevision·reviewFlags는 fields 안에, citations는 최상위에 둔다.
절 저장은 evidence·expected_revision을 최상위로 받는다. evidence.source_id의 S1은 citations[0],
S2는 citations[1]이다. Python은 이를 원래 출처 문자열로 변환하며 source와 충돌하면 거절한다.
evidence.source를 직접 쓰면 설명을 포함한 citations 항목 전체와 일치해야 한다.
이 요청별 S 라벨은 파이프라인의 불변 원문 문단 P 라벨과 별개다.

RPC 전 진단은 누락 근거·출처 불일치·변경하지 않은 필드의 근거를 함께 알려준다.
근거를 추론해 채우거나 revision을 갱신하지 않으며 JS 저장소의 최종 검증도 적용한다.
근거와 검토 checks에 수집 할당량은 없다. 각 항목의 원문·필드 지지 여부는 계속 검증한다.

## 직접 도구의 제안과 검토

직접 편집 경로의 direct_apply=true도 출처 충돌·동일인 불명·삭제·대규모 축약은 pending으로 끝난다.
false이면 모든 인물 쓰기가 pending이다. pending은 정상 종료이며 실패로 재시도하지 않는다.
수동 제안 approve는 note와 최초 revision 검사가 필요하다. 버전 없는 과거 제안은 최신 조사로 다시 제출한다.
confidence만으로 승인하지 않는다.

`agents/commulingo_reviewer.py`는 작성자 문맥을 물려받지 않는 전용 AgentSpec이다.
검토자는 조사와 내부 `commulingo_review_decision`만 사용하며 저장·승인 API를 직접 호출하지 않는다.
이 도구는 소유자와 commulingo_reviewer 호출자만 허용한다.
별도 review.timer는 비활성이고 자동 editor 검토는 파이프라인 내부에서 수행한다.

검토 checks는 citation_id(S1=source_refs[0]), 직접 조회한 P 문단 라벨, finding을 담는다.
check당 1~8개 라벨을 실제 citation/source/quote/finding으로 변환한다.
미등록 라벨은 해당 check의 수정 요청으로 반환하며 조용히 삭제하지 않는다.
작성자가 제공한 인용이나 검색 요약만으로 승인하지 않는다. 핵심 변경 사실과 검토 위험의 해소가 필요하다.
모든 출처를 하나씩 열거나 위키백과 밖 출처를 의무적으로 추가하지는 않는다.
자료 접근 불가·미해결 동일인은 내부 보류한다.

Telegram 수동 명령은 `/commulingo_review list|show|approve|reject|retry`이며 승인·반려에 사유가 필요하다.
자동 검토 요청·재알림은 보내지 않는다. `--notify-only`는 호환용 notifications_disabled를 반환한다.
기존 pending 제안의 수정·대체는 [원자적 공개 계약](commulingo_pipeline.md#독립-검토와-공개-반영)을 따른다.

## 보강 상태와 분류

직접 도구의 no_edit는 주제별 reason/status/sources를 기록한다.
complete/not_applicable은 180일, sources_unavailable은 90일 후 재검토하며 새 근거가 주제를 다시 열 수 있다.
no_edit와 pending_review는 별도로 집계한다. 자동 과제 선정은 파이프라인 planner가 담당한다.
gap 요청은 pending 인물·절 제안이 있으면 중복 등록하지 않고 승인 후 실제 카드와 연결을 확인한다.

국가 코드의 원본은 frontend `modern-country-codes.json`과 역사·지역 코드 집합이다.
Python 허용 목록과 국기·지도 등록을 함께 검증한다. 출신 배경은 출생지나 활동지가 아니라
문헌의 민족·가계·자기인식으로 판단하며 혼합 배경은 한영 라벨과 근거에 명시한다.
Jev의 그룹·관직·국가·출신·사망 유형·용어 분류 기준은 [Jev 연동](jev_system_one_adoption.md#분류-기준과-감사)에 모은다.

`commulingo_backfill_nationality.py`와 `commulingo_backfill_person_nationality.py`는 기본적으로 보고서만 만든다.
검토된 최신 revision·출처·근거가 있는 Admin spec을 `--apply-spec`으로 전달하면
공통 Admin CLI가 배치 전체를 한 트랜잭션으로 검증·반영한다. 단독 --apply와 직접 SQL 반영은 지원하지 않는다.

## 배포와 검증

frontend의 호스트 마운트 변경은 운영 변경이다. 공통 저장 계약 변경 시 Python과 frontend의 schema를 함께 맞춘다.
Python을 장기 import하는 서비스는 재시작이 필요하고 스케줄 프로세스는 다음 실행부터 새 코드를 읽는다.
private RPC의 코드 반영 경계는 [파이프라인 배포](commulingo_pipeline.md#legacy-호환과-배포-경계)를 따른다.
롤백 때도 기존 근거·보강 상태·검토·대기열 테이블은 보존한다.

`tests/test_commulingo_evidence_diagnostics.py`는 DB 없이 제출 진단을 검사한다.
`scripts/smoke_commulingo_maintainer.py`는 폐기한 lane의 실행 정책 대신 현재 공통 작성·저장 계약을
검사한다. `--extended`의 editor 검증 범위와 격리 방식은 [파이프라인 검증](commulingo_pipeline.md#검증과-효율-지표)을 따른다.
`tests/test_commulingo_editorial_selection.py`는 기존 도구의 선택·완료·pending 종료를 검사한다.
`tests/test_commulingo_person_rpc.py`의 실제 저장·충돌·승인 검사는 격리 컨테이너와 독립 DB에서만 실행한다.
frontend `test-commulingo-editorial-db.js`는 근거·검토·롤백·상태 전이를 검사한다.
