# CommuLingo 영속 편집 파이프라인

`commulingo_pipeline/`는 인물·용어의 신규 등록과 보강을 단계별 작업으로 실행한다.
진입점은 `scripts/commulingo_pipeline.py`다. 모듈 import나 조회 명령은 DDL이나
LLM 호출을 하지 않는다. 운영은 `phase=canary`, `legacy_shared_budget=true`, `term_editorial_service=true`다.
5분 타이머가 한 단계를 재개하며 작업군별 UTC 하루 한 건만 반영한다.
기존 작성 스크립트는 롤백을 위해 유지한다. `leninbot-commulingo-batch`는 사건·연결만
실행하고 과거 제안의 review 타이머도 유지한다.

사건 작업은 `knowledge_graph_search`를 사용하므로 `leninbot-commulingo-events.service`에
`neo4j_password` credential이 필요하다. 실행 CLI는 이 값이 없으면 LLM 호출 전에
exit 1로 중단한다(`--print-candidate` 조회는 제외). 유닛의 credential 변경은 설치와
daemon-reload 후 다음 타이머 실행부터 적용된다.

## 대기열과 선정

명시적 migration `scripts/schema_migrations.py --only commulingo-pipeline`이
PostgreSQL 작업·근거·artifact·비용·시범 반영 테이블을 생성한다.
작업은 kind, action, target, topic, baseline, reason, priority와 payload를 가진다.
같은 대상·주제의 활성 작업은 하나만 허용한다. 조사 시작 시 실제 저장소 revision을 읽고
research artifact에 고정한다. 계획의 baseline은 변경 감지용 타임스탬프이며 저장 권한 토큰이 아니다.

인물은 기본 정보·근거·한영 누락·국적·일화·상세 절, 용어는 정의·역사적 맥락·유사 개념
구별·사례·관계를 선정한다. 12개 절은 상한이며 분량은 품질 점수가 아니다.
운영자가 요청한 gap을 우선하고 네 작업군을 순환 배정한다. 검토·반영 준비 작업은 먼저
처리한다. frontend의 유효한 완료 주제와 대기 제안을 존중한다.

공개 연구 문서·사건·인물·미해결 gap의 내용 hash를 비교해 발견 작업을 만든다.
발견기는 실제 원문 언급을 제시하고 기존 이름·별칭을 검색한다. 한 문서에서 최대 네 후보,
명시적인 gap에서는 요청한 종류·한국어 이름과 일치하는 한 후보만 허용한다.
후보 생성과 처리한 문서 hash는 같은 트랜잭션에서 저장한다. 서로 다른 문서의 언급 수는
신규 후보 사이의 우선순위에만 사용하며, 역사적 중요성이나 출처 독립성을 증명하지 않는다.

## 단계와 복구

`discover → research → draft → validate → review → submit → complete` 중 필요한
단계만 실행한다. 편집 불필요 판단은 research 결과를 먼저 저장한 뒤 별도 judge 단계가
동일한 판단으로 보강 상태를 기록한다. 각 단계 완료 결과는 별도 artifact다. 120초 lease를 30초마다 갱신하고
단계는 480초로 제한한다. 소유권을 잃으면 실행을 취소하며 예전 lease의 결과 저장을 거부한다.
실패는 1시간 뒤 해당 단계부터 재시도하고 같은 단계의 3회 실패는 escalated로 남긴다.

조사기는 원문과 field별 주장·source ID·정확한 문자 범위를 제출한다. 작성기는 제한된
초안 도구와 사전 조회만 받는다. 실행기가 20~6000자의 원문 인용과 기준 revision을 붙인다.
작성 모델에는 evidence/revision을 수정하는 인자가 없다. 인물 상세 절도 같은 경로를 사용한다.
형식 오류는 기존 조사로 최대 두 번 수정한다. 필드별 근거 누락은 이전 주장·검증 오류를
조사기에 전달해 research로 돌리며, 같은 근거 실패가 세 번 누적되면 운영자 확인으로 남긴다.
예산 부족·초안 모드 대기는 실패 재시도 횟수에 포함하지 않는다. 충돌은 새 조사 단계로 돌리며 토큰을 자동 대입하지 않는다.

`sources_unavailable`은 조사 결과를 보존하고 90일 연기한다. complete/not_applicable은
관련 내용이 바뀌지 않으면 180일 동안 재선정을 억제한다. 기존 인물·용어는 공통 저장 서비스에도
그 판단을 기록한다. 글을 늘릴 필요가 없다는 판단은 정상 결과다.

원문은 URL·내용 SHA-256·취득/만료 시각으로 저장한다. authoring 캐시는 도구 이름과
정확한 인자 hash가 같을 때만 공유하며 캐시 적중은 취득 시각을 갱신하지 않는다.
14일 뒤 본문을 지우되 식별자·URL·hash를 보존한다. 저장된 인용은 원문 전체 캐시와 별개다.
독립 검토는 이 캐시를 사용하지 않고 기존 검토 도구로 직접 원문을 가져온다.

모든 새 편집은 검증과 독립 검토를 통과해야 반영된다. 실제 제출과 승인은 별개의
idempotency key를 쓰며 frontend 트랜잭션에 영수증을 저장한다. 프로세스가 저장 직후
죽어도 같은 결과를 되찾는다. 판단 불가 초안은 pending 제안과 기존 운영자 검토 대기열에
연결한다. 다음 tick은 운영자 승인·반려를 작업 상태와 gap 완료 여부에 반영한다.

## 저장 서비스와 호환성

frontend migration 179와 `commulingo-pipeline-service.js`가 필요하다. 공개 HTTP API는
추가하지 않는다. RPC는 read/validate/submit/review/enrichment이며 submit/review/enrichment에
idempotencyKey를 요구한다. validate는 SAVEPOINT 롤백으로 실제 저장 검증을 실행한다.
서비스가 없거나 실패하면 Python SQL로 우회하지 않는다.

용어 서비스는 field별 evidence, 전체 행·별칭·연결의 revision, 한영 부분 병합,
중복 별칭·분류·상위 용어 검증, 제안·승인·보강 상태를 소유한다. 기존 자료의 근거를
소급 생성하지 않는다. `term_editorial_service=true` 이후 기존 Python 용어 도구도 이 서비스로
연결된다. 이전 버전 없는 용어 제안은 기존 CLI 경로를 유지하며 새 자동 검토에 포함하지 않는다.
새 evidence가 있는 용어 제안은 기존 `/commulingo_review` 명령에서 처리한다.

## 예산과 운영 명령

`config/commulingo_pipeline.json`의 기본 일일 예산은 $3.39, 단계 예약은 $0.20,
검토 확보 비율은 30%다. UTC 호출 시작일 기준으로 PostgreSQL advisory lock 안에서
예약하고 응답 후 정산한다. 예약을 초과한 실제 비용은 overrun으로 보이며 후속 호출을 막는다.
응답 비용이 불명확한 실패는 예약액을 유지한다. 과거 날짜의 미정산분도 costs에서 확인할 수 있다.
예약만으로 외부 제공자의 단일 요청 비용 초과까지 차단하지는 못한다.

`legacy_shared_budget=true`이면 기존 사람·gap·사건의 공용 실행기, 검토기, 연결 생성기도
같은 예산을 예약한다. 배포 전 운영 장애를 막기 위해 기본값은 false이며 새 유료 실행 명령도
이 값이 false이면 거부한다. 기존 ledger 비용과 이 테이블 비용을 중복 합산하지 않는다.

```bash
venv/bin/python scripts/commulingo_pipeline.py plan
venv/bin/python scripts/commulingo_pipeline.py plan --apply
venv/bin/python scripts/commulingo_pipeline.py list
venv/bin/python scripts/commulingo_pipeline.py show 123
venv/bin/python scripts/commulingo_pipeline.py costs
venv/bin/python scripts/commulingo_pipeline.py metrics
venv/bin/python scripts/commulingo_pipeline.py run --limit 1
venv/bin/python scripts/commulingo_pipeline.py run --review --limit 4
venv/bin/python scripts/commulingo_pipeline.py retry 123
venv/bin/python scripts/commulingo_pipeline.py run --job-id 123 --publish --limit 4
```

run은 기본적으로 초안까지만 진행한다. `run --review`는 독립 검토까지 허용하되
승인 결과도 submit 앞에서 멈춘다. 판단 불가 초안은 운영자 pending 제안으로 남을 수 있다. --publish는 phase=canary/live에서만 가능하다.
`--job-id`는 지정한 작업만 정상 lease·예산·검토 제한 아래 재개한다.
tick은 변경분 계획·완료 검토 정리·원문 만료·한 단계 실행을 수행하고 phase를 따른다.
canary는 네 작업군별 UTC 하루 한 작업의 반영만 허용한다. 실패·장애 후 같은 작업은
원래 반영 슬롯을 재사용한다. 실패한 작업의 슬롯을 자동 회수하지 않는다.
metrics는 상태·단계별 형식 오류·기록된 비용·캐시 적중, costs는 실제 비용·미정산 예약·초과를 보여준다.

## 검증과 전환

단위/통합 검사: `tests/test_commulingo_pipeline.py`.
`COMMULINGO_PIPELINE_TEST_PORT`를 설정한 경우에만 localhost의
`commulingo_integrity_test` DB를 사용한다. lease 경쟁, 오래된 소유자의 저장, 동시 예산 예약,
영수증 키, 근거 범위·만료, 작성 revision 바인딩, 문서 hash, 발견 결과의 원자성, canary 제한을 검사한다.
frontend의 `test-commulingo-pipeline-db.js`는 별도 DB 가드 아래 실제 저장·검토·롤백을 검사한다.

전환 순서:

1. 기존 작업을 종료시키고 작성·검토 타이머를 정지한다. frontend 179와 Python migration을 적용하고
   검증한 frontend RPC 파일을 배포한다. 기존 자료·제안·검토 결과는 보존한다.
2. 새 UTC 날짜부터 shared budget을 활성화한다. 이전 실행 비용은 자동 소급 수집하지 않으므로
   같은 날 전환하려면 모든 기존 레인의 유휴 상태를 확인하고, UTC 자정부터 중단 시점까지
   journal의 결과 비용과 중복 없는 loop 감사 비용 중 큰 값을 빈 원장에 한 번만 이관한다.
   전환 후 journal을 다시 가져오면 이중 집계되므로 재이관하지 않는다. 사건·연결·기존 검토도 새 예산에 참여시킨다.
3. term_editorial_service를 활성화하고 phase=draft로 고정 사례를 평가한다. 네 작업군의 동일
   입력·출처 조건을 고정하고 정확성·근거 적합성·중요 누락 해소·한영 일치를 기존 결과와 비교한다.
   최초 형식 통과 95%는 목표이며 단위 테스트 통과로 달성했다고 간주하지 않는다.
4. 평가가 기존 이상이면 canary로 바꾸고 새 service/timer를 설치한다. 인물·용어·gap의 기존
   작성 타이머는 중복 실행하지 않는다. 사건·연결과 과거 제안 검토는 유지한다.
5. 중복 반영·revision 우회·예산 누락이 없고 품질 조건을 충족하면 live로 전환한다.

롤백은 새 타이머를 멈추고 진행 작업을 종료한 뒤 기존 작성기를 복구한다. 새 스키마와
artifact를 삭제하지 않는다. 아직 frontend/Python 코드를 되돌리지 않았다면 용어의 새 저장·검토
경계는 유지한다. 새 스키마가 필요한 서비스를 이전 파일과 섞어 실행하지 않는다.

건강 보고서 `scripts/commulingo_lane_health.py`는 과거 레인 비용을 보존하면서 새 파이프라인의
반영·실행·재시도·보류와 UTC 당일 공용 비용·예약을 조회한다. 신규 작업 비용만 기존 journal
합계에 더하며 공용 원장의 기존 레인 비용을 중복 합산하지 않는다. 중단된 인물 작성 레인의
무실행은 장애로 보지 않는다.

2026-09-09 전환 시 기존 당일 비용 $0.071381을 유휴 상태에서 한 번 이관했다.
초기 네 사례의 첫 형식 통과는 3/4였으며 95% 목표 달성이나 기존 대비 품질 우위는 아직
입증하지 않았다. 독립 검토를 통과한 건만 시범 반영하며 전체 live 확대는 추가 표본 평가 후 결정한다.

DeepSeek 모델 선택은 큐레이터·이벤트 큐레이터·리뷰어의 스펙과 runtime overlay에서 `deepseek_flash`로 통일한다. 공통 provider registry가 정식 `deepseek-flash`로 해석하며, 타이머 작업은 다음 프로세스 시작부터 읽는다. 옛 `deepseek_pro`는 저장값 호환용으로만 유지한다.
