# CommuLingo 영속 편집 파이프라인

인물·용어의 등록과 보강은 `commulingo_pipeline/`이 맡는다. 진입점은
`scripts/commulingo_pipeline.py`, 운영 설정은 `config/commulingo_pipeline.json`이다.
저장 도구의 계약은 [편집 서비스](commulingo_editorial.md), 판정 정책은 [Jev 연동](jev_system_one_adoption.md)을 따른다.

## 운영 경로와 승인 범위

운영 설정은 `workflow=editor`, `phase=live`다. 코드의 workflow 미지정 기본값인 `legacy`와 구분한다.
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

### 원문 캐시와 문단 라벨

`source_session.py`는 기존 PostgreSQL 원문 캐시와 job_sources를 재사용한다.
페이지를 이어 붙이거나 별도의 합본 원문을 만들지 않는다. `evidence.py`의 `Passages`는
스냅샷과 고정 문단 범위에 불변 `P1` 같은 라벨을 부여한다. 작성기는 라벨을 인용하고
코드가 원문 위치·인용문으로 변환한다. 존재하지 않는 라벨은 오류이며 주장을 몰래 삭제하지 않는다.

캐시 재사용은 최초 조회 시각과 만료를 보존한다. 체크포인트에서 만료된 자료를 제외해도
그 라벨을 새 자료에 재배정하지 않는다. 독립 검토자는 작성기의 원문 캐시 대신 직접 자료를 가져온다.

### 부분 수정과 체크포인트

| 세션 내부 도구 | 역할 |
|---|---|
| `commulingo_pipeline_result` | 전체 수정안 제출 |
| `commulingo_pipeline_repair` | JSON pointer의 set/remove로 부분 수정, 배열은 `/-`나 다음 인덱스 set으로 추가; 검증 성공 시 종료 |
| `commulingo_pipeline_cached_passages` | 기존 P 라벨의 유효한 원문을 네트워크 없이 조회 |
| `commulingo_pipeline_context` | 추가 필드 schema·현재 값 조회 |
| `commulingo_pipeline_research` | 필드와 이유를 명시해 사실 조사 재개 |

도구는 세션에만 주입하며 소유자·commulingo_curator 호출자 권한을 검사한다.
초안 접수는 잘못된 중첩 데이터도 먼저 보존하고 실제 schema로 검증한다.
오류는 해당 JSON pointer와 규칙을 반환하므로 정상 필드와 근거를 유지한 채 고칠 수 있다.
수정 요청 자체의 형식 오류는 내용 무진전 횟수에 넣지 않는다.

최초 문맥은 과제에 필요한 필드로 좁히되 저자 notes·인물 sections·original_proposal은 포함한다.
공통 WRITING_RULES를 적용하고 문장 분량은 상한의 80%를 여유 목표로 제시한다.
fields.notes는 공개 필드에서 제거해 비공개 notes와 중복 없이 합친다.
형식 수정 중에는 기존 근거를 사용하고 사전 항목 조회를 제한한다. 누락 근거나 사실 충돌은
해당 필드의 조사를 재개할 수 있다. 소개문·정의는 실제 저장 형식인 문자열로 제출한다.

`editor_checkpoint`는 초안·P 라벨·조회 인자·오류·분류 캐시와 수정 상태를 저장한다.
같은 baseline이면 재시작 후 복구하며 체크포인트 쓰기도 lease로 보호한다.
같은 수정안과 오류의 반복은 내부 보류한다. revision 충돌은 최신 상태의 조사로 돌려보낸다.

## 독립 검토와 공개 반영

`workflow.py`는 현재 문서와의 차이, 이전 수정안과의 차이 및 이전 검토를 제공한다.
검토자는 원문을 직접 가져와 핵심 변경 사실과 위험 항목을 확인한다.
`required_corrections`는 사실 오류의 필드 위치와 이유를 담으며 `optional_suggestions`와 구분한다.
선택 제안만으로 revise할 수 없고 내용·근거가 그대로인 거절안은 유료 재검토 전에 보류한다.
Jev는 주장과 인용, 검토 finding과 인용의 지지 관계를 판정한다. 최종 서술과 사실 검토는
작성자·독립 검토자가 담당한다. 인용 게이트 장애 정책은 분류 장애 정책과 다르다.

승인은 target/action/id/fields/sources의 정규화 SHA-256에 묶이며 baseline revision도 포함한다.
승인 후 수정안이 달라지면 다시 검토한다. frontend private RPC의 `publish`는 해시와 revision을
검증하고 submit·review·원 제안 대체·메모·영수증을 한 트랜잭션으로 반영한다.
실패하면 모두 롤백하며 동일 요청 재실행에는 기존 영수증을 반환한다. 옛 두 RPC 방식으로 우회하지 않는다.
유료 작업 전 `capabilities.atomicPublish`를 확인한다.

기존 pending 제안의 revise는 원 제안과 근거를 보존하는 수정 작업으로 이어진다.
수정안이 승인되기 전에는 원 제안을 대체하지 않으며 수동 처리된 원 제안을 덮어쓰지 않는다.

## 대기열·예산·복구

`Store`와 engine은 PostgreSQL 대기열, SKIP LOCKED lease, heartbeat, fencing을 공유한다.
lease를 잃은 작업자는 결과를 저장하지 못한다. 실패는 제한된 횟수만 재시도하고 초과하면 escalated로 남긴다.
`retry`는 deferred/escalated 작업을 재개한다. editor의 완료 단계 보류도 조사 단계로 복구한다.

예산은 예약과 실제 사용 원장을 기준으로 한다. 비용 미확정 요청의 예약은 임의로 해제하지 않는다.
일일 한도·검토 몫·단계 예약액은 운영 설정을 따르며 `legacy_shared_budget=true`로 기존 lane과 공유한다.
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

## Legacy 호환과 배포 경계

작업에 고정된 `payload.workflow=editor`는 이후 실행자의 기본 설정보다 우선하며 파생 등록 작업도 계승한다.
기존 미지정 작업은 research/draft/discover 경계에서 전환하고 validate/review/submit/judge 중인 작업은
기존 묶음을 완료하여 저장·검토 영수증을 보존한다. 기존 작성 스크립트와 handler는 이 호환 경로를 위해 남는다.

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

캐시 조회 도구는 기존 `passages` 또는 `source_id` 중 하나를 받는다.
라벨 없는 저장 페이지는 `source_cache.available_pages`의 `source_id`로 열어 라벨을 얻는다.
원문 시각과 만료는 유지하며 할당된 라벨은 초안이 아직 없어도 체크포인트에 보존한다.
잘못된 라벨은 대체하거나 무시하지 않고 실제 라벨과 복구 경로를 안내한다.

단계 JSON은 기존 필드를 유지하면서 `completed_stage`와 `disposition`을 추가한다.
`published`는 승인된 submit, `no_edit`는 공개 저장 없이 종료한 judge,
`failed`는 실행 실패, `budget_wait`는 예산 대기다. 그 외는 progress/held/deferred로 구분한다.
정리 결과는 별도 journal 로그로 기록한다. 일일 health 집계는 승인 기록 수와
고유 반영 작업 수, 마지막 공개 반영 시각, 무편집 완료·취소·실패·현재 예산 대기를 분리한다.

정리 회귀 검사는 확장 smoke에 포함된다. 실제 SQL·동시 쓰기 검사는
`COMMULINGO_CLEANUP_TEST_PORT=<임시 PostgreSQL 포트> venv/bin/python -m unittest discover -s tests -p test_commulingo_cleanup_db.py`
로 실행한다. DB명은 `commulingo_integrity_test`이며 전용 임시 schema를 생성·삭제한다.
운영 DB를 사용하지 않는다.
