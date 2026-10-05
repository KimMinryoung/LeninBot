# CommuLingo 보강 세션 (leninbot 쪽)

최종 확인: 2026-10-05 코드 트리.

CommuLingo 보강 파이프라인(작업 대기열, 후보 선정, 예산, 검증, 공개)은 2026-10-05부터 frontend 저장소가 운영한다(frontend `dev_docs/commulingo-agent-pipeline.md`, `services/commulingo-pipeline/`, `scripts/commulingo-pipeline`). leninbot은 그 파이프라인이 맡기는 **에이전트 세션**만 실행하는 일꾼이다([agent_worker.md](agent_worker.md)). 대기열·planner·엔진·레인·운영 일괄 스크립트는 삭제했다. 이전 운영 기록과 결정 이력은 git 기록(`git log -- dev_docs/commulingo_pipeline.md`)에 있다.

## 남은 것

| 모듈 | 역할 |
|---|---|
| `worker/commulingo.py` | 일꾼 작업 종류 `commulingo_editor`(research·draft), `commulingo_review`, `commulingo_discover`. 입력 `{job, artifacts, settings}`, 출력 `{stage, artifacts, notes, usage, costComplete}` |
| `commulingo/pipeline/editor.py` | 저자 세션: 원문 열람(P 문단), 나눠 제출 저장, 초안 수리, 인용 지지 검사, Jev 분류, 체크포인트 |
| `commulingo/pipeline/workflow.py` | 독립 검토 세션(`Review`) |
| `commulingo/pipeline/stages.py` | `model_call`, `Discover`, 공용 단계 헬퍼 |
| `commulingo/pipeline/store.py` | 출처 캐시(`commulingo_pipeline_sources`, `fetch_cache`, `job_sources`): leninbot 조사 인프라. 만료 본문은 `leninbot-worker`가 매시간 지운다 |
| `commulingo/pipeline/{author_draft,evidence,citation_gate,decisions,issues,patches,…}.py` | 세션이 쓰는 검증·근거·패치 규칙 |
| `agents/commulingo_curator.py`, `agents/commulingo_reviewer.py` | 저자·검토자 지시문과 모델 설정 |

세션은 CommuLingo를 관리자 MCP로만 읽고(`person_get`, `term_get`, `entry_lookup` 등), 초안은 `editorial_pipeline validate`로 검사하며, 쓰지 않는다. 판정 기록·검증·공개·검토 메모는 frontend 단계가 한다.

## 운영

- 일꾼: `journalctl -u leninbot-worker`. 대기열과 단계 진행은 frontend `scripts/commulingo-pipeline list|show|run|costs`.
- 정기 실행 여부와 일일 상한은 frontend `data/commulingo/pipeline-config.json`(`enabled`, `daily_cap_usd`, `stage_budget_usd`)이 정한다.
- 삭제한 leninbot 유닛(`leninbot-commulingo-{pipeline,maintainer,new,enrich,terms,gap,review,health,events,links,batch}`, `leninbot-event-backfill`, `leninbot-variant-scan`)은 저장소에서 지웠다. `/etc/systemd/system`에 남은 설치본은 비활성이며 root로 지운다.
- 대량 데이터 작업(백필, 표기 정규화, 감사)은 frontend 저장소에서 개발 도구로 직접 한다.
