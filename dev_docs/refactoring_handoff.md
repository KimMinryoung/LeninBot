# 리팩터링 인수인계: 남은 항목

2026-09-24 코드 리뷰 겸 리팩터링에서 찾아낸 항목 가운데 아직 처리하지 않은 것만 모은다. 첫 작업(브랜치 `claude/project-code-refactoring-t3byuo`)은 운영 접근이 없는 컨테이너에서 했고, 같은 날 운영 서버에서 이어서 처리했다(브랜치 `refactor/handoff-followups`). 처리한 항목은 이 파일에서 지웠고, 설계상 알아둘 내용은 해당 주제 문서로 옮겼다.

[문서 유지 원칙](README.md#문서-유지-원칙)대로 이 파일은 임시 목록이다. 항목을 처리하면 여기서 지우고, 목록이 비면 파일을 지운다. 줄 번호는 적지 않는다. 작업 전에 코드에서 다시 찾는다.

## A. 결정이 필요한 항목

### A7. legacy stage 삭제로 드러난 editor의 빠진 동작

2026-09-24에 legacy stage(`stages.Research/Draft/Review`, legacy `validate`/`submit`)를 지우고 테스트를 editor 경로로 옮기는 과정에서 두 가지가 드러났다. 둘 다 2026-09-20 editor 전환 때부터 운영에 없던 동작이고, 자세한 내용은 `commulingo_pipeline.md`에 있다.

- **검토 판정 노트:** legacy `Review`는 escalate/reject로 끝나면 사전 항목에 `note` RPC로 사유를 남겼다. editor는 판정을 작업의 review artifact에만 남긴다. editor의 `reject`는 `escalated`로 끝나고, legacy는 `complete`로 끝났다. 다음 작성자에게 사유를 보여 줄지 정한다.
- **이미 수동 처리된 원본을 교정하는 작업:** legacy `submit`은 원본 상태를 먼저 확인하고 조용히 끝냈다. editor는 `replacesSuggestionId`를 발행 RPC 하나에 담아 보낸다. frontend 서비스가 트랜잭션 안에서 거부하므로 원본은 바뀌지 않지만, 작업은 오류로 재시도·escalate 경로를 탄다. 조용히 완료할지 정한다.

### C4. `self_modification_core.py`

자동 편집이 권한 분류기에 막혀 적용하지 못했다. 봇의 자가수정 안전 모듈이라 사람이 직접 보고 적용한다.

- `request_user_approval`(`input()` 대기)과 `self_modify_with_safety`의 승인 단계는 도달할 수 없다. 유일한 호출부인 Telegram `/modify` 콜백은 `request_approval=False`로 부른다. 이 둘을 지우고, `request_approval=True`면 `ValueError`를 내게 한다.
- `git_backup_before_modification`/`git_reset_to_commit`은 git을 쓰지 않는다. `.bak.TIMESTAMP` 파일을 복사하고 복원한다. 이름을 `backup_file`/`restore_backup`으로 바꾸고, 옛 이름은 별칭으로 남긴다. `runtime_tools/filesystem.py`가 옛 이름을 import한다.
- 모듈을 `self_runtime/`으로 옮기고 루트에는 shim을 남긴다. 로그 경로는 `__file__` 기준이므로 `ops.paths.PROJECT_ROOT / "logs"`로 고정한다.

## B. 남은 분할

B 목록의 대형 함수 10개 가운데 9개를 분할했다. 모두 로직을 그대로 옮긴 분할이고, 분할 전후 호출 순서와 출력을 비교 스크립트로 대조했다. 남은 것:

- `scripts/commulingo_people_maintainer.py` `run_once`(약 240줄): 아직 분할하지 않았다.
- 분할 뒤에도 긴 함수:
  - `_validate_person`(약 250줄), `_validate_term`(약 155줄): 더 나누려면 검사 순서를 재배치해야 해서 그대로 두었다.
  - `self_runtime/tools.py` `_read_autonomous_project_detail`(약 220줄): 조회와 렌더링을 나누면 advisory 조회의 DB 호출 순서가 바뀐다. 거의 같은 "type X의 최신 이벤트" 쿼리 세 개도 합치지 않았다. 합치면 로직이 바뀐다.

## C. 구조와 레이어

### C1. CommuLingo 레인 공용 코드

레인 이름을 모듈 전역 덮어쓰기로 전달하던 문제는 해결했다. maintainer가 `COMMULINGO_SUGGESTED_BY`에서 레인 이름을 읽는다. 남은 것:

- `commulingo_people_parallel.py`, `commulingo_terms_maintainer.py`, `commulingo_gap_worker.py`는 여전히 `scripts/commulingo_people_maintainer.py`를 라이브러리로 import한다. 공용 헬퍼를 `runtime_tools/commulingo_lane.py` 같은 모듈로 옮기면 의존 방향이 바로잡힌다.
- 노트 작성자 `changedBy`는 어느 레인에서 돌든 `"commulingo-maintainer"`로 고정돼 있다. 레인별로 나눌지는 따로 정한다(동작 변경).

### C3. 루트의 도메인 모듈

- **옮기지 않은 모듈:** `site_publishing.py`, `research_store.py`, `publication_records.py`, `task_store.py`, `redis_state.py`, `prompt_context.py`, `skills_loader.py`, `audit_sink.py`
- **이유:** 옮기는 이득에 비해 import 경로 변경이 넓다. 옮긴다면 한 번에 하나씩, 루트에 re-export shim을 남기며 옮긴다.
- **처리한 것:** `creative_writer.py` shim은 지웠다.

## D. 알고 유지하는 것

아래는 확인한 뒤 의도적으로 그대로 두었다. 바꾸려면 이 메모를 참고한다.

- `bot_config.ANTHROPIC_CLIENT_KEY`/`MOONSHOT_CLIENT_KEY`: 쓰는 곳이 없지만 OPENAI/DEEPSEEK 쪽과 모양을 맞추려고 둔다.
- `bot_config.set_gateway_enforce_mode`: 코드에서 부르는 곳은 없지만 `security_gateway.md`에 운영 진입점으로 적혀 있다.
- `_slice_text`(`self_runtime/tools.py`, `mcp_gateway/tools.py`): offset이 길이를 넘을 때 반환하는 시작 위치가 다르다. 앞의 것은 길이로 자르고, 뒤의 것은 그대로 돌려준다. 합치면 한쪽 출력이 바뀐다.
- `KST` 재정의: 런타임 모듈은 `shared.KST`를 쓴다. standby에서 따로 도는 백업·복원 스크립트(`scripts/backup_*_to_r2.py`, `scripts/restore_db.py`)는 자체 정의를 둔다.
- 운영 스크립트의 `/home/grass/leninbot` 하드코딩: `scripts/*.sh`, `scripts/metrics_*.py`, 스모크 일부, `skills/kg-maintenance/scripts/*`. 서버 전용이라 둔다.

## E. 테스트 기준선 (2026-09-24, 운영 서버)

- `scripts/run_unit_tests.sh`: 1223개 OK, skip 23 (legacy stage 테스트 정리 후). 운영 서버에는 CommuLingo fixture가 있어 전부 로드된다.
- 에이전트 worktree가 저장소 안 `.claude/worktrees/`에 있는 동안에는 `test_llm_gateway_conformance`가 그 사본의 SDK 생성 코드를 잡아 실패한다. worktree를 지우면 통과한다.
- 리팩터링 전부터 실패하던 스모크(원인 미조사):
  - `smoke_commulingo_person_nationality_backfill`: audit coverage 실패
  - `smoke_deepseek_autonomous_harness`: `bot_config.py`에서 찾는 소스 문자열이 없음
  - `smoke_llm_routing_search`: 키워드 추측 라우팅 판정
  - `smoke_runtime`: `_assert_prompt_context`의 `format_subtask_results` 기대값 불일치, `_assert_read_self_autonomous_project_uses_note_table`
