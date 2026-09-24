# 리팩터링 인수인계: 진단했으나 적용하지 않은 항목

2026-09-24 코드 리뷰 겸 리팩터링(브랜치 `claude/project-code-refactoring-t3byuo`)에서 찾아냈지만 고치지 않은 항목을 모은다. 이 작업은 운영 DB·Redis·Telegram, 별도 저장소 `/home/grass/frontend`에 접근할 수 없는 클라우드 컨테이너에서 했다. 그래서 동작이 그대로인 변경만 적용했고, 동작이 바뀌거나 운영 상태를 확인해야 하는 항목은 여기 남긴다.

[문서 유지 원칙](README.md#문서-유지-원칙)대로 이 파일은 임시 목록이다. 항목을 처리하면 여기서 지우고, 설계상 알아둘 내용은 해당 주제 문서(`project_state.md`, `multi_agent_architecture.md`, `commulingo_pipeline.md` 등)로 옮긴다. 목록이 비면 파일을 지운다. 줄 번호는 위 브랜치의 마지막 커밋 기준이다. 작업 전에 코드에서 다시 확인한다.

## 이미 적용한 것 (참고)

- `llm/runtime_context.py`
  - `current_task_ctx`와 런타임 prelude 헬퍼를 여기로 옮겼다. 봇이 아닌 프로세스가 더는 `telegram.bot`를 import하지 않는다.
  - `telegram.bot`은 같은 객체를 예전 이름으로 다시 export한다.
- `telegram/bot.py`
  - `bot_main`의 중첩 함수 두 개를 모듈 수준으로 올렸다.
  - 실행 컨텍스트 생성은 `_owner_run_context`, provider 목록은 `CHAT_PROVIDERS`로 모았다.
- 흩어진 공용 로직을 하나씩으로 합쳤다.
  - `audit_sink.BatchedAuditWriter`: LLM 감사 큐와 도구 감사 큐의 공통 구현
  - 위임 헬퍼 (`self_runtime/tools.py`)
  - `recent_project_notes_with_total`: 자율 프로젝트 노트 조회
  - `READ_SELF_ALIASES`: read_self 별칭표
  - `_run_chat`: `handle_message`의 채팅 호출 세 곳
  - `_record_verification`: 검증 결과 기록
  - `task_store.load_task_metadata`: 태스크 metadata 파싱
- `shared.py`는 `KST`, `upload_to_r2`, `MODULE_ARCHITECTURE`만 남겼다. 참조가 없는 함수와 상수 약 20개도 지웠다.
- `ops/paths.py`: 체크아웃과 프런트엔드 경로를 한 곳에서 관리한다. 테스트의 `/home/grass/leninbot` 하드코딩도 없앴다.

## A. 동작이 바뀌어 결정이 필요한 항목

### A1. 레지스트리 스키마 정규화가 마지막에 실행되지 않음

- **위치:** `runtime_tools/registry.py:1587-1596`
- **증상:**
  - `_normalize_tool_schemas_inplace`의 주석은 "모든 TOOLS.append 뒤에 실행된다"고 말한다.
  - 실제로는 호출(1596) 뒤에 도구 4개가 더 추가된다(1599-1613): `PREPARE_MAIL_BRIEFING_TOOL`, `ROLEPLAY_MEMORY_TOOL`, `ROLEPLAY_STATE_TOOL`, `ROLEPLAY_PERSON_TOOL`.
  - `TOOLS = dedupe_tool_registry(TOOLS)`(1120)도 파일 중간에서 실행돼, 그 뒤에 붙는 약 25개 도구는 중복 제거 대상이 아니다.
  - `mail_runtime/inbox.py:218`의 `PREPARE_MAIL_BRIEFING_TOOL`에는 최상위 `additionalProperties`가 없다. 그래서 정규화되지 않고, OpenAI strict 모드 적용 대상인지가 다른 도구와 다르다.
- **제안:** dedupe와 정규화를 파일 끝으로 옮긴다.
- **검증:** OpenAI provider로 메일 브리핑 태스크와 역할극 도구 호출을 한 번씩 확인한다. 스키마가 바뀌므로 `scripts/smoke_tool_allowlists.py`도 돌린다.

### A2. `telegram_tasks`를 직접 INSERT하는 경로가 소유 함수를 우회함

- **소유 함수:** `task_store.create_task_in_db`(`multi_agent_architecture.md`). 이 함수는 priority를 정규화하고, 부모 태스크의 mission을 물려받고, 기본 agent를 처리한다.
- **우회하는 곳:**
  - `telegram/commands.py:563`: `/task`
  - `telegram/commands.py:1745`: `[CONTINUE_TASK:]` 자동 승격
  - `telegram/curate.py:321`: `/curate`
  - `telegram/tasks.py:2563`: schedule_worker
  - `telegram/tasks.py:2024`: 재시작 복구 시 자식 태스크 생성. restart 관련 컬럼이 많아 별도로 봐야 한다.
- **먼저 확인할 것:** DB 컬럼 기본값(`priority`, `available_at`, `status`)이 `create_task_in_db`가 넣는 값과 같은지 `scripts/query-db`로 확인한다. 다르면 위 경로의 동작이 바뀐다.

### A3. 모델 해석 로직이 세 곳에 있음

- **위치:**
  - `bot_config.py:640` `_get_model`
  - `bot_config.py:655` `_get_model_task`
  - `llm/runtime_profile.py:114-126`
- **차이:** 세 곳 모두 provider → resolver 분기가 같지만, 별칭을 검사하는 방식이 다르다.
  - `bot_config`는 `alias in _OPENAI_MODEL_MAP` 식으로 맵의 key를 본다. 여기에는 `gpt56` 같은 옛 별칭이 들어 있다.
  - `runtime_profile`은 `_TIER_MAP[provider].values()`를 본다.
- **제안:** `_get_model*`를 `(await resolve_runtime_profile("chat"|"task")).model_id`로 바꾼다.
- **선행 작업:** 저장된 config에 옛 별칭(`gpt56`, `gpt56terra`, `deepseek_pro` 등)이 남아 있을 때 결과가 같은지 테스트를 추가한다. `tests/test_deepseek_model_selection.py`가 출발점이다.

### A4. CommuLingo 파이프라인의 legacy stage

- **위치:**
  - `commulingo_pipeline/stages.py:1123-1127`의 `legacy` 묶음: `Research`(307), `Discover`(509), `Draft`(587), `Review`(987)
  - `Draft.__call__`만 약 320줄이다.
- **라우팅:** `workflow.py:171 routed_stages`는 `payload.workflow`가 없거나 `legacy`인 작업에만 legacy stage를 쓴다.
- **기본값 불일치:** `commulingo_pipeline/config.py:10`의 기본값은 `'legacy'`인데, `config/commulingo_pipeline.json:14`와 문서는 `editor`다.
- **지금 해도 되는 것:** `stages.Review`(991-1000)와 `workflow.py`(18-28)가 공유하는 revision 검사를 하나로 뽑는다.
- **삭제 조건:** 끝나지 않은 작업 중 `payload.workflow`가 비었거나 `legacy`인 것이 0건임을 `scripts/query-db`로 확인한 뒤 legacy 클래스를 지운다. 그다음 기본값을 `editor`로 맞춘다.

### A5. 자율 프로젝트 노트 이중 기록

- **증상:** `jobs/autonomous_project.py:585-600`은 노트를 `autonomous_project_notes` 테이블에 넣으면서 `autonomous_projects.research_notes` JSONB에도 계속 덧붙인다. 문서(`autonomous_project.md`)는 JSONB를 legacy fallback으로만 설명한다.
- **조건:** `scripts/schema_migrations.py --only autonomous-projects`로 테이블이 모든 환경에 있다고 보장된 뒤 JSONB 쓰기를 끊는다.
- **남은 읽기 경로:** `_recent_notes`(틱 프롬프트용)는 테이블이 비어 있으면 JSONB로 fallback하고 KST로 형식을 바꾼다. JSONB 쓰기를 끊을 때 이 경로도 같이 정리한다.

### A6. 그 밖의 작은 동작 변경

- **`register_handlers` ctx:** `telegram/bot.py`의 ctx dict에서 9개 키를 `telegram/`이 읽지 않는다. `current_datetime_str`, `format_current_model_context`, `format_system_alerts`, `clear_system_alert`, `deepseek_anthropic_client`, `build_skills_prompt`, `ALLOWED_USER_IDS`, `CLAUDE_MAX_TOKENS`, `email_approval_base_url`다. 다만 `_ctx`는 `telegram.curate.cmd_curate`와 `cmd_commulingo_review`에도 넘어가므로, 두 함수가 쓰는 키를 확인한 뒤 지운다. `commands.py`의 `_ctx.get(f"{provider}_client")`는 openai와 kimi에만 닿지만 `kimi_client`는 남겨야 한다.
- **명령 설명 중복:** `commands.py _HELP_TEXT`와 `bot.py`의 `BotCommand` 목록에 같은 설명이 두 번 있다. 표 하나에서 둘 다 만들도록 바꾼다.
- **`bot_config.py:73,76`:** `ANTHROPIC_CLIENT_KEY`와 `MOONSHOT_CLIENT_KEY`는 쓰는 곳이 없다. OPENAI/DEEPSEEK 쪽과 모양을 맞추려고 남겨 두었다.
- **`bot_config.py:553 set_gateway_enforce_mode`:** 코드에서 부르는 곳은 없지만 `security_gateway.md`에 운영 진입점으로 적혀 있다. 유지하려면 그대로 두고, 아니면 문서와 함께 지운다.
- **도구 감사 flush:** `security_gateway` 도구 감사 writer는 종료 시 flush하지 않는다(`BatchedAuditWriter(flush_at_exit=False)`). 수명이 짧은 프로세스에서는 감사 행이 빠질 수 있다. 켤지는 따로 정한다.

## B. 검증 수단이 부족해 미룬 대형 분할

모두 로직을 그대로 옮기는 분할이지만, 관련 테스트가 외부 데이터에 의존하거나 운영 경로 스모크가 필요해서 미뤘다.

| 함수 | 위치 | 줄 수 | 분할 방향 |
|---|---|---|---|
| `_validate` | `runtime_tools/commulingo_people.py:1663-2394` | 732 | 공통 전처리 뒤 `target_type`별 함수(person, person_section, history_event_person, history_event, history_event_section, term, office_row)로 dict 분기 |
| `_build_project_tools` | `jobs/autonomous_project.py:558-873` | 316 | project_id와 무관한 스키마 목록(약 110줄)을 모듈 상수로 올림 |
| `_chat_with_tools` | `telegram/bot.py:1265-1574` | 310 | profile·toolset 해석 / 컨텍스트 조립 / provider 분기로 나눔 |
| `_execute_one_tick` | `jobs/autonomous_project.py:1861-2151` | 291 | |
| `_exec_read_autonomous_project` | `self_runtime/tools.py:3060-3329` | 270 | 목록 조회와 상세 조회 |
| `process_task` | `telegram/tasks.py:1643-1901` | 259 | |
| `cmd_project` | `telegram/commands.py:2376-2624` | 249 | 하위 명령별 함수 |
| `run_once` | `scripts/commulingo_people_maintainer.py:1344-1581` | 238 | C1과 함께 |
| `_exec_web_read_self` | `services/web_chat.py:441-666` | 226 | 공개 content_type별 함수 |
| `recover_processing_tasks_on_startup` | `telegram/tasks.py:1904-2124` | 221 | |

테스트 전제: CommuLingo 테스트 약 120개는 `/home/grass/frontend/data/commulingo/*.json`(계약, 국적 정책, 활동 스키마·카탈로그)이 있어야 로드된다. 운영 서버에서 돌리거나, 프런트엔드 저장소의 실제 파일을 `COMMULINGO_PERSON_CONTRACT`, `COMMULINGO_NATIONALITY_POLICY`, `COMMULINGO_ACTIVITY_SCHEMA`, `COMMULINGO_ACTIVITY_CATALOG`로 가리킨 뒤 `scripts/run_unit_tests.sh`를 돌린다. `_validate`를 분할하기 전에 이 테스트 전체가 통과하는 기준선부터 확보한다.

## C. 구조와 레이어

### C1. `scripts/commulingo_people_maintainer.py`는 이름만 legacy

- **문서 상태가 엇갈림:** `project_state.md`는 이 스크립트를 한 곳에서 "legacy, inactive"로, 서비스 표에서는 활성 레인으로 설명한다.
- **다른 스크립트가 라이브러리로 씀:** `commulingo_people_parallel.py:28`, `commulingo_terms_maintainer.py:30`, `commulingo_gap_worker.py:73`, 스모크 스크립트 1개, 테스트 3개가 import한다.
- **모듈 전역을 바꿔 씀:** `commulingo_people_parallel.py:61-63`과 `commulingo_gap_worker.py:103-111`이 `maintainer.LOCK_PATH`, `completed_run_count`, `latest_maintainer_edit`를 덮어쓴다. 레인 이름 `'commulingo-maintainer'`가 코드에 고정돼 있기 때문이다(192, 1262).
- **죽은 인자:** `_call_curator_stage(before_count=...)`(1269)는 `before_count`를 읽지 않는다. 그런데 `gap_worker.py:97-101`의 주석은 이 인자가 쓰인다고 설명한다.
- **중복:** `completed_run_count`가 `commulingo_terms_maintainer.py`와 `commulingo_gap_worker.py`에 똑같이 두 벌 있다.
- **제안:**
  - 레인 문맥(`suggested_by`, lock 경로)을 인자로 넘긴다.
  - 공용 헬퍼를 `runtime_tools/commulingo_lane.py` 같은 모듈로 옮긴다.
  - 모듈 전역 덮어쓰기와 `before_count`를 없앤다.
  - 운영 유닛이 실제로 활성인지 확인해 문서를 한쪽으로 맞춘다.

### C2. 라이브러리 패키지가 `scripts/`를 import함

- `commulingo_pipeline/draft_repair.py:5` → `scripts.commulingo_write_session`
- `commulingo_pipeline/stages.py:992`, `workflow.py:21` → `scripts.commulingo_person_reviewer`
- `site_publishing.py:808` → `scripts.comic_composer`
- `jobs/experience_writer.py:318` → `scripts.ingest_pending_curations`
- `telegram/commands.py:654,1405` → `scripts.metrics_snapshot`, `scripts.llm_balances`

**제안:** 재사용되는 함수를 패키지로 옮기고, 스크립트는 그 함수를 import하는 얇은 진입점으로 만든다.

### C3. 루트의 도메인 모듈

README는 루트에 호환용 모듈만 둔다고 하지만, 실제 구현이 루트에 있다.

- **모듈:** `site_publishing.py`(943줄), `research_store.py`, `publication_records.py`, `task_store.py`, `redis_state.py`, `prompt_context.py`, `skills_loader.py`, `self_modification_core.py`, `audit_sink.py`
- **순수 호환 shim:** `creative_writer.py`는 `from writer import *`뿐이다. 쓰는 곳은 `api_routes/writer.py`와 `scripts/schema_migrations.py:126` 두 곳이다.
- **제안:** 한 번에 하나씩 패키지로 옮기고 루트에는 다시 export하는 shim을 남긴다. `creative_writer.py`는 두 호출부를 `writer`로 바꾼 뒤 지운다.

### C4. `self_modification_core.py`

- **도달할 수 없는 코드:** `request_user_approval`(약 578-642)는 `input()`에서 대기한다. `request_approval=True`일 때만 불리는데, 그렇게 부르는 곳이 없다.
- **이름과 실제 동작이 다름:** `git_backup_before_modification`와 `git_reset_to_commit`은 git을 쓰지 않고 `.bak.TIMESTAMP` 파일을 복사·복원한다. 승인 메시지는 여전히 `git reset --hard`를 안내한다.
- **제안:**
  - 대기 경로를 지운다.
  - 함수 이름을 `backup_file`/`restore_backup`으로 바꾸고, 기존 이름은 별칭으로 남긴다.
  - 모듈을 `self_runtime/`로 옮긴다.

### C5. 작은 중복

| 중복 | 위치 | 비고 |
|---|---|---|
| `josa` | `jobs/kg_sync_commulingo.py:185`, `kg_runtime/doc_extract.py:41` | 동일 |
| `_gemini_usage`, `_anthropic_usage` | `llm/call_registry.py`, `llm/instrumented_clients.py` | 거의 동일 |
| `_purge_cloudflare_sync` | `runtime_tools/research.py`, `runtime_tools/post_edit.py` | 약 82% 유사 |
| `_slice_text` | `self_runtime/tools.py`, `mcp_gateway/tools.py` | None 처리만 다름 |
| `_atomic_write` | `runtime_tools/research.py` | fsync 없음. `translation_runtime/storage.atomic_write`의 약한 버전 |
| `KST = timezone(timedelta(hours=9))` | 약 12개 모듈에서 재정의 | 기준 정의는 `shared.KST` |

## D. 문서·설정 불일치 (운영 서버 확인 필요)

- **존재하지 않는 timer:** `project_state.md` Runtime Map에는 `leninbot-commulingo-{maintainer,new,enrich,terms}.timer`가 적혀 있다. 저장소 `systemd/`에는 이 이름의 `.service`만 있고 `.timer`는 없다. 서버에 설치된 unit을 `systemctl list-timers`로 확인한 뒤 문서를 고친다.
- **옛 배치 참조:** `systemd/leninbot-commulingo-maintainer.service`의 주석이 2026-09-20 폐지된 daily batch 순서를 여전히 언급한다. `leninbot-commulingo-batch.*`는 빈 mask 파일이다.
- **옛 동기화 주석:** `runtime_tools/commulingo_people.py` 상단 주석은 "`NATIONALITY_CODES`를 maintainer와 동기화해야 한다"고 말한다. 지금은 maintainer가 이 모듈에서 값을 가져오므로 동기화할 대상이 없다. 같은 이유로 `scripts/smoke_commulingo_nationality_policy.py`의 동기화 검증은 항상 통과한다.
- **문서에 없는 failover:** `_chat_with_tools`의 DeepSeek 분기에는 Anthropic 호환 경로에서 OpenAI 호환 경로로 넘어가는 failover가 있다. `llm_provider_architecture.md`의 Provider Paths 도식에는 이 경로가 없다.
- **남은 하드코딩 경로:** 운영 스크립트에 `/home/grass/leninbot`이 남아 있다: `scripts/*.sh`, `scripts/metrics_*.py`, `scripts/smoke_kimi_k3.py`, `scripts/smoke_luna_call_sites.py`, `skills/kg-maintenance/scripts/*`. 런타임 코드는 `ops/paths.py`로 옮겼다. 스크립트는 서버 전용이라 그대로 두었다.

## E. 테스트 환경

`scripts/run_unit_tests.sh`를 운영 서버가 아닌 곳에서 돌리면 다음 이유로 실패한다.

- **외부 데이터:** 테스트 약 120개가 `/home/grass/frontend/data/commulingo/*.json`을 요구한다. 위 B의 환경변수로 대신 지정할 수 있다.
- **requirements에 없는 패키지:**
  - `tavily`(`tavily-python`): `web_gateway/app.py`와 테스트 3개가 직접 import한다.
  - `pydantic_settings`: 테스트 4개가 간접적으로 요구한다.
  - 둘 다 `requirements.txt`에 없다.
- **기준선:** 이 브랜치에서 fixture를 넣고 돌린 결과는 1222개 중 실패 4, 에러 17(환경 요인)이다. 원본 HEAD와 같다.
