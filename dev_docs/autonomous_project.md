# Autonomous Project Loop — T0 Pilot

## 1. Purpose

Cyber-Lenin이 사용자 입력 없이도 **장기 프로젝트**를 스스로 전진시키는 자율 에이전트 레이어. 매시간 한 번 깨어나 현재 active 프로젝트 중 오래 안 돈 것 1개를 골라 bounded 라운드 예산 안에서 한 스텝 진행한다.

**핵심 차이 (vs 기존 diary/experience_writer)**: diary는 고정 스키마의 일기 한 편을 매번 새로 생산. experience_writer는 지난 24시간을 요약. 자율 프로젝트 루프는 **상태를 누적**한다 — 매 tick이 이전 tick의 노트·plan 위에 쌓이고, 여러 날에 걸쳐 하나의 목표를 심화시킨다.

## 2. Tier System

외부 영향력 투사 기능은 티어로 분리 — goal 이 열려있을수록 누적 오정렬 리스크가 크기 때문.

| Tier | 허용 액션 | 게이트 |
|------|-----------|--------|
| **T0** (현재 구현) | 리서치, KG 쓰기, 내부 노트, plan 수정, **cyber-lenin.com 내 게시** (research / hub / static pages) | 없음 — 자율 |
| T1 (미구현) | 외부 플랫폼(Twitter/X, reddit, 디시, clien 등) 포스팅, 이메일 발송, A2A 통신 | 로그 + 사후 감사 |
| T2 (미구현) | 새 계정 개설, 외부 에이전트 직접 연락, 재정 집행 | 매 건 휴먼 승인 |

AgentSpec.tools가 이 경계를 강제한다. T0 spec은 cyber-lenin.com 내 3개 publish 도구를 포함하되, 외부 도구(`send_email`, `a2a_send`, `browse_web`, `generate_image`, `save_diary`, `write_file`)는 **들어있지 않다**. 프롬프트 `tier-constraints` 섹션이 이 경계를 에이전트에게도 명시.

### 왜 "cyber-lenin.com 내 게시"가 T0인가

외부 플랫폼 행동은 오정렬 누적 리스크가 크다 — 남의 집에서 일어나는 일은 철회 비용이 크고, 우리 평판에 묶인다. 반면 우리 도메인 내 게시는:
- reversible (같은 slug overwrite, archive, 수동 삭제 모두 즉시 가능)
- 블래스트 레디어스가 cyber-lenin.com 으로 한정
- 인프라·로그·모니터링이 이미 우리 통제 하에 있음

따라서 "외부 행동 vs 내부 작업"이 아니라 **"우리 도메인 vs 타인 도메인"** 이 T0의 실제 경계.

## 3. State Machine

```
                 create
                   │
                   v
            ┌───────────────┐
            │  researching  │◄──┐
            └───────┬───────┘   │ (리서치 성숙 → 재진입)
                    │           │
                    │ plan 구체화 │
                    v           │
            ┌───────────────┐   │
            │   planning    │───┘
            └───────┬───────┘
                    │
                    │ 일시 중단  ┌───────────┐
                    ├──────────► │  paused   │
                    │            └─────┬─────┘
                    │                  │ resume
                    │ ◄────────────────┘
                    │
                    │ 종료
                    v
            ┌───────────────┐
            │   archived    │ (terminal)
            └───────────────┘
```

- **active states** (`_pick_next_project`이 선택하는 것): `researching`, `planning`
- `paused` / `archived` 는 스케줄러가 건너뜀
- state 전환은 agent가 `set_project_state` 도구로 수행 (rationale 필수) 또는 사용자가 CLI로 수행

## 4. Architecture

### 파일 구성

| 경로 | 역할 |
|------|------|
| `agents/autonomous.py` | `AUTONOMOUS_PROJECT` AgentSpec — 페르소나, 프롬프트 IR, 도구 화이트리스트, 라운드/예산 한도 |
| `autonomous_project.py` | 런타임 — 테이블 부트스트랩, 프로젝트 스코프 도구, tick 실행, 텔레그램 알림 |
| `scripts/autonomous_cli.py` | 운영 CLI — create / list / show / events / edit / pause / resume / archive / tick |
| `systemd/leninbot-autonomous.service` | oneshot 서비스, `python -m autonomous_project` 실행 |
| `systemd/leninbot-autonomous.timer` | `OnCalendar=*-*-* *:17:00 Asia/Seoul` — 매시 :17 KST |

### DB 스키마 (Supabase Postgres)

```sql
CREATE TABLE autonomous_projects (
    id            SERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    topic         TEXT NOT NULL,
    goal          TEXT NOT NULL,       -- 이 프로젝트의 디렉티브, 매 tick 프롬프트에 주입
    state         VARCHAR(20) NOT NULL DEFAULT 'researching',
    plan          JSONB NOT NULL DEFAULT '{"goals": [], "steps": []}',
    research_notes JSONB NOT NULL DEFAULT '[]',  -- list of {turn, text, sources, created_at}
    turn_count    INT NOT NULL DEFAULT 0,
    last_run_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE autonomous_project_events (
    id         SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES autonomous_projects(id) ON DELETE CASCADE,
    event_type VARCHAR(40) NOT NULL,   -- tick_start|tick_end|tick_error|note_added|plan_revised|state_transition|project_edited|project_created
    content    TEXT,
    meta       JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

테이블은 `_ensure_tables()` 가 첫 CLI 호출 시 `CREATE IF NOT EXISTS` 로 멱등 생성.

## 5. Tick Lifecycle

```
systemd timer (*:17 KST)
  └─► python -m autonomous_project
        └─► run_tick()
              ├─ _pick_next_project() — state IN (researching, planning) ORDER BY last_run_at NULLS FIRST
              │
              ├─ _run_one_tick(project):
              │    ├─ get_agent("autonomous_project")  ← spec, tools 화이트리스트, max_rounds
              │    ├─ _build_project_tools(project_id)  ← add_research_note / revise_plan / set_project_state (project_id 클로저)
              │    ├─ _build_task_prompt()  ← <project> <goal> <state> <plan> <recent-notes> 주입
              │    ├─ log_event(tick_start)
              │    ├─ claude_loop.chat_with_tools(...)  ← max_rounds=6, budget=$0.40, provider=claude
              │    ├─ UPDATE autonomous_projects SET turn_count+1, last_run_at=NOW()
              │    ├─ log_event(tick_end, meta={cost, rounds})
              │    └─ _notify_telegram(project, result_text, actions, runtime)
              │
              └─ (tick 완료)
```

tick 중간에 발생하는 세 가지 영속화:
1. `add_research_note` → `autonomous_projects.research_notes` jsonb append + `note_added` 이벤트
2. `revise_plan` → `autonomous_projects.plan` overwrite + `plan_revised` 이벤트 (이전 plan을 meta에 저장)
3. `set_project_state` → `autonomous_projects.state` 변경 + `state_transition` 이벤트

`finalization_tools=["add_research_note", "revise_plan", "set_project_state"]` 로 지정되어 있어 예산/라운드 한도로 강제 종료될 때도 이 세 도구는 호출 가능하다 — 작업 데이터가 라운드 컷오프에 사라지지 않도록.

## 5a. LLM Provider & Model

- **Provider: Claude 고정** (`AgentSpec.provider="claude"`). `/config` 의 provider 설정(openai/local)과 무관하게 항상 Anthropic API 사용. 이유: 복잡한 장기 리서치·KG 구조화에서 가장 안정적이며, 같은 이유로 diary 에이전트도 Claude 고정
- **Tier: config 따름** — `_config["task_model"]` (high/medium/low) 를 `_TIER_MAP["claude"]` 로 매핑하여 모델 해석:
  - high → Claude Opus 4.7
  - medium → Claude Sonnet 4.6
  - low → Claude Haiku 4.5
- `/config` 로 task_model 을 내리면 다음 tick부터 즉시 반영. 비용 조절 메커니즘
- 해석 코드: `autonomous_project._run_one_tick` 안의 `_TIER_MAP.get("claude", {}).get(_task_tier, ...)` → `_get_model_by_alias(alias)`. `telegram_bot._chat_with_tools` 의 `provider in ("claude", "openai")` 분기와 동일 패턴

## 6. Agent Toolset (T0)

리서치 (읽기 전용):
- `web_search`, `fetch_url` — 외부 웹
- `vector_search` — lenin_corpus pgvector (core_theory / modern_analysis)
- `knowledge_graph_search` — Neo4j Graphiti
- `read_self`, `recall_experience`, `get_finance_data`

영속화 — 프로젝트 상태:
- `add_research_note`, `revise_plan`, `set_project_state` — 프로젝트 DB 상태 변경 (project_id 클로저로 스코프됨)
- `write_kg_structured` — KG 구조화 fact 쓰기

게시 — cyber-lenin.com 내 (T0 경계 안):
- `publish_research(title, content, filename?)` — markdown 문서 → `/reports/research/{slug}` (`.md` 없는 공개 URL). 연재 시리즈·장문 에세이·정세 분석용
- `publish_hub_curation(title, source_url, source_title, selection_rationale, context, …)` — 구조화 DB 레코드 → `/hub/{slug}`. 외부 한국어 글 큐레이션용. 각 필드가 분리되어 품질 일관성 확보
- `publish_static_page(slug, title, html_body, summary?)` — 샌드박스된 HTML 페이지 → `/p/{slug}`. 위키 스타일 레퍼런스·시각적 레이아웃용. slug 정규식 검증(alphanumeric + dash), html_body 는 inner content 만(site 가 레이아웃 감쌈), DOMPurify 로 클라이언트 sanitize

명시적으로 **없는 것** (T0 경계):
- `send_email`, `a2a_send`, `browse_web`, `generate_image` — 외부 송출
- `save_diary` — diary 에이전트 전담 채널
- `write_file`, `patch_file` — 범용 파일/코드 수정 (공격 표면 넓음)
- `delegate`, `multi_delegate`, `run_agent` — 다른 에이전트 호출

### 게시 도구의 샌드박스 설계

- `publish_research` — 기존 도구. `/home/grass/leninbot/research/*.md` 로만 쓰기 (경로 traversal 차단)
- `publish_hub_curation` — DB 테이블 `hub_curations` 로만 INSERT. slug 자동 고유화(2, 3, … 접미사)
- `publish_static_page` — `/home/grass/leninbot/static_pages/{slug}.json` 으로만 쓰기. slug 정규식 `^[a-z0-9][a-z0-9-]{0,79}$` 검증, resolved path 가 sandbox 디렉토리 안에 있는지 이중 확인, `<html>/<body>/<head>` 태그 포함 거부
- 모든 게시 도구가 `finalization_tools` 에 포함되어 라운드/예산 cutoff 에도 호출 가능 → 작업 결과물 손실 방지

## 7. Scheduler & Notification

### 타이머

- `leninbot-autonomous.timer`: `OnCalendar=*-*-* *:17:00 Asia/Seoul`
- 매시 :17 선택 이유: diary(0/6/12/18) · experience_writer(00:30 UTC) · 기타 타이머와 충돌 회피
- `Persistent=true` — 서버가 꺼져 있던 시간 동안의 과거 tick은 부팅 후 재개되지 않음(권장 동작, 누적 tick 폭주 방지)

### 매 tick Telegram 알림

`_notify_telegram` 이 tick 종료 직후 `TELEGRAM_CHAT_ID` (관리자 개인 채팅)로 발송. aiogram 단독 Bot 인스턴스. 실패해도 tick 데이터는 이미 커밋된 뒤라 손실 없음.

메시지 포맷 (plain text, no markdown):
```
🤖 자율 프로젝트 #{id} tick {N} 완료
프로젝트: {title}
상태: {state}  /  라운드: {rounds_used}  /  비용: ${cost:.3f}

[노트 k/M] {note text, up to 900 chars}
[Plan 수정] {rationale, up to 600 chars}
[상태전환] {from} → {to}  —  {reason, up to 400 chars}

[자가비평] {마지막 문단, up to 500 chars}
```

액션별 내용(카운트가 아니라 **실제 본문**)을 보여준다. 도구 호출이 하나도 없으면 "(도구 호출 없음 — agent가 저장 없이 종료)" 로 명시.

## 8. Operations — CLI

```
venv/bin/python scripts/autonomous_cli.py <command>
```

| 명령 | 용도 |
|------|------|
| `create --title T --topic X --goal P` | 새 프로젝트 생성 (state=researching) |
| `list` | 전체 목록 (state, turn_count, last_run_at, title) |
| `show <id>` | 단건 상세 — goal, plan, 최근 5 notes |
| `events <id> [--limit N]` | 이벤트 로그 (기본 30건) |
| `edit <id> [--title T] [--topic X] [--goal P \| --goal-file F]` | 필드별 수정 + `project_edited` 이벤트 기록 |
| `pause <id>` / `resume <id>` / `archive <id>` | 상태 전환 (스케줄러 대상에서 제외/포함/종결) |
| `tick` | 수동으로 한 번 즉시 실행 (테스트용) |

goal 이 multi-line일 때는 `--goal-file /path/to/file.txt` 또는 `--goal-file -` (stdin) 사용.

### 방향 전환 워크플로우

소폭 조정:
```
pause 1 → edit 1 --goal ... → resume 1
```

대폭 방향 전환 (기존 notes/plan이 새 방향과 어긋나 혼란을 줄 경우):
```
archive 1 → create --title ... --goal ... → (새 프로젝트로 시작)
```

## 9. Design Decisions

### max_rounds = 6 (not 3)

사용자 요구는 "3턴의 기회"였지만, `claude_loop.py`의 라운드 한도 경고가 `max_rounds - 2` 라운드에 주입된다. max_rounds=3이면 경고가 라운드 1에 주입되어 tool_result 순서를 깨고 Claude API 400 발생. **6으로 버퍼**를 주면 경고가 라운드 4에 주입되어 자연스러운 "마무리 지시" 타이밍이 된다. 여전히 작은 bounded wake (analyst=50, diary=30 대비).

### Round-limit warning injection 순서 버그 수정

원본 `claude_loop.py:844-852`는 `tool_results.insert(0, warning_block)` 로 경고 텍스트를 tool_result 앞에 prepend. 이는 Claude의 "tool_use id는 바로 다음 user turn의 tool_result 블록과 즉시 매칭돼야 한다" 규칙 위반 → 400 에러. `append()` 로 변경해 tool_result 뒤에 붙도록. 모든 에이전트(diary, analyst, browser 등)에 영향 있는 수정이었음.

### 핸들러 시그니처: `args: dict` → 명시적 kwargs

프로젝트 스코프 도구(`add_research_note` 등)를 처음에 `async def _handle(args: dict)` 로 작성했는데, `tool_loop_common.py` 의 디스패처가 `inspect.signature()` 로 허용 kwargs를 판단 → `text`, `sources` 등을 "unknown kwargs"로 드롭하고 실행 실패. 기존 에이전트 도구(`_exec_read_diary(limit=5, keyword=None)` 스타일)와 맞춰 **명시적 kwargs** 로 재작성.

### 알림 포맷: 카운트 대신 내용

초기 구현은 "노트 +1, plan 유지, 상태전환 아니오" 같은 메타 카운트를 보냈는데 사용자 피드백 — "뭘 했다는 게 없잖아". `_collect_tick_actions` 로 교체해 이벤트 로그에서 **본문**(노트 텍스트, plan rationale, 전환 이유)을 읽어 표시. 카운트는 agent 자가비평에 흡수.

### `pause` vs `archive`

`pause` 는 일시 보류 — `resume` 으로 복귀 가능. 사용자가 검토 중이거나 잠시 막고 싶을 때. `archive` 는 종결 — 되돌릴 수 있으나 의미적으로 "이 프로젝트는 끝났다". 스케줄러는 둘 다 대상에서 제외.

### sudoers NOPASSWD 제약

`/etc/sudoers.d/leninbot` 의 NOPASSWD 규칙은 `systemctl enable leninbot-*` 와 `systemctl start leninbot-*` 를 개별로 허용. 합성 커맨드 `enable --now` 는 매칭이 안 되어 비번 요구. 배포 스크립트에서는 항상 두 커맨드로 분리해야 함.

## 10. Current Limitations

1. **단일 프로젝트 tick / 시간**: 여러 active 프로젝트가 있어도 매시 1개만 돈다 (oldest `last_run_at`). 24 tick/일로 분산되므로 5개 프로젝트면 각각 ~5번/일.
2. **KG group_id 일관성**: 프롬프트 rules가 `autonomous_project_{project_id}` 를 제안하지만 강제는 아님. 초기 tick들은 기존 `agent_knowledge` 그룹에 썼다. 프로젝트별 격리가 필요하면 추후 도구 래퍼에서 강제해야 함.
3. **Goal 변경 감지 없음**: 사용자가 `edit` 으로 goal 을 바꿔도 agent는 변경 사실 자체를 모른다 — 단지 다음 tick에서 새 텍스트를 읽을 뿐. 누적된 notes/plan이 새 방향과 어긋난 경우 agent가 자각하지 못할 수 있다. **"goal 이 최근 수정됨"** 플래그를 프롬프트에 주입하는 게 개선 방향.
4. **턴간 플래너 없음**: 매 tick이 독립된 wake. 프로젝트 수준의 상위 플래너(지난 10 tick 회고 → 다음 10 tick 방향 재설정)가 없다. 현재는 plan JSONB + self-critique 마지막 문단으로 대신.
5. **Critic 부재**: 사용자 초안에서는 Planner+Executor+Critic 3자 구조였으나, MVP에서는 agent의 자기비평(`[자가비평]` 섹션)으로 대체. 별도 critic 호출을 붙이면 preference 정렬 점수·드리프트 감지가 가능해지지만 비용이 2~3배.
6. **비용 상한 없음**: 개별 tick은 `budget_usd=$0.40` 로 bounded지만 프로젝트 전체 비용 상한은 없음. 1주일 168 tick × $0.40 = 최대 $67/프로젝트.

## 11. Roadmap

- [ ] **goal_recently_changed 플래그**: `edit` 시 플래그 ON, 다음 tick 프롬프트에 "goal 이 방금 바뀌었다 — 기존 plan·notes와 새 goal 정합성을 점검하라" 주입, tick 완료 시 OFF.
- [ ] **Project-scoped KG group_id 강제**: 도구 래퍼에서 `group_id=f"autonomous_project_{pid}"` 강제 주입, 프로젝트별 격리된 지식 네트워크.
- [ ] **Critic 분리**: 별도 LLM 호출로 preference 정렬 점수·드리프트 감지, 임계치 이하면 `state_transition → paused` 자동 + 관리자 알림.
- [ ] **T1 게이트 설계**: 공개 포스팅·이메일 발송 같은 외부 액션의 화이트리스트·가드레일·로그. T0에서 누적된 plan 중 "T1 승인 필요" 표시된 항목을 사용자가 한 번에 검토·승인하는 UI.
- [ ] **텔레그램 커맨드**: `/projects`, `/project <id> show` 등으로 CLI 없이 모바일에서 관리.

## 12. Change History

- **2026-04-18 (초기)**: 파일럿 MVP 구축. 프로젝트 #1 "한국 진보주의자 온라인 생태계" 시드. systemd timer enable. 버그 3건 수정 (핸들러 시그니처, 경고 prepend, max_rounds). 텔레그램 알림을 카운트에서 내용으로 교체. `edit` CLI 추가.
- **2026-04-18 (재정의)**: 프로젝트 목적 재정의 — 기존 커뮤니티 **조사**에서 cyber-lenin.com 의 **자율 건설**로 초점 이동. T0 경계를 "외부 행동 없음" 에서 **"우리 도메인 내 게시 허용"** 으로 재해석.
  - `#1` archive, `#2 "Cyber-Lenin Node 건설"` 생성
  - 새 모듈 `site_publishing.py` — `publish_hub_curation` (DB), `publish_static_page` (sandbox HTML)
  - 새 DB 테이블 `hub_curations` + 파일 샌드박스 `/home/grass/leninbot/static_pages/`
  - 새 API 엔드포인트 `/hub`, `/hub/{slug}`, `/pages`, `/pages/{slug}`
  - 프론트엔드 (BichonWebpage): 새 라우트 `/hub`, `/p/{slug}`, 뷰 템플릿 `hub.ejs`, `hub-view.ejs`, `page-view.ejs`, 네비 "허브" 추가, server.js 마운트
  - agents/autonomous.py 확장: 3개 게시 도구 추가, prompt에 `building-modalities` 섹션, budget $0.40→$0.60 (게시 라운드 여유)
