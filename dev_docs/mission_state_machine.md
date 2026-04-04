# Mission Context System — State Machine Spec

## 1. Purpose

채팅과 태스크 간 맥락 공유. 하나의 **과제**(goal)에 매핑되는 공유 타임라인.

## 2. States

| State | 의미 | 진입 | 탈출 |
|-------|------|------|------|
| `(none)` | 미션 없음 | 초기 상태 / 미션 종료 후 | `/task` → `active` |
| `active` | 과제 진행 중. 이벤트 기록 가능. 시스템 프롬프트에 주입됨. | `create_mission()` | `close_mission()` → `done` |
| `done` | 과제 완료/종료. 읽기 전용. 시스템 프롬프트에 주입 안 됨. | `close_mission()` | (terminal) |

**불변식**: 사용자당 `active` 미션은 최대 1개.

## 3. State Transitions

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    v                                      │
(none) ──/task──► active ──close──► done                   │
                    │                                      │
                    │  이벤트 기록 (timeline grows)         │
                    │  - context_capture                   │
                    │  - task_created                      │
                    │  - tool_result                       │
                    │  - finding                           │
                    │  - decision                          │
                    │  - task_completed                    │
                    │                                      │
                    └──────────────────────────────────────┘
                     (이벤트는 상태를 바꾸지 않음)
```

## 4. Transition Triggers

### 4.1 `(none)` → `active`: 미션 생성

| Trigger | 위치 | 조건 |
|---------|------|------|
| `/task` 커맨드 | `telegram_bot.py` `cmd_task` | `get_active_mission(uid)` 이 None |

**부작용**:
- 최근 채팅 5턴을 `context_capture` 이벤트로 캡처
- 태스크에 `mission_id` FK 설정
- `task_created` 이벤트 기록

**미션이 이미 active이면**: 새 미션을 생성하지 않고, 기존 미션에 태스크를 합류시킴.

### 4.2 `active` → `done`: 미션 종료

| Trigger | 위치 | 주체 |
|---------|------|------|
| `mission(action="close")` 도구 | `telegram_tools.py` `build_mission_handler` | **에이전트** (채팅/태스크 모두) |
| `/mission close` 커맨드 | `telegram_bot.py` `cmd_mission` | **사용자** |
| `/clear` 커맨드 | `telegram_bot.py` `cmd_clear` | **사용자** |
| 24시간 이벤트 없음 | `telegram_mission.py` `get_active_mission` | **시스템** (접근 시 자동) |

## 5. Events (Timeline)

이벤트는 `active` 상태에서만 기록되어야 함. 상태를 변경하지 않음.

| event_type | source | 발생 시점 | 내용 |
|------------|--------|----------|------|
| `context_capture` | `system` | 미션 생성 시 | 최근 채팅 5턴 |
| `task_created` | `system` / `task#N` | 미션에 태스크 연결 시 | 태스크 내용 요약 |
| `tool_result` | `chat` | 채팅 중단 시 (예산/한도) | 수행한 도구 호출 목록 |
| `finding` | `task#N` | `save_finding` 도구 호출 | 중간 발견 (≤2000자) |
| `decision` | `task#N` | `request_continuation` 호출 | 진행 요약 + 다음 단계 |
| `decision` | `system` | 서비스 재시작/종료 | handoff/shutdown 기록 |
| `task_completed` | `task#N` | 태스크 완료/실패 시 | 결과 요약 또는 에러 |

## 6. mission_id 상속

태스크는 생성 시 `mission_id`를 FK로 저장. 이후 실행/이벤트는 이 값을 직접 사용.

```
/task (user) ──────────────── mission_id = active mission (없으면 생성)
  ├─ request_continuation ── child task ← parent.mission_id (DB 상속)
  │    └─ child의 child ──── grandchild ← parent.mission_id (재귀)
  ├─ auto-escalation ─────── mission_id = active mission
  └─ service restart ─────── child task ← parent.mission_id (recovery)

create_task (자율, user_id=0) ── mission_id = 최신 active mission
스케줄 태스크 ──────────────────── mission_id = NULL (미션과 무관)
```

`shared.create_task_in_db`: `parent_task_id`가 있으면 부모의 `mission_id`를 자동 상속.

## 7. Context Injection

### 채팅 (handle_message)
- `build_mission_context(user_id)` → active 미션의 최근 10 이벤트를 시스템 프롬프트 끝에 추가
- 미션 없으면 주입 없음

### 태스크 (process_task)
- `task["mission_id"]`로 직접 조회
- 미션의 최근 15 이벤트를 태스크 content 앞에 주입
- mission_id가 없으면 (스케줄 태스크 등) legacy scratchpad fallback

## 8. Edge Cases & Guards

| 상황 | 처리 |
|------|------|
| 미션 없이 `save_finding` 호출 | "No mission linked" 반환, 데이터 유실 없음 (에러 메시지로 안내) |
| `done` 미션에 이벤트 INSERT 시도 | `add_mission_event`가 상태 체크 후 무시 (guard 적용됨) |
| 동시에 2개 태스크가 미션 생성 시도 | `LIMIT 1` + application-level 체크로 1개만 생성 (race 가능) |
| user_id=0 (자율 생성) 미션 조회 | 전역 최신 active 미션 사용 |
| 24시간 만료 중 이벤트 기록 race | 만료가 먼저 처리되면 이벤트는 orphan됨 |

### 8.1 적용된 Guard

1. **`add_mission_event` 상태 체크**: `done` 미션에 이벤트 기록 시 무시 (양쪽 모두 적용)
2. **`done` → `active` 전이 불가**: `close_mission()`은 단방향. 코드에 역전이 경로 없음.

## 9. Database Schema

### PostgreSQL (Telegram)
```sql
CREATE TABLE telegram_missions (
    id          SERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL,
    title       TEXT NOT NULL,
    status      VARCHAR(20) DEFAULT 'active',  -- 'active' | 'done'
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    closed_at   TIMESTAMPTZ
);

CREATE TABLE telegram_mission_events (
    id          SERIAL PRIMARY KEY,
    mission_id  INTEGER NOT NULL REFERENCES telegram_missions(id),
    source      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_mission_events_timeline ON telegram_mission_events(mission_id, created_at);

ALTER TABLE telegram_tasks ADD COLUMN mission_id INTEGER REFERENCES telegram_missions(id);
```

### SQLite (Local Agent)
```sql
CREATE TABLE missions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    status      TEXT DEFAULT 'active',
    created_at  TEXT DEFAULT (datetime('now', 'localtime')),
    closed_at   TEXT
);

CREATE TABLE mission_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id  INTEGER NOT NULL REFERENCES missions(id),
    source      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX idx_mission_events_timeline ON mission_events(mission_id, created_at);

ALTER TABLE tasks ADD COLUMN mission_id INTEGER;
```

## 10. Revision History

| Date | Change |
|------|--------|
| 2026-03-21 | Initial implementation: telegram agent system |
| 2026-03-21 | Explicit mission_id inheritance across all task paths |
| 2026-03-21 | Agent close tool, /mission command, 24h auto-expire |
| 2026-03-21 | State machine spec document created |
