"""Private, user-scoped persistent notes for the standalone roleplay bot."""
from __future__ import annotations

import json
import sqlite3
import re
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from tool_gateway.security import get_caller
from runtime_tools.roleplay_clock import TEMPORAL_SCHEMA, interpret_clock, validate_temporal
from runtime_tools.roleplay_dynamics import (METRICS, CONDITION_SCHEMA, with_defaults, validate_conditions, advance)

MEMORY_PATH = Path(__file__).resolve().parents[1] / "output" / "roleplay_memory.sqlite3"
MAX_NOTES = 30


@contextmanager
def _connection():
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MEMORY_PATH, timeout=10)
    try:
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS notes (user_id TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL, PRIMARY KEY(user_id, key))")
            conn.execute("CREATE TABLE IF NOT EXISTS character_state (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS people (user_id TEXT NOT NULL, person_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(user_id, person_id))")
            conn.execute("CREATE TABLE IF NOT EXISTS state_history (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL)")
            yield conn
    finally:
        conn.close()


def load_notes(user_id: str | int) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute("SELECT key, content FROM notes WHERE user_id = ? ORDER BY key", (str(user_id),)).fetchall()
    return [{"key": key, "content": content} for key, content in rows]


def roleplay_memory(action: str, key: str = "", content: str = "") -> str:
    caller = get_caller()
    if caller.interface != "telegram" or caller.agent_name != "roleplay" or not caller.is_owner or not caller.user_id:
        raise PermissionError("Roleplay owner context required")
    user_id = str(caller.user_id)
    if action == "list":
        return json.dumps(load_notes(user_id), ensure_ascii=False)
    if action not in {"save", "delete"}:
        raise ValueError("Unknown memory action")
    key = key.strip()
    content = content.strip()
    if not key or len(key) > 80:
        raise ValueError("Memory key must contain 1–80 characters")
    if action == "save" and not 1 <= len(content) <= 800:
        raise ValueError("Memory content must contain 1–800 characters")
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if action == "delete":
            count = conn.execute("DELETE FROM notes WHERE user_id = ? AND key = ?", (user_id, key)).rowcount
            return json.dumps({"deleted": bool(count), "key": key}, ensure_ascii=False)
        exists = conn.execute("SELECT 1 FROM notes WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM notes WHERE user_id = ?", (user_id,)).fetchone()[0]
        if not exists and count >= MAX_NOTES:
            raise ValueError("Memory is full (30 notes); merge or delete an existing note first")
        conn.execute("INSERT INTO notes VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET content = excluded.content", (user_id, key, content))
    return json.dumps({"saved": True, "key": key}, ensure_ascii=False)


ROLEPLAY_MEMORY_TOOL = {
    "name": "roleplay_memory",
    "description": "예조프 전용 영구 메모. list로 읽고 save로 같은 key의 메모를 생성·교체하고 delete로 삭제한다. 사용자별 최대 30개. 확정된 관계·사건·선호·표현 교정을 짧게 저장한다. /new 이후에도 유지된다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "save", "delete"]},
            "key": {"type": "string", "maxLength": 80},
            "content": {"type": "string", "maxLength": 800},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


SCENE_TEXT_FIELDS = {"period", "location", "last_event", "unresolved", "goal", "avoid", "next_action"}

STATE_DEFAULTS = {**{key: "" for key in sorted(SCENE_TEXT_FIELDS)}, "participants": [], "hunger": None, "fatigue": None, "pain": None, "tension": None,
                  "body": "미설정", "mood": "미설정", "scene": "미설정", "reason": "아직 설정되지 않음"}


def load_state(user_id: str | int) -> dict:
    with _connection() as conn:
        row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (str(user_id),)).fetchone()
    return with_defaults({**STATE_DEFAULTS, **(json.loads(row[0]) if row else {})})


def roleplay_state(action: str, changes: dict | None = None, reason: str = "", *,
                   expected_revision: int | None = None,
                   adjustment: str = "", event_id: str = "",
                   metric_reasons: dict | None = None, temporal: dict | None = None,
                   event_type: str = "other") -> str:
    caller = get_caller()
    if caller.interface != "telegram" or caller.agent_name != "roleplay" or not caller.is_owner or not caller.user_id:
        raise PermissionError("Roleplay owner context required")
    if action == "read":
        return json.dumps(load_state(caller.user_id), ensure_ascii=False)
    if action == "history":
        with _connection() as conn:
            rows = conn.execute("SELECT payload FROM state_history WHERE user_id = ? ORDER BY id DESC LIMIT 20", (str(caller.user_id),)).fetchall()
        records = [json.loads(row[0]) for row in rows]
        for record in records:
            for side in ("before", "after"):
                record[side] = {k: record[side].get(k) for k in (*METRICS, "scene_minute", "activity", "sleep_quality", "threat", "injuries", "clock")}
        return json.dumps(records, ensure_ascii=False)
    if action not in {"update", "reset", "time"}:
        raise ValueError("Unknown state action")
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
        raise ValueError("A scene-based reason (1–300 characters) is required")
    if changes is not None and not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    changes = changes or {}
    if action == "time":
        validate_temporal(temporal)
        if not isinstance(event_id, str) or not 1 <= len(event_id.strip()) <= 100:
            raise ValueError("Time interpretation needs a stable event_id for retries")
    elif temporal is not None:
        raise ValueError("temporal is only accepted by action=time")
    if event_type not in {"other", "meal", "sleep", "injury", "treatment"}:
        raise ValueError("Unknown event_type")
    if action != "update" and changes:
        raise ValueError("Only update accepts changes; time uses already saved conditions")
    validate_conditions(changes)
    for key, value in changes.items():
        if key in CONDITION_SCHEMA:
            continue
        if key in {"hunger", "fatigue", "pain", "tension"}:
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError("State values must be integers from 0 to 100")
        elif key == "participants":
            if not isinstance(value, list) or len(value) > 12 or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value):
                raise ValueError("participants must be up to 12 distinct person IDs")
        elif key in SCENE_TEXT_FIELDS:
            if not isinstance(value, str) or len(value) > 300:
                raise ValueError("Scene fields must be strings of at most 300 characters")
        elif key in {"body", "mood", "scene"}:
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= 300:
                raise ValueError("State descriptions must contain 1–300 characters")
        else:
            raise ValueError("Unknown state field")
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for person_id in changes.get("participants", []) if action == "update" else []:
            if not conn.execute("SELECT 1 FROM people WHERE user_id = ? AND person_id = ?", (str(caller.user_id), person_id)).fetchone():
                raise ValueError("Register each participant with roleplay_person before using its ID")
        row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (str(caller.user_id),)).fetchone()
        before = with_defaults({**STATE_DEFAULTS, **(json.loads(row[0]) if row else {})})
        # Safe replay of an already applied absolute time/event never adds a second delta.
        if action in {"update", "time"} and event_id and event_id in before["recent_events"]:
            return json.dumps(before, ensure_ascii=False)
        if type(expected_revision) is not int or expected_revision != before["revision"]:
            raise ValueError(f"Read current state and retry with expected_revision={before['revision']}")
        if action == "time":
            state = interpret_clock(before, temporal, advance)
            state["recent_events"] = (before["recent_events"] + [event_id])[-100:]
        elif action == "reset":
            state = with_defaults(dict(STATE_DEFAULTS))
        else:
            numeric = set(changes) & set(METRICS)
            if numeric:
                if adjustment not in {"initialize", "event", "correction"}:
                    raise ValueError("Numeric update needs adjustment=initialize/event/correction; use action=time for time effects")
                if adjustment == "initialize" and any(before[k] is not None for k in numeric):
                    raise ValueError("initialize only fills unknown metrics; existing values require event/correction")
                if not isinstance(metric_reasons, dict) or set(metric_reasons) != numeric or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 300 for v in metric_reasons.values()):
                    raise ValueError("Provide metric_reasons for every numeric field being set")
                if not isinstance(event_id, str) or not 1 <= len(event_id.strip()) <= 100:
                    raise ValueError("Numeric update needs a stable event_id; reuse it when retrying the same event")
            state = {**before, **changes}
            if all(k in changes for k in CONDITION_SCHEMA):
                state["conditions_initialized"] = True
            if numeric:
                state["recent_events"] = (before["recent_events"] + [event_id])[-100:]
                state["event_timestamps"] = (before["event_timestamps"] + [{
                    "event_id": event_id, "type": event_type, "scene_minute": state["scene_minute"],
                    "clock": {k: state["clock"][k] for k in ("date", "year", "time", "daypart", "relative_day", "certainty", "unquantified_gaps")},
                    "reason": reason.strip(),
                }])[-50:]
        state["revision"] = before["revision"] + 1
        state["reason"] = reason.strip()
        audit = {"revision": state["revision"], "action": action, "reason": reason.strip(),
                 "adjustment": adjustment, "event_id": event_id, "metric_reasons": metric_reasons,
                 "temporal": temporal, "source_scope_id": caller.scope_id, "before": before, "after": state}
        conn.execute("INSERT INTO state_history(user_id, revision, payload) VALUES (?, ?, ?)",
                     (str(caller.user_id), state["revision"], json.dumps(audit, ensure_ascii=False)))
        conn.execute("DELETE FROM state_history WHERE user_id = ? AND id NOT IN (SELECT id FROM state_history WHERE user_id = ? ORDER BY id DESC LIMIT 1000)", (str(caller.user_id), str(caller.user_id)))
        conn.execute("INSERT INTO character_state VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload",
                     (str(caller.user_id), json.dumps(state, ensure_ascii=False)))
    return json.dumps(state, ensure_ascii=False)


ROLEPLAY_STATE_TOOL = {
    "name": "roleplay_state",
    "description": "시간은 action=time + temporal(근거 인용·해석·현재/과거/계획·명시/추정/미상) + event_id. 현재 advance만 elapsed_minutes를 계산, anchor는 시계 기준 설정, next_day는 날짜만 진행, reference는 회상·계획 기록. 시간 직접 advance 호출은 폐지. 변경 시 현재 expected_revision 필수. update는 활동·부상·장면·목적 설정. 수치 직접 수정은 initialize/event/correction과 event_id·metric_reasons 필요. history는 최근 변경 이력. 역할극 인물의 지속 상태표. read/update/reset. 수치는 0(없음)–100(극심). 장면 속 원인에 따라 변경하고 reason에 근거를 남긴다. 현실 시계로 자동 변화하지 않는다. period=시기, location=장소, participants=등록된 인물 ID, last_event=완료 사건, unresolved=미해결 질문, goal=예조프의 당면 목적, avoid=피하려는 결과, next_action=시도할 행동. 빈 문자열/빈 participants로 해소된 항목을 비운다. reset은 새 장면에 상태와 목적을 초기화한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "history", "update", "reset", "time"]},
            "changes": {"type": "object", "properties": {
                **CONDITION_SCHEMA,
                **{k: {"type": "integer", "minimum": 0, "maximum": 100} for k in ["hunger", "fatigue", "pain", "tension"]},
                **{k: {"type": "string", "maxLength": 300} for k in ["body", "mood", "scene", *sorted(SCENE_TEXT_FIELDS)]},
                "participants": {"type": "array", "items": {"type": "string"}, "maxItems": 12, "uniqueItems": True},
            }, "additionalProperties": False},
            "expected_revision": {"type": "integer", "minimum": 0},
            "temporal": TEMPORAL_SCHEMA,
            "event_type": {"type": "string", "enum": ["other", "meal", "sleep", "injury", "treatment"]},
            "adjustment": {"type": "string", "enum": ["initialize", "event", "correction"]},
            "event_id": {"type": "string", "maxLength": 100},
            "metric_reasons": {"type": "object", "properties": {k: {"type": "string", "maxLength": 300} for k in METRICS}, "additionalProperties": False},
            "reason": {"type": "string", "maxLength": 300, "description": "changes 안이 아닌 최상위 변경 이유"},
        }, "required": ["action"], "additionalProperties": False,
    },
}


PERSON_FIELDS = {"name": 100, "identity": 400, "relationship": 400,
                 "observed": 600, "reported": 600, "inferred": 400}


def _owner_id() -> str:
    caller = get_caller()
    if caller.interface != "telegram" or caller.agent_name != "roleplay" or not caller.is_owner or not caller.user_id:
        raise PermissionError("Roleplay owner context required")
    return str(caller.user_id)


def load_people(user_id: str | int) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute("SELECT person_id, payload FROM people WHERE user_id = ? ORDER BY person_id", (str(user_id),)).fetchall()
    return [{"person_id": pid, **json.loads(payload)} for pid, payload in rows]


def people_context(user_id: str | int, participants: list[str]) -> dict:
    people = load_people(user_id)
    return {
        "index": [{k: p[k] for k in ("person_id", "name", "aliases", "identity")} for p in people],
        "present": [p for p in people if p["person_id"] in participants],
    }


def _name_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def roleplay_person(action: str, person_id: str = "", query: str = "", changes: dict | None = None) -> str:
    user_id = _owner_id()
    if action in {"list", "read"}:
        people = load_people(user_id)
        if action == "read":
            if not person_id and not query.strip():
                raise ValueError("read requires a person_id or name/alias query")
            people = [p for p in people if p["person_id"] == person_id] if person_id else [
                p for p in people if _name_key(query) in {_name_key(p["name"]), *(_name_key(a) for a in p["aliases"])}]
        else:
            people = [{k: p[k] for k in ("person_id", "name", "aliases", "identity")} for p in people]
        return json.dumps({"matches": people, "ambiguous": action == "read" and len(people) > 1}, ensure_ascii=False)
    if action not in {"save", "delete"}:
        raise ValueError("Unknown person action")
    if not isinstance(person_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", person_id):
        raise ValueError("Use a stable lowercase person_id (1–64 letters/digits/_/-)")
    if changes is not None and not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    changes = changes or {}
    for key, value in changes.items():
        if key == "aliases":
            if not isinstance(value, list) or len(value) > 8 or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 100 for v in value):
                raise ValueError("Use up to 8 nonempty aliases of at most 100 characters")
        elif key not in PERSON_FIELDS or not isinstance(value, str) or len(value) > PERSON_FIELDS[key] or (key == "name" and not value.strip()):
            raise ValueError("Unknown or invalid person field")
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if action == "delete":
            count = conn.execute("DELETE FROM people WHERE user_id = ? AND person_id = ?", (user_id, person_id)).rowcount
            row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                state = json.loads(row[0])
                state["revision"] = state.get("revision", 0) + 1
                state["participants"] = [pid for pid in state.get("participants", []) if pid != person_id]
                conn.execute("UPDATE character_state SET payload = ? WHERE user_id = ?", (json.dumps(state, ensure_ascii=False), user_id))
            return json.dumps({"deleted": bool(count), "person_id": person_id})
        row = conn.execute("SELECT payload FROM people WHERE user_id = ? AND person_id = ?", (user_id, person_id)).fetchone()
        if not row:
            if not changes.get("name"):
                raise ValueError("A new person requires a name")
            if conn.execute("SELECT COUNT(*) FROM people WHERE user_id = ?", (user_id,)).fetchone()[0] >= 30:
                raise ValueError("Person limit reached (30); update or delete an existing record")
        person = {**{k: "" for k in PERSON_FIELDS}, "aliases": []}
        if row:
            person.update(json.loads(row[0]))
        person.update(changes)
        conn.execute("INSERT INTO people VALUES (?, ?, ?) ON CONFLICT(user_id, person_id) DO UPDATE SET payload = excluded.payload", (user_id, person_id, json.dumps(person, ensure_ascii=False)))
    return json.dumps({"saved": True, "person_id": person_id, **person}, ensure_ascii=False)


ROLEPLAY_PERSON_TOOL = {
    "name": "roleplay_person",
    "description": "사용자별 인물 기록. list=이름·ID 목록, read=person_id 또는 이름/별칭(query) 조회, save=생성/부분 수정, delete=삭제. 같은 인물은 안정된 ID를 유지한다. 이름이 같아도 자동 병합하지 않는다. identity=인물 식별 정보와 적용 시기, relationship=예조프와의 관계, observed=직접 관찰한 사실, reported=전해 들은 말과 출처, inferred=미확정 추측. 목록은 최대 30명. 필드 수정은 해당 필드 전체를 교체하므로 기존 유효 정보를 보존한다.",
    "input_schema": {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "read", "save", "delete"]},
            "person_id": {"type": "string", "maxLength": 64},
            "query": {"type": "string", "maxLength": 100},
            "changes": {"type": "object", "properties": {
                **{k: {"type": "string", "maxLength": limit} for k, limit in PERSON_FIELDS.items()},
                "aliases": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 8},
            }, "additionalProperties": False},
        }, "required": ["action"], "additionalProperties": False,
    },
}
