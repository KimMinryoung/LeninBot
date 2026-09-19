"""Private, user-scoped persistent notes for the standalone roleplay bot."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from tool_gateway.security import get_caller
from runtime_tools.roleplay_clock import TEMPORAL_SCHEMA, interpret_clock, validate_temporal
from runtime_tools.roleplay_dynamics import (METRICS, CONDITION_SCHEMA, RESOLVE_EVENT_KINDS, RESOLVE_INTENSITY,
                                             RESOLVE_EVENT_HISTORY, with_defaults, validate_conditions, advance,
                                             carry_injury_progress, injury_pain_floor, reconcile_injuries,
                                             isolation_stage, resolve_event_delta)

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


def _owner_id() -> str:
    caller = get_caller()
    if caller.interface != "telegram" or caller.agent_name != "roleplay" or not caller.is_owner or not caller.user_id:
        raise PermissionError("Roleplay owner context required")
    return str(caller.user_id)


def load_notes(user_id: str | int) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute("SELECT key, content FROM notes WHERE user_id = ? ORDER BY key", (str(user_id),)).fetchall()
    return [{"key": key, "content": content} for key, content in rows]


def _key_tokens(key: str) -> set[str]:
    return {t for t in re.split(r"[\s\-–—_()\[\]·,:/]+", _name_key(key)) if len(t) >= 2}


def _similar_keys(existing: list[str], key: str) -> list[str]:
    """Keys sharing a topic word: a hint to update one note instead of piling up variants."""
    tokens = _key_tokens(key)
    return [k for k in existing if k != key and _key_tokens(k) & tokens][:5]


def roleplay_memory(action: str, key: str = "", content: str = "") -> str:
    user_id = _owner_id()
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
        keys = [row[0] for row in conn.execute("SELECT key FROM notes WHERE user_id = ? ORDER BY key", (user_id,))]
        if key not in keys and len(keys) >= MAX_NOTES:
            raise ValueError("Memory is full (30 notes); merge or delete an existing note first")
        conn.execute("INSERT INTO notes VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET content = excluded.content", (user_id, key, content))
    result = {"saved": True, "key": key}
    similar = _similar_keys(keys, key)
    if similar:
        result["similar_keys"] = similar
        result["hint"] = "같은 주제의 메모가 이미 있으면 그 key로 갱신하고 중복 메모는 delete로 정리"
    return json.dumps(result, ensure_ascii=False)


ROLEPLAY_MEMORY_TOOL = {
    "name": "roleplay_memory",
    "description": "예조프 전용 영구 메모. list로 읽고 save로 같은 key의 메모를 생성·교체하고 delete로 삭제한다. 사용자별 최대 30개. 확정된 관계·사건·선호·표현 교정을 짧게 저장한다. 같은 주제는 새 key를 만들지 말고 기존 key를 갱신한다. /new 이후에도 유지된다.",
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


# ── People ────────────────────────────────────────────────────────────
PERSON_FIELDS = {"name": 100, "identity": 400, "relationship": 400,
                 "observed": 600, "reported": 600, "inferred": 400, "commulingo_id": 64}
PERSON_NOTE_FIELDS = {"relationship": 400, "observed": 600, "reported": 600, "inferred": 400}
# Items are checked in code so a stray review sentence or an unwrapped record is
# folded into the canonical shape instead of failing the whole state change.
PERSON_UPDATES_SCHEMA = {"type": "array", "maxItems": 12, "items": {"type": ["object", "string"]}}
PERSON_REVIEW_ALIASES = {"person_review_note", "people_review", "review", "person_note", "people_note"}


def _name_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_person_id(value: str) -> str:
    """Fold case/spacing into the stable-ID form; anything else must be re-chosen."""
    if not isinstance(value, str):
        return ""
    folded = re.sub(r"\s+", "_", unicodedata.normalize("NFKC", value).strip().casefold())
    folded = re.sub(r"[^a-z0-9_-]", "", folded).lstrip("_-")
    return folded if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", folded) else ""


def _validate_commulingo_id(person_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", person_id):
        raise ValueError("Invalid CommuLingo person ID")
    from db import query_one
    if not query_one("SELECT id FROM commulingo_people WHERE id = %s", (person_id,)):
        raise ValueError("CommuLingo person not found; search the dictionary before linking")


def _with_dictionary_link(person: dict) -> dict:
    cid = person.get("commulingo_id", "")
    return {**person, "commulingo_id": cid,
            "commulingo_url": f"https://cyber-lenin.com/commulingo/people/{cid}" if cid else ""}


def load_people(user_id: str | int) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute("SELECT person_id, payload FROM people WHERE user_id = ? ORDER BY person_id", (str(user_id),)).fetchall()
    return [_with_dictionary_link({"person_id": pid, **json.loads(payload)}) for pid, payload in rows]


def people_context(user_id: str | int, participants: list[str]) -> dict:
    people = load_people(user_id)
    return {
        "index": [{k: p[k] for k in ("person_id", "name", "aliases", "identity", "commulingo_id", "commulingo_url")} for p in people],
        "present": [p for p in people if p["person_id"] in participants],
    }


def _resolve_person_ids(conn, user_id: str, given: list[str], warnings: list[str], what: str) -> list[str]:
    """Map IDs, or names/aliases the model used instead, onto registered person IDs."""
    rows = conn.execute("SELECT person_id, payload FROM people WHERE user_id = ?", (user_id,)).fetchall()
    people = {pid: json.loads(payload) for pid, payload in rows}
    by_name: dict[str, set[str]] = {}
    for pid, person in people.items():
        for label in (person.get("name", ""), *person.get("aliases", [])):
            if label:
                by_name.setdefault(_name_key(label), set()).add(pid)
    resolved = []
    for raw in given:
        pid = raw if raw in people else _normalize_person_id(raw) if _normalize_person_id(raw) in people else ""
        if not pid:
            candidates = by_name.get(_name_key(raw), set())
            if len(candidates) > 1:
                raise ValueError(f"'{raw}' matches several people {sorted(candidates)}; use the person_id")
            pid = next(iter(candidates), "")
        if not pid:
            registered = {k: v.get("name", "") for k, v in people.items()}
            raise ValueError(f"Unknown person '{raw}' in {what}. Registered IDs: {json.dumps(registered, ensure_ascii=False)}. Register new people with roleplay_person(save) first")
        if pid != raw:
            warnings.append(f"{what}의 '{raw}'는 등록된 ID '{pid}'로 해석함")
        if pid not in resolved:
            resolved.append(pid)
    return resolved


def _normalize_person_updates(updates, review, warnings: list[str]):
    if isinstance(review, str) and len(review) > 300:
        review = review[:300]
        warnings.append("person_review를 300자로 잘라 저장함")
    if updates is None:
        return None, review
    if isinstance(updates, dict):
        updates = [updates]
    if not isinstance(updates, list) or len(updates) > 12:
        raise ValueError("person_updates must be a list of at most 12 records")
    normalized = []
    for item in updates:
        if isinstance(item, str):
            # A review sentence placed in the list instead of person_review.
            review = f"{review} {item}".strip() if isinstance(review, str) and review.strip() else item
            warnings.append("person_updates 안의 문장은 person_review로 옮김")
            continue
        if not isinstance(item, dict):
            raise ValueError("Each person update is {person_id, changes:{relationship/observed/reported/inferred}}")
        item = dict(item)
        pid = item.pop("person_id", None) or item.pop("id", None)
        fields = item.pop("changes", None)
        if fields is None:
            fields, item = item, {}
        if item:
            raise ValueError(f"Unexpected keys in a person update: {sorted(item)}")
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError("Each person update needs person_id")
        if not isinstance(fields, dict):
            raise ValueError("Person changes must be an object")
        fields = dict(fields)
        fields.pop("reason", None)
        if not fields:
            raise ValueError("Person changes must contain at least one of relationship/observed/reported/inferred")
        for key, value in fields.items():
            if key not in PERSON_NOTE_FIELDS:
                raise ValueError(f"Person update fields are {sorted(PERSON_NOTE_FIELDS)}; got '{key}'")
            if not isinstance(value, str) or len(value) > PERSON_NOTE_FIELDS[key]:
                raise ValueError(f"'{key}' must be a string of at most {PERSON_NOTE_FIELDS[key]} characters")
        normalized.append({"person_id": pid.strip(), "changes": fields})
    return normalized, review


def _apply_person_updates(conn, user_id: str, updates: list[dict], warnings: list[str]) -> list[dict]:
    applied = []
    for update in updates:
        pid = _resolve_person_ids(conn, user_id, [update["person_id"]], warnings, "person_updates")[0]
        row = conn.execute("SELECT payload FROM people WHERE user_id = ? AND person_id = ?", (user_id, pid)).fetchone()
        person = json.loads(row[0])
        person.update(update["changes"])
        conn.execute("UPDATE people SET payload = ? WHERE user_id = ? AND person_id = ?", (json.dumps(person, ensure_ascii=False), user_id, pid))
        applied.append({"person_id": pid, "changes": update["changes"]})
    return applied


# ── Character state ───────────────────────────────────────────────────
SCENE_TEXT_FIELDS = {"period", "location", "last_event", "unresolved", "goal", "avoid", "next_action"}
DESCRIPTION_FIELDS = {"body", "mood", "scene"}
CHANGE_KEYS = set(CONDITION_SCHEMA) | set(METRICS) | SCENE_TEXT_FIELDS | DESCRIPTION_FIELDS | {"participants"}
INTERVAL_ALIASES = {"interval", "conditions", "elapsed_conditions", "interval_condition"}
ADJUSTMENTS = {"initialize", "event", "correction"}
EVENT_TYPES = {"other", "meal", "sleep", "injury", "treatment"}
REVIEW_TRIGGERS = {"last_event", "participants"}

STATE_DEFAULTS = {**{key: "" for key in sorted(SCENE_TEXT_FIELDS)}, "participants": [], **{key: None for key in METRICS},
                  "body": "미설정", "mood": "미설정", "scene": "미설정", "reason": "아직 설정되지 않음"}
VIEW_CLOCK_KEYS = ("date", "year", "time", "daypart", "relative_day", "certainty", "elapsed_complete", "unquantified_gaps")


def load_state(user_id: str | int) -> dict:
    with _connection() as conn:
        row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (str(user_id),)).fetchone()
    return with_defaults({**STATE_DEFAULTS, **(json.loads(row[0]) if row else {})})


def state_view(state: dict) -> dict:
    """The model-facing state: everything it decides with, none of the replay bookkeeping."""
    view = {"revision": state.get("revision", 0)}
    for key in METRICS:
        value = state.get(key)
        view[key] = None if value is None else round(value, 1)
    for key in ("body", "mood", "scene", *sorted(SCENE_TEXT_FIELDS), "participants", "activity", "sleep_quality",
                "threat", "injuries", "conditions_initialized", "scene_minute", "last_calculated_minute", "time_basis", "reason"):
        view[key] = state.get(key)
    if not view.get("period"):
        view.pop("period", None)
    clock = state.get("clock") or {}
    view["clock"] = {k: clock.get(k) for k in VIEW_CLOCK_KEYS}
    interpretation = clock.get("last_interpretation")
    if interpretation:
        view["clock"]["last_interpretation"] = {k: interpretation.get(k) for k in ("relation", "operation", "source_quote", "interpretation")}
    view["pain_floor"] = injury_pain_floor(state.get("injuries") or [])
    calculation = state.get("last_calculation")
    if calculation:
        view["last_calculation"] = {k: calculation.get(k) for k in ("from_minute", "to_minute", "basis", "before", "after", "tension_target", "isolation_stage", "injury_changes", "healed") if calculation.get(k) not in (None, [])}
    if state.get("resolve_events"):
        view["last_resolve_event"] = state["resolve_events"][-1]
    view["calm_hours"] = round(state.get("calm_minutes", 0) / 60, 1)
    view["isolation_hours"] = round(state.get("isolation_minutes", 0) / 60, 1)
    stage = isolation_stage(state.get("isolation_minutes", 0))
    view["isolation_stage"] = f"{stage['label']}: {stage['description']}" if stage else None
    unset = [key for key in METRICS if state.get(key) is None]
    if unset:
        # A null among numbers is easy to skim past; name the gap and what closes it.
        view["unset_metrics"] = unset
        view["unset_metrics_note"] = "미설정 수치는 시간 계산에서 제외됨. 장면 근거로 changes에 값을 넣어 initialize"
    return view


def _normalize_changes(changes, extra: dict, warnings: list[str]):
    """Fold the shapes the model sends (reason inside changes, scene fields at top level,
    injuries keyed by id) into one changes object; reject only what cannot be interpreted."""
    if changes is None:
        changes = {}
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    changes = dict(changes)
    inner_reason = changes.pop("reason", "")
    if inner_reason:
        warnings.append("changes 안의 reason은 최상위 reason으로 옮겨 적용함")
    moved = [key for key in list(extra) if key in CHANGE_KEYS]
    for key in moved:
        changes.setdefault(key, extra.pop(key))
    if moved:
        warnings.append(f"최상위 인자 {moved}는 changes 안의 필드로 옮겨 적용함")
    injury_upserts = {}
    for key in list(changes):
        if key in CHANGE_KEYS:
            continue
        value = changes[key]
        if isinstance(value, dict) and "severity" in value:
            injury_upserts[key] = changes.pop(key)
        else:
            raise ValueError(f"Unknown state field '{key}'. changes accepts: {sorted(CHANGE_KEYS)}")
    return changes, inner_reason if isinstance(inner_reason, str) else "", injury_upserts


def _merge_injury_upserts(existing: list, changes: dict, upserts: dict, warnings: list[str]) -> None:
    merged = {item["id"]: dict(item) for item in changes.get("injuries", existing) if isinstance(item, dict) and "id" in item}
    for key, value in upserts.items():
        item = {"trend": "stable", "treated": False, **merged.get(key, {}), **value}
        item["id"] = value.get("id", key)
        merged[item["id"]] = item
    changes["injuries"] = list(merged.values())
    warnings.append(f"부상 {sorted(upserts)}은 injuries 목록에 병합함 (부상은 injuries 배열로 제출)")


def _normalize_temporal(temporal, warnings: list[str]) -> dict:
    if not isinstance(temporal, dict):
        raise ValueError("time needs temporal {relation, certainty, source_quote, interpretation, operation}")
    temporal = dict(temporal)
    operation = temporal.get("operation")
    if operation == "advance":
        dropped = [k for k in ("date", "year", "time", "daypart") if k in temporal]
        for key in dropped:
            temporal.pop(key)
        if dropped:
            warnings.append(f"advance는 elapsed_minutes로 시계를 계산하므로 {dropped}는 무시함")
    elif "elapsed_minutes" in temporal:
        temporal.pop("elapsed_minutes")
        warnings.append(f"{operation}에서는 elapsed_minutes를 쓰지 않아 무시함 (구간 계산은 advance)")
    validate_temporal(temporal)
    return temporal


def _auto_event_id(scope_id, action: str, changes: dict, temporal) -> str:
    digest = hashlib.sha1(json.dumps({"a": action, "c": changes, "t": temporal}, sort_keys=True,
                                     ensure_ascii=False, default=str).encode()).hexdigest()[:10]
    return f"auto-{scope_id or 'noscope'}-{digest}"


def _validate_changes(changes: dict) -> None:
    validate_conditions(changes)
    for key, value in changes.items():
        if key in CONDITION_SCHEMA:
            continue
        if key in METRICS:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("State values must be finite numbers from 0 to 100")
        elif key == "participants":
            if not isinstance(value, list) or len(value) > 12 or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value):
                raise ValueError("participants must be up to 12 distinct person IDs")
        elif key in SCENE_TEXT_FIELDS:
            if not isinstance(value, str) or len(value) > 300:
                raise ValueError("Scene fields must be strings of at most 300 characters")
        elif not isinstance(value, str) or not 1 <= len(value.strip()) <= 300:
            raise ValueError("State descriptions must contain 1–300 characters")


def _apply_changes(base: dict, changes: dict, *, adjustment: str, metric_reasons, reason: str,
                   event_id: str, event_type: str, warnings: list[str]) -> tuple[dict, str, dict | None]:
    """Apply an immediate update on top of ``base`` (the saved state, or the state after an interval)."""
    numeric = set(changes) & set(METRICS)
    state = {**base, **{k: v for k, v in changes.items() if k not in numeric}}
    if not numeric:
        return state, adjustment, metric_reasons
    if not adjustment:
        adjustment = "initialize" if all(base[k] is None for k in numeric) else "event"
        warnings.append(f"adjustment 미지정: {adjustment}로 처리함 (수치 직접 변경은 initialize/event/correction)")
    if adjustment not in ADJUSTMENTS:
        raise ValueError("adjustment must be initialize/event/correction")
    if adjustment == "event" and "resolve" in numeric:
        raise ValueError("의지(resolve)는 event로 직접 쓰지 않는다. resolve_event={kind, intensity}로 사건을 보내면 코드가 표에 따라 계산한다. "
                         f"kind: {', '.join(f'{k}={v[1]}' for k, v in RESOLVE_EVENT_KINDS.items())}; intensity 1 스침/2 보통/3 극심. 잘못된 값의 정정만 adjustment=correction")
    reasons = dict(metric_reasons) if isinstance(metric_reasons, dict) else {}
    unknown = set(reasons) - set(METRICS)
    if unknown:
        raise ValueError(f"metric_reasons keys must be metrics; got {sorted(unknown)}")
    missing = sorted(numeric - set(reasons))
    for key in missing:
        reasons[key] = reason
    if missing:
        warnings.append(f"metric_reasons가 없는 {missing}는 reason을 근거로 기록함")
    if any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 300 for v in reasons.values()):
        raise ValueError("Each metric reason must contain 1–300 characters")
    kept = []
    for key in sorted(numeric):
        if adjustment == "initialize" and base[key] is not None:
            kept.append(key)
            continue
        state[key] = changes[key]
    # A shock (a wound, or tension pushed up by an event) ends the calm streak that eases tension.
    if adjustment == "event" and (event_type == "injury" or ("tension" in numeric and base["tension"] is not None and changes["tension"] > base["tension"])):
        state["calm_minutes"] = 0
    if kept:
        warnings.append(f"initialize는 미설정 값만 채움: {kept}는 기존 값 유지 (바꾸려면 adjustment=event 또는 correction)")
    state["recent_events"] = (base["recent_events"] + [event_id])[-100:] if event_id not in base["recent_events"] else base["recent_events"]
    state["event_timestamps"] = (base["event_timestamps"] + [{
        "event_id": event_id, "type": event_type, "scene_minute": state["scene_minute"],
        "clock": {k: state["clock"][k] for k in ("date", "year", "time", "daypart", "relative_day", "certainty", "unquantified_gaps")},
        "reason": reason,
    }])[-50:]
    return state, adjustment, {k: reasons[k] for k in sorted(numeric)}


def _normalize_resolve_event(value, action: str, warnings: list[str]):
    if value in (None, {}, ""):
        return None
    if action not in {"update", "time"}:
        raise ValueError("resolve_event is only accepted by update/time")
    if isinstance(value, str):
        value = {"kind": value}
    if not isinstance(value, dict):
        raise ValueError("resolve_event must be {kind, intensity, note?}")
    item = dict(value)
    kind = item.pop("kind", None) or item.pop("type", None)
    intensity = item.pop("intensity", None)
    note = item.pop("note", "") or item.pop("reason", "")
    if item:
        raise ValueError(f"resolve_event accepts kind/intensity/note; got {sorted(item)}")
    if intensity is None:
        intensity = 2
        warnings.append("resolve_event.intensity 미지정: 2(보통)로 처리함")
    if isinstance(intensity, float) and intensity.is_integer():
        intensity = int(intensity)
    if kind not in RESOLVE_EVENT_KINDS or type(intensity) is not int or intensity not in RESOLVE_INTENSITY:
        raise ValueError(f"resolve_event.kind는 {', '.join(f'{k}({v[1]})' for k, v in RESOLVE_EVENT_KINDS.items())} 중 하나, intensity는 1 스침/2 보통/3 극심")
    if not isinstance(note, str) or len(note) > 300:
        raise ValueError("resolve_event.note must be a string of at most 300 characters")
    return {"kind": kind, "intensity": intensity, "note": note.strip()}


def _apply_resolve_event(state: dict, event: dict, *, event_id: str, reason: str) -> dict:
    """Take the table's toll on resolve (after any interval and the direct changes of the same call)."""
    if state["resolve"] is None:
        raise ValueError("resolve is unset; initialize it in changes with adjustment=initialize before sending resolve_event")
    delta, factors = resolve_event_delta(state, event["kind"], event["intensity"])
    before = state["resolve"]
    state["resolve"] = round(max(0, min(100, before + delta)), 4)
    record = {"event_id": event_id, "kind": event["kind"], "intensity": event["intensity"], "delta": delta,
              "from": before, "to": state["resolve"], "factors": factors, "scene_minute": state["scene_minute"],
              "note": event["note"] or reason}
    state["resolve_events"] = (state.get("resolve_events", []) + [record])[-RESOLVE_EVENT_HISTORY:]
    return record


def roleplay_state(action: str, changes: dict | None = None, reason: str = "", *,
                   expected_revision: int | None = None,
                   adjustment: str = "", event_id: str = "",
                   metric_reasons: dict | None = None, temporal: dict | None = None,
                   event_type: str = "other", interval_conditions: dict | None = None,
                   person_updates: list | None = None, person_review: str = "",
                   resolve_event: dict | None = None, **extra) -> str:
    caller = get_caller()
    user_id = _owner_id()
    if action == "read":
        return json.dumps(state_view(load_state(user_id)), ensure_ascii=False)
    if action == "history":
        with _connection() as conn:
            rows = conn.execute("SELECT payload FROM state_history WHERE user_id = ? ORDER BY id DESC LIMIT 20", (user_id,)).fetchall()
        records = [json.loads(row[0]) for row in rows]
        for record in records:
            for side in ("before", "after"):
                record[side] = {k: record[side].get(k) for k in (*METRICS, "scene_minute", "activity", "sleep_quality", "threat", "injuries", "clock")}
        return json.dumps(records, ensure_ascii=False)
    if action not in {"update", "reset", "time"}:
        raise ValueError("Unknown state action: read/history/update/time/reset")

    # ── Normalize the call into the canonical shape, recording what was reinterpreted.
    warnings: list[str] = []
    extra = dict(extra)
    for alias in list(extra):
        if alias in PERSON_REVIEW_ALIASES:
            person_review = person_review or extra.pop(alias)
            warnings.append(f"{alias}는 person_review로 해석함")
        elif alias in INTERVAL_ALIASES:
            interval_conditions = interval_conditions or extra.pop(alias)
            warnings.append(f"{alias}는 interval_conditions로 해석함")
    if isinstance(changes, dict) and isinstance(changes.get("metric_reasons"), dict):
        inner_reasons = dict(changes).pop("metric_reasons")
        changes = {k: v for k, v in changes.items() if k != "metric_reasons"}
        metric_reasons = {**inner_reasons, **(metric_reasons if isinstance(metric_reasons, dict) else {})}
        warnings.append("changes 안의 metric_reasons는 최상위 인자로 옮겨 적용함")
    changes, inner_reason, injury_upserts = _normalize_changes(changes, extra, warnings)
    if extra:
        raise ValueError(f"Unknown argument(s) {sorted(extra)}. Accepted: action, changes, reason, expected_revision, temporal, interval_conditions, person_updates, person_review, adjustment, event_id, metric_reasons, event_type, resolve_event")
    resolve_event = _normalize_resolve_event(resolve_event, action, warnings)
    person_updates, person_review = _normalize_person_updates(person_updates, person_review, warnings)
    if action == "time":
        temporal = _normalize_temporal(temporal, warnings)
    elif temporal is not None:
        raise ValueError("temporal is only accepted by action=time")
    if event_type not in EVENT_TYPES:
        raise ValueError("event_type must be other/meal/sleep/injury/treatment")
    if not isinstance(reason, str) or not reason.strip():
        candidates = [inner_reason, (temporal or {}).get("interpretation", ""),
                      *(metric_reasons.values() if isinstance(metric_reasons, dict) else []), person_review or ""]
        reason = next((c for c in candidates if isinstance(c, str) and c.strip()), "")
        if not reason:
            raise ValueError("A scene-based reason (1–300 characters) is required")
        if not inner_reason:
            warnings.append("reason 미지정: 시간 해석·수치 근거·검토 문구로 대체함")
    reason = reason.strip()
    if len(reason) > 300:
        reason = reason[:300]
        warnings.append("reason을 300자로 잘라 저장함")
    timed_interval = action == "time" and temporal["relation"] == "current" and temporal["operation"] in {"advance", "until"}
    if timed_interval:
        if interval_conditions is None and "activity" in changes:
            interval_conditions = {k: changes[k] for k in CONDITION_SCHEMA if k in changes}
            warnings.append("interval_conditions 미지정: changes의 활동·조건을 지난 구간에도 적용함. 지난 구간이 달랐다면 interval_conditions로 구분")
        if not isinstance(interval_conditions, dict) or "activity" not in interval_conditions or set(interval_conditions) - set(CONDITION_SCHEMA):
            raise ValueError("Supply interval_conditions={activity: rest/light/moderate/strenuous/sleep, sleep_quality?, threat?, injuries?} describing the interval that just elapsed; changes describe the state after it")
        validate_conditions(interval_conditions)
    elif interval_conditions is not None:
        warnings.append("interval_conditions는 현재 시간의 advance/until에만 쓰이므로 무시함")
        interval_conditions = None
    if not isinstance(event_id, str) or not 1 <= len(event_id.strip()) <= 100:
        if action == "time" or set(changes) & set(METRICS) or resolve_event:
            event_id = _auto_event_id(caller.scope_id, action, changes, temporal)
            warnings.append(f"event_id 미지정: {event_id}로 자동 생성함. 같은 사건을 재시도할 때만 재사용")
        else:
            event_id = ""
    event_id = event_id.strip()
    _validate_changes(changes)

    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (user_id,)).fetchone()
        before = with_defaults({**STATE_DEFAULTS, **(json.loads(row[0]) if row else {})})
        # Safe replay of an already applied absolute time/event never adds a second delta.
        if event_id and event_id in before["recent_events"]:
            return json.dumps({**state_view(before), "replayed": True,
                               "note": f"event_id '{event_id}'는 이미 반영되어 다시 적용하지 않음. 새 사건이면 다른 event_id를 사용"}, ensure_ascii=False)
        if expected_revision != before["revision"]:
            if caller.scope_id and before.get("last_scope_id") == caller.scope_id:
                warnings.append(f"expected_revision {expected_revision}은 오래됐지만 같은 턴의 연속 변경이라 현재 revision {before['revision']}에 적용함")
            elif expected_revision is None and before["revision"] == 0:
                pass
            else:
                raise ValueError(f"Read current state and retry with expected_revision={before['revision']}")
        if "participants" in changes:
            changes["participants"] = _resolve_person_ids(conn, user_id, changes["participants"], warnings, "participants")
        if injury_upserts:
            _merge_injury_upserts(before["injuries"], changes, injury_upserts, warnings)
            validate_conditions({"injuries": changes["injuries"]})
        if interval_conditions and "injuries" in interval_conditions:
            interval_conditions["injuries"] = carry_injury_progress(before["injuries"], interval_conditions["injuries"])
        if action == "time":
            interval_state = {**before, **(interval_conditions or {})}
            if interval_conditions and all(k in interval_conditions for k in CONDITION_SCHEMA):
                interval_state["conditions_initialized"] = True
            base = interpret_clock(interval_state, temporal, advance)
            base["recent_events"] = (before["recent_events"] + [event_id])[-100:]
            if (base.get("last_calculation") or {}).get("threat_relieved"):
                warnings.append(f"현장에 아무도 없는 {base['last_calculation']['to_minute'] - base['last_calculation']['from_minute']}분의 휴식·수면 구간이라 "
                                f"위협 {interval_state['threat']}은 uncertain으로 계산·저장함. 방문·호출 예고 같은 실제 위협이 이어지면 changes.threat로 되돌리고 그 근거를 reason에 적음")
            if "injuries" in changes:
                changes["injuries"] = reconcile_injuries(interval_state["injuries"], base["injuries"], changes["injuries"])
        elif action == "reset":
            base = with_defaults(dict(STATE_DEFAULTS))
            base["revision"] = before["revision"]
        else:
            base = before
            if "injuries" in changes:
                changes["injuries"] = carry_injury_progress(before["injuries"], changes["injuries"])
        state, adjustment, metric_reasons = _apply_changes(
            base, changes, adjustment=adjustment, metric_reasons=metric_reasons, reason=reason,
            event_id=event_id, event_type=event_type, warnings=warnings)
        if all(k in changes or k in (interval_conditions or {}) for k in CONDITION_SCHEMA):
            state["conditions_initialized"] = True
        applied_resolve = _apply_resolve_event(state, resolve_event, event_id=event_id, reason=reason) if resolve_event else None
        applied_people = _apply_person_updates(conn, user_id, person_updates or [], warnings)
        state["revision"] = before["revision"] + 1
        state["reason"] = reason
        state["last_scope_id"] = caller.scope_id
        audit = {"revision": state["revision"], "action": action, "reason": reason,
                 "adjustment": adjustment, "event_id": event_id, "metric_reasons": metric_reasons,
                 "temporal": temporal, "interval_conditions": interval_conditions,
                 "person_updates": applied_people, "person_review": person_review, "warnings": warnings,
                 "resolve_event": applied_resolve,
                 "source_scope_id": caller.scope_id, "before": before, "after": state}
        conn.execute("INSERT INTO state_history(user_id, revision, payload) VALUES (?, ?, ?)",
                     (user_id, state["revision"], json.dumps(audit, ensure_ascii=False)))
        conn.execute("DELETE FROM state_history WHERE user_id = ? AND id NOT IN (SELECT id FROM state_history WHERE user_id = ? ORDER BY id DESC LIMIT 1000)", (user_id, user_id))
        conn.execute("INSERT INTO character_state VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload",
                     (user_id, json.dumps(state, ensure_ascii=False)))
        present = [json.loads(r[0]).get("name", pid) for pid in state["participants"]
                   for r in conn.execute("SELECT payload FROM people WHERE user_id = ? AND person_id = ?", (user_id, pid))]
    result = state_view(state)
    review_needed = timed_interval or (action == "update" and bool(set(changes) & REVIEW_TRIGGERS))
    if review_needed and person_updates is None and not (person_review or "").strip():
        who = f"현장 인물 {present}의" if present else "관련 인물의"
        result["people_reminder"] = f"인물 기록 검토 없이 저장함. {who} 새 행동·발언·관계 변화가 있으면 person_updates 또는 roleplay_person(save)로 갱신"
    if warnings:
        result["warnings"] = warnings
    return json.dumps(result, ensure_ascii=False)


ROLEPLAY_STATE_TOOL = {
    "name": "roleplay_state",
    "description": "역할극 인물의 지속 상태표. read=현재 상태, history=최근 변경, update=즉시 변경, time=장면 시간 진행(+같은 호출의 changes로 그 뒤의 상태), reset=새 장면. 한 사건은 한 호출로: time에는 temporal(source_quote/interpretation/relation current·past·plan/certainty/operation anchor·advance·until·next_day·correct·reference)과 지난 구간의 interval_conditions.activity, 그리고 구간 뒤 달라진 것을 changes에 함께 넣는다. changes: hunger/fatigue/pain/tension과 정신 수치 clarity(명료함)/humiliation(굴욕)(0–100, 시간 경과가 아닌 식사·부상·굴욕 사건 같은 즉시 사건만; resolve(의지)는 initialize/correction만 직접 쓰고 사건은 resolve_event={kind,intensity}로 보내 코드가 계산), body/mood/scene, location/participants(등록된 인물 ID)/last_event/unresolved/goal/avoid/next_action, activity/sleep_quality/threat/injuries(전체 목록 교체). 빈 문자열·빈 배열로 해소된 항목을 비운다. expected_revision=현재 revision. person_updates=[{person_id, changes:{observed/reported/relationship/inferred}}]로 인물 기록을 같은 트랜잭션에서 갱신하고, 새 정보가 없으면 person_review에 이유. 누락·형식 차이는 가능한 한 해석해 적용하고 결과의 warnings에 알린다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "history", "update", "reset", "time"]},
            "changes": {"type": "object", "properties": {
                **CONDITION_SCHEMA,
                **{k: {"type": "number", "minimum": 0, "maximum": 100} for k in METRICS},
                **{k: {"type": "string", "maxLength": 300} for k in ["body", "mood", "scene", *sorted(SCENE_TEXT_FIELDS)]},
                "participants": {"type": "array", "items": {"type": "string"}, "maxItems": 12, "uniqueItems": True},
            }, "additionalProperties": True},
            "reason": {"type": "string", "maxLength": 300, "description": "장면 속 변경 이유 (최상위 인자)"},
            "expected_revision": {"type": "integer", "minimum": 0},
            "temporal": TEMPORAL_SCHEMA,
            "interval_conditions": {"type": "object", "properties": CONDITION_SCHEMA, "additionalProperties": False,
                                    "description": "방금 지나간 구간의 활동·수면·위협·부상 (activity 필수)"},
            "person_updates": PERSON_UPDATES_SCHEMA,
            "person_review": {"type": "string", "maxLength": 300},
            "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
            "adjustment": {"type": "string", "enum": sorted(ADJUSTMENTS)},
            "event_id": {"type": "string", "maxLength": 100},
            "metric_reasons": {"type": "object", "properties": {k: {"type": "string", "maxLength": 300} for k in METRICS}, "additionalProperties": False},
            "resolve_event": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": list(RESOLVE_EVENT_KINDS),
                         "description": "; ".join(f"{k}={v[1]}" for k, v in RESOLVE_EVENT_KINDS.items())},
                "intensity": {"type": "integer", "minimum": 1, "maximum": 3, "description": "1 스침, 2 보통, 3 극심"},
                "note": {"type": "string", "maxLength": 300, "description": "이 사건이 의지에 미친 영향의 장면 근거"},
            }, "required": ["kind", "intensity"], "additionalProperties": False,
                "description": "의지(resolve)를 깎거나 돌리는 사건. 수치는 코드가 종류·강도·상태(통증 60↑, 피로 70↑, 고립)·반복으로 계산한다"},
        }, "required": ["action"], "additionalProperties": True,
    },
}


def roleplay_person(action: str, person_id: str = "", query: str = "", changes: dict | None = None, reason: str = "") -> str:
    user_id = _owner_id()
    if action in {"list", "read"}:
        people = load_people(user_id)
        if action == "read":
            if not person_id and not query.strip():
                raise ValueError("read requires a person_id or name/alias query")
            wanted = {person_id, _normalize_person_id(person_id)} - {""}
            people = [p for p in people if p["person_id"] in wanted] if person_id else [
                p for p in people if _name_key(query) in {_name_key(p["name"]), *(_name_key(a) for a in p["aliases"])}]
        else:
            people = [{k: p[k] for k in ("person_id", "name", "aliases", "identity", "commulingo_id", "commulingo_url")} for p in people]
        return json.dumps({"matches": people, "ambiguous": action == "read" and len(people) > 1}, ensure_ascii=False)
    if action not in {"save", "delete"}:
        raise ValueError("Unknown person action")
    warnings = []
    normalized = _normalize_person_id(person_id)
    if not normalized:
        raise ValueError("Use a stable lowercase Latin person_id (1–64 letters/digits/_/-), e.g. 'young_guard'")
    if normalized != person_id:
        warnings.append(f"person_id '{person_id}'는 '{normalized}'로 정규화함")
        person_id = normalized
    if changes is not None and not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    changes = dict(changes or {})
    changes.pop("reason", None)
    changes.pop("person_id", None)
    for key, value in changes.items():
        if key == "aliases":
            if not isinstance(value, list) or len(value) > 8 or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 100 for v in value):
                raise ValueError("Use up to 8 nonempty aliases of at most 100 characters")
        elif key not in PERSON_FIELDS or not isinstance(value, str) or len(value) > PERSON_FIELDS[key] or (key == "name" and not value.strip()):
            raise ValueError(f"Unknown or invalid person field '{key}'. Fields: {sorted(PERSON_FIELDS)}, aliases")
    if action == "save" and changes.get("commulingo_id"):
        _validate_commulingo_id(changes["commulingo_id"])
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if action == "delete":
            count = conn.execute("DELETE FROM people WHERE user_id = ? AND person_id = ?", (user_id, person_id)).rowcount
            row = conn.execute("SELECT payload FROM character_state WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                state = json.loads(row[0])
                state["revision"] = state.get("revision", 0) + 1
                state["last_scope_id"] = get_caller().scope_id
                state["participants"] = [pid for pid in state.get("participants", []) if pid != person_id]
                conn.execute("UPDATE character_state SET payload = ? WHERE user_id = ?", (json.dumps(state, ensure_ascii=False), user_id))
            return json.dumps({"deleted": bool(count), "person_id": person_id, **({"warnings": warnings} if warnings else {})}, ensure_ascii=False)
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
    result = {"saved": True, **_with_dictionary_link({"person_id": person_id, **person})}
    if warnings:
        result["warnings"] = warnings
    return json.dumps(result, ensure_ascii=False)


ROLEPLAY_PERSON_TOOL = {
    "name": "roleplay_person",
    "description": "사용자별 인물 기록. commulingo_people로 동일 인물임을 확인한 사전 ID를 changes.commulingo_id에 저장하면 공식 링크를 연결한다. 빈 문자열은 연결 해제. list=이름·ID 목록, read=person_id 또는 이름/별칭(query) 조회, save=생성/부분 수정, delete=삭제. person_id는 소문자 라틴 고정 식별자(예: young_guard). 같은 인물은 안정된 ID를 유지하고 이름이 같아도 자동 병합하지 않는다. identity=인물 식별 정보와 적용 시기, relationship=예조프와의 관계, observed=직접 관찰한 사실, reported=전해 들은 말과 출처, inferred=미확정 추측. 목록은 최대 30명. 필드 수정은 해당 필드 전체를 교체하므로 기존 유효 정보를 보존한다.",
    "input_schema": {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "read", "save", "delete"]},
            "person_id": {"type": "string", "maxLength": 64},
            "query": {"type": "string", "maxLength": 100},
            "changes": {"type": "object", "properties": {
                **{k: {"type": "string", "maxLength": limit} for k, limit in PERSON_FIELDS.items()},
                "aliases": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 8},
            }, "additionalProperties": True},
            "reason": {"type": "string", "maxLength": 300, "description": "선택: 변경 이유 (기록용)"},
        }, "required": ["action"], "additionalProperties": False,
    },
}
