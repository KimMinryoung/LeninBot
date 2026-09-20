"""Bounded fictional appointments and consequences; no wall-clock jobs or narration."""
from copy import deepcopy

MAX_ACTIVE_EVENTS = 20
MAX_EVENTS = 100
TERMINAL = {"completed", "cancelled"}


ALONE_EVENT_EXPIRY_MINUTES = 1440


def refresh_events(state):
    events = state.get("story_events", [])
    by_id = {event["id"]: event for event in events}
    for event in events:
        if event["status"] in TERMINAL:
            continue
        dependency = by_id.get(event.get("after_event"))
        if dependency and dependency["status"] == "cancelled":
            event.update(status="cancelled", outcome="선행 사건 취소", resolved_minute=state["scene_minute"])
        elif event.get("when_alone"):
            # A cue for the next scene the character spends alone, never a clock stop.
            if event["status"] == "ready" and state["scene_minute"] - event.get("ready_minute", state["scene_minute"]) >= ALONE_EVENT_EXPIRY_MINUTES:
                event.update(status="cancelled", outcome="혼자 남은 장면이 이어지지 않아 시기를 놓침", resolved_minute=state["scene_minute"])
            elif event["status"] == "pending" and not state.get("participants") and (not dependency or dependency["status"] == "completed"):
                event.update(status="ready", ready_minute=state["scene_minute"])
        elif ((not dependency or dependency["status"] == "completed")
              and event.get("due_minute", state["scene_minute"]) <= state["scene_minute"]):
            event["status"] = "ready"
    return state


def apply_story_updates(state, updates):
    if not isinstance(updates, list) or len(updates) > MAX_ACTIVE_EVENTS:
        raise ValueError("story_updates must be a list of at most 20 operations")
    state = deepcopy(state)
    events = state.setdefault("story_events", [])
    for update in updates:
        if not isinstance(update, dict) or set(update) - {"op", "id", "title", "source", "due_minute", "after_event", "outcome", "when_alone", "kind", "date"}:
            raise ValueError("Invalid story update fields")
        if "kind" in update and (not isinstance(update["kind"], str) or not 1 <= len(update["kind"]) <= 32):
            raise ValueError("kind is a short label such as routine or track")
        if "date" in update and (not isinstance(update["date"], str) or len(update["date"]) != 10):
            raise ValueError("date is YYYY-MM-DD")
        op, eid = update.get("op"), update.get("id")
        if not isinstance(eid, str) or not 1 <= len(eid.strip()) <= 64:
            raise ValueError("Story event needs a stable id of 1–64 characters")
        by_id = {event["id"]: event for event in events}
        old = by_id.get(eid)
        if op == "schedule":
            if "outcome" in update:
                raise ValueError("A scheduled event has no completed outcome")
            for key in ("title", "source"):
                if not isinstance(update.get(key), str) or not 1 <= len(update[key].strip()) <= 300:
                    raise ValueError(f"Scheduled event needs {key} (1–300 characters)")
            if "when_alone" in update and update["when_alone"] is not True:
                raise ValueError("when_alone is either true or omitted")
            if "due_minute" not in update and "after_event" not in update and not update.get("when_alone"):
                raise ValueError("Scheduled event needs due_minute, after_event or when_alone")
            if "due_minute" in update and (type(update["due_minute"]) is not int or update["due_minute"] < 0):
                raise ValueError("due_minute is an absolute nonnegative scene minute")
            if "after_event" in update and (not isinstance(update["after_event"], str) or update["after_event"] not in by_id):
                raise ValueError("after_event must name an already registered event")
            fields = {k: v for k, v in update.items() if k != "op"}
            if old:
                if any(old.get(k) != v for k, v in fields.items()) or any(k in old and k not in fields for k in ("due_minute", "after_event")):
                    raise ValueError("Event id already exists; cancel it and use a new id for a changed plan")
                continue  # retries must never revive a completed/cancelled event
            if update.get("due_minute", state["scene_minute"]) < state["scene_minute"]:
                raise ValueError("Cannot schedule a new event in the past")
            events.append({**fields, "status": "pending", "created_minute": state["scene_minute"]})
        elif op in {"complete", "cancel"}:
            if set(update) - {"op", "id", "outcome"}:
                raise ValueError("complete/cancel accept id and outcome only")
            if not old:
                raise ValueError("Unknown story event id")
            outcome = update.get("outcome")
            if not isinstance(outcome, str) or not 1 <= len(outcome.strip()) <= 300:
                raise ValueError("Record the completed outcome or cancellation reason")
            status = "completed" if op == "complete" else "cancelled"
            if old["status"] in TERMINAL:
                if old["status"] != status or old.get("outcome") != outcome:
                    raise ValueError("A resolved story event cannot be rewritten")
                continue
            if op == "complete" and old["status"] != "ready":
                raise ValueError("Event is not ready; advance to its time or complete its prerequisite first")
            old.update(status=status, outcome=outcome, resolved_minute=state["scene_minute"])
        else:
            raise ValueError("story_updates.op must be schedule/complete/cancel")
        refresh_events(state)
    if sum(e["status"] not in TERMINAL for e in events) > MAX_ACTIVE_EVENTS:
        raise ValueError("At most 20 active story events")
    # Keep referenced prerequisites and recent outcomes. Active plans are never evicted.
    referenced = {e.get("after_event") for e in events}
    while len(events) > MAX_EVENTS:
        removable = next((e for e in events if e["status"] in TERMINAL and e["id"] not in referenced), None)
        if removable is None:
            raise ValueError("Story event history is full; reset for a new scene")
        events.remove(removable)
    state["story_interrupt"] = None
    return state


def blocking_events(state, horizon_minutes):
    """Active clock-stopping events that are ready now or due within the horizon.
    Cues (when_alone) and far-off milestones never block a day skip."""
    by_id = {e["id"]: e for e in state.get("story_events", [])}
    result = []
    for event in by_id.values():
        if event["status"] in TERMINAL or event.get("when_alone"):
            continue
        dependency = by_id.get(event.get("after_event"))
        if dependency and dependency["status"] != "completed":
            continue
        if event["status"] == "ready" or event.get("due_minute", state["scene_minute"]) <= state["scene_minute"] + horizon_minutes:
            result.append(event)
    return result


def advance_to_event(state, target, basis, advance_fn):
    """Stop at the first eligible event; effects at the requested endpoint are deferred."""
    state = refresh_events(deepcopy(state))
    by_id = {e["id"]: e for e in state.get("story_events", [])}
    candidates = []
    for event in by_id.values():
        if event["status"] in TERMINAL or event.get("when_alone"):
            continue
        dependency = by_id.get(event.get("after_event"))
        if dependency and dependency["status"] != "completed":
            continue
        due = max(state["scene_minute"], event.get("due_minute", state["scene_minute"]))
        if due <= target:
            candidates.append(due)
    stop = min(candidates) if candidates else target
    result = advance_fn(state, stop, basis)
    result["story_interrupt"] = None
    refresh_events(result)
    ready = [e["id"] for e in result.get("story_events", []) if e["status"] == "ready"]
    if ready:
        result["story_interrupt"] = {"event_ids": ready, "requested_minute": target,
                                     "stopped_minute": stop, "remaining_minutes": target - stop}
    return result


STORY_UPDATES_SCHEMA = {
    "type": "array", "maxItems": MAX_ACTIVE_EVENTS,
    "description": "update에서만 사용. schedule=근거 있는 예정 사건, complete=도래한 사건의 실제 결과, cancel=취소. due_minute는 누적 장면 분, after_event는 선행 사건 ID. 완료 예정은 완료 사실이 아님.",
    "items": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["schedule", "complete", "cancel"]},
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        "title": {"type": "string", "maxLength": 300},
        "source": {"type": "string", "maxLength": 300},
        "due_minute": {"type": "integer", "minimum": 0},
        "after_event": {"type": "string", "maxLength": 64},
        "when_alone": {"type": "boolean", "description": "true면 인물이 혼자 남는 순간 ready가 되는 장면 신호. 시계를 멈추지 않으며 하루 안에 다루지 않으면 취소"},
        "kind": {"type": "string", "maxLength": 32, "description": "routine(일과)·track(실존 연표) 등 코드가 만든 사건의 종류. 보통 생략"},
        "date": {"type": "string", "maxLength": 10, "description": "연표 사건의 실제 날짜 YYYY-MM-DD. 보통 생략"},
        "outcome": {"type": "string", "maxLength": 300},
    }, "required": ["op", "id"], "additionalProperties": False},
}
