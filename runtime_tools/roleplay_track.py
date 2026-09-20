"""Optional historical track: the documented milestones of Yezhov's last year as pending
story events, so the director can see where the played story sits against the record.
Dates follow the owner's saved source notes; the game does not verify them further."""
from datetime import date

from runtime_tools.roleplay_dynamics import clock_minute_of_day
from runtime_tools.roleplay_story import apply_story_updates

TRACK_KIND = "track"
TRACK_SOURCE = "실존 연표(사용자 사료 메모 기반)"
TRACK_MORNING_MINUTE = 6 * 60
RELEASE_OUTCOME = "궤도 해제"
MILESTONES = (
    ("1939-04-30", "조서: 음모 가담자 66명 지목"),
    ("1939-06-10", "기소장 작성 (남색 혐의 포함)"),
    ("1939-08-04", "조서: 대량 작전을 음모의 이익으로 썼다고 인정"),
    ("1940-02-01", "최종 기소 (남색 혐의 제외)"),
    ("1940-02-02", "군사법정 비공개 재판 시작"),
    ("1940-02-03", "최후 진술"),
    ("1940-02-04", "처형 (일부 자료 02-06)"),
)


def event_id(day):
    return f"track-{day}"


def enable(state):
    """Schedule the milestones still ahead of the scene date. Needs a dated, timed clock."""
    clock = state.get("clock") or {}
    now = clock_minute_of_day(state)
    if not clock.get("date") or now is None:
        raise ValueError("실존 궤도는 장면의 날짜와 시각이 알려진 뒤에 켤 수 있음")
    today = date.fromisoformat(clock["date"])
    existing = {e["id"] for e in state.get("story_events", [])}
    updates = []
    for day, title in MILESTONES:
        when = date.fromisoformat(day)
        if when <= today or event_id(day) in existing:
            continue
        due = state["scene_minute"] + (when - today).days * 1440 - now + TRACK_MORNING_MINUTE
        updates.append({"op": "schedule", "id": event_id(day), "kind": TRACK_KIND, "date": day,
                        "title": f"{day} {title}", "source": TRACK_SOURCE, "due_minute": due})
    if not updates:
        raise ValueError("남은 실존 연표 사건이 없음 (장면 날짜가 연표 끝을 지났거나 이미 등록됨)")
    result = apply_story_updates(state, updates)
    result["track"] = {"enabled": True, "enabled_minute": state["scene_minute"], "from_date": clock["date"]}
    return result


def disable(state):
    updates = [{"op": "cancel", "id": e["id"], "outcome": RELEASE_OUTCOME} for e in state.get("story_events", [])
               if e.get("kind") == TRACK_KIND and e["status"] in {"pending", "ready"}]
    result = apply_story_updates(state, updates) if updates else dict(state)
    result["track"] = {**(state.get("track") or {}), "enabled": False}
    return result


def summary(state):
    """What the director sees: next milestone, matches and departures so far."""
    events = [e for e in state.get("story_events", []) if e.get("kind") == TRACK_KIND]
    if not events and not (state.get("track") or {}).get("enabled"):
        return None
    matched = [e for e in events if e["status"] == "completed"]
    departed = [e for e in events if e["status"] == "cancelled" and e.get("outcome") != RELEASE_OUTCOME]
    upcoming = sorted((e for e in events if e["status"] in {"pending", "ready"}), key=lambda e: e.get("due_minute", 0))
    nxt = None
    if upcoming:
        remaining = upcoming[0].get("due_minute", state["scene_minute"]) - state["scene_minute"]
        nxt = {"title": upcoming[0]["title"], "days": max(0, remaining) // 1440, "status": upcoming[0]["status"]}
    return {"enabled": bool((state.get("track") or {}).get("enabled")), "next": nxt,
            "matched": len(matched), "departed": len(departed), "remaining": len(upcoming)}
