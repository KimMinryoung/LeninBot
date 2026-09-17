"""Deterministic fictional state progression; rates are game tuning, not medicine."""
from copy import deepcopy
from runtime_tools.roleplay_clock import clock_defaults

METRICS = ("hunger", "fatigue", "pain", "tension")
ACTIVITIES = {"rest": -2.0, "light": 2.0, "moderate": 5.0, "strenuous": 10.0, "sleep": -8.0}
SLEEP_QUALITY = {"poor": 0.25, "normal": 1.0, "good": 1.25}
THREAT_TARGETS = {"safe": 10.0, "uncertain": 40.0, "threatening": 75.0, "immediate": 90.0}
DYNAMICS_DEFAULTS = {
    "revision": 0, "scene_minute": 0, "last_calculated_minute": 0,
    "time_basis": "현재 저장 상태를 기준 시점(0분)으로 삼음. 이전 경과 시간은 재계산하지 않음.",
    "activity": "rest", "sleep_quality": "normal", "threat": "uncertain",
    "injuries": [], "conditions_initialized": False, "recent_events": [],
    "last_calculation": None, "clock": None, "event_timestamps": [],
}


def with_defaults(state):
    result = {**deepcopy(DYNAMICS_DEFAULTS), **state}
    result["clock"] = clock_defaults(result.get("clock"))
    return result


def validate_conditions(changes):
    enums = {"activity": ACTIVITIES, "sleep_quality": SLEEP_QUALITY, "threat": THREAT_TARGETS}
    for key, allowed in enums.items():
        if key in changes and changes[key] not in allowed:
            raise ValueError(f"Invalid {key}")
    if "injuries" in changes:
        injuries = changes["injuries"]
        if not isinstance(injuries, list) or len(injuries) > 8:
            raise ValueError("Use at most 8 ongoing injuries")
        ids = set()
        for item in injuries:
            required = {"id", "description", "severity", "trend", "treated"}
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("Each injury needs id/description/severity/trend/treated")
            if any(not isinstance(item[k], str) or not 1 <= len(item[k].strip()) <= limit for k, limit in (("id", 64), ("description", 200))):
                raise ValueError("Invalid injury description or ID")
            if item["id"] in ids:
                raise ValueError("Duplicate injury ID")
            ids.add(item["id"])
            if type(item["severity"]) is not int or not 1 <= item["severity"] <= 3:
                raise ValueError("Injury severity must be 1–3")
            if item["trend"] not in {"stable", "worsening", "recovering"} or type(item["treated"]) is not bool:
                raise ValueError("Invalid injury trend/treatment")


def advance(state, target_minute, time_basis):
    """Apply only the as-yet unaccounted interval using previously saved conditions."""
    if type(target_minute) is not int or target_minute < 0:
        raise ValueError("target_minute must be an absolute nonnegative scene minute")
    start = state["last_calculated_minute"]
    if target_minute < start:
        raise ValueError("Cannot move backwards; reset for a different scene")
    if target_minute == start:
        return state
    if target_minute - start > 1440:
        raise ValueError("Advance at most 1440 minutes per interval; split longer passages")
    if not state["conditions_initialized"]:
        raise ValueError("First update activity, sleep_quality, threat and injuries from the current scene")
    if not isinstance(time_basis, str) or not 1 <= len(time_basis.strip()) <= 300:
        raise ValueError("Explain the fictional elapsed time in time_basis (1–300 characters)")
    result = deepcopy(state)
    hours = (target_minute - start) / 60
    activity = state["activity"]
    fatigue_rate = ACTIVITIES[activity]
    if activity == "sleep":
        fatigue_rate *= SLEEP_QUALITY[state["sleep_quality"]]
    movement = {"rest": 0, "sleep": 0, "light": 0.1, "moderate": 0.5, "strenuous": 1.5}[activity]
    pain_rate = 0.0
    for injury in state["injuries"]:
        trend = injury["trend"]
        drift = (0.25 if injury["treated"] else 0.5) if trend == "worsening" else (-0.5 if trend == "recovering" else 0)
        pain_rate += injury["severity"] * (drift + movement)
    tension = state["tension"]
    target = THREAT_TARGETS[state["threat"]]
    tension_delta = 0 if tension is None else max(-6 * hours, min(6 * hours, target - tension))
    deltas = {"hunger": 3 * hours, "fatigue": fatigue_rate * hours,
              "pain": pain_rate * hours, "tension": tension_delta}
    for key, delta in deltas.items():
        if state[key] is not None:
            result[key] = round(max(0, min(100, state[key] + delta)), 4)
    result.update(scene_minute=target_minute, last_calculated_minute=target_minute, time_basis=time_basis.strip())
    result["last_calculation"] = {
        "from_minute": start, "to_minute": target_minute, "basis": time_basis.strip(),
        "conditions": {k: deepcopy(state[k]) for k in ("activity", "sleep_quality", "threat", "injuries")},
        "before": {k: state[k] for k in METRICS}, "after": {k: result[k] for k in METRICS},
    }
    return result


CONDITION_SCHEMA = {
    "activity": {"type": "string", "enum": list(ACTIVITIES)},
    "sleep_quality": {"type": "string", "enum": list(SLEEP_QUALITY)},
    "threat": {"type": "string", "enum": list(THREAT_TARGETS)},
    "injuries": {"type": "array", "maxItems": 8, "items": {
        "type": "object", "properties": {
            "id": {"type": "string", "maxLength": 64},
            "description": {"type": "string", "maxLength": 200},
            "severity": {"type": "integer", "minimum": 1, "maximum": 3},
            "trend": {"type": "string", "enum": ["stable", "worsening", "recovering"]},
            "treated": {"type": "boolean"},
        }, "required": ["id", "description", "severity", "trend", "treated"], "additionalProperties": False,
    }},
}
