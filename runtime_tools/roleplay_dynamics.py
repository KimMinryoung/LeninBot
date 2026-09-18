"""Deterministic fictional state progression; rates are game tuning, not medicine."""
from copy import deepcopy
from runtime_tools.roleplay_clock import clock_defaults

PHYSICAL_METRICS = ("hunger", "fatigue", "pain", "tension")
# Mental axes: resolve/clarity read 100 = strong/lucid, humiliation reads 100 = extreme.
MENTAL_METRICS = ("resolve", "clarity", "humiliation")
METRICS = PHYSICAL_METRICS + MENTAL_METRICS
RESOLVE_BY_THREAT = {"safe": 2.0, "uncertain": 0.5, "threatening": -1.5, "immediate": -3.0}
ACTIVITIES = {"rest": -2.0, "light": 2.0, "moderate": 5.0, "strenuous": 10.0, "sleep": -8.0}
SLEEP_QUALITY = {"poor": 0.25, "normal": 1.0, "good": 1.25}
THREAT_TARGETS = {"safe": 10.0, "uncertain": 40.0, "threatening": 75.0, "immediate": 90.0}
# Tension settles toward a target that threat sets but quiet hours, resolve,
# sleep and pain move: an uneventful day in a cell should be felt.
HABITUATION_MAX = 15.0      # target drops 1 per calm hour (safe/uncertain), up to this
RESOLVE_TENSION_SPAN = 10.0  # resolve 100 lowers the target by this, resolve 0 raises it
SLEEP_TENSION_RELIEF = 10.0
PAIN_TENSION_PENALTY = 5.0   # while pain is at or above PAIN_HINDERS_REST
MAX_INJURIES = 12
# Wounds imply a baseline of pain that rest alone cannot remove; several wounds
# combine noisy-or style so a dozen bruises do not add up past a burn.
PAIN_FLOOR_BY_SEVERITY = {1: 4.0, 2: 8.0, 3: 14.0}
TREATED_FLOOR_FACTOR = 0.75
PAIN_FLOOR_APPROACH = 3.0  # per hour, when an event left pain below the floor
PAIN_DRIFT = {"worsening": {True: 0.25, False: 0.5}, "recovering": {True: -0.25, False: -0.15}, "stable": {True: 0.0, False: 0.0}}
MOVEMENT_PAIN = {"rest": 0, "sleep": 0, "light": 0.1, "moderate": 0.5, "strenuous": 1.5}
# Scene minutes of recovering/worsening before severity moves one step.
HEALING_STEP = {True: 1440, False: 2880}
WORSENING_STEP = {True: 2880, False: 1440}
PAIN_HINDERS_REST = 50  # at or above this, rest and sleep recover half the fatigue
DYNAMICS_DEFAULTS = {
    **{key: None for key in METRICS},  # legacy states lack the mental axes; unset stays unset
    "revision": 0, "scene_minute": 0, "last_calculated_minute": 0,
    "time_basis": "현재 저장 상태를 기준 시점(0분)으로 삼음. 이전 경과 시간은 재계산하지 않음.",
    "activity": "rest", "sleep_quality": "normal", "threat": "uncertain",
    "injuries": [], "conditions_initialized": False, "recent_events": [], "calm_minutes": 0,
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
        if not isinstance(injuries, list) or len(injuries) > MAX_INJURIES:
            raise ValueError(f"Use at most {MAX_INJURIES} ongoing injuries; merge related wounds into one entry")
        ids = set()
        for item in injuries:
            required = {"id", "description", "severity", "trend", "treated"}
            if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"progress_minutes"}:
                raise ValueError("Each injury needs id/description/severity/trend/treated")
            if "progress_minutes" in item and (type(item["progress_minutes"]) is not int or item["progress_minutes"] < 0):
                raise ValueError("progress_minutes is a nonnegative integer kept by the server")
            if any(not isinstance(item[k], str) or not 1 <= len(item[k].strip()) <= limit for k, limit in (("id", 64), ("description", 200))):
                raise ValueError("Invalid injury description or ID")
            if item["id"] in ids:
                raise ValueError("Duplicate injury ID")
            ids.add(item["id"])
            if type(item["severity"]) is not int or not 1 <= item["severity"] <= 3:
                raise ValueError("Injury severity must be 1–3")
            if item["trend"] not in {"stable", "worsening", "recovering"} or type(item["treated"]) is not bool:
                raise ValueError("Invalid injury trend/treatment")


def mental_rates(state):
    """Hourly drift of the mental axes from the conditions at the start of the interval."""
    fatigue = state["fatigue"] if state["fatigue"] is not None else 0
    pain = state["pain"] if state["pain"] is not None else 0
    threat, activity = state["threat"], state["activity"]
    sleeping = activity == "sleep"
    resolve = RESOLVE_BY_THREAT[threat] - (1 if fatigue > 70 else 0) - (1 if pain > 60 else 0)
    if sleeping and state["sleep_quality"] != "poor":
        resolve += 1
    if sleeping:
        clarity = 6 * SLEEP_QUALITY[state["sleep_quality"]]
    else:
        clarity = -(2 if fatigue > 80 else 1 if fatigue > 60 else 0) - (1 if pain > 60 else 0) - (1 if threat == "immediate" else 0)
        if activity == "rest" and fatigue <= 60:
            clarity += 1
    return {"resolve": resolve, "clarity": clarity, "humiliation": -0.5 if threat == "safe" else 0.0}


def injury_pain_floor(injuries):
    remaining = 1.0
    for injury in injuries:
        floor = PAIN_FLOOR_BY_SEVERITY[injury["severity"]] * (TREATED_FLOOR_FACTOR if injury["treated"] else 1)
        remaining *= 1 - floor / 100
    return round((1 - remaining) * 100, 4)


def carry_injury_progress(previous, injuries):
    """Keep the server-side healing clock across a model-sent injury list; a changed trend restarts it."""
    before = {i["id"]: i for i in previous}
    result = []
    for injury in injuries:
        item = dict(injury)
        old = before.get(item["id"])
        if "progress_minutes" not in item:
            # Records saved before the healing clock existed carry no progress yet.
            item["progress_minutes"] = old.get("progress_minutes", 0) if old and old.get("trend") == item["trend"] else 0
        result.append(item)
    return result


def reconcile_injuries(before, after_interval, sent):
    """A list the model sends after an interval usually echoes what it saw before it.
    Keep the server's healing where the model merely echoed; keep the model's
    values where it changed severity or trend; drop echoes of wounds that healed."""
    old_by_id = {i["id"]: i for i in before}
    cur_by_id = {i["id"]: i for i in after_interval}
    result = []
    for injury in sent:
        item = dict(injury)
        old, cur = old_by_id.get(item["id"]), cur_by_id.get(item["id"])
        echoed = old is not None and item["severity"] == old["severity"] and item["trend"] == old["trend"]
        if echoed and cur is None:
            continue
        if echoed:
            item["severity"] = cur["severity"]
            item.setdefault("progress_minutes", cur.get("progress_minutes", 0))
        elif cur is not None and item["trend"] == cur["trend"]:
            item.setdefault("progress_minutes", cur.get("progress_minutes", 0))
        else:
            item.setdefault("progress_minutes", 0)
        result.append(item)
    return result


def progress_injuries(injuries, minutes):
    """Advance healing/worsening clocks; return (updated list, [{id, from, to}] for severity moves)."""
    updated, changes = [], []
    for injury in injuries:
        item = {**injury, "progress_minutes": injury.get("progress_minutes", 0)}
        if item["trend"] == "recovering":
            step, direction = HEALING_STEP[item["treated"]], -1
        elif item["trend"] == "worsening":
            step, direction = WORSENING_STEP[item["treated"]], 1
        else:
            updated.append(item)
            continue
        item["progress_minutes"] += minutes
        severity = item["severity"]
        while item["progress_minutes"] >= step and 0 < severity and not (direction > 0 and severity >= 3):
            item["progress_minutes"] -= step
            severity += direction
        if direction > 0 and severity >= 3:
            item["progress_minutes"] = min(item["progress_minutes"], step)
        if severity != item["severity"]:
            changes.append({"id": item["id"], "from": item["severity"], "to": severity})
        if severity > 0:
            updated.append({**item, "severity": severity})
    return updated, changes


def tension_target(state):
    calm_hours = state.get("calm_minutes", 0) / 60
    target = THREAT_TARGETS[state["threat"]] - min(HABITUATION_MAX, calm_hours)
    if state["resolve"] is not None:
        target -= (state["resolve"] - 50) / 50 * RESOLVE_TENSION_SPAN
    if state["pain"] is not None and state["pain"] >= PAIN_HINDERS_REST:
        target += PAIN_TENSION_PENALTY
    if state["activity"] == "sleep":
        target -= SLEEP_TENSION_RELIEF
    return round(max(5.0, min(95.0, target)), 4)


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
        raise ValueError("Time cannot be calculated until activity, sleep_quality, threat and injuries are all set for this scene: pass all four in interval_conditions (or set them with update first)")
    if not isinstance(time_basis, str) or not 1 <= len(time_basis.strip()) <= 300:
        raise ValueError("Explain the fictional elapsed time in time_basis (1–300 characters)")
    result = deepcopy(state)
    hours = (target_minute - start) / 60
    activity = state["activity"]
    pain = state["pain"]
    fatigue_rate = ACTIVITIES[activity]
    if activity == "sleep":
        fatigue_rate *= SLEEP_QUALITY[state["sleep_quality"]]
    if activity in ("rest", "sleep") and pain is not None and pain >= PAIN_HINDERS_REST:
        fatigue_rate /= 2
    movement = MOVEMENT_PAIN[activity]
    pain_rate = sum(i["severity"] * (PAIN_DRIFT[i["trend"]][i["treated"]] + movement) for i in state["injuries"])
    floor = injury_pain_floor(state["injuries"])
    if pain is None:
        pain_delta = 0
    elif pain < floor:
        pain_delta = min(floor - pain, max(pain_rate, PAIN_FLOOR_APPROACH) * hours)
    else:
        pain_delta = max(floor - pain, pain_rate * hours)
    tension = state["tension"]
    target = tension_target(state)
    tension_delta = 0 if tension is None else max(-6 * hours, min(6 * hours, target - tension))
    calm = state["threat"] in ("safe", "uncertain")
    result["calm_minutes"] = state.get("calm_minutes", 0) + (target_minute - start) if calm else 0
    deltas = {"hunger": 3 * hours, "fatigue": fatigue_rate * hours,
              "pain": pain_delta, "tension": tension_delta}
    deltas.update({k: v * hours for k, v in mental_rates(state).items()})
    for key, delta in deltas.items():
        if state[key] is not None:
            result[key] = round(max(0, min(100, state[key] + delta)), 4)
    result["injuries"], injury_changes = progress_injuries(state["injuries"], target_minute - start)
    result.update(scene_minute=target_minute, last_calculated_minute=target_minute, time_basis=time_basis.strip())
    result["last_calculation"] = {
        "from_minute": start, "to_minute": target_minute, "basis": time_basis.strip(),
        "conditions": {k: deepcopy(state[k]) for k in ("activity", "sleep_quality", "threat", "injuries")},
        "before": {k: state[k] for k in METRICS}, "after": {k: result[k] for k in METRICS},
        "pain_floor": floor, "tension_target": target, "injury_changes": injury_changes,
        "healed": [c["id"] for c in injury_changes if c["to"] == 0],
    }
    return result


CONDITION_SCHEMA = {
    "activity": {"type": "string", "enum": list(ACTIVITIES)},
    "sleep_quality": {"type": "string", "enum": list(SLEEP_QUALITY)},
    "threat": {"type": "string", "enum": list(THREAT_TARGETS)},
    "injuries": {"type": "array", "maxItems": MAX_INJURIES, "items": {
        "type": "object", "properties": {
            "id": {"type": "string", "maxLength": 64},
            "description": {"type": "string", "maxLength": 200},
            "severity": {"type": "integer", "minimum": 1, "maximum": 3},
            "trend": {"type": "string", "enum": ["stable", "worsening", "recovering"]},
            "treated": {"type": "boolean"},
            "progress_minutes": {"type": "integer", "minimum": 0, "description": "서버가 유지하는 회복·악화 누적 분. 보통 생략"},
        }, "required": ["id", "description", "severity", "trend", "treated"], "additionalProperties": False,
    }},
}
