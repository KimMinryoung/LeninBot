"""Deterministic fictional state progression; rates are game tuning, not medicine."""
from copy import deepcopy
from runtime_tools.roleplay_clock import clock_defaults

PHYSICAL_METRICS = ("hunger", "fatigue", "pain", "tension")
# Mental axes: resolve/clarity read 100 = strong/lucid, humiliation reads 100 = extreme.
MENTAL_METRICS = ("resolve", "clarity", "humiliation")
METRICS = PHYSICAL_METRICS + MENTAL_METRICS
RESOLVE_BY_THREAT = {"safe": 2.0, "uncertain": 0.5, "threatening": -1.5, "immediate": -3.0}
SLEEP_CLARITY_RATE = 4.0
# Recovery of the mental axes slows as they climb: quiet time and sleep restore
# a shaken mind but never manufacture a perfect one. Drains are not scaled.
MENTAL_RECOVERY_CEILING = 90.0
MENTAL_RECOVERY_SPAN = 40.0
ACTIVITIES = {"rest": -2.0, "light": 2.0, "moderate": 5.0, "strenuous": 10.0, "sleep": -8.0}
SLEEP_QUALITY = {"poor": 0.25, "normal": 1.0, "good": 1.25}
THREAT_TARGETS = {"safe": 10.0, "uncertain": 40.0, "threatening": 75.0, "immediate": 90.0}
# Tension settles toward a target that threat sets but quiet hours, resolve,
# sleep and pain move: an uneventful day in a cell should be felt.
HABITUATION_MAX = 10.0      # target drops 1 per calm hour (safe/uncertain), up to this
RESOLVE_TENSION_SPAN = 8.0   # resolve 100 lowers the target by this, resolve 0 raises it
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
# Pain above the floor the wounds imply is acute (a blow, a struggle) and subsides
# on its own: it halves every N hours by activity, never during exertion.
ACUTE_PAIN_HALF_LIFE_HOURS = {"rest": 2.0, "sleep": 2.0, "light": 3.0, "moderate": 6.0, "strenuous": None}
# Long passages are computed in steps so thresholds crossed inside them (pain
# easing below 60, an isolation stage starting, calm hours piling up) count.
CALCULATION_STEP_MINUTES = 60
# Resting or sleeping with nobody present for at least this long means the
# threat is not in the room: threatening/immediate compute as uncertain and the
# stored threat eases too (a visitor's return is a tension event, then a new threat).
ALONE_THREAT_RELIEF_MINUTES = 60
ALONE_THREAT_RELIEF_ACTIVITIES = ("rest", "sleep")
# Solitary confinement: hours without anyone present accumulate; a visit of
# 30 minutes or more breaks the streak, a shorter one (a meal pushed through
# the door) only takes a few hours off it. Effects add to the other drifts.
ISOLATION_RESET_MINUTES = 30
ISOLATION_BRIEF_CONTACT_RELIEF = 240
ISOLATION_STAGES = (  # (from_hours, label, description, clarity/h, resolve/h, tension target)
    (168, "왜곡", "지각 왜곡·환각의 경계, 무감동 또는 충동성", -0.75, -0.5, 15.0),
    (72, "침식", "침입적 사고 반복, 사소한 접촉 과대평가, 집중 붕괴", -0.5, -0.25, 10.0),
    (24, "단절", "시간 감각 흐려짐, 소리·발소리에 과민, 상상 대화", -0.25, 0.0, 5.0),
)
DYNAMICS_DEFAULTS = {
    **{key: None for key in METRICS},  # legacy states lack the mental axes; unset stays unset
    "revision": 0, "scene_minute": 0, "last_calculated_minute": 0,
    "time_basis": "현재 저장 상태를 기준 시점(0분)으로 삼음. 이전 경과 시간은 재계산하지 않음.",
    "activity": "rest", "sleep_quality": "normal", "threat": "uncertain",
    "injuries": [], "conditions_initialized": False, "recent_events": [], "calm_minutes": 0, "isolation_minutes": 0,
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
    """Hourly drift of the mental axes from the conditions at the start of the interval.
    Gains thin out near the ceiling (see _diminish); drains always apply in full."""
    fatigue = state["fatigue"] if state["fatigue"] is not None else 0
    pain = state["pain"] if state["pain"] is not None else 0
    threat, activity = state["threat"], state["activity"]
    sleeping = activity == "sleep"
    stage = isolation_stage(state.get("isolation_minutes", 0)) or {"resolve": 0.0, "clarity": 0.0}
    by_threat = RESOLVE_BY_THREAT[threat]
    resolve_gain = max(0.0, by_threat) + (1 if sleeping and state["sleep_quality"] != "poor" else 0)
    resolve_drain = min(0.0, by_threat) - (1 if fatigue > 70 else 0) - (1 if pain > 60 else 0) + stage["resolve"]
    if sleeping:
        clarity_gain = SLEEP_CLARITY_RATE * SLEEP_QUALITY[state["sleep_quality"]]
        clarity_drain = 0.0
    else:
        # Quiet rest sharpens the mind, except in solitary once isolation has set in.
        clarity_gain = 1.0 if activity == "rest" and fatigue <= 60 and not stage.get("label") else 0.0
        clarity_drain = -(2 if fatigue > 80 else 1 if fatigue > 60 else 0) - (1 if pain > 60 else 0) - (1 if threat == "immediate" else 0)
    clarity_drain += stage["clarity"]
    return {"resolve": _diminish(resolve_gain, state["resolve"]) + resolve_drain,
            "clarity": _diminish(clarity_gain, state["clarity"]) + clarity_drain,
            "humiliation": -0.5 if threat == "safe" else 0.0}


def _diminish(rate, value):
    if rate <= 0 or value is None:
        return rate
    return rate * max(0.0, min(1.0, (MENTAL_RECOVERY_CEILING - value) / MENTAL_RECOVERY_SPAN))


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


def isolation_stage(minutes):
    """(label, description, clarity/h, resolve/h, tension target add) for hours alone; None below a day."""
    hours = minutes / 60
    for from_hours, label, description, clarity, resolve, tension in ISOLATION_STAGES:
        if hours >= from_hours:
            return {"label": label, "description": description, "clarity": clarity, "resolve": resolve, "tension": tension}
    return None


def isolation_after(state, minutes):
    """Minutes alone after an interval: someone present resets or relieves the streak."""
    current = state.get("isolation_minutes", 0)
    if not state.get("participants"):
        return current + minutes
    if minutes >= ISOLATION_RESET_MINUTES:
        return 0
    return max(0, current - ISOLATION_BRIEF_CONTACT_RELIEF)


def tension_target(state):
    calm_hours = state.get("calm_minutes", 0) / 60
    target = THREAT_TARGETS[state["threat"]] - min(HABITUATION_MAX, calm_hours)
    stage = isolation_stage(state.get("isolation_minutes", 0))
    if stage:
        target += stage["tension"]
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
    minutes = target_minute - start
    threat_relieved = (not state.get("participants") and minutes >= ALONE_THREAT_RELIEF_MINUTES
                       and state["activity"] in ALONE_THREAT_RELIEF_ACTIVITIES
                       and state["threat"] in ("threatening", "immediate"))
    conditions = {k: deepcopy(state[k]) for k in ("activity", "sleep_quality", "threat", "injuries")}
    if threat_relieved:
        conditions["threat"] = "uncertain"
    result = deepcopy(state)
    result["threat"] = conditions["threat"]
    injury_changes = []
    targets, tension_targets = [], []
    cursor = start
    while cursor < target_minute:
        step_end = min(target_minute, cursor + CALCULATION_STEP_MINUTES)
        result, step_changes, target = _step(result, step_end - cursor)
        injury_changes.extend(step_changes)
        tension_targets.append(target)
        cursor = step_end
    result.update(scene_minute=target_minute, last_calculated_minute=target_minute, time_basis=time_basis.strip())
    result["last_calculation"] = {
        "from_minute": start, "to_minute": target_minute, "basis": time_basis.strip(),
        "conditions": conditions,
        "before": {k: state[k] for k in METRICS}, "after": {k: result[k] for k in METRICS},
        "pain_floor": injury_pain_floor(state["injuries"]), "tension_target": tension_targets[-1],
        "injury_changes": _merge_injury_changes(injury_changes),
        "isolation_stage": (isolation_stage(state.get("isolation_minutes", 0)) or {}).get("label"),
        "healed": [c["id"] for c in _merge_injury_changes(injury_changes) if c["to"] == 0],
        "threat_relieved": threat_relieved,
    }
    return result


def _merge_injury_changes(changes):
    merged = {}
    for change in changes:
        entry = merged.setdefault(change["id"], {"id": change["id"], "from": change["from"], "to": change["to"]})
        entry["to"] = change["to"]
    return [c for c in merged.values() if c["from"] != c["to"]]


def _step(state, minutes):
    """One calculation step from the conditions at its start; returns (state, injury changes, tension target)."""
    result = deepcopy(state)
    hours = minutes / 60
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
        half_life = ACUTE_PAIN_HALF_LIFE_HOURS[activity]
        acute = (pain - floor) * (0.5 ** (hours / half_life) if half_life else 1)
        pain_delta = max(floor, floor + acute + pain_rate * hours) - pain
    tension = state["tension"]
    target = tension_target(state)
    tension_delta = 0 if tension is None else max(-6 * hours, min(6 * hours, target - tension))
    calm = state["threat"] in ("safe", "uncertain")
    result["calm_minutes"] = state.get("calm_minutes", 0) + minutes if calm else 0
    result["isolation_minutes"] = isolation_after(state, minutes)
    deltas = {"hunger": 3 * hours, "fatigue": fatigue_rate * hours,
              "pain": pain_delta, "tension": tension_delta}
    deltas.update({k: v * hours for k, v in mental_rates(state).items()})
    for key, delta in deltas.items():
        if state[key] is not None:
            result[key] = round(max(0, min(100, state[key] + delta)), 4)
    result["injuries"], injury_changes = progress_injuries(state["injuries"], minutes)
    return result, injury_changes, target


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
