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
ACTIVITIES = {"rest": -2.0, "light": 2.0, "moderate": 5.0, "strenuous": 10.0, "sleep": -8.0, "restrained": 2.0, "self_care": 1.0, "focused_work": 3.0}
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
MOVEMENT_PAIN = {"rest": 0, "sleep": 0, "light": 0.1, "moderate": 0.5, "strenuous": 1.5, "restrained": 0, "self_care": 0.05, "focused_work": 0.1}
# Scene minutes of recovering/worsening before severity moves one step.
HEALING_STEP = {True: 1440, False: 2880}
WORSENING_STEP = {True: 2880, False: 1440}
PAIN_HINDERS_REST = 50  # at or above this, rest and sleep recover half the fatigue
# Pain above the floor the wounds imply is acute (a blow, a struggle) and subsides
# on its own: it halves every N hours by activity, never during exertion.
ACUTE_PAIN_HALF_LIFE_HOURS = {"rest": 2.0, "sleep": 2.0, "light": 3.0, "moderate": 6.0, "strenuous": None, "restrained": None, "self_care": 3.0, "focused_work": 3.0}
# Long passages are computed in steps so thresholds crossed inside them (pain
# easing below 60, an isolation stage starting, calm hours piling up) count.
CALCULATION_STEP_MINUTES = 1
# Resting or sleeping with nobody present for at least this long means the
# threat is not in the room: threatening/immediate compute as uncertain and the
# stored threat eases too (a visitor's return is a tension event, then a new threat).
ALONE_THREAT_RELIEF_MINUTES = 60
ALONE_THREAT_RELIEF_ACTIVITIES = ("rest", "sleep")
# Resolve is the lever of an interrogation, so what an event takes from it is
# a table, not the model's mood: the model names the kind and how hard it hit.
RESOLVE_EVENT_KINDS = {  # kind: (base delta at intensity 2, label)
    "beating": (-10.0, "구타·고문"),
    "sexual_coercion": (-8.0, "성적 강요(구형 미분류 기록)"),
    "sexual_harassment": (-1.0, "비접촉 성희롱"),
    "sexual_assault": (-3.0, "비삽입 성추행"),
    "rape": (-8.0, "삽입을 동반한 성폭행"),
    "public_submission": (-5.0, "증인 앞 복종·공개 굴욕"),
    "threat_to_kin": (-6.0, "가족·측근 언급 협박"),
    "futile_effort": (-3.0, "자술서 물리기·헛수고"),
    "kindness": (3.0, "배려·양보"),
    "recognition": (5.0, "능력·기여·쓸모 인정"),
    "agency": (6.0, "작은 선택권 행사"),
    "small_success": (5.0, "작은 과제의 성취"),
    "boundary_respected": (6.0, "거절·경계의 존중"),
    "support": (4.0, "지지적인 교류"),
    "setback": (-3.0, "구체적인 시도 실패"),
    "betrayal": (-6.0, "믿었던 약속의 파기"),
    "interrogation": (-2.0, "집중 심문"),
    "coerced_confession": (-4.0, "강요된 자백"),
    "implicating_others": (-5.0, "타인 연루 진술"),
}
RESOLVE_INTENSITY = {1: 0.5, 2: 1.0, 3: 1.5}  # 스침 / 보통 / 극심
RESOLVE_EVENT_CAP = 15.0            # one event never takes more than this
RESOLVE_STATE_FACTOR = 1.25         # each of pain >= 60, fatigue >= 70, an isolation stage
RESOLVE_PAIN_THRESHOLD = 60
RESOLVE_FATIGUE_THRESHOLD = 70
RESOLVE_REPEAT_WINDOW_MINUTES = 120  # the same kind again within this window counts half
RESOLVE_REPEAT_FACTOR = 0.5
RESOLVE_EVENT_HISTORY = 100
# Contact quality, not bodies in the room, determines isolation relief. These
# coefficients describe fictional accumulated burden, not clinical exposure hours.
SOCIAL_CONTACTS = ("unknown", "none", "incidental", "hostile", "meaningful")
ISOLATION_MODES = ("unknown", "solitary", "ordinary")
REQUIRED_CONDITIONS = ("activity", "sleep_quality", "threat", "injuries")
ISOLATION_RECOVERY_RATE = 2
ISOLATION_STAGES = (  # (from_hours, label, description, clarity/h, resolve/h, tension target)
    (168, "왜곡", "지각 착오·무감동·충동성 등이 나타날 가능성", -0.75, -0.5, 15.0),
    (72, "침식", "생각의 반복·접촉에 대한 민감함·집중 저하 가능성", -0.5, -0.25, 10.0),
    (24, "단절", "시간감각 변화·소리에 대한 과민·혼잣말 가능성", -0.25, 0.0, 5.0),
)
DYNAMICS_DEFAULTS = {
    **{key: None for key in METRICS},  # legacy states lack the mental axes; unset stays unset
    "revision": 0, "scene_minute": 0, "last_calculated_minute": 0,
    "time_basis": "현재 저장 상태를 기준 시점(0분)으로 삼음. 이전 경과 시간은 재계산하지 않음.",
    "activity": "rest", "sleep_quality": "normal", "threat": "uncertain",
    "injuries": [], "conditions_initialized": False, "recent_events": [], "calm_minutes": 0, "isolation_minutes": 0,
    "last_calculation": None, "clock": None, "event_timestamps": [], "resolve_events": [],
    "social_contact": "unknown", "isolation_mode": "unknown", "alone_rest_minutes": 0,
    "wakefulness_minutes": 0, "story_events": [], "story_interrupt": None, "metric_remainders": {},
}


def with_defaults(state):
    result = {**deepcopy(DYNAMICS_DEFAULTS), **state}
    result["clock"] = clock_defaults(result.get("clock"))
    return result


def validate_conditions(changes):
    enums = {"activity": ACTIVITIES, "sleep_quality": SLEEP_QUALITY, "threat": THREAT_TARGETS,
             "social_contact": SOCIAL_CONTACTS, "isolation_mode": ISOLATION_MODES}
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
    Gains thin out near the ceiling; low resolve softens further losses."""
    fatigue = state["fatigue"] if state["fatigue"] is not None else 0
    pain = state["pain"] if state["pain"] is not None else 0
    threat, activity = state["threat"], state["activity"]
    sleeping = activity == "sleep"
    stage = isolation_stage(state.get("isolation_minutes", 0)) or {"resolve": 0.0, "clarity": 0.0}
    by_threat = RESOLVE_BY_THREAT[threat]
    resolve_gain = max(0.0, by_threat) + (1 if sleeping and threat in ("safe", "uncertain") and state["sleep_quality"] != "poor" else 0)
    resolve_drain = min(0.0, by_threat) - (1 if fatigue > 70 else 0) - (1 if pain > 60 else 0) + stage["resolve"]
    if sleeping:
        clarity_gain = SLEEP_CLARITY_RATE * SLEEP_QUALITY[state["sleep_quality"]]
        clarity_drain = 0.0
    else:
        # Quiet rest sharpens the mind, except in solitary once isolation has set in.
        clarity_gain = 1.0 if activity == "rest" and fatigue <= 60 and not stage.get("label") else 0.0
        clarity_drain = -(2 if fatigue > 80 else 1 if fatigue > 60 else 0) - (1 if pain > 60 else 0) - (1 if threat == "immediate" else 0)
    clarity_drain += stage["clarity"]
    # Low resolve limits further losses; an unthreatened recovery activity must
    # provide a playable route out of collapse even before complete safety.
    quiet = threat in ('safe', 'uncertain') and activity != 'restrained'
    if quiet and activity in ('rest', 'sleep', 'self_care', 'focused_work'):
        resolve_gain += 1.5 * max(0, 1 - (state['resolve'] or 0) / 30)
    recovery = {'self_care': (3.0, 2.0, 5.0), 'focused_work': (2.0, 3.0, 2.0)}.get(activity)
    humiliation_rate = -1.5 if threat == 'safe' else (-0.75 if quiet and activity in ('rest','sleep') else 0.0)
    if quiet and recovery:
        strain = .5 if fatigue > 70 or pain > 60 else 1.0
        resolve_gain += recovery[0] * strain
        clarity_gain += recovery[1] * strain
        humiliation_rate -= recovery[2] * strain
    resolve_drain *= resolve_loss_scale(state.get('resolve'))
    return {"resolve": _diminish(resolve_gain, state["resolve"]) + resolve_drain,
            "clarity": _diminish(clarity_gain, state["clarity"]) + clarity_drain,
            "humiliation": humiliation_rate}


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


def resolve_loss_scale(value):
    return 1.0 if value is None else .25 + .75 * min(1.0, max(0.0, value) / 40)


def event_repeat_scale(state, kind):
    """Repeated rewards within 3h: full, half, quarter, then no extra reward."""
    positive = RESOLVE_EVENT_KINDS.get(kind, (0, ''))[0] > 0
    window = 180 if positive else RESOLVE_REPEAT_WINDOW_MINUTES
    count = sum(e.get('kind') == kind and 0 <= state.get('scene_minute', 0) - e.get('scene_minute', 0) <= window
                for e in state.get('resolve_events', []))
    return ((1.0, .5, .25)[count] if count < 3 else 0.0) if positive else (.5 if count else 1.0)


def resolve_event_delta(state, kind, intensity):
    """What an event of this kind takes from (or gives to) resolve under the current state.
    Returns (delta, factors): drains scale with pain/fatigue/isolation and halve when the
    same kind repeats within the window; positive repeats diminish too; drains stop at the cap."""
    if kind not in RESOLVE_EVENT_KINDS:
        raise ValueError(f"resolve_event.kind must be one of {sorted(RESOLVE_EVENT_KINDS)}")
    if type(intensity) is not int or intensity not in RESOLVE_INTENSITY:
        raise ValueError("resolve_event.intensity is 1 (스침), 2 (보통) or 3 (극심)")
    base, label = RESOLVE_EVENT_KINDS[kind]
    delta = base * RESOLVE_INTENSITY[intensity]
    factors = {"base": base, "intensity": RESOLVE_INTENSITY[intensity]}
    if delta > 0:
        factor = event_repeat_scale(state, kind)
        delta *= factor
        if factor != 1:
            factors['repeat'] = factor
        value = state.get('resolve')
        delta *= 1.0 if value is None else max(0.0, min(1.0, (95 - value) / 20))
    if delta < 0:
        landing = resolve_loss_scale(state.get('resolve'))
        delta *= landing
        if landing != 1:
            factors['low_resolve'] = landing
        pain = state.get("pain") or 0
        fatigue = state.get("fatigue") or 0
        weights = {"pain": pain >= RESOLVE_PAIN_THRESHOLD, "fatigue": fatigue >= RESOLVE_FATIGUE_THRESHOLD,
                   "isolation": bool(isolation_stage(state.get("isolation_minutes", 0)))}
        for name, applies in weights.items():
            if applies:
                delta *= RESOLVE_STATE_FACTOR
                factors[name] = RESOLVE_STATE_FACTOR
        recent = [e for e in state.get("resolve_events", []) if e.get("kind") == kind
                  and state.get("scene_minute", 0) - e.get("scene_minute", 0) <= RESOLVE_REPEAT_WINDOW_MINUTES]
        if recent:
            delta *= RESOLVE_REPEAT_FACTOR
            factors["repeat"] = RESOLVE_REPEAT_FACTOR
        if delta < -RESOLVE_EVENT_CAP:
            delta = -RESOLVE_EVENT_CAP
            factors["cap"] = RESOLVE_EVENT_CAP
    return round(delta, 4), {**factors, "label": label}


def isolation_stage(minutes):
    """(label, description, clarity/h, resolve/h, tension target add) for hours alone; None below a day."""
    hours = minutes / 60
    for from_hours, label, description, clarity, resolve, tension in ISOLATION_STAGES:
        if hours >= from_hours:
            return {"label": label, "description": description, "clarity": clarity, "resolve": resolve, "tension": tension}
    return None


def effective_contact(state):
    contact = state.get("social_contact", "unknown")
    # A stale meaningful-contact condition must not survive someone's departure.
    if not state.get("participants"):
        return "none"
    return "incidental" if contact == "unknown" else contact


def isolation_after(state, minutes):
    """Accumulate burden through routine/hostile contact; recover gradually."""
    current = state.get("isolation_minutes", 0)
    if effective_contact(state) == "meaningful":
        return max(0, current - minutes * ISOLATION_RECOVERY_RATE)
    if state.get("isolation_mode") == "ordinary":
        return max(0, current - minutes)
    return current + minutes


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
    threat_relieved = False
    conditions = {k: deepcopy(state[k]) for k in REQUIRED_CONDITIONS}
    conditions.update({k: state.get(k, "unknown") for k in ("social_contact", "isolation_mode")})
    result = deepcopy(state)
    injury_changes = []
    tension_targets = []
    cursor = start
    while cursor < target_minute:
        step_end = min(target_minute, cursor + CALCULATION_STEP_MINUTES)
        alone_rest = not result.get("participants") and result["activity"] in ALONE_THREAT_RELIEF_ACTIVITIES
        if (alone_rest and result.get("alone_rest_minutes", 0) >= ALONE_THREAT_RELIEF_MINUTES
                and result["threat"] in ("threatening", "immediate")):
            result["threat"] = "uncertain"
            threat_relieved = True
        result["alone_rest_minutes"] = result.get("alone_rest_minutes", 0) + step_end - cursor if alone_rest else 0
        result, step_changes, target = _step(result, step_end - cursor)
        if (alone_rest and result["alone_rest_minutes"] >= ALONE_THREAT_RELIEF_MINUTES
                and result["threat"] in ("threatening", "immediate")):
            result["threat"] = "uncertain"
            threat_relieved = True
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
        "isolation_stage": (isolation_stage(result.get("isolation_minutes", 0)) or {}).get("label"),
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
    # Preserve sub-display precision across calls, so splitting a passage cannot
    # change either rounding or the minute-by-minute integration path.
    state = {**state, **{key: state[key] + state.get("metric_remainders", {}).get(key, 0)
                        for key in METRICS if state[key] is not None}}
    result = deepcopy(state)
    hours = minutes / 60
    activity = state["activity"]
    pain = state["pain"]
    fatigue_rate = ACTIVITIES[activity]
    awake = state.get("wakefulness_minutes", 0)
    result["wakefulness_minutes"] = (max(0, awake - minutes * 2 * SLEEP_QUALITY[state["sleep_quality"]])
                                      if activity == "sleep" else awake + minutes)
    if activity == "rest" and state["fatigue"] is not None:
        # Quiet waking rest cannot substitute for sleep. Its recovery floor rises
        # with accumulated waking time, and prolonged wakefulness adds fatigue.
        floor = min(80, awake / 60 * 2)
        fatigue_rate = -2.0 if state["fatigue"] > floor else (2.0 if awake >= 16 * 60 else 0.0)
        if fatigue_rate < 0:
            fatigue_rate = max(fatigue_rate, (floor - state["fatigue"]) / hours)
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
            raw = max(0, min(100, state[key] + delta))
            result[key] = round(raw, 4)
            result.setdefault("metric_remainders", {})[key] = raw - result[key]
    result["injuries"], injury_changes = progress_injuries(state["injuries"], minutes)
    return result, injury_changes, target


CONDITION_SCHEMA = {
    "social_contact": {"type": "string", "enum": list(SOCIAL_CONTACTS),
                       "description": "none 고립 / incidental 배식·점검 / hostile 심문·위협 / meaningful 지속적 지지 대화 / unknown 미확인. 사람의 존재만으로 meaningful로 두지 않음"},
    "isolation_mode": {"type": "string", "enum": list(ISOLATION_MODES),
                       "description": "solitary 강제 독방·사회적 격리 / ordinary 일상적 생활 / unknown 미확인. ordinary는 잔여 고립 부담을 서서히 회복"},
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
