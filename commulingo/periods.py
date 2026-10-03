"""Structured career/office periods — mirror of frontend data/commulingo/career-period.js.

The table columns are the source of truth (frontend migrations 270–272): start/end
year-month-day, a qualifier per side, an ongoing flag and, only for periods the
columns cannot say, a bilingual override label. Nothing parses a label; the text
is formatted from the columns. Keep the rules in step with the JS module.
"""
from __future__ import annotations

import calendar

START_QUALS = ("circa", "decade", "early", "mid", "late", "after", "summer")
END_QUALS = START_QUALS + ("until", "open", "unknown")
_DECADE_QUALS = {"decade", "early", "mid", "late"}
_NO_DATE = {"open", "unknown"}

PERIOD_COLUMNS = ("start_year", "start_month", "start_day", "end_year", "end_month", "end_day",
                  "start_qual", "end_qual", "ongoing", "period_label_ko", "period_label_en")

_POINT_SCHEMA = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3,
                 "description": "[year, month?, day?]"}
PERIOD_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "start": _POINT_SCHEMA,
        "end": _POINT_SCHEMA,
        "startQual": {"type": "string", "enum": list(START_QUALS)},
        "endQual": {"type": "string", "enum": list(END_QUALS)},
        "ongoing": {"type": "boolean"},
        "label": {"type": "object", "additionalProperties": False,
                  "properties": {"ko": {"type": "string"}, "en": {"type": "string"}}, "required": ["ko", "en"]},
    },
    "description": (
        "Period as numbers, never a label string. {\"start\": [1918, 7], \"end\": [1918, 9]} → 1918.07–09; "
        "omit end for a single date. Qualifiers: circa (about), decade (start [1920] = the 1920s), "
        "early/mid/late (of that decade), after (that date or later), summer; end-only: until (no start), "
        "open (continued, end not recorded: 1945–), unknown (1979–?). ongoing: true for 'to the present'. "
        "label {ko,en} ONLY when numbers cannot say it (conflicting sources: '1963 또는 1964'); still give start."
    ),
}


def _point(value, side):
    if value is None:
        return None
    if (not isinstance(value, list) or not 1 <= len(value) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
        raise ValueError(f"period.{side} must be [year, month?, day?] of integers")
    year, month, day = (list(value) + [None, None])[:3]
    if not 1 <= year <= 2100:
        raise ValueError(f"period.{side} year out of range")
    if month is not None and not 1 <= month <= 12:
        raise ValueError(f"period.{side} month must be 1–12")
    if day is not None and not 1 <= day <= calendar.monthrange(year, month)[1]:
        raise ValueError(f"period.{side} day out of range")
    return year, month, day


def period_columns(period) -> dict:
    """Validated period object → column values. Raises ValueError (mirrors periodColumns)."""
    if isinstance(period, str):
        raise ValueError("period is a string; send {start: [year, month?, day?], end, startQual, endQual, ongoing, label}")
    if not isinstance(period, dict):
        raise ValueError("period is required")
    extra = set(period) - {"start", "end", "startQual", "endQual", "ongoing", "label"}
    if extra:
        raise ValueError(f"period has unknown keys {', '.join(sorted(extra))}")
    start, end = _point(period.get("start"), "start"), _point(period.get("end"), "end")
    sq, eq = period.get("startQual") or None, period.get("endQual") or None
    ongoing = period.get("ongoing") is True
    if sq is not None and sq not in START_QUALS:
        raise ValueError(f"period.startQual must be one of {', '.join(START_QUALS)}")
    if eq is not None and eq not in END_QUALS:
        raise ValueError(f"period.endQual must be one of {', '.join(END_QUALS)}")
    if sq and not start:
        raise ValueError("period.startQual needs a start")
    if eq in _NO_DATE and end:
        raise ValueError(f"period.endQual {eq} takes no end date")
    if eq and eq not in _NO_DATE and not end:
        raise ValueError(f"period.endQual {eq} needs an end")
    if eq == "until" and start:
        raise ValueError("period.endQual until is for a period with no start")
    for point, qual, side in ((start, sq, "start"), (end, eq, "end")):
        if qual in _DECADE_QUALS and (point[1] is not None or point[0] % 10):
            raise ValueError(f"period.{side}Qual {qual} needs a decade year (1920) without month")
    if ongoing and (end or eq):
        raise ValueError("period.ongoing excludes an end")
    label = period.get("label")
    if label is not None:
        if not isinstance(label, dict) or not str(label.get("ko") or "").strip() or not str(label.get("en") or "").strip():
            raise ValueError("period.label must have non-empty ko and en")
        label = {"ko": label["ko"].strip(), "en": label["en"].strip()}
    if not (start or end or ongoing or label or eq):
        raise ValueError("period is empty")
    if start and end and tuple(v or 0 for v in end) < tuple(v or 0 for v in start):
        raise ValueError("period.end precedes start")
    s, e = start or (None, None, None), end or (None, None, None)
    return dict(zip(PERIOD_COLUMNS, (*s, *e, sq, eq, ongoing,
                                      label["ko"] if label else None, label["en"] if label else None)))


def _fmt_point(year, month, day, qual, lang):
    ko = lang == "ko"
    if qual == "decade":
        return f"{year}년대" if ko else f"{year}s"
    if qual in ("early", "mid", "late"):
        if ko:
            return f"{year}년대 " + {"early": "초", "mid": "중반", "late": "후반"}[qual]
        return {"early": "early ", "mid": "mid-", "late": "late "}[qual] + f"{year}s"
    if qual == "summer":
        return f"{year} 여름" if ko else f"summer {year}"
    base = str(year) + (f".{month:02d}" if month else "") + (f".{day:02d}" if day else "")
    if qual == "circa":
        return f"{base}{'' if month else '년'}경" if ko else f"c. {base}"
    if qual == "after":
        return f"{base} 이후" if ko else f"after {base}"
    return base


def format_period(row: dict, lang: str = "ko") -> str:
    """Row columns → display text (same output as the frontend formatPeriod)."""
    if not row:
        return ""
    label = row.get(f"period_label_{lang}") or row.get("period_label_ko") or row.get("period_label_en")
    if label:
        return label
    sy, sm, sd = row.get("start_year"), row.get("start_month"), row.get("start_day")
    ey, em, ed = row.get("end_year"), row.get("end_month"), row.get("end_day")
    sq, eq = row.get("start_qual"), row.get("end_qual")
    s = _fmt_point(sy, sm, sd, sq, lang) if sy is not None else ""
    present = "present" if lang == "en" else "현재"
    if row.get("ongoing"):
        return f"{s}–{present}" if s else present
    if eq == "open":
        return f"{s}–"
    if eq == "unknown":
        return f"{s}–?"
    if ey is None:
        return s
    if eq == "until" or sy is None:
        return "–" + _fmt_point(ey, em, ed, None, lang)
    e = _fmt_point(ey, em, ed, eq, lang)
    if not sq and not eq and sm and em and sy == ey:
        if sm == em and sd and ed:
            e = "" if sd == ed else f"{ed:02d}"
        elif not sd and not ed:
            e = "" if sm == em else f"{em:02d}"
    elif not sq and not eq and not sm and not em and sy == ey:
        e = ""
    return f"{s}–{e}" if e else s


def career_period_text(entry: dict, lang: str = "ko") -> str:
    """Display text of a career entry from the person store ({y: {ko,en}}) or a patch ({period})."""
    y = (entry or {}).get("y")
    if isinstance(y, dict):
        return y.get(lang) or y.get("ko") or y.get("en") or ""
    if isinstance(y, str):
        return y
    try:
        return format_period(period_columns((entry or {}).get("period")), lang)
    except ValueError:
        return ""
