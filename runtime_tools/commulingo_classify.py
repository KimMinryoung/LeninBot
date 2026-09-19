"""Assign a CommuLingo person's group and role with a System One model.

The person create API used to require the writing model to pick ``groupId``
(9 groups) and ``role`` (16 Soviet offices or 10 categories) beside the prose.
Those are closed-set editorial judgements: on 2026-09-19 a Jev audit of all
2,341 stored people found 106 misfiled ones, and the operator decided the
runner should assign the classification after the draft instead of the
writer choosing it. ``classify_person`` takes the drafted fields (name,
years, citizenship, epithet, career, bio) and returns the group and role
with confidences; the pipeline draft stage and the person create tool fill
them in when the writer left them out. Offices are only offered for Soviet
and successor-state citizens (offices are Soviet institutions).

The criteria carry the editorial rules the operator confirmed on 2026-09-19
(dev_docs/jev_system_one_adoption.md 4.11.1); the audit script imports the
same rules so the two never drift.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FEATURE = "commulingo_person_classification"
SOVIET_CITIZENSHIP = frozenset({"soviet", "russia"})
# Successor states of union republics: their citizens may legitimately hold a
# Soviet office (a republic first secretary coded with today's state).
SOVIET_SUCCESSORS = frozenset({"armenia", "azerbaijan", "belarus", "estonia", "georgia", "kazakhstan", "kyrgyzstan",
                               "latvia", "lithuania", "moldova", "tajikistan", "turkmenistan", "ukraine", "uzbekistan"})
DEFAULT_ACCEPT = 0.7

GROUP_RULES = {
    "old-regime": "Inside the Russian Empire before October 1917: tsarist officials and generals, White commanders of the "
                  "civil war, and the revolutionaries and thinkers (Chernyshevsky, Plekhanov, Zasulich) whose activity peaked "
                  "before 1917 or who died before the Soviet state formed.",
    "bolshevik": "Soviet citizens whose public role peaked 1917–1940: Old Bolsheviks, civil-war commanders, early Soviet "
                 "officials, many purged in the 1930s.",
    "stalin-era": "Soviet citizens whose public role peaked 1929–1953: enforcers, officials, commanders, scientists and "
                  "artists of the Stalin years, including victims of the terror whose careers began under Stalin.",
    "thaw": "Soviet citizens whose public role peaked 1953–1985: thaw and stagnation, space and science, reformers, dissidents.",
    "perestroika": "Soviet citizens whose public role peaked 1985–1991 or who presided over the end of the USSR and the first "
                   "post-Soviet years.",
    "international-revolutionary": "Anyone OUTSIDE the Soviet state apparatus on the revolutionary or socialist side: "
                                   "communists, socialists, leaders and officials of socialist states (Poland, Hungary, "
                                   "Czechoslovakia, East Germany, China, Cuba, Vietnam...), their reformers and dissidents. "
                                   "A non-Soviet citizen never belongs to a Soviet era group.",
    "foreign-statesmen": "Non-communist politicians, diplomats and generals of other states who negotiated with or confronted "
                         "the USSR: presidents, prime ministers, foreign ministers, ambassadors, monarchs.",
    "international-counterrevolutionary": "Rulers, soldiers and movements outside the USSR that fought revolution and the "
                                          "socialist camp by force: fascists, military dictators, anti-communist insurgents. "
                                          "Not Russian Whites (old-regime).",
    "scholar": "ONLY historians and social scientists who researched and interpreted this history (Soviet studies, Marxist "
               "theory scholarship). NOT natural scientists, engineers or physicians — those belong to the era group of their "
               "Soviet career.",
}

OFFICE_RULES = {
    "party-leadership": "Politburo/Presidium members and Central Committee secretaries at the all-union top, and the General "
                        "Secretary; Lenin belongs here.",
    "party-secretariat-cadres": "The Secretariat, Orgburo and cadres apparatus, AND regional/oblast/city first secretaries "
                                "(Moscow, Leningrad, Sverdlovsk...) and Komsomol leaders — the party machine below the top.",
    "nationalities-federal": "Union-republic first secretaries and republic heads of government (Ukraine, Kazakhstan, "
                             "the Baltics, Caucasus, Central Asia, Moldova...) and the institutions managing nationalities "
                             "and the federal structure.",
    "ideology-propaganda": "Pro-Soviet ideologues who ran ideology, censorship and propaganda for the party (Suslov, Zhdanov "
                           "line). Never a dissident or a scientist who published critical essays.",
    "science-nuclear-space": "Scientists, chief designers and administrators of the atomic, missile and space programmes, "
                             "including physicists who later dissented (Sakharov).",
    "state-security": "Cheka/GPU/NKVD/MGB/KGB command line, military counter-intelligence (Special Departments, SMERSH) "
                      "and GRU chiefs.",
    "defence": "War Commissariat and Ministry of Defence: commanders and marshals, not intelligence chiefs.",
    "head-of-government": "Chairmen of Sovnarkom / Council of Ministers and their deputies; an ambiguous line — prefer the "
                          "person's defining office when they also held one.",
    "comintern": "Comintern functionaries of any nationality (Dimitrov, Kolarov, Manuilsky).",
}


def offices_allowed(citizenship_code: str | None) -> bool:
    return (citizenship_code or "") in SOVIET_CITIZENSHIP | SOVIET_SUCCESSORS


def build_questions(groups: list[dict], offices: list[dict], categories: list[dict], soviet: bool) -> dict:
    group_criteria = {g["id"]: f"{g['title_en']} ({g.get('range_label') or ''}). {GROUP_RULES.get(g['id'], g.get('blurb_en') or '')}"
                      for g in groups}
    role_criteria = {c["id"]: f"CATEGORY {c['label_en']} / {c['label_ko']}" for c in categories}
    if soviet:
        role_criteria.update({o["id"]: f"OFFICE {o['title_en']} ({o.get('range_label') or ''}): {OFFICE_RULES.get(o['id'], '')}"
                              for o in offices})
    return {
        "group": {"type": "choice", "criteria": group_criteria,
                  "instructions": "Which dictionary group does this person belong to? Soviet citizens go to the era in which "
                                  "their public role peaked; people outside the Soviet state go to one of the four "
                                  "non-Soviet groups."},
        "role": {"type": "choice", "criteria": role_criteria,
                 "instructions": ("Which single role identifies this person? Choose a catalogued OFFICE only when the career "
                                  "shows they held it; otherwise the closest CATEGORY." if soviet else
                                  "Which category best identifies this non-Soviet person's role in this history?")},
    }


def _text(value, lang: str) -> str:
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("ko") or value.get("en") or "")
    return str(value or "")


def state_from_fields(fields: dict) -> dict:
    """Decision state from the create API's ``fields`` (bio may still be a sentence list)."""
    def joined(value, lang):
        text = (value or {}).get(lang) if isinstance(value, dict) else value
        return " ".join(text) if isinstance(text, list) else str(text or "")
    name = fields.get("name") or {}
    if not name:
        name = {lang: " ".join(p for p in (_text(fields.get("givenName"), lang), _text(fields.get("familyName"), lang)) if p)
                for lang in ("ko", "en")}
    career = fields.get("career") or []
    return {"name": _text(name, "ko") or _text(name, "en"), "years": fields.get("years"),
            "citizenship": (fields.get("citizenship") or {}).get("code"), "epithet": _text(fields.get("epithet"), "ko"),
            "career": [f"{_text(c.get('r'), 'ko')} ({c.get('y')})" for c in career if isinstance(c, dict)],
            "bio_ko": joined(fields.get("bio"), "ko"), "bio_en": joined(fields.get("bio"), "en")}


def load_catalogs() -> tuple[list[dict], list[dict], list[dict]]:
    from runtime_tools.commulingo_people import _list_categories, _list_groups, _list_offices
    return _list_groups(), _list_offices(), _list_categories()


def classify_person(fields: dict, *, catalogs=None, decide=None) -> dict | None:
    """Group and role for a drafted person, or None when the model is unavailable.

    Returns {"groupId", "role": {"officeId"|"category"}, "confidence": {"group", "role"},
    "low_confidence": bool, "model"}. ``low_confidence`` (below the entry's
    ``thresholds.accept``) means the caller should have the independent
    reviewer confirm the classification rather than drop it: the best choice
    still beats the writer guessing.
    """
    from llm.call_registry import decide_detailed, resolve

    profile = resolve(FEATURE)
    extra = profile.extra or {}
    if not extra.get("enabled", True):
        return None
    accept = float((extra.get("thresholds") or {}).get("accept", DEFAULT_ACCEPT))
    groups, offices, categories = catalogs or load_catalogs()
    if not groups or not categories:
        return None
    state = state_from_fields(fields)
    soviet = offices_allowed(state["citizenship"])
    result = (decide or decide_detailed)(FEATURE, state, build_questions(groups, offices, categories, soviet),
                                         label="person-classification")
    decision = result.decision
    if decision is None:
        logger.warning("person classification unavailable: %s", result.error)
        return None
    group, role = decision.choice("group"), decision.choice("role")
    office_ids = {o["id"] for o in offices}
    if group not in {g["id"] for g in groups} or role not in office_ids | {c["id"] for c in categories}:
        return None
    conf = {"group": round(decision.confidence("group") or 0.0, 3), "role": round(decision.confidence("role") or 0.0, 3)}
    return {"groupId": group, "role": {"officeId": role} if role in office_ids else {"category": role},
            "confidence": conf, "low_confidence": min(conf.values()) < accept, "model": decision.model}


def fill_classification(fields: dict, classification: dict | None) -> dict:
    """Copy of ``fields`` with the assigned group/role; the writer's own values
    are only kept when no classification came back."""
    out = dict(fields)
    if classification is None:
        return out
    out.pop("group", None)
    out["groupId"] = classification["groupId"]
    out["role"] = dict(classification["role"])
    return out
