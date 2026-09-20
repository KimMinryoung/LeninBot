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


TERM_FEATURE = "commulingo_term_classification"
# Topic-first: a French Revolution faction is "factions", a foreign camp is
# "repression"; "international" is relations between states and the world
# movement; "contemporary" is present-day capitalism. Against the 1,086 stored
# terms (themselves writer-chosen) this agreed 813/1,086, 628/714 at conf ≥0.85.
TERM_RULES = {
    "theory": "Ideologies, doctrines, -isms, concepts, theoretical and historiographical terms of Marxism, socialism and "
              "their critics.",
    "economy": "Economic policies, institutions, campaigns and economic concepts — Soviet planning as well as monetary and "
               "industrial policy anywhere and in any era.",
    "party-state": "Party and state organs, offices, congresses, constitutions, military commands and political events of a "
                   "state or party (Soviet, Russian, or a revolutionary state abroad such as revolutionary France).",
    "factions": "Intra-party factions, oppositions, platforms and line struggles, in any party.",
    "repression": "Terror, security organs, camps, trials, repressive laws, censorship and rehabilitation, in any state.",
    "nationalities": "Nationalities policy, ethnic questions, deportations, republic and minority statuses.",
    "culture": "Culture, education, science and technology, arts, media and everyday life.",
    "international": "Relations BETWEEN states and the world movement: diplomacy, treaties, wars and campaigns between "
                     "states, international organizations and payment systems, foreign communist parties, liberation "
                     "fronts and the Cold War order.",
    "korea": "Korean politics, economy and society, any era.",
    "contemporary": "Present-day capitalism since the 1990s: today's labour, finance, technology, AI, platforms, climate "
                    "and policy debates.",
}


def _profile(feature: str) -> tuple[bool, float]:
    """(enabled, accept threshold) of a classification registry entry."""
    from llm.call_registry import resolve
    extra = resolve(feature).extra or {}
    return bool(extra.get("enabled", True)), float((extra.get("thresholds") or {}).get("accept", DEFAULT_ACCEPT))


def load_term_categories() -> list[dict]:
    """Rows of commulingo_term_categories, or the built-in fallback pairs."""
    try:
        from db import query
        rows = query("SELECT id, label_ko, label_en FROM commulingo_term_categories ORDER BY sort_order, id") or []
    except Exception as exc:  # no DB in this process: keep the tool usable
        logger.warning("term categories unavailable for classification (%s); using the built-in list", exc)
        rows = []
    if not rows:
        from runtime_tools.commulingo_people import _TERM_CATEGORY_FALLBACK
        rows = [{"id": slug, "label_ko": slug, "label_en": label} for slug, label in _TERM_CATEGORY_FALLBACK]
    return rows


def term_questions(categories: list[dict]) -> dict:
    return {"category": {"type": "choice", "instructions": "Which glossary category does this term belong to?",
                         "criteria": {c["id"]: f"{c['label_en']} / {c['label_ko']}. {TERM_RULES.get(c['id'], '')}"
                                      for c in categories}}}


def term_state(fields: dict) -> dict:
    def joined(value, lang):
        text = (value or {}).get(lang) if isinstance(value, dict) else value
        return " ".join(text) if isinstance(text, list) else str(text or "")
    period = fields.get("period")
    aliases = fields.get("aliases") or {}
    return {"term": {"ko": _text(fields.get("term"), "ko"), "en": _text(fields.get("term"), "en")},
            "aliases": (aliases.get("ko") or []) + (aliases.get("en") or []) if isinstance(aliases, dict) else aliases,
            "parent_term": fields.get("parentId"),
            "definition": {"ko": joined(fields.get("definition"), "ko"), "en": joined(fields.get("definition"), "en")},
            "period": _text(period, "ko") if isinstance(period, dict) else period,
            "body_ko": joined(fields.get("body"), "ko")[:1500], "body_en": joined(fields.get("body"), "en")[:800]}


def classify_term(fields: dict, *, categories=None, decide=None) -> dict | None:
    """{"category", "confidence", "low_confidence", "model"} for a drafted term, or None when unavailable."""
    from llm.call_registry import decide_detailed

    enabled, accept = _profile(TERM_FEATURE)
    if not enabled:
        return None
    categories = categories or load_term_categories()
    result = (decide or decide_detailed)(TERM_FEATURE, term_state(fields), term_questions(categories),
                                         label="term-classification")
    decision = result.decision
    if decision is None:
        logger.warning("term classification unavailable: %s", result.error)
        return None
    category = decision.choice("category")
    if category not in {c["id"] for c in categories}:
        return None
    conf = round(decision.confidence("category") or 0.0, 3)
    return {"category": category, "confidence": conf, "low_confidence": conf < accept, "model": decision.model}


def fill_term_category(fields: dict, classification: dict | None) -> dict:
    """The classifier's category; nothing to fall back on when it is unavailable."""
    out = dict(fields)
    if classification is None:
        return out
    out["category"] = classification["category"]
    return out


CODES_FEATURE = "commulingo_person_codes"
FATE_CRITERIA = {
    "unconfirmed": "not stated, disputed, unknown, or the person is still living",
    "executed": "executed after a sentence or purge",
    "assassinated": "assassinated by an attacker",
    "murdered": "murdered outside a judicial process",
    "killed": "killed in war or action",
    "suicide": "died by suicide",
    "deposed": "removed from power and lived on",
    "exile": "died in exile or emigration",
    "natural": "died a natural death (illness, old age)",
}
# Without these rules the origin code matched the writer 42/49 (a Jewish
# background became "israel"); with them 48/49, the miss at 0.49 confidence.
ORIGIN_INSTRUCTIONS = (
    "Which code names the national or ethnic background stated in the label and the source excerpts? Rules: the "
    "background is a people or nation, never a birthplace, place of activity or citizenship (Radek = poland though born "
    "in today's Ukraine; Yezhov = russia, an ethnic Russian born in Lithuania; a Soviet official of a non-Russian "
    "nationality keeps that nation: Sillari = estonia, Gumbaridze = georgia, never a blanket russia). A Jewish family "
    "background takes the code of the country or region the family came from (russia, ukraine, belarus, poland, "
    "lithuania, hungary, germany...), NEVER israel unless the person was born in Israel or Palestine. A mixed "
    "background takes the code the label names first. Follow the label when it names a nation."
)


def _living(years) -> bool:
    return str(years or "").strip().endswith("–")


def person_code_questions(fields: dict, citizenship_codes, origin_codes=()) -> dict:
    """One question per code object on the card that still lacks its code:
    citizenship.code, nationalOrigin.code and fate.kind (not asked for a living person)."""
    def missing(key, code):
        return isinstance(fields.get(key), dict) and not fields[key].get(code)
    questions = {}
    if missing("nationalOrigin", "code") and origin_codes:
        questions["nationalOrigin"] = {"type": "choice", "criteria": {c: c for c in origin_codes},
                                       "instructions": ORIGIN_INSTRUCTIONS}
    if missing("citizenship", "code"):
        questions["citizenship"] = {
            "type": "choice", "criteria": {c: c for c in citizenship_codes},
            "instructions": "Which state code matches the citizenship label and the source excerpts? soviet for Soviet "
                            "citizens; the state of most of the person's public life."}
    if isinstance(fields.get("fate"), dict) and fields["fate"].get("kind") is None and not _living(fields.get("years")):
        questions["fate"] = {
            "type": "choice", "criteria": FATE_CRITERIA,
            "instructions": "How did this person's life or career end, according to the fate label and the source excerpts?"}
    return questions


def _living_fate(fields: dict) -> dict:
    """A living person's fate is the empty kind, decided without a call."""
    if isinstance(fields.get("fate"), dict) and _living(fields.get("years")):
        return {"fate": {"kind": "", "confidence": 1.0, "low_confidence": False}}
    return {}


def person_code_state(fields: dict, claims: dict | None) -> dict:
    """Labels the writer wrote plus the research excerpts for those fields."""
    claims = claims or {}
    card = state_from_fields(fields)
    return {"name": card["name"], "years": fields.get("years"), "epithet": card["epithet"],
            "bio_ko": card["bio_ko"][:1200], "moment_ko": card["moment_ko"],
            "citizenship_label": (fields.get("citizenship") or {}).get("label"),
            "citizenship_claims": claims.get("citizenship", [])[:4],
            "origin_label": (fields.get("nationalOrigin") or {}).get("label"),
            "origin_claims": claims.get("nationalOrigin", [])[:4],
            "fate_label": (fields.get("fate") or {}).get("label"),
            "fate_claims": claims.get("fate", [])[:4]}


def _codes_from(decision, questions, fields, accept) -> dict:
    """Code verdicts from a decision that answered person_code_questions."""
    from runtime_tools.commulingo_people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES
    out = _living_fate(fields)
    for key, field, valid in (("nationalOrigin", "code", _NATIONAL_ORIGIN_CODES), ("citizenship", "code", _NATIONALITY_CODES),
                              ("fate", "kind", FATE_CRITERIA)):
        choice = decision.choice(key) if key in questions else None
        if choice not in valid:
            continue
        conf = round(decision.confidence(key) or 0.0, 3)
        value = "" if key == "fate" and choice == "unconfirmed" else choice
        out[key] = {field: value, "confidence": conf, "low_confidence": conf < accept}
    out["model"] = decision.model
    return out


def classify_person_codes(fields: dict, *, claims: dict | None = None, decide=None) -> dict | None:
    """{"citizenship": {"code", "confidence", "low_confidence"}, "fate": {"kind", ...}} for the code
    objects still lacking a code, or None when the model is unavailable. ``claims``
    maps field name to [{"claim", "excerpt"}] from the research artifact."""
    from llm.call_registry import decide_detailed
    from runtime_tools.commulingo_people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES

    enabled, accept = _profile(CODES_FEATURE)
    if not enabled:
        return None
    questions = person_code_questions(fields, sorted(_NATIONALITY_CODES), sorted(_NATIONAL_ORIGIN_CODES))
    if not questions:
        return _living_fate(fields)
    result = (decide or decide_detailed)(CODES_FEATURE, person_code_state(fields, claims), questions,
                                         label="person-codes")
    decision = result.decision
    if decision is None:
        logger.warning("person code classification unavailable: %s", result.error)
        return None
    return _codes_from(decision, questions, fields, accept)


def fill_person_codes(fields: dict, codes: dict | None) -> dict:
    """Copy of ``fields`` with citizenship.code / nationalOrigin.code / fate.kind
    set from the classification."""
    out = dict(fields)
    if not codes:
        return out
    for field, key in (("citizenship", "code"), ("nationalOrigin", "code"), ("fate", "kind")):
        judged = codes.get(field)
        if not judged or not isinstance(out.get(field), dict):
            continue
        obj = dict(out[field])
        obj[key] = judged[key]
        out[field] = obj
    return out


def missing_person_codes(fields: dict) -> list[str]:
    """Code objects on the card that still lack their code/kind."""
    missing = []
    for field in ("citizenship", "nationalOrigin"):
        if isinstance(fields.get(field), dict) and not fields[field].get("code"):
            missing.append(f"{field}.code")
    if isinstance(fields.get("fate"), dict) and fields["fate"].get("kind") is None:
        missing.append("fate.kind")
    return missing


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
    origin = fields.get("nationalOrigin") or fields.get("origin") or {}
    fate = fields.get("fate") or {}
    return {"name": _text(name, "ko") or _text(name, "en"), "years": fields.get("years"),
            "citizenship": (fields.get("citizenship") or {}).get("code") or _text((fields.get("citizenship") or {}).get("label"), "en"),
            "national_origin": origin.get("code") or _text(origin.get("label"), "en"),
            "epithet": _text(fields.get("epithet"), "ko"),
            "career": [f"{_text(c.get('r'), 'ko')} ({c.get('y')})" for c in career if isinstance(c, dict)],
            "bio_ko": joined(fields.get("bio"), "ko"), "bio_en": joined(fields.get("bio"), "en"),
            "moment_ko": joined(fields.get("moment"), "ko"),
            "fate": (fate.get("kind") or "") + (" · " + _text(fate.get("label"), "ko") if fate.get("label") else "")}


def load_catalogs() -> tuple[list[dict], list[dict], list[dict]]:
    from runtime_tools.commulingo_people import _list_categories, _list_groups, _list_offices
    return _list_groups(), _list_offices(), _list_categories()


CLASSIFY_EVIDENCE_FIELDS = ("bio", "career", "moment", "years")
EVIDENCE_PER_FIELD = 4
EVIDENCE_CHARS = 900


def evidence_for(claims: dict | None, fields=CLASSIFY_EVIDENCE_FIELDS) -> list[dict]:
    """Research excerpts for the classifier: the writer's card summarises them,
    but an office or era named only in the sources still counts. Capped so the
    decision state stays about the person, not the whole research."""
    out = []
    for field in fields:
        for c in (claims or {}).get(field, [])[:EVIDENCE_PER_FIELD]:
            out.append({"field": field, "claim": c.get("claim"), "excerpt": str(c.get("excerpt") or "")[:EVIDENCE_CHARS]})
    return out


def _person_from(decision, role_key, groups, offices, categories, accept) -> dict | None:
    """The group/role verdict of a decision, or None when a choice is off the catalogs."""
    group, role = decision.choice("group"), decision.choice(role_key)
    office_ids = {o["id"] for o in offices}
    if group not in {g["id"] for g in groups} or role not in office_ids | {c["id"] for c in categories}:
        return None
    conf = {"group": round(decision.confidence("group") or 0.0, 3), "role": round(decision.confidence(role_key) or 0.0, 3)}
    return {"groupId": group, "role": {"officeId": role} if role in office_ids else {"category": role},
            "confidence": conf, "low_confidence": min(conf.values()) < accept, "model": decision.model}


def classify_person(fields: dict, *, catalogs=None, claims: dict | None = None, decide=None) -> dict | None:
    """Group and role for a drafted person whose codes are settled, or None when
    the model is unavailable: the card request without its code questions.

    Returns {"groupId", "role": {"officeId"|"category"}, "confidence": {"group", "role"},
    "low_confidence": bool, "model"}. ``low_confidence`` (below the entry's
    ``thresholds.accept``) means the caller should have the independent
    reviewer confirm the classification rather than drop it: the best choice
    still beats the writer guessing.
    """
    card = classify_person_card(fields, catalogs=catalogs, claims=claims, decide=decide, codes=False)
    return card["person"] if card else None


def person_card_state(fields: dict, claims: dict | None) -> dict:
    """The group/role state plus the code labels and their excerpts: one state for the whole card."""
    state = state_from_fields(fields)
    evidence = evidence_for(claims)
    if evidence:
        state["research_excerpts"] = evidence
    codes = person_code_state(fields, claims)
    for key in ("citizenship_label", "citizenship_claims", "origin_label", "origin_claims", "fate_label", "fate_claims"):
        state[key] = codes[key]
    return state


def person_card_questions(fields: dict, groups, offices, categories, citizenship_codes, origin_codes, codes=True) -> dict:
    """The card's questions: missing codes, group, and role. The role's options
    depend on the citizenship (offices are Soviet institutions): when the card
    already carries the code, one role question fits it; when this request
    decides the citizenship, the role is asked both ways and code picks one."""
    questions = person_code_questions(fields, citizenship_codes, origin_codes) if codes else {}
    questions["group"] = build_questions(groups, offices, categories, soviet=True)["group"]
    if "citizenship" in questions:
        questions["role_soviet"] = build_questions(groups, offices, categories, soviet=True)["role"]
        questions["role_non_soviet"] = build_questions(groups, offices, categories, soviet=False)["role"]
    else:
        soviet = offices_allowed((fields.get("citizenship") or {}).get("code"))
        questions["role"] = build_questions(groups, offices, categories, soviet=soviet)["role"]
    return questions


def classify_person_card(fields: dict, *, catalogs=None, claims: dict | None = None, decide=None, codes=True) -> dict | None:
    """One request for a person's card: {"codes": classify_person_codes shape,
    "person": classify_person shape or None}, or None when the model is unavailable.

    Two requests (codes, then group/role once the citizenship was known) sent
    the same ~8k-token card twice and doubled the latency.
    """
    from llm.call_registry import decide_detailed
    from runtime_tools.commulingo_people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES

    enabled, accept = _profile(FEATURE)
    if not enabled:
        return None
    groups, offices, categories = catalogs or load_catalogs()
    if not groups or not categories:
        return None
    questions = person_card_questions(fields, groups, offices, categories, sorted(_NATIONALITY_CODES),
                                      sorted(_NATIONAL_ORIGIN_CODES), codes=codes)
    result = (decide or decide_detailed)(FEATURE, person_card_state(fields, claims), questions,
                                         label="person-card" if codes else "person-classification")
    decision = result.decision
    if decision is None:
        logger.warning("person card classification unavailable: %s", result.error)
        return None
    verdicts = _codes_from(decision, questions, fields, accept) if codes else {}
    if "role" in questions:
        role_key = "role"
    else:
        citizenship = (verdicts.get("citizenship") or {}).get("code")
        role_key = "role_soviet" if offices_allowed(citizenship) else "role_non_soviet"
    return {"codes": verdicts, "person": _person_from(decision, role_key, groups, offices, categories, accept)}


def fill_classification(fields: dict, classification: dict | None) -> dict:
    """Copy of ``fields`` with the assigned group/role. The writer never
    classifies: the schema it drafts against has no such fields, so a
    classification that came back always lands, low confidence included
    (the review stage gets that as a risk line)."""
    out = dict(fields)
    if classification is None:
        return out
    out.pop("group", None)
    out["groupId"] = classification["groupId"]
    out["role"] = dict(classification["role"])
    return out
