"""Assign a CommuLingo person's group and primary activity with a System One model.

The person create API used to require the writing model to pick ``groupId``
and a ``role`` (a Soviet office or a role category) beside the prose.
Those are closed-set editorial judgements: on 2026-09-19 a Jev audit of all
2,341 stored people found 106 misfiled ones, and the operator decided the
runner should assign the classification after the draft instead of the
writer choosing it. The legacy role was retired on 2026-09-30 (the frontend
dropped the person role tables): every person now carries sourced
``activities`` with exactly one primary activity. ``classify_person`` takes
the drafted fields (name, years, citizenship, epithet, career, bio) and the
cited evidence and returns the group and the primary activity (function,
affiliation, evidence) with confidences; the pipeline and the person create
tool fill them in when the writer left them out.

The group criteria carry the editorial rules the operator confirmed on
2026-09-19 (dev_docs/jev_system_one_adoption.md 4.11.1); the audit script
imports the same rules so the two never drift.
"""
from __future__ import annotations

import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

FEATURE = "commulingo_person_classification"
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
    "china-old-regime": "Chinese citizens on the side the Communists fought, 1894–1949: late-Qing officials and generals "
                        "(Yuan Shikai), warlords (Zhang Zuolin), Kuomintang politicians, diplomats and generals (Chiang "
                        "Kai-shek, Wang Jingwei, T. V. Soong). NOT the Communists' foreign statesmen counterparts.",
    "china-revolution": "Chinese Communists and their allies whose public role peaked 1911–1949 or who MADE the revolution "
                        "(May Fourth radicals, party founders, Long March leaders, Yan'an leadership): Chen Duxiu, Li Dazhao, "
                        "Qu Qiubai, Wang Ming, Mao, Zhou Enlai, Zhu De, Liu Shaoqi; Sun Yat-sen and the republican "
                        "revolutionaries of 1911 too. Founders stay here even when they ruled the People's Republic later.",
    "china-mao-era": "Chinese citizens whose public role peaked 1949–1976: officials, commanders, writers and victims of "
                     "land reform, the anti-rightist campaign, the Great Leap and the Cultural Revolution (Peng Dehuai, Lin "
                     "Biao, Kang Sheng, Jiang Qing, Hua Guofeng).",
    "china-reform": "Chinese citizens whose public role peaked after 1976: the reform leadership (Deng Xiaoping, Hu Yaobang, "
                    "Zhao Ziyang, Jiang Zemin), the dissidents of Democracy Wall and 1989, and the post-1989 leadership.",
    "france-revolution": "The French Revolution and the Napoleonic era, 1774–1830: Louis XVI and his ministers, the "
                         "revolutionaries of the assemblies, clubs and Paris Commune (Sieyès, Robespierre, Danton, Marat, "
                         "Hébert, Babeuf), the royalists, émigrés and Vendée leaders who fought them, the generals of the "
                         "republican armies, Napoleon and his marshals and administrators, the Restoration kings to the July "
                         "Revolution, and foreign commanders of the coalition wars (Howe, Nelson). Socialists of the 1830s and "
                         "after (Cabet, Blanqui) are world groups.",
    # The world shelf (migration 218) is sorted by era like the Soviet one, not by camp: revolutionaries,
    # statesmen and their opponents of one era sit together. Camp is the person's activities and the
    # position collections (counterrevolution, dissident, ...), not the group.
    "world-before-1917": "People outside the Soviet, Chinese and French Revolution shelves whose public role peaked before "
                         "1917: early socialists and 1848, the First and Second Internationals and the Paris Commune "
                         "(Marx, Engels, Bakunin, Bebel, Jaurès), and the statesmen, generals and monarchs of the imperial "
                         "age (Bismarck, Wilhelm II). A party founder whose defining role is leading it after 1917 goes to "
                         "world-interwar. A non-Soviet citizen never belongs to a Soviet era group.",
    "world-interwar": "People outside the Soviet and Chinese shelves whose public role peaked 1917–1945: the revolutionary "
                      "wave after October (Luxemburg, Béla Kun, Gramsci), Comintern and interwar communists and socialists, "
                      "interwar statesmen and diplomats, fascism and Nazism, the Spanish Civil War and the Second World War "
                      "(Hitler, Churchill, Franco), and foreign advisers to the Chinese revolution (Borodin, Otto Braun). "
                      "A Second World War leader stays here even when he lived long after. A non-Soviet citizen never "
                      "belongs to a Soviet era group.",
    "world-cold-war": "People outside the Soviet and Chinese shelves whose public role peaked after 1945: the people's "
                      "democracies and their reformers and dissidents, decolonization and national liberation (Ho Chi Minh, "
                      "Castro, Guevara, Cabral), Cold War statesmen and anti-communist regimes (Pinochet), 1956, 1968 "
                      "and 1989, and anyone later. A non-Soviet citizen never belongs to a Soviet era group.",
    "scholar": "ONLY historians and social scientists who researched and interpreted this history (Soviet studies, Marxist "
               "theory scholarship). NOT natural scientists, engineers or physicians — those belong to the era group of their "
               "Soviet career.",
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
        from commulingo.people import _TERM_CATEGORY_FALLBACK
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
    from commulingo.people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES
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
    from commulingo.people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES

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


# Stage one of the group decision is arithmetic, not a model call. The
# groups span three centuries — the French Revolution, the Soviet and Chinese
# eras, the wider modern world — and a person's life years already rule most of
# them out: Robespierre (1758–1794) cannot belong to a group of 1871–2016 or to
# the Soviet eras. The model then chooses only among the groups whose era
# overlaps the person's adult life (from 16 to death, or to today). Windows are
# the years a group's people were active, wider than the display range_label.
# A group missing here is never filtered out, and years that cannot be parsed
# leave every group open.
GROUP_ERAS = {
    "old-regime": (1700, 1917), "bolshevik": (1890, 1940), "stalin-era": (1924, 1953), "thaw": (1945, 1991),
    "perestroika": (1975, None),
    "china-old-regime": (1880, 1949), "china-revolution": (1895, 1949), "china-mao-era": (1940, 1976),
    "china-reform": (1970, None),
    "france-revolution": (1774, 1830),
    "world-before-1917": (1800, 1930), "world-interwar": (1895, 1960), "world-cold-war": (1925, None),
    "scholar": (1850, None),
}
ADULT_AGE = 16

# The dictionary files French Revolution figures by their camp in the
# Revolution, so their defining activity is the one in it. Lafayette (job
# 63829, 2026-09-26) was first filed by his Continental Army command, the
# Feuillant National Guard commander only after an operator correction.
# Other shelves keep the plain "defining career" rule.
FRENCH_REVOLUTION_GROUPS = frozenset({"france-revolution"})
FRENCH_REVOLUTION_BASIS = (" This person is filed on the French Revolution shelf: prefer the excerpt documenting their role"
                           " in the Revolution and its wars, 1789–1815, the side, club, faction or regime they served,"
                           " over an earlier or later career elsewhere (the American war, exile, a later"
                           " reign) even when that career is better known.")
_YEAR = r"(\d{3,4})(?:/(\d{3,4}))?\??"
_YEARS = re.compile(r"^\s*(?:c\.\s*)?(?:" + _YEAR + r"|\?)\s*[–-]\s*(?:(?:" + _YEAR + r"|\?)(\s*이후)?)?\s*$")


def active_span(years, today: int | None = None) -> tuple[int, int] | None:
    """(first, last) year of a person's adult life from a years label such as
    '1758–1794', 'c. 1729/1730–1800', '1950–' (living) or '1900–1950 이후';
    None when neither end is a year."""
    today = today or date.today().year
    m = _YEARS.match(str(years or ""))
    if not m:
        return None
    birth = int(m.group(1)) if m.group(1) else None  # the earlier of two candidate years
    death = int(m.group(4) or m.group(3)) if m.group(3) else None
    if m.group(5) or (death is None and not re.search(r"[–-]\s*\?", str(years))):
        death = today  # died at an unknown date after the year, or still living
    if birth is None and death is None:
        return None
    if death is None:  # 'born 1900, death unknown': a lifetime, capped at today
        death = min(birth + 90, today)
    return ((birth + ADULT_AGE) if birth is not None else death - 50, death)


def groups_for_years(groups: list[dict], years, today: int | None = None) -> list[dict]:
    """Stage one: the groups whose era overlaps the person's adult life."""
    span = active_span(years, today)
    if span is None:
        return groups
    today = today or date.today().year
    first, last = span
    kept = [g for g in groups
            if g["id"] not in GROUP_ERAS
            or (GROUP_ERAS[g["id"]][0] <= last and first <= (GROUP_ERAS[g["id"]][1] or today))]
    return kept or groups


def group_question(groups: list[dict]) -> dict:
    """The dictionary group question over the (era-filtered) groups."""
    group_criteria = {g["id"]: f"{g['title_en']} ({g.get('range_label') or ''}). {GROUP_RULES.get(g['id'], g.get('blurb_en') or '')}"
                      for g in groups}
    return {"type": "choice", "criteria": group_criteria,
            "instructions": "Which dictionary group does this person belong to? Soviet citizens go to the era in which "
                            "their public role peaked; Chinese citizens go to the china-* group of their era or side; "
                            "people outside both states go to the world group of the era in which their public role peaked; people of the French Revolution and Napoleon go to france-revolution. Historians researching this history use scholar regardless of nationality; actors or targets in historical events can retain the era of their activity."}


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


def load_catalogs() -> tuple[list[dict], list[dict]]:
    """(groups, offices). The role categories were retired with the legacy person role."""
    from commulingo.people import _list_groups, _list_offices
    return _list_groups(), _list_offices()


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


def classify_person(fields: dict, *, catalogs=None, claims: dict | None = None, decide=None) -> dict | None:
    """Group and primary activity for a drafted person whose codes are settled,
    or None when the model is unavailable or the card has no cited activity
    evidence: the card request without its code questions.

    Returns {"groupId", "activities": [primary], "confidence", "low_confidence": bool, "model"}. ``low_confidence`` (below the entry's
    ``thresholds.accept``) means the caller should have the independent
    reviewer confirm the classification rather than drop it: the best choice
    still beats the writer guessing.
    """
    card = classify_person_card(fields, catalogs=catalogs, claims=claims, decide=decide, codes=False)
    return card["person"] if card else None


def person_card_state(fields: dict, claims: dict | None) -> dict:
    """The group/activity state plus the code labels and their excerpts: one state for the whole card."""
    state = state_from_fields(fields)
    evidence = evidence_for(claims)
    if evidence:
        state["research_excerpts"] = evidence
    codes = person_code_state(fields, claims)
    for key in ("citizenship_label", "citizenship_claims", "origin_label", "origin_claims", "fate_label", "fate_claims"):
        state[key] = codes[key]
    return state


def person_card_questions(fields: dict, groups, citizenship_codes, origin_codes, codes=True) -> dict:
    """The card's questions: missing codes and the group. The activity
    questions are added by ``classify_person_card`` from the cited evidence."""
    questions = person_code_questions(fields, citizenship_codes, origin_codes) if codes else {}
    questions["group"] = group_question(groups)
    return questions


def classify_person_card(fields: dict, *, catalogs=None, claims: dict | None = None, decide=None, codes=True) -> dict | None:
    """Classify a card, returning codes and a person assignment, or None.

    Sourced activities select the function, then the supporting evidence,
    then its affiliation in dependent requests.
    Each stage sees the preceding choice; Jev question heads are independent.
    """
    from llm.call_registry import decide_detailed, Decision
    from commulingo.people import _NATIONAL_ORIGIN_CODES, _NATIONALITY_CODES

    enabled, accept = _profile(FEATURE)
    if not enabled:
        return None
    groups = (catalogs or load_catalogs())[0]
    if not groups:
        return None
    # Stage one: the life years narrow the era groups; the model picks among them.
    groups = groups_for_years(groups, fields.get("years"))
    questions = person_card_questions(fields, groups, sorted(_NATIONALITY_CODES),
                                      sorted(_NATIONAL_ORIGIN_CODES), codes=codes)
    from commulingo.activities import activity_evidence, activity_questions, activity_person_from, activity_basis_question, excerpt_years, load_catalog
    basis = activity_evidence(fields, claims)
    if not basis:
        logger.warning("activity classification requires cited career/bio evidence excerpts")
        return None
    activity_catalog = load_catalog()
    questions['activity_function'] = activity_questions(activity_catalog, basis)['activity_function']
    state = person_card_state(fields, claims)
    state['cited_activity_evidence'] = basis
    result = (decide or decide_detailed)(FEATURE, state, questions,
                                         label="person-card" if codes else "person-classification")
    decision = result.decision
    if decision is None:
        logger.warning("person card classification unavailable: %s", result.error)
        return None
    verdicts = _codes_from(decision, questions, fields, accept) if codes else {}
    # Jev questions run independently. Later choices must see the actual
    # earlier answer, never instructions referring to another parallel question.
    function = decision.choice('activity_function')
    if function not in {f['id'] for f in activity_catalog['functions']}:
        return None
    state['selected_activity_function'] = function
    # Evidence first, then the organization: the chosen excerpt's years
    # (or the adult life when it names none) decide which affiliations
    # existed and are offered, so an 1830 excerpt is never read as service
    # to the First Republic (job 63829, Lafayette, 2026-09-26).
    basis_q = activity_basis_question(activity_catalog, basis, function, decision.choice('group'))
    basis_result = (decide or decide_detailed)(FEATURE, state,
        {'activity_basis': basis_q}, label='person-activity-basis')
    if basis_result.decision is None:
        return None
    basis_choice = basis_result.decision.choice('activity_basis')
    if not isinstance(basis_choice, str) or not basis_choice.isdigit() or int(basis_choice) >= len(basis):
        return None
    chosen = basis[int(basis_choice)]
    state['selected_activity_evidence'] = chosen
    span = active_span(fields.get("years"))
    years = excerpt_years(chosen.get('excerpt'), span)
    window = (years[0], years[-1]) if years else span
    state['selected_activity_years'] = list(window) if window else None
    affiliation_q = activity_questions(activity_catalog, basis, window)['activity_affiliation']
    affiliation_q['instructions'] += ' The function and excerpt are fixed by selected_activity_function and selected_activity_evidence in the state.'
    affiliation_result = (decide or decide_detailed)(FEATURE, state,
        {'activity_affiliation': affiliation_q}, label='person-activity-affiliation')
    if affiliation_result.decision is None:
        return None
    affiliation = affiliation_result.decision.choice('activity_affiliation')
    if affiliation not in set(affiliation_q['criteria']):
        return None
    decision = Decision(answers={**decision.answers,
        'activity_affiliation': affiliation_result.decision.answers['activity_affiliation'],
        'activity_basis': basis_result.decision.answers['activity_basis']}, model=decision.model)
    person = activity_person_from(decision, activity_catalog, basis, {g['id'] for g in groups}, accept)
    return {"codes": verdicts, "person": person}


def fill_classification(fields: dict, classification: dict | None) -> dict:
    """Copy of ``fields`` with the assigned group and primary activity. The writer never
    classifies: the schema it drafts against has no such fields, so a
    classification that came back always lands, low confidence included
    (the review stage gets that as a risk line)."""
    out = dict(fields)
    if classification is None:
        return out
    out.pop("group", None)
    out["groupId"] = classification["groupId"]
    out.pop("role", None)  # the legacy person role was retired; activities replace it
    if classification.get("activities"):
        out["activities"] = classification["activities"]
    return out
