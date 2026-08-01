#!/usr/bin/env python3
"""Hermetic smoke checks for the dedicated CommuLingo maintainer."""

from pathlib import Path
import sys
import json
import asyncio
from types import SimpleNamespace
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.commulingo_curator import COMMULINGO_CURATOR
from runtime_tools.commulingo_people import (
    DENSE_SENTENCE_CHARS,
    FIELD_LIMITS,
    sentence_budget,
    COMMULINGO_EVENT_LINK_TOOL,
    COMMULINGO_OFFICE_ROW_SAVE_TOOL,
    COMMULINGO_PERSON_CREATE_TOOL,
    COMMULINGO_PERSON_UPDATE_TOOL,
    COMMULINGO_SECTION_SAVE_TOOL,
    COMMULINGO_TERM_CREATE_TOOL,
    CommulingoInputError,
    _merge_patronymic_patch,
    _nationality_values,
    _patronymic_problem,
    _person_create_nationality_problem,
    _validate,
    normalize_commulingo_write,
)
import scripts.commulingo_people_maintainer as maintainer


class EmptyCursor:
    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return None


CURSOR = EmptyCursor()


class SectionCursor:
    """A person carrying one section: slug 'taken-slug', heading '이미 있는 제목'."""

    def __init__(self):
        self._sql, self._params = "", ()

    def execute(self, sql, params=None):
        self._sql, self._params = sql, params or ()

    def fetchone(self):
        if "commulingo_person_sections" in self._sql:
            return {"ok": 1} if self._params[-1] == "taken-slug" else None
        return {"ok": 1}  # the target person exists

    def fetchall(self):
        return [{"slug": "taken-slug", "heading_ko": "이미 있는 제목",
                 "heading_en": "An Existing Heading"}]


SECTION_CURSOR = SectionCursor()


assert COMMULINGO_CURATOR.provider == "deepseek"
assert COMMULINGO_CURATOR.model == "deepseek_pro"
NARROW_TOOLS = {
    "commulingo_person_create", "commulingo_person_update",
    "commulingo_section_save", "commulingo_event_link", "commulingo_term_create",
}
assert set(COMMULINGO_CURATOR.tools) == {
    "wiki_search", "wiki_get", "web_search", "fetch_url", "commulingo_people", *NARROW_TOOLS,
}
assert set(COMMULINGO_CURATOR.terminal_tools) == NARROW_TOOLS
assert set(COMMULINGO_CURATOR.finalization_tools) == NARROW_TOOLS
assert COMMULINGO_CURATOR.max_rounds == 16
assert COMMULINGO_CURATOR.max_input_tokens == 160_000
assert COMMULINGO_CURATOR.max_output_tokens == 16_000
assert COMMULINGO_CURATOR.max_output_continuations == 2
assert COMMULINGO_CURATOR.thinking_policy == "tool_loop"
assert "Verified nicknames" in COMMULINGO_CURATOR.prompt_ir.identity
assert "given name + surname ONLY" in COMMULINGO_CURATOR.prompt_ir.identity
assert "Birthplace and work in the Ukrainian SSR alone never suffice" in maintainer.CARD_STYLE_GUIDANCE
assert "Jewish ancestry alone does not create a separate" in maintainer.CARD_STYLE_GUIDANCE
# The prescribed sentence count and the ceiling it is written under must agree.
# When they disagreed by one sentence (4-5 prescribed, 4 affordable), 17 of 19
# curator tool rejections in a day were a fifth bio sentence overflowing both
# languages — each a paid round spent discovering the arithmetic.
_BIO_BUDGET = sentence_budget("bio")
_KO_COST, _EN_COST = DENSE_SENTENCE_CHARS
assert _BIO_BUDGET * _KO_COST <= FIELD_LIMITS["bio"][0], "prescription exceeds the Korean ceiling"
assert _BIO_BUDGET * _EN_COST <= FIELD_LIMITS["bio"][1], "prescription exceeds the English ceiling"
assert f"{_BIO_BUDGET - 1}-{_BIO_BUDGET} sentences" in maintainer.MAJOR_BIO_SENTENCES
assert maintainer.MAJOR_BIO_SENTENCES in maintainer.CARD_STYLE_GUIDANCE
# The curator prompt states the same derived count, and no token is left unfilled.
assert f"{_BIO_BUDGET - 1}–{_BIO_BUDGET} sentences for a major" in COMMULINGO_CURATOR.prompt_ir.identity
assert "__" not in COMMULINGO_CURATOR.prompt_ir.identity
assert "already embeds cyrillicPatronymic" in _validate(
    CURSOR, "person", "update", "example", {
        "cyrillic": "Михаил Петрович Фриновский",
        "patronymic": {"ko": "페트로비치", "en": "Petrovich"},
        "cyrillicPatronymic": "Петрович",
    }
)
assert "contains '북한'" in _validate(
    CURSOR, "person", "update", "example",
    {"bio": {"ko": "북한 관련 문장", "en": "A sentence"}},
)
assert "bio is too long" in _validate(
    CURSOR, "person", "update", "example",
    {"bio": {"ko": "가" * 381, "en": "A sentence"}},
)
# moment used to have no ceiling at all; 308-character moments reached live cards
assert "moment is too long" in _validate(
    CURSOR, "person", "update", "example",
    {"moment": {"ko": "가" * 141, "en": "A sentence"}},
)
assert "one sentence, two at most" in _validate(
    CURSOR, "person", "update", "example",
    {"moment": {"ko": "가", "en": "x" * 301}},
)
# The same topic under a second slug is what two lanes enriching one person produced.
assert "already covers this topic" in _validate(
    SECTION_CURSOR, "person_section", "create", "example",
    {"slug": "a-different-slug",
     "heading": {"ko": "이미 있는 제목", "en": "An Existing Heading"},
     "body": {"ko": "본문", "en": "Body"}},
)
assert "Do NOT" in _validate(
    SECTION_CURSOR, "person_section", "create", "example",
    {"slug": "taken-slug", "heading": {"ko": "새 제목", "en": "New Heading"},
     "body": {"ko": "본문", "en": "Body"}},
)
assert not _validate(
    SECTION_CURSOR, "person_section", "create", "example",
    {"slug": "a-different-slug", "heading": {"ko": "새 제목", "en": "New Heading"},
     "body": {"ko": "본문", "en": "Body"}},
)
assert "epithet is too long" in _validate(
    CURSOR, "person", "update", "example",
    {"epithet": {"ko": "짧은 표현", "en": "x" * 141}},
)
assert "fate.label is too long" in _validate(
    CURSOR, "person", "update", "example",
    {"fate": {"kind": "natural", "label": {"ko": "가" * 23, "en": "Died of illness"}}},
)
assert "unknown patch key" not in (_validate(
    CURSOR, "person", "update", "example",
    {"citizenship": {"code": "mali", "label": {"ko": "말리", "en": "Mali"}}},
) or "")
assert "no flag icon" in _validate(
    CURSOR, "person", "update", "example",
    {"citizenship": {"code": "mali", "label": {"ko": "말리", "en": "Mali"}}},
)
assert "citizenship must be" in _validate(
    CURSOR, "person", "update", "example", {"citizenship": "vietnam"},
)
assert "non-standard person-name spelling" in _validate(
    CURSOR, "person", "update", "example",
    {"bio": {"ko": "베리아의 심복으로 일했다.", "en": "Worked under Beria."}},
)
assert "non-standard person-name spelling" in _validate(
    CURSOR, "person_section", "create", "example",
    {"slug": "x", "heading": {"ko": "제목", "en": "T"},
     "body": {"ko": "투하체프스키 재판에 관여했다.", "en": "Involved."}, "sources": []},
)
from runtime_tools.commulingo_people import _find_name_variants
assert _find_name_variants(
    {"body": {"ko": "그는 “베리아 동지에게 보고하라”라고 적었다. 베리야는 침묵했다.", "en": ""}}
) == []
assert _find_name_variants(
    {"career": [{"y": "1938", "r": {"ko": "베리아의 부관", "en": "Beria's deputy"}}]}
) == [("베리아", "베리야")]

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "config.json"
    path.write_text('{"mode":"enrich","new_person_every":4,"recent_days":7}', encoding="utf-8")
    cfg = maintainer.load_config(path)
    assert cfg["mode"] == "enrich"
    assert cfg["new_person_every"] == 4
    assert cfg["recent_days"] == 7
    assert cfg["new_person_cooldown_runs"] == 6
    assert cfg["enrich_failure_cooldown_runs"] == 6
    assert cfg["enrich_non_soviet_revolutionaries"] is True
    assert cfg["new_person_focus"] == "all"

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "config.json"
    path.write_text('{"new_person_focus":"unsupported"}', encoding="utf-8")
    try:
        maintainer.load_config(path)
        raise AssertionError("unsupported people focus should fail")
    except ValueError as exc:
        assert "new_person_focus" in str(exc)

candidate = {
    "id": "example",
    "name_ko": "예시",
    "name_en": "Example",
    "group_id": "thaw",
    "bio_chars": 40,
    "has_epithet": 0,
    "career_count": 1,
    "section_count": 0,
    "event_count": 0,
    "has_moment": 0,
    "has_role": 1,
}
task = maintainer.build_task("enrich", candidate)
assert "example" in task and "get_person" in task and "one available narrow write" in task
assert "Do not create a section" in task and "has epithet: False" in task
assert "commulingo_event_link" in task and "linked historical events: 0" in task
new_task = maintainer.build_task("new", None)
assert "search_people" in new_task and "commulingo_person_create" in new_task
assert "Every new person requires both citizenship and nationalOrigin" in new_task
assert "either the citizenship or nationalOrigin flag code is unset" in task
_ROSTER_ROWS = [
    {"group_id": "stalin-era", "title_ko": "스탈린 시대의 사람들", "range_label": "1929–1953",
     "name_ko": "니콜라이 예조프", "name_en": "Nikolai Yezhov"},
    {"group_id": "stalin-era", "title_ko": "스탈린 시대의 사람들", "range_label": "1929–1953",
     "name_ko": "라브렌티 베리야", "name_en": "Lavrentiy Beria"},
    {"group_id": "international-revolutionary", "title_ko": "비소련 혁명가", "range_label": "1871–2016",
     "name_ko": "로자 룩셈부르크", "name_en": "Rosa Luxemburg"},
]

original_db_query = maintainer.db_query
try:
    maintainer.db_query = lambda sql, params=None: _ROSTER_ROWS
    focused_discovery = maintainer.build_discovery_task("soviet_institutions")
    scoped_discovery = maintainer.build_discovery_task(
        "soviet_institutions", None, ("stalin-era",),
    )
    full_roster = maintainer.registered_person_roster()
    scoped_roster = maintainer.registered_person_roster(("stalin-era",))
finally:
    maintainer.db_query = original_db_query
assert "Do not select a non-Soviet revolutionary" in focused_discovery
assert "Soviet institution" in focused_discovery and "list_offices" in focused_discovery
# discovery must be able to read absence off a roster instead of guessing at it
assert "ALREADY IN THE DICTIONARY" in focused_discovery
assert "니콜라이 예조프(Nikolai Yezhov), 라브렌티 베리야(Lavrentiy Beria)" in focused_discovery

# The roster carries an era heading per group so the model can read the one
# section its candidate belongs to instead of a single undivided wall.
assert "### 스탈린 시대의 사람들 (1929–1953) — 2명" in full_roster, full_roster
assert "### 비소련 혁명가 (1871–2016) — 1명" in full_roster, full_roster

# Narrowing keeps the in-focus era in full and NAMES the omitted one with its
# count, so absence outside the roster is never read as a gap.
assert "니콜라이 예조프(Nikolai Yezhov)" in scoped_roster
assert "로자 룩셈부르크" not in scoped_roster, scoped_roster
assert "비소련 혁명가 1명" in scoped_roster, scoped_roster
assert "여기 없다는 사실이 빈자리라는 뜻은 아니다" in scoped_roster
# The omitted era loses its listing, not just its heading. (Char length is not
# the assertion: on a three-row fixture the omission notice outweighs the two
# names it replaces, though on the real 1,095-card roster it does not.)
assert "### 비소련 혁명가 (1871–2016)" not in scoped_roster, scoped_roster
assert scoped_roster.count("(") < full_roster.count("(")
assert "비소련 혁명가 1명" in scoped_discovery

# Focus drives the group list; an explicit config list overrides it.
assert maintainer.roster_groups_for_focus(
    {"new_person_focus": "soviet_institutions", "roster_groups": []}
) == ("bolshevik", "stalin-era", "thaw", "perestroika")
assert maintainer.roster_groups_for_focus(
    {"new_person_focus": "all", "roster_groups": []}
) == (), "focus=all must cover every group"
assert maintainer.roster_groups_for_focus(
    {"new_person_focus": "soviet_institutions", "roster_groups": ["thaw"]}
) == ("thaw",)

original_db_query = maintainer.db_query
captured_selection = {}
try:
    def capture_selection(sql, params):
        captured_selection["sql"] = sql
        captured_selection["params"] = params
        return []
    maintainer.db_query = capture_selection
    assert maintainer.select_sparse_person(30, enrich_non_soviet_revolutionaries=False) is None
finally:
    maintainer.db_query = original_db_query
assert captured_selection["params"]["enrich_non_soviet_revolutionaries"] is False
assert "excluded_role.category_id = 'non-soviet-revolutionary'" in captured_selection["sql"]

assert maintainer.choose_mode(
    {**cfg, "mode": "auto", "new_person_every": 1},
    state={"new_cooldown_remaining": 2},
) == "enrich"

legacy_patch = {
    "slug": "example-person", "nameKo": "예시", "nameEn": "Example",
    "bioKo": "한국어", "bioEn": "English", "epithetKo": "긴장",
    "epithetEn": "Tension", "fateKind": "natural", "yearsLabel": "1900–1980",
    "category": "revolutionary",
    "sources": ["https://example.com/bio — biography"],
    "confidence": 0.91,
}
normalized, citations, confidence, repairs = normalize_commulingo_write(
    "person", "example-person", legacy_patch, [], None,
)
assert normalized["name"] == {"ko": "예시", "en": "Example"}
assert normalized["bio"] == {"ko": "한국어", "en": "English"}
assert normalized["fate"]["kind"] == "natural"
assert normalized["role"] == {"category": "revolutionary"}
assert normalized["years"] == "1900–1980"
assert "slug" not in normalized and repairs
assert citations == ["https://example.com/bio — biography"]
assert confidence == 0.91

person_fields = COMMULINGO_PERSON_CREATE_TOOL["input_schema"]["properties"]["fields"]["properties"]
term_fields = COMMULINGO_TERM_CREATE_TOOL["input_schema"]["properties"]["fields"]["properties"]
assert {"citizenship", "nationalOrigin"} <= set(person_fields)
create_required = set(COMMULINGO_PERSON_CREATE_TOOL["input_schema"]["properties"]["fields"]["required"])
assert {"citizenship", "nationalOrigin"} <= create_required
valid_nationality = {"code": "russia", "label": {"ko": "러시아", "en": "Russia"}}
assert _person_create_nationality_problem({"citizenship": valid_nationality, "origin": valid_nationality}) is None
assert "nationalOrigin" in _person_create_nationality_problem({"citizenship": valid_nationality})
assert "citizenship" in _person_create_nationality_problem({"origin": valid_nationality})
assert "nationalOrigin" in _validate(
    CURSOR, "person", "create", "missing-origin", {"citizenship": valid_nationality},
)
assert "citizenship" in _validate(
    CURSOR, "person", "create", "missing-citizenship", {"origin": valid_nationality},
)
assert "origin" not in person_fields
assert "sources" not in person_fields and "term" not in person_fields
assert "sources" not in term_fields and "bio" not in term_fields

# Glossary metadata: an entry written without a category renders as
# 'Uncategorized', and a single-string period leaks one language onto the other
# page, so the schema demands both and the years that sort the entry.
assert {"category", "startYear", "endYear"} <= set(term_fields)
assert term_fields["period"]["type"] == "object"
assert set(term_fields["period"]["properties"]) == {"ko", "en"}
term_required = set(COMMULINGO_TERM_CREATE_TOOL["input_schema"]["properties"]["fields"]["required"])
assert {"period", "category"} <= term_required
_dated_term = {
    "term": {"ko": "용어", "en": "Term"},
    "definition": {"ko": "정의.", "en": "Definition."},
    "aliases": {"ko": ["용어"], "en": ["term"]},
    "period": {"ko": "1930년대–1991", "en": "1930s–1991"},
    "startYear": 1930, "endYear": 1991, "category": "economy",
}
assert _validate(CURSOR, "term", "create", "ok-term", _dated_term) is None
assert "category is required" in _validate(
    CURSOR, "term", "create", "no-category", {k: v for k, v in _dated_term.items() if k != "category"},
)
assert "category must be one of" in _validate(
    CURSOR, "term", "create", "bad-category", {**_dated_term, "category": "misc"},
)
assert "period is required" in _validate(
    CURSOR, "term", "create", "no-period", {k: v for k, v in _dated_term.items() if k != "period"},
)
assert "must be an object" in _validate(
    CURSOR, "term", "create", "string-period", {**_dated_term, "period": "1930-1991"},
)
assert "startYear is required" in _validate(
    CURSOR, "term", "create", "no-start", {**_dated_term, "startYear": None},
)
assert "before startYear" in _validate(
    CURSOR, "term", "create", "reversed-years", {**_dated_term, "endYear": 1900},
)
# An undated concept keeps null years without tripping the year guard.
assert _validate(CURSOR, "term", "create", "concept-term", {
    **_dated_term, "period": {"ko": "개념", "en": "Concept"},
    "startYear": None, "endYear": None,
}) is None
assert COMMULINGO_PERSON_UPDATE_TOOL["input_schema"]["additionalProperties"] is False
assert COMMULINGO_SECTION_SAVE_TOOL["input_schema"]["additionalProperties"] is False
assert COMMULINGO_EVENT_LINK_TOOL["input_schema"]["additionalProperties"] is False
office_fields = COMMULINGO_OFFICE_ROW_SAVE_TOOL["input_schema"]["properties"]["fields"]["properties"]
assert set(office_fields) == {"sortOrder", "years", "body", "personId", "name", "note"}
assert "bio" not in office_fields and "term" not in office_fields

stored_patronymic = {"ko": "이바노비치", "en": "Ivanovich", "native": "Иванович"}
merged_patronymic = _merge_patronymic_patch(
    {"cyrillicPatronymic": "Петрович"}, stored_patronymic,
)
assert merged_patronymic["ko"] == "이바노비치"
assert merged_patronymic["en"] == "Ivanovich"
assert merged_patronymic["native"] == "Петрович"
assert "requires cyrillicPatronymic" in _patronymic_problem(
    {"ko": "이바노비치", "en": "Ivanovich", "native": "", "invalid": ""},
    "Иван Иванов",
)

normalized_origin, _, _, origin_repairs = normalize_commulingo_write(
    "person", "example", {"nationalOrigin": {"code": "poland", "label": {"ko": "폴란드", "en": "Poland"}}},
    ["https://example.com — national background"], None,
)
assert normalized_origin["origin"]["code"] == "poland"
assert "nationalOrigin->origin" in origin_repairs
assert _nationality_values(
    {"origin": {"code": "georgia", "label": {"ko": "조지아", "en": "Georgia"}}},
    "origin",
) == ("georgia", "그루지야", "Georgia")
assert _nationality_values(
    {"citizenship": {"code": "georgia", "label": {"ko": "조지아", "en": "Georgia"}}},
    "citizenship",
) == ("georgia", "조지아", "Georgia")

normalized_terms, _, _, term_repairs = normalize_commulingo_write(
    "person",
    "example",
    {
        "bio": {"ko": "조지아 공산당에서 활동했다.", "en": "Worked in Georgia."},
        "citizenship": {"code": "georgia", "label": {"ko": "조지아", "en": "Georgia"}},
        "career": [{"y": "1930", "r": {"ko": "조지아 당 서기", "en": "Party secretary"}}],
    },
    ["https://example.com — terminology"],
    None,
)
assert normalized_terms["bio"]["ko"] == "그루지야 공산당에서 활동했다."
assert normalized_terms["career"][0]["r"]["ko"] == "그루지야 당 서기"
assert normalized_terms["citizenship"]["label"]["ko"] == "조지아"
assert "조지아->그루지야 in Korean content" in term_repairs

try:
    normalize_commulingo_write(
        "person", "example", {"definition": {"ko": "x", "en": "x"}},
        ["https://example.com — source"], None,
    )
    raise AssertionError("cross-target field should fail")
except CommulingoInputError as exc:
    assert exc.code == "unknown_field"

original_query_one = maintainer.db_query_one
original_query = maintainer.db_query
try:
    duplicate_queries = []
    maintainer.db_query_one = lambda sql, *_a, **_kw: duplicate_queries.append(sql) or None
    roster = [{"id": "c-l-r-james", "name_ko": "C. L. R. 제임스", "name_en": "C. L. R. James"}]
    maintainer.db_query = lambda sql, params=None: roster
    candidate_payload = {
        "id": "example-person", "name_ko": "예시",
        "name_en": "Example Person", "reason": "gap",
        "source_url": "https://example.com/bio",
    }
    discovered = maintainer.validate_discovered_candidate(candidate_payload)
    assert discovered["id"] == "example-person"
    assert "commulingo_person_aliases" in duplicate_queries[-1]

    # a different transliteration of a registered card is still a duplicate
    try:
        maintainer.validate_discovered_candidate({
            "id": "clr-james", "name_ko": "C.L.R. 제임스",
            "name_en": "C.L.R. James", "reason": "gap",
            "source_url": "https://example.com/bio",
        })
        raise AssertionError("respelled duplicate should be rejected")
    except ValueError as exc:
        assert "under a different spelling" in str(exc)
finally:
    maintainer.db_query_one = original_query_one
    maintainer.db_query = original_query

async def assert_dsml_retry():
    original_chat = maintainer.chat_with_tools
    original_count = maintainer.completed_run_count
    original_query = maintainer.db_query_one
    original_roster = maintainer.db_query
    calls = []
    async def fake_chat(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return "<｜｜DSML｜｜tool_calls>"
        await _kwargs["tool_handlers"]["commulingo_candidate_select"](**{
            "id": "retry-person", "name_ko": "재시도",
            "name_en": "Retry Person", "reason": "gap",
            "source_url": "https://example.com/retry",
        })
        return "selected"
    try:
        maintainer.chat_with_tools = fake_chat
        maintainer.completed_run_count = lambda: 10
        maintainer.db_query_one = lambda *_a, **_kw: None
        maintainer.db_query = lambda sql, params=None: []
        policy = SimpleNamespace(
            max_output_continuations=2, max_rounds=16, max_output_tokens=16000,
            max_input_tokens=160000, budget_usd=0.35,
            thinking_policy="disabled", thinking_budget_tokens=8192,
        )
        spec = SimpleNamespace(
            name="commulingo_curator", finalization_tools=[], terminal_tools=[],
            render_prompt=lambda **_kw: "system",
        )
        box = {}
        candidate_handler = maintainer.build_candidate_select_handler(box)
        _result, _tracker, found = await maintainer._call_curator_stage(
            task="discover", spec=spec, model="deepseek-v4-pro",
            tools=[maintainer.COMMULINGO_CANDIDATE_SELECT_TOOL],
            handlers={"commulingo_candidate_select": candidate_handler},
            policy=policy, stage="test discovery",
            expect_edit=False, before_count=10,
            finalization_tools=["commulingo_candidate_select"],
            terminal_tools=["commulingo_candidate_select"], candidate_box=box,
        )
        assert len(calls) == 2
        assert found["id"] == "retry-person"
    finally:
        maintainer.chat_with_tools = original_chat
        maintainer.completed_run_count = original_count
        maintainer.db_query_one = original_query
        maintainer.db_query = original_roster

asyncio.run(assert_dsml_retry())

async def assert_structured_retry_error():
    async def fake_write(**_kwargs):
        return 'Error: {"ok":false,"error":{"code":"unknown_field","message":"bad field","retryable":true}}'
    wrapped = maintainer.build_retrying_write_handler(fake_write)
    try:
        await wrapped(fields={})
        raise AssertionError("structured error should become a retryable tool failure")
    except ValueError as exc:
        assert "unknown_field" in str(exc) and "retryable=true" in str(exc)

asyncio.run(assert_structured_retry_error())

async def assert_discovery_search_bound():
    calls = []

    async def fake_people(**kwargs):
        calls.append(kwargs)
        return "[]"

    box = {}
    handlers = maintainer.build_bounded_discovery_handlers(
        {"commulingo_people": fake_people}, box,
    )
    budget = maintainer.DISCOVERY_SEARCH_BUDGET
    # Every in-budget search carries the payload plus what is left, counting down.
    for spent in range(1, budget + 1):
        body = await handlers["commulingo_people"](action="search_people", q="x")
        assert body.startswith("[]"), body
        assert f"{budget - spent} of {budget}" in body, body
    assert len(calls) == budget

    # Past the cap the call no longer reaches the database and no longer raises:
    # an error here is what made the model retry instead of moving on.
    for _ in range(3):
        body = await handlers["commulingo_people"](action="search_people", q="overflow")
        payload = json.loads(body)
        assert payload["search_budget_spent"] is True and payload["remaining"] == 0
        assert payload["next_action"] == "commulingo_candidate_select"
    assert len(calls) == budget, "over-budget searches must not hit the handler"

    # Non-search actions are never counted or capped.
    assert await handlers["commulingo_people"](action="list_groups") == "[]"
    assert len(calls) == budget + 1

asyncio.run(assert_discovery_search_bound())

def assert_rejected_candidate_memory():
    # A duplicate rejection is remembered; a malformed-slug rejection is not,
    # because that person may still be a genuine gap.
    box = {}
    maintainer.record_rejected_candidate(
        box, {"name_ko": "블라디미르 첼로메이", "name_en": "Vladimir Chelomey"},
        "candidate duplicates existing person vladimir-chelomey",
    )
    assert box["rejected"] == [
        {"label": "블라디미르 첼로메이(Vladimir Chelomey)", "existing_id": "vladimir-chelomey"}
    ], box
    # Recording the same person twice must not grow the list.
    maintainer.record_rejected_candidate(
        box, {"name_ko": "블라디미르 첼로메이", "name_en": "Vladimir Chelomey"},
        "candidate duplicates existing person vladimir-chelomey",
    )
    assert len(box["rejected"]) == 1, box

    note = maintainer.rejected_candidate_note(box["rejected"])
    assert "블라디미르 첼로메이(Vladimir Chelomey)" in note and "vladimir-chelomey" in note
    assert maintainer.rejected_candidate_note([]) == ""

    # The discovery task must actually carry the note, or the memory is inert.
    task = maintainer.build_discovery_task("all", box["rejected"])
    assert "ALREADY PROPOSED AND REJECTED" in task and "Vladimir Chelomey" in task

    # load_state is a whitelist; the list has to survive a save/load round trip.
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        maintainer.save_state(
            {"new_cooldown_remaining": 3, "rejected_candidates": box["rejected"]}, path,
        )
        reloaded = maintainer.load_state(path)
    assert reloaded["new_cooldown_remaining"] == 3
    assert reloaded["rejected_candidates"] == box["rejected"], reloaded
    # Junk entries are dropped rather than crashing the run.
    assert maintainer._clean_rejected([{"label": ""}, "nope", {"label": "ok"}]) == [
        {"label": "ok", "existing_id": ""}
    ]

assert_rejected_candidate_memory()


def assert_enrich_failure_cooldown():
    """A card that could not be enriched steps aside instead of being re-picked.

    Only an applied edit writes a revision, so the DB-side cooldown never sees a
    failed run: without this, the hourly lane re-picks the same unresearchable
    card every hour and spends the full three attempts on it again.
    """
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        maintainer.save_state(
            {
                "new_cooldown_remaining": 0,
                "rejected_candidates": [],
                "failed_candidates": [
                    {"id": "mikhail-kozlovsky", "runs_left": 6},
                    {"id": "expired", "runs_left": 0},
                ],
            },
            path,
        )
        reloaded = maintainer.load_state(path)
    # An entry whose cooldown has run out is dropped on load, not carried at 0.
    assert reloaded["failed_candidates"] == [{"id": "mikhail-kozlovsky", "runs_left": 6}], reloaded
    # Junk entries are dropped rather than crashing the run.
    assert maintainer._clean_failed(
        [{"id": ""}, "nope", {"id": "x", "runs_left": "abc"}, {"id": "ok", "runs_left": 2}]
    ) == [{"id": "ok", "runs_left": 2}]

    seen = []

    def fake_select(recent, forced, incomplete, non_soviet, exclude_ids=None):
        seen.append(list(exclude_ids or []))
        return None

    original_select = maintainer.select_sparse_person
    try:
        maintainer.select_sparse_person = fake_select
        maintainer.select_claimable_person(
            maintainer.load_config(Path("/nonexistent")), "",
            exclude_ids=["mikhail-kozlovsky"],
        )
    finally:
        maintainer.select_sparse_person = original_select
    assert seen == [["mikhail-kozlovsky"]], seen


assert_enrich_failure_cooldown()

print("commulingo maintainer smoke ok")
