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
original_db_query = maintainer.db_query
try:
    maintainer.db_query = lambda sql, params=None: [
        {"name_ko": "니콜라이 예조프", "name_en": "Nikolai Yezhov"},
        {"name_ko": "라브렌티 베리야", "name_en": "Lavrentiy Beria"},
    ]
    focused_discovery = maintainer.build_discovery_task("soviet_institutions")
finally:
    maintainer.db_query = original_db_query
assert "Do not select a non-Soviet revolutionary" in focused_discovery
assert "Soviet institution" in focused_discovery and "list_offices" in focused_discovery
# discovery must be able to read absence off a roster instead of guessing at it
assert "ALREADY IN THE DICTIONARY" in focused_discovery
assert "니콜라이 예조프(Nikolai Yezhov), 라브렌티 베리야(Lavrentiy Beria)" in focused_discovery

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
    async def fake_people(**_kwargs):
        return "[]"
    box = {}
    handlers = maintainer.build_bounded_discovery_handlers(
        {"commulingo_people": fake_people}, box,
    )
    for _ in range(6):
        assert await handlers["commulingo_people"](action="search_people", q="x") == "[]"
    try:
        await handlers["commulingo_people"](action="search_people", q="overflow")
        raise AssertionError("discovery search limit should fail closed")
    except ValueError as exc:
        assert "discovery_search_limit" in str(exc)

asyncio.run(assert_discovery_search_bound())

print("commulingo maintainer smoke ok")
