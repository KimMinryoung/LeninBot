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
from runtime_tools.commulingo_people import _validate
import scripts.commulingo_people_maintainer as maintainer


assert COMMULINGO_CURATOR.provider == "deepseek"
assert COMMULINGO_CURATOR.model == "deepseek_pro"
assert set(COMMULINGO_CURATOR.tools) == {"wiki_search", "wiki_get", "web_search", "fetch_url", "commulingo_people", "commulingo_edit"}
assert COMMULINGO_CURATOR.terminal_tools == ["commulingo_edit"]
assert COMMULINGO_CURATOR.max_rounds == 16
assert COMMULINGO_CURATOR.max_input_tokens == 160_000
assert COMMULINGO_CURATOR.max_output_tokens == 16_000
assert COMMULINGO_CURATOR.max_output_continuations == 2
assert COMMULINGO_CURATOR.thinking_policy == "tool_loop"
assert "Verified nicknames" in COMMULINGO_CURATOR.prompt_ir.identity
assert "given name + surname ONLY" in COMMULINGO_CURATOR.prompt_ir.identity
assert "already includes cyrillicPatronymic" in _validate(
    None, "person", "update", "example", {"cyrillic": "Михаил Петрович Фриновский", "cyrillicPatronymic": "Петрович"}
)
assert "contains '북한'" in _validate(
    None, "person", "update", "example",
    {"bio": {"ko": "북한 관련 문장", "en": "A sentence"}},
)
assert "bio is too long" in _validate(
    None, "person", "update", "example",
    {"bio": {"ko": "가" * 321, "en": "A sentence"}},
)
assert "epithet is too long" in _validate(
    None, "person", "update", "example",
    {"epithet": {"ko": "짧은 표현", "en": "x" * 141}},
)
assert "fate.label is too long" in _validate(
    None, "person", "update", "example",
    {"fate": {"kind": "natural", "label": {"ko": "가" * 23, "en": "Died of illness"}}},
)
assert "unknown patch key" not in (_validate(
    None, "person", "update", "example",
    {"citizenship": {"code": "mali", "label": {"ko": "말리", "en": "Mali"}}},
) or "")
assert "no flag icon" in _validate(
    None, "person", "update", "example",
    {"citizenship": {"code": "mali", "label": {"ko": "말리", "en": "Mali"}}},
)
assert "citizenship must be" in _validate(
    None, "person", "update", "example", {"citizenship": "vietnam"},
)
assert "non-standard person-name spelling" in _validate(
    None, "person", "update", "example",
    {"bio": {"ko": "베리아의 심복으로 일했다.", "en": "Worked under Beria."}},
)
assert "non-standard person-name spelling" in _validate(
    None, "person_section", "create", "example",
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
assert "example" in task and "get_person" in task and "one commulingo_edit" in task
assert "Do not create a section" in task and "has epithet: False" in task
assert "history_event_person" in task and "linked historical events: 0" in task
assert "initial" not in maintainer._PERSON_PATCH_KEYS
assert "history_event_person" in __import__("runtime_tools.commulingo_people", fromlist=["COMMULINGO_EDIT_TOOL"]).COMMULINGO_EDIT_TOOL["input_schema"]["properties"]["target_type"]["enum"]
assert "initial" not in __import__("runtime_tools.commulingo_people", fromlist=["COMMULINGO_EDIT_TOOL"]).COMMULINGO_EDIT_TOOL["input_schema"]["properties"]["patch"]["properties"]
new_task = maintainer.build_task("new", None)
assert "search_people" in new_task and "action='create'" in new_task

assert maintainer.choose_mode(
    {**cfg, "mode": "auto", "new_person_every": 1},
    state={"new_cooldown_remaining": 2},
) == "enrich"

legacy_patch = {
    "slug": "example-person", "nameKo": "예시", "nameEn": "Example",
    "bioKo": "한국어", "bioEn": "English", "epithetKo": "긴장",
    "epithetEn": "Tension", "fateKind": "natural", "yearsLabel": "1900–1980",
    "category": "revolutionary",
}
normalized, repairs = maintainer.normalize_commulingo_patch(
    "person", "example-person", legacy_patch
)
assert normalized["name"] == {"ko": "예시", "en": "Example"}
assert normalized["bio"] == {"ko": "한국어", "en": "English"}
assert normalized["fate"]["kind"] == "natural"
assert normalized["role"] == {"category": "revolutionary"}
assert normalized["years"] == "1900–1980"
assert "slug" not in normalized and repairs

original_query_one = maintainer.db_query_one
try:
    maintainer.db_query_one = lambda *_a, **_kw: None
    candidate_line = "CANDIDATE_JSON: " + json.dumps({
        "id": "example-person", "name_ko": "예시",
        "name_en": "Example Person", "reason": "gap",
        "source_url": "https://example.com/bio",
    }, ensure_ascii=False)
    discovered = maintainer.parse_discovered_candidate(candidate_line)
    assert discovered["id"] == "example-person"
finally:
    maintainer.db_query_one = original_query_one

async def assert_dsml_retry():
    original_chat = maintainer.chat_with_tools
    original_count = maintainer.completed_run_count
    original_query = maintainer.db_query_one
    calls = []
    async def fake_chat(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return "<｜｜DSML｜｜tool_calls>"
        return "CANDIDATE_JSON: " + json.dumps({
            "id": "retry-person", "name_ko": "재시도",
            "name_en": "Retry Person", "reason": "gap",
            "source_url": "https://example.com/retry",
        }, ensure_ascii=False)
    try:
        maintainer.chat_with_tools = fake_chat
        maintainer.completed_run_count = lambda: 10
        maintainer.db_query_one = lambda *_a, **_kw: None
        policy = SimpleNamespace(
            max_output_continuations=2, max_rounds=16, max_output_tokens=16000,
            max_input_tokens=160000, budget_usd=0.35,
            thinking_policy="disabled", thinking_budget_tokens=8192,
        )
        spec = SimpleNamespace(
            name="commulingo_curator", finalization_tools=[], terminal_tools=[],
            render_prompt=lambda **_kw: "system",
        )
        _result, _tracker, found = await maintainer._call_curator_stage(
            task="discover", spec=spec, model="deepseek-v4-pro", tools=[],
            handlers={}, policy=policy, stage="test discovery",
            expect_edit=False, before_count=10,
        )
        assert len(calls) == 2
        assert found["id"] == "retry-person"
    finally:
        maintainer.chat_with_tools = original_chat
        maintainer.completed_run_count = original_count
        maintainer.db_query_one = original_query

asyncio.run(assert_dsml_retry())

print("commulingo maintainer smoke ok")
