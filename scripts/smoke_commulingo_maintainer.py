#!/usr/bin/env python3
"""Hermetic smoke checks for the dedicated CommuLingo maintainer."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.commulingo_curator import COMMULINGO_CURATOR
from runtime_tools.commulingo_people import _validate
import scripts.commulingo_people_maintainer as maintainer


assert COMMULINGO_CURATOR.provider == "deepseek"
assert COMMULINGO_CURATOR.model == "deepseek_pro"
assert set(COMMULINGO_CURATOR.tools) == {"web_search", "fetch_url", "commulingo_people", "commulingo_edit"}
assert COMMULINGO_CURATOR.terminal_tools == ["commulingo_edit"]
assert "given name + surname ONLY" in COMMULINGO_CURATOR.prompt_ir.identity
assert "already includes cyrillicPatronymic" in _validate(
    None, "person", "update", "example", {"cyrillic": "Михаил Петрович Фриновский", "cyrillicPatronymic": "Петрович"}
)

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "config.json"
    path.write_text('{"mode":"enrich","new_person_every":4,"recent_days":7}', encoding="utf-8")
    cfg = maintainer.load_config(path)
    assert cfg["mode"] == "enrich"
    assert cfg["new_person_every"] == 4
    assert cfg["recent_days"] == 7

candidate = {
    "id": "example",
    "name_ko": "예시",
    "name_en": "Example",
    "group_id": "thaw",
    "bio_chars": 40,
    "career_count": 1,
    "section_count": 0,
    "has_moment": 0,
    "has_role": 1,
}
task = maintainer.build_task("enrich", candidate)
assert "example" in task and "get_person" in task and "one commulingo_edit" in task
new_task = maintainer.build_task("new", None)
assert "search_people" in new_task and "action='create'" in new_task

print("commulingo maintainer smoke ok")
