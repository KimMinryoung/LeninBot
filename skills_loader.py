"""skills_loader.py — Loads skill definitions from skills/ directory.

Parses SKILL.md frontmatter and injects relevant skills into the system prompt.
Skills are loaded once at startup and cached.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"

# Parsed skill: {name, description, instructions, allowed_tools}
_skills_cache: list[dict] = []
_skills_loaded = False


def _parse_skill_md(path: Path) -> Optional[dict]:
    """Parse a SKILL.md file. Returns dict or None on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("skills_loader: cannot read %s: %s", path, e)
        return None

    # Extract YAML frontmatter between --- delimiters
    fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not fm_match:
        logger.warning("skills_loader: no frontmatter in %s", path)
        return None

    frontmatter_raw = fm_match.group(1)
    instructions = fm_match.group(2).strip()

    # Simple key: value parsing (no full YAML parser needed)
    meta = {}
    for line in frontmatter_raw.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')

    name = meta.get("name", path.parent.name)
    description = meta.get("description", "")
    allowed_tools = meta.get("allowed-tools", "").split()

    return {
        "name": name,
        "description": description,
        "allowed_tools": allowed_tools,
        "instructions": instructions,
    }


def load_skills() -> list[dict]:
    """Load all skills from SKILLS_DIR. Cached after first call."""
    global _skills_loaded, _skills_cache

    if _skills_loaded:
        return _skills_cache

    if not SKILLS_DIR.exists():
        logger.info("skills_loader: skills/ directory not found, skipping")
        _skills_loaded = True
        return []

    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            logger.warning("skills_loader: %s has no SKILL.md", skill_dir.name)
            continue
        skill = _parse_skill_md(skill_md)
        if skill:
            skills.append(skill)
            logger.info("skills_loader: loaded skill '%s'", skill["name"])

    _skills_cache = skills
    _skills_loaded = True
    logger.info("skills_loader: %d skills loaded", len(skills))
    return skills


def build_skills_prompt() -> str:
    """Build a compact skills index for the system prompt.

    Only includes skill names + trigger descriptions (not full instructions).
    The model should read_file("skills/<name>/SKILL.md") when it needs to execute a skill.
    """
    skills = load_skills()
    if not skills:
        return ""

    lines = ["\n\n## Available Skills"]
    lines.append(
        "스킬이 트리거되면 read_file(\"skills/<name>/SKILL.md\")로 전체 지침을 읽은 뒤 따를 것.\n"
    )

    for skill in skills:
        lines.append(f"- **{skill['name']}**: {skill['description']}")

    return "\n".join(lines)


def get_skill_by_name(name: str) -> Optional[dict]:
    """Retrieve a specific skill by name."""
    for skill in load_skills():
        if skill["name"] == name:
            return skill
    return None


def reload_skills() -> int:
    """Force reload skills from disk. Returns count of loaded skills."""
    global _skills_loaded
    _skills_loaded = False
    return len(load_skills())
