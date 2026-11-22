"""CommuLingo fate kinds and their default card labels.

The kinds live in the frontend's person-editorial-contract.json (``fateKinds``),
which the store validates against; this module only reads it. Fate is the
manner of death (2026-11-22 owner decision): falls from power, exile and
expulsion belong in the career timeline, not here. An empty label shows the
kind's default (자연사, 처형, 사인 미상 …).
"""
from __future__ import annotations

import json

from ops.paths import commulingo_data_file

FATE_KINDS: dict[str, dict] = json.loads(commulingo_data_file(
    "person-editorial-contract.json", "COMMULINGO_PERSON_CONTRACT").read_text())["fateKinds"]


def fate_label(kind: str | None, label: str | None, lang: str = "ko") -> str:
    """The card label: the stored one, or the kind's default when it is empty."""
    text = (label or "").strip()
    if text:
        return text
    entry = FATE_KINDS.get(kind or "")
    return entry.get("en" if lang == "en" else "ko", "") if entry else ""
