"""CommuLingo reads for leninbot, through the frontend's admin MCP.

CommuLingo is a separate service (frontend dev_docs/commulingo-admin-mcp.md):
leninbot never queries its tables. The curator tools' readers and the write
validators in commulingo/people.py ask this module, and tests replace
`people._reads` with a fake. Short in-process caches keep one agent turn from
asking the same question many times; they never outlive a minute.
"""
from __future__ import annotations

import time

from commulingo.mcp_client import CommuLingoToolError, call_tool

_TTL = 60.0


class McpReads:
    def __init__(self):
        self._cache: dict[tuple, tuple[float, object]] = {}

    def _cached(self, key, fetch, ttl=_TTL):
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        value = fetch()
        self._cache[key] = (now, value)
        return value

    def clear(self):
        self._cache.clear()

    # ── existence and single records ─────────────────────────────────
    def existing(self, kind: str, ids) -> set[str]:
        ids = [str(i) for i in ids if i not in (None, "")]
        if not ids:
            return set()
        return set(call_tool("entries_exist", {"kind": kind, "ids": ids})["existing"])

    def exists(self, kind: str, entry_id) -> bool:
        return str(entry_id) in self.existing(kind, [entry_id])

    def person(self, person_id: str) -> dict | None:
        """Editorial state (person_get), or None."""
        try:
            return self._cached(("person", person_id), lambda: call_tool("person_get", {"id": person_id})["person"], ttl=5)
        except CommuLingoToolError as exc:
            if exc.status == 404:
                return None
            raise

    def event(self, event_id: str) -> dict | None:
        """Stored event (event_raw): card, full bodies, timeline, focus, sides, people."""
        try:
            return call_tool("event_raw", {"id": event_id})["event"]
        except CommuLingoToolError as exc:
            if exc.status == 404:
                return None
            raise

    def term(self, term_id: str) -> dict | None:
        """{id, parentId, hasChildren} for an existing term, or None."""
        found = call_tool("entries_exist", {"kind": "term", "ids": [term_id]})
        return next(iter(found.get("terms") or []), None)

    def term_editorial(self, term_id: str) -> dict | None:
        try:
            return call_tool("term_get", {"id": term_id})["term"]
        except CommuLingoToolError as exc:
            if exc.status == 404:
                return None
            raise

    def office(self, office_id: str) -> dict | None:
        try:
            return call_tool("office_get", {"id": office_id})["office"]
        except CommuLingoToolError as exc:
            if exc.status == 404:
                return None
            raise

    def label_owner(self, kind: str, label: str, entry_id: str = "") -> str | None:
        """Id of the person/term whose id, name or alias equals label (case-insensitive)."""
        label = (label or "").strip()
        if not label:
            return None
        return call_tool("entry_lookup", {"kind": kind, "id": entry_id, "label": label})["existingId"]

    def event_title_match(self, label: str) -> str | None:
        label = (label or "").strip()
        if not label:
            return None
        return call_tool("entry_lookup", {"kind": "term", "label": label})["eventTitleMatch"]

    # ── lists ────────────────────────────────────────────────────────
    def dataset(self, name: str) -> list[dict]:
        def fetch():
            rows, offset = [], 0
            while True:
                page = call_tool("dataset_rows", {"dataset": name, "limit": 5000, "offset": offset}, timeout=300)
                rows += page["rows"]
                offset += len(page["rows"])
                if page["done"]:
                    return rows
        return self._cached(("dataset", name), fetch)

    def groups(self) -> list[dict]:
        return self._cached(("groups",), lambda: call_tool("groups_list")["groups"])

    def people_search(self, q="", group_id="", limit=30, function_id="", affiliation_ids=()) -> list[dict]:
        return call_tool("people_search", {"q": q, "groupId": group_id, "functionId": function_id,
                                           "affiliationIds": list(affiliation_ids), "limit": limit})["people"]

    def _paged(self, tool: str, arguments: dict, key: str = "items") -> list[dict]:
        items, offset = [], 0
        while True:
            page = call_tool(tool, {**arguments, "limit": 100, "offset": offset})
            items += page[key]
            offset += len(page[key])
            if not page[key] or offset >= page["total"]:
                return items

    def events(self, q: str = "") -> list[dict]:
        return self._cached(("events", q), lambda: self._paged("event_search", {"q": q}))

    def terms(self, q: str = "") -> list[dict]:
        return self._cached(("terms", q), lambda: self._paged("term_search", {"q": q}))

    def offices(self) -> list[dict]:
        return self._cached(("offices",), lambda: call_tool("offices_list")["offices"])

    def suggestions(self, status: str = "", limit: int = 30, target_type: str = "", target_id: str = "") -> list[dict]:
        return call_tool("suggestions_list", {"status": status or "all", "limit": limit,
                                              "targetType": target_type, "targetId": target_id})["items"]


reads = McpReads()
