"""Durable, task-scoped research and rejected-draft recovery for curator runs.

This is evidence reuse, not a search quota. Gateway authorization/validation
still precedes every wrapped handler, including reads served from memory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
import time
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

from provenance.runtime import _wrap_external
from tool_gateway.observations import argument_rejection_observer
from tool_gateway.results import ToolFailure, ToolRejection
from scripts.commulingo_write_session import draft_id, prepare_write, repair_schema

logger = logging.getLogger(__name__)
STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "commulingo_research.sqlite3"
READS = frozenset({"web_search", "fetch_url", "wiki_search", "wiki_get", "vector_search"})
WRITES = frozenset({
    "commulingo_person_create", "commulingo_person_update", "commulingo_section_save",
    "commulingo_event_link", "commulingo_term_create", "commulingo_term_update",
    "commulingo_event_update", "commulingo_event_section_save", "commulingo_office_row_save",
})
RETENTION = 14 * 86400
GUIDANCE = """
RESEARCH CONTINUITY: Reuse the dated evidence, URLs and draft below before any
new research. Repeating the same read retrieves saved evidence; use the same
arguments to recover its complete text. Search only for a specific unresolved
fact, not paraphrases of answered questions. Search snippets remain leads, not
verified source text. Check current dictionary state before writing; saved
research does not prove that a prior write succeeded.
An attempt_output is an unverified prior assistant draft, not source evidence.
WRITE REPAIR: For schema/format/length/punctuation errors, retain the complete
draft and citations and correct only the rejected fields. Do not restart the
investigation. Resolve invalid internal references with commulingo_people's
lookup/list actions. Never fabricate facts to satisfy a schema. If the schema
cannot represent the evidence, use commulingo_no_edit when available or report
the exact blocker. No successful write is replayed from this memory.
When a rejected draft_id is available, use the SAME tool with draft_id and repairs
to change only rejected arguments (JSON pointer path, op=set/remove, value for set).
For person fields use paths like /fields/bio/ko; event section text is /body/ko.
For evidence use source_id S1=citations[0], S2=citations[1], avoiding duplicate strings.
"""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ResearchMemory:
    def __init__(self, identity: str, *, path: Path | None = None):
        self.scope = hashlib.sha256(identity.encode()).hexdigest()
        self.path = path or STORE_PATH
        self.schemas = {}
        self.baseline = None
        self.outcomes = []
        self.metrics = {"read_calls": 0, "cache_hits": 0, "write_calls": 0, "write_rejections": 0}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS evidence (scope TEXT, tool TEXT, args TEXT, result TEXT, ts REAL, PRIMARY KEY(scope,tool,args))")
            db.execute("CREATE TABLE IF NOT EXISTS drafts (scope TEXT PRIMARY KEY, payload TEXT, ts REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS source_cache (tool TEXT, args TEXT, result TEXT, ts REAL, PRIMARY KEY(tool,args))")
            db.execute("DELETE FROM evidence WHERE ts < ?", (time.time() - RETENTION,))
            db.execute("DELETE FROM drafts WHERE ts < ?", (time.time() - RETENTION,))
            db.execute("DELETE FROM source_cache WHERE ts < ?", (time.time() - RETENTION,))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _draft(self):
        with self._db() as db:
            row = db.execute("SELECT payload FROM drafts WHERE scope=?", (self.scope,)).fetchone()
        return json.loads(row[0]) if row else None

    def rejected(self, name, args, error, *, repair=True):
        if name not in WRITES:
            return
        if "draft_id" in args:
            return  # a malformed repair must not overwrite the recoverable full draft
        payload = _json({"tool": name, "args": args, "error": str(error), "repair_only": repair})
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO drafts VALUES (?,?,?)", (self.scope, payload, time.time()))

    def _clear_draft(self):
        with self._db() as db:
            db.execute("DELETE FROM drafts WHERE scope=?", (self.scope,))

    @staticmethod
    def _ttl(name, args, result=""):
        if name == "web_search":
            if args.get("topic") in {"news", "finance"} or args.get("time_range"):
                return 60
            if result.startswith("No results for:"):
                return 300
        return RETENTION

    def _lookup(self, name, args):
        if args.get("use_cache") is False:
            return None
        with self._db() as db:
            row = db.execute("SELECT result,ts FROM evidence WHERE scope=? AND tool=? AND args=?",
                             (self.scope, name, _json(args))).fetchone()
            if not row and name in {"fetch_url", "wiki_get"}:
                row = db.execute("SELECT result,ts FROM source_cache WHERE tool=? AND args=?",
                                 (name, _json(args))).fetchone()
        if row and time.time() - row[1] < self._ttl(name, args, row[0]):
            # Make shared raw text visible in this task's dated research context.
            with self._db() as db:
                db.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?)",
                           (self.scope, name, _json(args), row[0], row[1]))
            return row[0]
        return None

    def wrap(self, handlers):
        wrapped = dict(handlers)
        for name, handler in handlers.items():
            is_write = name in WRITES
            if name not in READS and not is_write:
                continue

            async def call(_name=name, _handler=handler, _write=is_write, **args):
                if not _write:
                    self.metrics["read_calls"] += 1
                    cached = self._lookup(_name, args)
                    if cached is not None:
                        self.metrics["cache_hits"] += 1
                        logger.info("curator research reuse tool=%s scope=%s", _name, self.scope[:12])
                        return cached
                    draft = self._draft()
                    if draft and draft.get("repair_only"):
                        raise ToolRejection("Repair the saved write arguments using existing evidence first. "
                                            "This is a format/reference correction, not a new research task. "
                                            "Use commulingo_people for internal reference lookups.")
                try:
                    if _write:
                        self.metrics["write_calls"] += 1
                        args = prepare_write(_name, args, self._draft(), self.schemas.get(_name), self.baseline)
                    result = _handler(**args)
                    if inspect.isawaitable(result):
                        result = await result
                except ToolRejection as exc:
                    if _write:
                        args = getattr(exc, "canonical_args", args)
                        self.metrics["write_rejections"] += 1
                        message = str(exc)
                        repair = any(marker in message.lower() for marker in (
                            "unknown_field", "invalid_reference", "em dash", "character",
                            "required property", "unknown argument", "field shape", "schema",
                        ))
                        self.rejected(_name, args, message, repair=repair)
                        draft = self._draft()
                        if draft and draft["tool"] == _name:
                            raise ToolRejection(message + f"\nSaved draft_id={draft_id(draft)}. "
                                "Retry this SAME tool with only draft_id and repairs "
                                "[{op:'set',path:'/fields/bio/ko',value:'corrected text'}]; "
                                "choose the actual rejected path. Do not re-emit unchanged text.") from exc
                    raise
                if _write:
                    if not isinstance(result, ToolFailure) and not str(result).startswith("Error:"):
                        self.outcomes.append({"tool": _name, "result": str(result),
                            "target": args.get("person_id") or args.get("event_id") or args.get("term_id") or args.get("target_id")})
                        draft = self._draft()
                        if draft and draft.get("tool") == _name:
                            self._clear_draft()
                elif isinstance(result, str) and not isinstance(result, ToolFailure) and not result.startswith(("Error:", "Tool execution failed")):
                    # Keep complete recoverable results, never cache a truncated replacement.
                    if len(result) <= 120000:
                        with self._db() as db:
                            db.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?)",
                                       (self.scope, _name, _json(args), result, time.time()))
                            if _name in {"fetch_url", "wiki_get"}:
                                db.execute("INSERT OR REPLACE INTO source_cache VALUES (?,?,?,?)",
                                           (_name, _json(args), result, time.time()))
                return result

            wrapped[name] = call
        return wrapped

    def context(self):
        with self._db() as db:
            rows = db.execute("SELECT tool,args,result,ts FROM evidence WHERE scope=? ORDER BY ts DESC",
                              (self.scope,)).fetchall()
        pieces = []
        for name, args, result, ts in rows:
            if time.time() - ts >= self._ttl(name, json.loads(args), result):
                continue
            piece = _json({"tool": name, "args": json.loads(args), "retrieved_at_unix": ts,
                           "excerpt": result[:1800], "complete_chars": len(result)})
            if sum(map(len, pieces)) + len(piece) > 24000:
                break
            pieces.append(piece)
        draft = self._draft()
        if draft:
            # Full rejected payload is necessary to repair it without researching again.
            pieces.insert(0, _json({"draft_id": draft_id(draft), "last_rejected_write": draft}))
        return GUIDANCE + (_wrap_external("\n".join(pieces), "curator_research_memory") if pieces else "")

    async def chat(self, chat, messages, **kwargs):
        def schema_rejected(name, args, error):
            if name in WRITES:
                self.metrics["write_calls"] += 1
                self.metrics["write_rejections"] += 1
            self.rejected(name, args, error)
        token = argument_rejection_observer.set(schema_rejected)
        try:
            messages = [*messages, {"role": "user", "content": self.context()}]
            tools = deepcopy(kwargs.get("tools") or [])
            for tool in tools:
                name = tool.get("name")
                if name in WRITES and "input_schema" in tool:
                    self.schemas[name] = deepcopy(tool["input_schema"])
                    tool["input_schema"] = repair_schema(tool["input_schema"], self.baseline)
            if "tools" in kwargs:
                kwargs["tools"] = tools
            kwargs["tool_handlers"] = self.wrap(kwargs["tool_handlers"])
            result = await chat(messages, **kwargs)
            if isinstance(result, str) and result:
                with self._db() as db:
                    db.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?)",
                               (self.scope, "attempt_output", "{}", result[:120000], time.time()))
            return result
        finally:
            argument_rejection_observer.reset(token)
