"""One temporal and document-visibility contract for every KG read path.

valid_at <= as_of < invalid_at (null endpoints are unbounded).
With no as_of, historical assertions remain searchable. expired_at denotes
supersession, not the end of a historical tenure.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
from graph_memory.fact_version import canonical_date


def document_visibility():
    from kg_runtime.search import _run_rows
    rows = _run_rows("MATCH (n:Entity:Document) UNWIND coalesce(n.external_ids, []) AS ref "
                     "RETURN ref, n.uuid AS uuid, coalesce(n.source_active, true) AS active")
    research = [r for r in rows if r['ref'].startswith('research:')]
    public = set()
    if research:
        from db import query
        # Fail closed if current publication status cannot be established.
        public = {'research:' + r['slug'] for r in query(
            "SELECT slug FROM research_documents WHERE status = 'public'")}
    hidden = [r for r in rows if not r['active'] or
              (r['ref'].startswith('research:') and r['ref'] not in public)]
    return [r['ref'] for r in hidden], [r['uuid'] for r in hidden]


@dataclass
class ReadPolicy:
    as_of: str | None = None
    include_expired: bool = False
    _visibility: tuple | None = field(default=None, repr=False)

    def __post_init__(self):
        self.as_of = canonical_date(self.as_of)

    @property
    def hidden(self):
        if self._visibility is None:
            self._visibility = document_visibility()
        return self._visibility

    def params(self):
        refs, nodes = self.hidden
        return dict(as_of=self.as_of, include_expired=self.include_expired,
                    hidden_refs=refs, hidden_nodes=nodes)

    def allows(self, edge):
        if edge.get('doc_ref') in self.hidden[0]:
            return False
        if edge.get('expired_at') and not self.include_expired:
            return False
        if self.as_of:
            start, end = (canonical_date(edge.get(k)) for k in ('valid_at', 'invalid_at'))
            if start and start > self.as_of or end and end <= self.as_of:
                return False
        return True


def edge_where(var='r'):
    return (f"($include_expired OR {var}.expired_at IS NULL) "
            f"AND ({var}.doc_ref IS NULL OR NOT {var}.doc_ref IN $hidden_refs) "
            f"AND ($as_of IS NULL OR (({var}.valid_at IS NULL OR datetime({var}.valid_at) <= datetime($as_of)) "
            f"AND ({var}.invalid_at IS NULL OR datetime({var}.invalid_at) > datetime($as_of))))")


_current = ContextVar('kg_read_policy', default=None)


def current_policy():
    return _current.get() or ReadPolicy()
