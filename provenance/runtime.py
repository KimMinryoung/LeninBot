"""Runtime provenance and external-source wrapping helpers."""

import contextvars as _contextvars
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))

_kg_provenance_ctx = _contextvars.ContextVar("kg_provenance_buffer", default=None)


class ProvenanceBuffer:
    """Per-agent-run record of external sources touched and KG content read."""

    def __init__(self, agent: str = "agent", mission_id: int | None = None):
        self.agent = agent
        self.mission_id = mission_id
        # Each entry: {"tool", "source", "domain", "ts"}
        self.external_calls: list[dict] = []
        # Recent KG retrieval results, normalized text snippets
        self.kg_reads: list[str] = []

    def record_external(self, tool: str, source: str) -> None:
        from urllib.parse import urlparse as _up
        domain = ""
        try:
            raw = source
            for prefix in ("url:", "search:", "document:", "file:", "web_search:"):
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                    break
            if "://" in raw:
                domain = _up(raw).netloc.lower()
            elif raw.startswith("/") or raw.startswith("data/"):
                domain = "local-file"
        except Exception:
            pass
        self.external_calls.append({
            "tool": tool,
            "source": source[:300],
            "domain": domain,
            "ts": datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        # Cap to avoid unbounded growth on long agent runs
        if len(self.external_calls) > 64:
            self.external_calls = self.external_calls[-64:]

    def record_kg_read(self, text: str) -> None:
        if text:
            self.kg_reads.append(text[:5000])
            if len(self.kg_reads) > 16:
                self.kg_reads = self.kg_reads[-16:]

    def infer_trust_tier(self) -> str:
        """corroborated (≥2 independent domains) > single (1 domain) > unverified."""
        if not self.external_calls:
            return "unverified"
        domains = {c["domain"] for c in self.external_calls if c["domain"] and c["domain"] != "local-file"}
        if len(domains) >= 2:
            return "corroborated"
        if domains or any(c["domain"] == "local-file" for c in self.external_calls):
            return "single"
        return "unverified"

    def recent_sources(self, limit: int = 8) -> list[str]:
        seen, out = set(), []
        for c in reversed(self.external_calls):
            s = c["source"]
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
        return list(reversed(out))


def get_provenance_buffer() -> ProvenanceBuffer | None:
    return _kg_provenance_ctx.get()


def init_provenance_buffer(agent: str = "agent", mission_id: int | None = None) -> ProvenanceBuffer:
    buf = ProvenanceBuffer(agent=agent, mission_id=mission_id)
    _kg_provenance_ctx.set(buf)
    return buf


def _wrap_external(content: str, source: str) -> str:
    """Wrap tool output that came from an untrusted external source.

    Neutralizes any nested authority-impersonation tags (<user>, <system>,
    <assistant>, <external>, <operator>, <tool_use>, <tool_result>) so they
    cannot be used to spoof a higher-trust frame from inside the envelope.
    """
    if not content:
        return content
    import re as _re
    def _neutralize(m):
        return m.group(0).replace("<", "⟨").replace(">", "⟩")
    content = _re.sub(
        r"</?(?:user|system|assistant|external|operator|tool_use|tool_result)\b[^>]*>",
        _neutralize,
        content,
        flags=_re.IGNORECASE,
    )
    safe_source = source.replace('"', "'")[:200]
    return f'<external source="{safe_source}">\n{content}\n</external>'


