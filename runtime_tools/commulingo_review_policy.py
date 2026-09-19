"""Pure decision validation. Only source text fetched in this review counts."""
import re
import json
import hashlib
from urllib.parse import urlsplit

DECISION_TOOL = {"name": "commulingo_review_decision", "description": "Submit one independently researched review decision; does not directly write dictionary content.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["approve", "revise", "reject", "escalate"]},
            "reason": {"type": "string", "minLength": 20},
            "needs_research": {"type": "boolean", "description": "For revise: false for corrections supported by retained research, true only if new source support is needed. Explain the missing fact in reason."},
            "resolved_risks": {"type": "array", "description": "Exact strings copied from suggestion.risks; put explanations in reason/findings, not in risk identifiers.", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "properties": {
                    "citation": {"type": "string", "description": "Copy one COMPLETE suggestion.source_refs entry verbatim, including its URL and any annotation. Never shorten or rename it."},
                    "citation_id": {"type": "string", "pattern": "^S[1-9][0-9]*$", "description": "S1 is suggestion.source_refs[0], S2 is source_refs[1]."},
                    "source_id": {"type": "string", "description": "Review source ID shown with text retrieved during this review (R...), or that source's URL."},
                    "quote": {"type": "string", "minLength": 20, "maxLength": 1000, "description": "Contiguous passage copied exactly from that retrieved text, 20..1000 characters. No ellipsis or paraphrase."},
                    "finding": {"type": "string", "description": "Your Korean explanation of what this quote verifies."},
                },
                "required": ["source_id", "quote", "finding"],
                "oneOf": [{"required": ["citation"], "not": {"required": ["citation_id"]}}, {"required": ["citation_id"], "not": {"required": ["citation"]}}],
                }},
        }, "required": ["decision", "reason", "resolved_risks", "checks"]}}


# Characters that differ between a page and what a model types back from it:
# typographic quotes and dashes, non-breaking and zero-width spaces, ellipsis.
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u3000": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "", "\u00ad": "",
    "\u2026": "...",
})


def normalize(text):
    """Whitespace-collapsed, quote/dash-folded, case-folded text for matching.

    Every substitution keeps two renderings of the same passage equal and
    never makes different passages equal. Exact matching lost 63 reviews in
    the week to 2026-09-19 to curly quotes, en dashes and NBSPs.
    """
    return re.sub(r"\s+", " ", str(text).translate(_FOLD)).strip().casefold()


def locate(body, quote):
    """(start, end) of quote in body under normalize(), in body's own offsets; None if absent."""
    folded, index = [], []
    pending_space = False
    for i, ch in enumerate(str(body).translate(_FOLD)):
        if ch.isspace():
            pending_space = bool(folded)
            continue
        if pending_space:
            folded.append(" ")
            index.append(i)
            pending_space = False
        for c in ch.casefold():
            folded.append(c)
            index.append(i)
    needle = normalize(quote)
    if not needle:
        return None
    at = "".join(folded).find(needle)
    if at < 0:
        return None
    return index[at], index[at + len(needle) - 1] + 1


def review_source(url, body, snapshots):
    """Immutable source pages scoped to this review, never the author's cache."""
    source_id = "R" + hashlib.sha256((url + "\n" + body).encode()).hexdigest()[:16]
    snapshots[source_id] = {"url": url, "body": body}
    return source_id, body


def resolve_review_checks(value, proposal, snapshots):
    """Expand citation IDs and locate each quote; returns the persistable decision.

    A check names a review source (its displayed R-id or URL) and copies a
    passage; the passage is located with typography folded, in that source
    first and then in any other source of this review, and persisted as the
    exact text at that location. Nothing is counted or numbered.
    """
    from copy import deepcopy
    value = deepcopy(value)
    by_url = {}
    for sid, snap in snapshots.items():
        by_url.setdefault(snap["url"], []).append(sid)
    for check in value.get("checks", []):
        if not isinstance(check, dict):
            raise ValueError("each check must be an object")
        if "citation_id" in check:
            identifier = check.pop("citation_id")
            match = re.fullmatch(r"S([1-9][0-9]*)", str(identifier))
            refs = proposal.get("source_refs") or []
            index = int(match[1]) - 1 if match else -1
            if "citation" in check or not 0 <= index < len(refs):
                raise ValueError("citation_id must select an original source_refs entry; do not also supply citation")
            check["citation"] = refs[index]
        if "source_id" in check:
            named = check.pop("source_id")
            candidates = [named] if named in snapshots else by_url.get(named, [])
            if not candidates:
                available = "; ".join(f"{sid} ({snap['url']})" for sid, snap in snapshots.items()) or "none fetched yet"
                raise ValueError(f"select a review source_id shown with text retrieved during this review. Available: {available}")
            quote = str(check.get("quote") or "")
            located = None
            for sid in candidates + [s for s in snapshots if s not in candidates]:
                span = locate(snapshots[sid]["body"], quote)
                if span:
                    located = (snapshots[sid], span)
                    break
            if not located:
                raise ValueError("quote not found in the retrieved text of that source (or any other fetched in this "
                                 "review): copy 20..1000 characters exactly as displayed, without ellipsis")
            snapshot, (start, end) = located
            check["source"] = snapshot["url"]
            check["quote"] = snapshot["body"][start:end]
    return value


def external_url(url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme in {"http", "https"} and host and not (host == "cyber-lenin.com" or host.endswith(".cyber-lenin.com"))


def validate_decision(value, proposal, fetched):
    required = {"decision", "reason", "resolved_risks", "checks"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - {"needs_research"}:
        raise ValueError("decision, reason, resolved_risks and checks required")
    if "needs_research" in value and type(value["needs_research"]) is not bool:
        raise ValueError("needs_research must be a boolean")
    decision = value["decision"]
    if decision not in {"approve", "revise", "reject", "escalate"} or not isinstance(value["reason"], str) or len(value["reason"].strip()) < 20:
        raise ValueError("valid decision and substantive reason required")
    risks = value["resolved_risks"]
    checks = value["checks"]
    if not isinstance(risks, list) or any(not isinstance(r, str) for r in risks):
        raise ValueError("resolved_risks must list review risks")
    if not isinstance(checks, list):
        raise ValueError("checks must be an array")
    for index, check in enumerate(checks, 1):
        if not isinstance(check, dict) or set(check) != {"citation", "source", "quote", "finding"} or any(not isinstance(v, str) or not v.strip() for v in check.values()):
            raise ValueError("each check needs citation, source, quote and finding")
        quote = normalize(check["quote"])
        if not external_url(check["source"]):
            raise ValueError(f"check {index}: select an external source fetched during this review")
        if len(quote) < 20:
            raise ValueError(f"check {index}: quote has {len(quote)} normalized characters; at least 20 required. Copy a longer passage")
        if quote not in normalize(fetched.get(check["source"], "")):
            raise ValueError(f"check {index}: quote must occur in source text fetched during this review; copy it exactly from the displayed text")
    if decision in {"approve", "revise", "reject"} and not checks:
        raise ValueError("approve/revise/reject requires retrieved evidence; otherwise escalate")
    if decision == "approve":
        # Approval needs verified quotes and every named risk resolved. It does
        # not need a check per cited reference or a non-Wikipedia source: those
        # requirements forced extra fetches and lost otherwise sound reviews at
        # the round limit (operator relaxed them 2026-09-17).
        if not set(proposal.get("risks") or []).issubset(set(risks)):
            raise ValueError("approval must resolve every review risk; resolved_risks must include these exact identifiers (explanations belong in reason/findings): " + json.dumps(sorted(set(proposal.get("risks") or []) - set(risks)), ensure_ascii=False))
    return value
