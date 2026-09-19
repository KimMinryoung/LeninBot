"""Pure decision validation. Only source text fetched in this review counts."""
import re
import json
import hashlib
import unicodedata
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
    return _fold_marks(re.sub(r"\s+", " ", str(text).translate(_FOLD)).strip().casefold())


def _fold_marks(text):
    """Drop combining marks (Russian Wikipedia stress accents: Собра́ние) that a model may or may not copy."""
    decomposed = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M"))


def locate(body, quote, min_prefix=None):
    """(start, end) of quote in body under normalize(), in body's own offsets; None if absent."""
    folded, index = [], []
    pending_space = False
    for i, ch in enumerate(str(body).translate(_FOLD)):
        if unicodedata.category(ch).startswith("M"):
            continue
        if ch.isspace():
            pending_space = bool(folded)
            continue
        if pending_space:
            folded.append(" ")
            index.append(i)
            pending_space = False
        for c in _fold_marks(ch.casefold()):
            folded.append(c)
            index.append(i)
    needle = normalize(quote)
    if not needle:
        return None
    haystack = "".join(folded)
    at = haystack.find(needle)
    # A model's copy usually goes wrong late (a dropped footnote marker, a
    # rewritten bracket), so the longest matching prefix of at least
    # min_prefix folded characters still pins the passage; the caller stores
    # the source's own text at that place, never the model's copy.
    while at < 0 and min_prefix and len(needle) > min_prefix:
        needle = needle[:max(min_prefix, len(needle) - 20)].rstrip()
        at = haystack.find(needle)
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

    A check whose source or passage cannot be found is dropped, not fatal: the
    decision keeps the checks that did locate and records the dropped ones under
    ``dropped_checks``. Only a decision with checks and none locatable is
    refused. (One drifted copy among 9..39 checks bounced whole decisions 159
    times on 2026-09-19; the reviewer could not tell which and resubmitted.)
    """
    from copy import deepcopy
    value = deepcopy(value)
    by_url = {}
    for sid, snap in snapshots.items():
        by_url.setdefault(snap["url"], []).append(sid)
    kept, dropped = [], []
    for index, check in enumerate(value.get("checks", []), 1):
        if not isinstance(check, dict):
            raise ValueError(f"check {index}: each check must be an object")
        if "citation_id" in check:
            identifier = check.pop("citation_id")
            match = re.fullmatch(r"S([1-9][0-9]*)", str(identifier))
            refs = proposal.get("source_refs") or []
            index_ref = int(match[1]) - 1 if match else -1
            if "citation" in check or not 0 <= index_ref < len(refs):
                raise ValueError(f"check {index}: citation_id must select an original source_refs entry; do not also supply citation")
            check["citation"] = refs[index_ref]
        if "source_id" in check:
            named = check.pop("source_id")
            quote = str(check.get("quote") or "")
            candidates = [named] if named in snapshots else by_url.get(named, [])
            if not candidates:
                dropped.append({"check": index, "source_id": named, "quote": quote[:80], "reason": "source not retrieved in this review"})
                continue
            located = None
            # Same prefix fallback as the research lane (evidence.locate_claim_quotes):
            # a copy that drifts after 40 folded characters still pins the passage.
            for sid in candidates + [s for s in snapshots if s not in candidates]:
                span = locate(snapshots[sid]["body"], quote, min_prefix=40)
                if span:
                    located = (snapshots[sid], span)
                    break
            if not located:
                dropped.append({"check": index, "source_id": named, "quote": quote[:80], "reason": "quote not found in retrieved text"})
                continue
            snapshot, (start, end) = located
            body = snapshot["body"]
            if end - start < len(quote):
                # A prefix match ends where the copy drifted, possibly mid-word; persist
                # through the end of that sentence (bounded) so the stored quote reads whole.
                stop = re.search(r"[.!?…]+(?=\s|$)|\n", body[end:end + min(len(quote) + 50, 300)])
                end = end + stop.end() if stop else end
            check["source"] = snapshot["url"]
            check["quote"] = body[start:end]
        kept.append(check)
    if dropped and not kept:
        available = "; ".join(f"{sid} ({snap['url']})" for sid, snap in snapshots.items()) or "none fetched yet"
        heads = "; ".join(f"check {d['check']}: {d['reason']} ({d['quote'][:60]!r})" for d in dropped)
        raise ValueError("no check could be verified — " + heads + ". Cite a review source_id shown with retrieved text "
                         f"and copy 20..1000 characters exactly as displayed, without ellipsis. Available: {available}")
    value["checks"] = kept
    if dropped:
        value["dropped_checks"] = dropped
    return value


def external_url(url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme in {"http", "https"} and host and not (host == "cyber-lenin.com" or host.endswith(".cyber-lenin.com"))


def validate_decision(value, proposal, fetched):
    required = {"decision", "reason", "resolved_risks", "checks"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - {"needs_research", "dropped_checks"}:
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
