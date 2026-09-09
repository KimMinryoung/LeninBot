"""Pure decision validation. Only source text fetched in this review counts."""
import re
import json
import hashlib
from urllib.parse import urlsplit

DECISION_TOOL = {"name": "commulingo_review_decision", "description": "Submit one independently researched review decision; does not directly write dictionary content.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["approve", "reject", "escalate"]},
            "reason": {"type": "string", "minLength": 20},
            "resolved_risks": {"type": "array", "description": "Exact strings copied from suggestion.risks; put explanations in reason/findings, not in risk identifiers.", "items": {"type": "string"}},
            "checks": {"type": "array", "maxItems": 30, "items": {"type": "object", "additionalProperties": False,
                "properties": {
                    "citation": {"type": "string", "description": "Copy one COMPLETE suggestion.source_refs entry verbatim, including its URL and any annotation. Never shorten or rename it."},
                    "source": {"type": "string", "description": "URL whose text you actually retrieved during this review."},
                    "quote": {"type": "string", "description": "Exact contiguous quotation from the retrieved body, at least 20 characters. No ellipsis or paraphrase."},
                    "finding": {"type": "string", "description": "Your Korean explanation of what this quote verifies."},
                    "citation_id": {"type": "string", "pattern": "^S[1-9][0-9]*$", "description": "S1 is suggestion.source_refs[0], S2 is source_refs[1]."},
                    "source_id": {"type": "string", "description": "Exact review source ID returned by fetch_url/wiki_get."},
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1},
                },
                "required": ["finding"],
                "allOf": [
                    {"oneOf": [{"required": ["citation"], "not": {"required": ["citation_id"]}}, {"required": ["citation_id"], "not": {"required": ["citation"]}}]},
                    {"oneOf": [{"required": ["source", "quote"], "not": {"required": ["source_id"]}}, {"required": ["source_id", "line_start", "line_end"], "not": {"anyOf": [{"required": ["source"]}, {"required": ["quote"]}]}}]},
                ]}},
        }, "required": ["decision", "reason", "resolved_risks", "checks"]}}


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def review_source(url, body, snapshots):
    """Immutable source pages scoped to this review, never the author's cache."""
    source_id = "R" + hashlib.sha256((url + "\n" + body).encode()).hexdigest()[:16]
    # Some extractors emit an entire page on one line. Display bounded chunks
    # while preserving every original character for exact quote reconstruction.
    lines = [line[i:i+240] for line in body.splitlines(keepends=True)
             for i in range(0, len(line), 240)]
    snapshots[source_id] = {"url": url, "lines": lines}
    numbered = "\n".join(f"{i}: {line.rstrip()}" for i, line in enumerate(snapshots[source_id]["lines"], 1))
    return source_id, numbered


def resolve_review_checks(value, proposal, snapshots):
    """Expand explicit IDs/ranges into the legacy, persistable decision contract."""
    from copy import deepcopy
    value = deepcopy(value)
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
            snapshot = snapshots.get(check.pop("source_id"))
            start, end = check.pop("line_start", None), check.pop("line_end", None)
            if (not snapshot or type(start) is not int or type(end) is not int
                    or not 1 <= start <= end <= len(snapshot["lines"])):
                raise ValueError("select an existing review source_id and valid inclusive line_start/line_end")
            if "source" in check or "quote" in check:
                raise ValueError("use a source range or literal source/quote, not both")
            check["source"] = snapshot["url"]
            check["quote"] = "".join(snapshot["lines"][start-1:end])
            if len(check["quote"]) > 6000:
                raise ValueError("select a narrower source range (at most 6000 characters)")
    return value


def external_url(url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme in {"http", "https"} and host and not (host == "cyber-lenin.com" or host.endswith(".cyber-lenin.com"))


def validate_decision(value, proposal, fetched):
    if not isinstance(value, dict) or set(value) != {"decision", "reason", "resolved_risks", "checks"}:
        raise ValueError("decision, reason, resolved_risks and checks required")
    decision = value["decision"]
    if decision not in {"approve", "reject", "escalate"} or not isinstance(value["reason"], str) or len(value["reason"].strip()) < 20:
        raise ValueError("valid decision and substantive reason required")
    risks = value["resolved_risks"]
    checks = value["checks"]
    if not isinstance(risks, list) or any(not isinstance(r, str) for r in risks):
        raise ValueError("resolved_risks must list review risks")
    if not isinstance(checks, list) or len(checks) > 30:
        raise ValueError("checks must be an array of at most 30 items")
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"citation", "source", "quote", "finding"} or any(not isinstance(v, str) or not v.strip() for v in check.values()):
            raise ValueError("each check needs citation, source, quote and finding")
        quote = normalize(check["quote"])
        if not external_url(check["source"]) or len(quote) < 20 or quote not in normalize(fetched.get(check["source"], "")):
            raise ValueError("quote must occur in source text fetched during this review")
    if decision in {"approve", "reject"} and not checks:
        raise ValueError("approve/reject requires retrieved evidence; otherwise escalate")
    if decision == "approve":
        if not set(proposal.get("source_refs") or []).issubset({c["citation"] for c in checks}):
            raise ValueError("approval must independently verify every cited reference; copy these missing source_refs verbatim into checks[].citation: " + json.dumps(sorted(set(proposal.get("source_refs") or []) - {c["citation"] for c in checks}), ensure_ascii=False))
        if not any(not (urlsplit(c["source"]).hostname or "").endswith("wikipedia.org") for c in checks):
            raise ValueError("approval requires an independent source outside Wikipedia")
        if not set(proposal.get("risks") or []).issubset(set(risks)):
            raise ValueError("approval must resolve every review risk; resolved_risks must include these exact identifiers (explanations belong in reason/findings): " + json.dumps(sorted(set(proposal.get("risks") or []) - set(risks)), ensure_ascii=False))
    return value
