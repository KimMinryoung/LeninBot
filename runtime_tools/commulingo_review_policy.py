"""Pure decision validation. Only source text fetched in this review counts."""
import re
from urllib.parse import urlsplit

DECISION_TOOL = {"name": "commulingo_review_decision", "description": "Submit one independently researched review decision; does not directly write dictionary content.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["approve", "reject", "escalate"]},
            "reason": {"type": "string", "minLength": 20},
            "resolved_risks": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "maxItems": 30, "items": {"type": "object", "additionalProperties": False,
                "properties": {key: {"type": "string"} for key in ("citation", "source", "quote", "finding")},
                "required": ["citation", "source", "quote", "finding"]}},
        }, "required": ["decision", "reason", "resolved_risks", "checks"]}}


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


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
            raise ValueError("approval must independently verify every cited reference")
        if not any(not (urlsplit(c["source"]).hostname or "").endswith("wikipedia.org") for c in checks):
            raise ValueError("approval requires an independent source outside Wikipedia")
        if not set(proposal.get("risks") or []).issubset(set(risks)):
            raise ValueError("approval must resolve every review risk")
    return value
