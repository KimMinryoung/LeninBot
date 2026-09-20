"""Pure decision validation. Only source text fetched in this review counts."""
import re
import json
import hashlib
from urllib.parse import urlsplit

from commulingo_pipeline.evidence import MAX_PASSAGES, Passages

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
                    "passages": {"type": "array", "minItems": 1, "maxItems": MAX_PASSAGES,
                                 "items": {"type": "string", "pattern": "^R[0-9a-f]{16}@[0-9]+$"},
                                 "description": "The labels shown in brackets at the start of the retrieved paragraphs that verify this finding (for example R5c5d56d06d4d1f95@6933): one to three paragraphs of one retrieved source, copied exactly."},
                    "finding": {"type": "string", "description": "Your Korean explanation of what those passages verify."},
                },
                "required": ["passages", "finding"],
                "oneOf": [{"required": ["citation"], "not": {"required": ["citation_id"]}}, {"required": ["citation_id"], "not": {"required": ["citation"]}}],
                }},
        }, "required": ["decision", "reason", "resolved_risks", "checks"]}}


def review_source(url, body, snapshots, passages, base=0):
    """Register one fetched slice for this review and render it with passage labels.

    Returns (source_id, labelled text). ``base`` is the slice's offset in its
    page so labels stay distinct across pages of one URL.
    """
    source_id = "R" + hashlib.sha256((url + "\n" + body).encode()).hexdigest()[:16]
    snapshots[source_id] = {"url": url, "body": body}
    return source_id, passages.show(source_id, body, base=base)


def resolve_review_checks(value, proposal, snapshots, passages):
    """Expand citation IDs and turn passage labels into the persisted decision.

    A check cites labels shown with retrieved text (``Passages.resolve``:
    one check per source and per contiguous range). A check none of whose
    labels was shown is dropped, not fatal: the decision keeps the checks
    that resolve and records the rest under ``dropped_checks``. Only a
    decision with checks and none resolvable is refused.
    """
    from copy import deepcopy
    value = deepcopy(value)
    kept, dropped = [], []
    for index, check in enumerate(value.get("checks", []), 1):
        if not isinstance(check, dict):
            raise ValueError(f"check {index}: each check must be an object")
        if "citation_id" in check:
            identifier = check.pop("citation_id")
            match = re.fullmatch(r"S([1-9][0-9]*)", str(identifier))
            refs = proposal.get("source_refs") or []
            ref_index = int(match[1]) - 1 if match else -1
            if "citation" in check or not 0 <= ref_index < len(refs):
                raise ValueError(f"check {index}: citation_id must select an original source_refs entry; do not also supply citation")
            check["citation"] = refs[ref_index]
        labels = [str(label) for label in (check.pop("passages", None) or [])]
        if not labels:
            raise ValueError(f"check {index}: passages must list the labels shown in brackets before the retrieved paragraphs")
        try:
            ranges, unknown = passages.resolve(labels, lambda sid: (snapshots.get(sid) or {}).get("body"))
        except ValueError as exc:
            raise ValueError(f"check {index}: {exc}") from exc
        if not ranges:
            dropped.append({"check": index, "labels": labels, "reason": "passage label not shown in this review: " + ", ".join(unknown)})
            continue
        for sid, start, end in ranges:
            kept.append({**check, "source": snapshots[sid]["url"], "quote": snapshots[sid]["body"][start:end]})
    if dropped and not kept:
        available = "; ".join(f"{sid} ({snap['url']})" for sid, snap in snapshots.items()) or "none fetched yet"
        heads = "; ".join(f"check {d['check']}: {d['reason']}" for d in dropped)
        raise ValueError("no check could be verified — " + heads + ". Cite passage labels exactly as shown in brackets "
                         f"before the retrieved paragraphs. Sources retrieved in this review: {available}")
    value["checks"] = kept
    if dropped:
        value["dropped_checks"] = dropped
    return value


def external_url(url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme in {"http", "https"} and host and not (host == "cyber-lenin.com" or host.endswith(".cyber-lenin.com"))


def validate_decision(value, proposal):
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
        if not external_url(check["source"]):
            raise ValueError(f"check {index}: select an external source fetched during this review")
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
