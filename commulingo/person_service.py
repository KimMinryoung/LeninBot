"""CommuLingo editorial store calls, through the frontend's admin MCP; never falls back to Python SQL."""
import json
from pathlib import Path

from commulingo.mcp_client import CommuLingoToolError, call_tool, read_entry


def call_person_service(request: dict):
    """Person/term editorial store: read, submit, review, enrichment, note."""
    request = dict(request)
    command = request.pop("command", None)
    target = request.pop("target", None) or "person"
    if command == "read":
        return read_entry(target, request.get("id"))
    changed_by = request.pop("changedBy", None)
    arguments = {"command": command, "target": target, "request": request}
    if changed_by:
        arguments["changedBy"] = changed_by
    return call_tool("editorial_store", arguments)


def apply_person_spec(path):
    """Apply an explicit sourced/versioned spec via the atomic Admin upsert (MCP people_upsert)."""
    spec = json.loads(Path(path).read_text())
    people = spec if isinstance(spec, list) else spec.get("people")
    arguments = {"people": people, "dryRun": False}
    if isinstance(spec, dict) and spec.get("changedBy"):
        arguments["changedBy"] = spec["changedBy"]
    try:
        result = call_tool("people_upsert", arguments)
    except CommuLingoToolError as exc:
        raise RuntimeError(f"rejected: {exc.payload.get('error')}") from exc
    lines = []
    for row in result["results"]:
        lines.append(f"{row['status']} {row['id']} (edit {row['suggestionId']})")
        lines += [f"  {section['status']} section {section['slug']}" for section in row["sections"]]
    lines.append(f"committed {len(result['results'])} person(s)")
    return "\n".join(lines)
