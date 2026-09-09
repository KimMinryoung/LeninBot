"""Run-local draft repair using the same authorized narrow write tools."""
from copy import deepcopy
import hashlib
import json

from tool_gateway.validation import validate_tool_arguments, ToolArgumentValidationError
from tool_gateway.results import ToolRejection


def draft_id(draft):
    return hashlib.sha256(json.dumps({"tool": draft["tool"], "args": draft["args"]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def repair_schema(schema, baseline=None):
    schema = deepcopy(schema)
    if baseline:
        fields = schema.get("properties", {}).get("fields", {})
        if "expectedRevision" in fields.get("required", []):
            fields["required"].remove("expectedRevision")
        if "expected_revision" in schema.get("required", []):
            schema["required"].remove("expected_revision")
    properties = schema.setdefault("properties", {})
    properties.update({
        "draft_id": {"type": "string", "description": "Exact ID of the rejected draft returned by this tool."},
        "repairs": {"type": "array", "minItems": 1, "maxItems": 30,
            "description": "Repair only rejected arguments; paths are JSON pointers such as /fields/bio/ko. The complete draft is revalidated before submission.",
            "items": {"type": "object", "additionalProperties": False,
                "properties": {"path": {"type": "string", "pattern": "^/"},
                    "op": {"type": "string", "enum": ["set", "remove"]}, "value": {}},
                "required": ["path", "op"]}},
    })
    required = schema.pop("required", [])
    schema["if"] = {"anyOf": [{"required": ["draft_id"]}, {"required": ["repairs"]}]}
    schema["then"] = {"type": "object", "additionalProperties": False,
        "properties": {k: properties[k] for k in ("draft_id", "repairs")},
        "required": ["draft_id", "repairs"]}
    schema["else"] = {"required": required}
    return schema


def prepare_write(name, args, draft, schema=None, baseline=None):
    args = deepcopy(args)
    if "draft_id" in args:
        if set(args) != {"draft_id", "repairs"} or not draft or draft["tool"] != name or args["draft_id"] != draft_id(draft):
            raise ToolRejection("draft_id does not identify this tool's current rejected draft")
        repaired = deepcopy(draft["args"])
        for edit in args["repairs"]:
            parts = [part.replace("~1", "/").replace("~0", "~") for part in edit["path"].split("/")[1:]]
            if not parts or any(part in {"person_id", "event_id", "term_id", "target_id", "expectedRevision", "expected_revision", "action"} for part in parts):
                raise ToolRejection("draft repair cannot change target, action or revision; reconcile state in a new run")
            node = repaired
            try:
                for part in parts[:-1]:
                    node = node[int(part)] if isinstance(node, list) else node[part]
                key = int(parts[-1]) if isinstance(node, list) else parts[-1]
                if isinstance(key, int) and not 0 <= key < len(node):
                    raise KeyError(key)
                if edit["op"] == "remove":
                    del node[key]
                elif edit["op"] == "set" and "value" in edit:
                    node[key] = edit["value"]
                else:
                    raise KeyError("set requires value")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ToolRejection(f"invalid repair path or value: {edit['path']}") from exc
        # Whole-object replacement must not indirectly change immutable fields.
        for path in (("person_id",), ("event_id",), ("term_id",), ("target_id",), ("action",),
                     ("expected_revision",), ("fields", "expectedRevision")):
            def value(obj):
                for key in path:
                    obj = obj.get(key) if isinstance(obj, dict) else None
                return obj
            if value(repaired) != value(draft["args"]):
                raise ToolRejection("draft repair cannot change target, action or revision")
        args = repaired
    if baseline and name in {"commulingo_person_update", "commulingo_section_save"}:
        if args.get("person_id") != baseline["id"]:
            raise ToolRejection("write must target the commissioned person")
        holder = args.setdefault("fields", {}) if name == "commulingo_person_update" else args
        key = "expectedRevision" if name == "commulingo_person_update" else "expected_revision"
        if holder.get(key, baseline["revision"]) != baseline["revision"]:
            raise ToolRejection("revision differs from the commissioned snapshot; reconcile in a new run")
        holder[key] = baseline["revision"]
    if schema:
        try:
            args = validate_tool_arguments(name, args, schema=schema, risk_class="write")
        except ToolArgumentValidationError as exc:
            error = ToolRejection(str(exc))
            error.canonical_args = args
            raise error from exc
    return args
