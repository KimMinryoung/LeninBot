"""Central validation and normalization for model-emitted tool arguments."""

from __future__ import annotations

import copy
import math
import re
from email.utils import parseaddr
from pathlib import PurePath
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator


class ToolArgumentValidationError(ValueError):
    """Raised when model-emitted tool arguments violate the executable schema."""


_URL_KEYS = frozenset({
    "agent_url",
    "public_url",
    "source_url",
    "start_url",
    "url",
})
_PATH_KEYS = frozenset({
    "file_path",
    "local_path",
    "output_path",
    "path",
})
_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def tool_schema_map(tool_definitions: list[dict] | None) -> dict[str, dict]:
    """Return ``tool name -> JSON Schema`` for Anthropic or OpenAI tool payloads."""
    schemas: dict[str, dict] = {}
    for tool in tool_definitions or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            schema = fn.get("parameters")
        else:
            name = str(tool.get("name") or "").strip()
            schema = tool.get("input_schema")
        if name and isinstance(schema, dict):
            schemas[name] = schema
    return schemas


def _schema_with_closed_top_level(schema: dict) -> dict:
    normalized = copy.deepcopy(schema)
    if normalized.get("type") == "object" or "properties" in normalized:
        normalized.setdefault("type", "object")
        normalized.setdefault("additionalProperties", False)
    return normalized


def _apply_top_level_defaults(args: dict, schema: dict) -> dict:
    normalized = dict(args)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return normalized
    for key, spec in properties.items():
        if (
            key not in normalized
            and isinstance(spec, dict)
            and "default" in spec
        ):
            normalized[key] = copy.deepcopy(spec["default"])
    return normalized


def _format_jsonschema_error(error) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    location = f" at '{path}'" if path else ""
    # jsonschema's maxLength message echoes the whole offending value, which
    # for a card bio is the entire paragraph — the model then re-counts by
    # hand. State the counts instead so one retry can trim precisely.
    if error.validator == "maxLength" and isinstance(error.instance, str):
        limit = int(error.validator_value)
        length = len(error.instance)
        return (
            f"value is {length} characters, {length - limit} over the "
            f"{limit}-character limit{location}"
        )
    message = error.message
    if len(message) > 300:
        message = message[:300] + "… [truncated]"
    return f"{message}{location}"


def _validate_url(key: str, value: Any) -> None:
    if value in (None, ""):
        return
    if not isinstance(value, str):
        raise ToolArgumentValidationError(f"{key} must be a string")
    if len(value) > 4096 or any(ord(ch) < 0x20 or ch.isspace() for ch in value):
        raise ToolArgumentValidationError(f"{key} contains whitespace/control data or is too long")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ToolArgumentValidationError(f"{key} is not a valid URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ToolArgumentValidationError(f"{key} must use http or https")
    if not parsed.hostname:
        raise ToolArgumentValidationError(f"{key} must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ToolArgumentValidationError(f"{key} must not contain URL credentials")


def _validate_path(key: str, value: Any) -> None:
    if value in (None, ""):
        return
    if not isinstance(value, str):
        raise ToolArgumentValidationError(f"{key} must be a string")
    if len(value) > 4096 or "\x00" in value:
        raise ToolArgumentValidationError(f"{key} contains NUL data or is too long")
    if any(ord(ch) < 0x20 for ch in value):
        raise ToolArgumentValidationError(f"{key} contains control characters")
    try:
        PurePath(value)
    except Exception as exc:
        raise ToolArgumentValidationError(f"{key} is not a valid path: {exc}") from exc


def _validate_json_numbers(value: Any, path: str = "arguments") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolArgumentValidationError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, f"{path}[{index}]")


def _validate_pay_arguments(args: dict) -> None:
    for key, value in args.items():
        lowered = str(key).lower()
        if lowered.startswith("amount_") or lowered in {"amount", "max_usdc"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolArgumentValidationError(f"{key} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ToolArgumentValidationError(f"{key} must be finite and greater than zero")
    to_address = args.get("to_address")
    if to_address is not None and (
        not isinstance(to_address, str) or not _ETH_ADDRESS_RE.fullmatch(to_address)
    ):
        raise ToolArgumentValidationError("to_address must be a 20-byte 0x Ethereum address")


def _validate_recipient_arguments(tool_name: str, args: dict) -> None:
    if tool_name == "send_email" and "to" in args:
        recipients = args["to"]
        if not isinstance(recipients, list) or not recipients:
            raise ToolArgumentValidationError("to must contain at least one email address")
        if len(recipients) > 50:
            raise ToolArgumentValidationError("to contains too many recipients")
        for recipient in recipients:
            if not isinstance(recipient, str) or len(recipient) > 320:
                raise ToolArgumentValidationError("email recipient is invalid or too long")
            _display, address = parseaddr(recipient)
            if address != recipient.strip() or "@" not in address or "\n" in recipient:
                raise ToolArgumentValidationError(f"invalid email recipient: {recipient!r}")

    for key in ("recipient", "recipient_id"):
        if key not in args:
            continue
        value = args[key]
        if not isinstance(value, (str, int)) or not str(value).strip():
            raise ToolArgumentValidationError(f"{key} must identify a recipient")
        if len(str(value)) > 320 or any(ch in str(value) for ch in "\r\n"):
            raise ToolArgumentValidationError(f"{key} is too long or contains a newline")


def _validate_confirmation_nonce(args: dict) -> None:
    if "confirmation_nonce" not in args:
        return
    nonce = args["confirmation_nonce"]
    if not isinstance(nonce, str) or len(nonce.strip()) < 16:
        raise ToolArgumentValidationError(
            "confirmation_nonce must be an opaque string of at least 16 characters"
        )


def validate_tool_arguments(
    tool_name: str,
    args: dict,
    *,
    schema: dict | None,
    risk_class: str,
) -> dict:
    """Validate and return normalized arguments.

    Provider schemas historically omitted ``additionalProperties``. The
    executable boundary closes only the top-level object by default so a model
    cannot smuggle misspelled arguments that are silently discarded.
    """
    if not isinstance(args, dict):
        raise ToolArgumentValidationError("tool arguments must be a JSON object")

    normalized = dict(args)
    if schema is not None:
        executable_schema = _schema_with_closed_top_level(schema)
        normalized = _apply_top_level_defaults(normalized, executable_schema)
        errors = sorted(
            Draft202012Validator(executable_schema).iter_errors(normalized),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            raise ToolArgumentValidationError(_format_jsonschema_error(errors[0]))

    _validate_json_numbers(normalized)

    for key, value in normalized.items():
        lowered = str(key).lower()
        if lowered in _URL_KEYS or lowered.endswith("_url"):
            _validate_url(str(key), value)
        if lowered in _PATH_KEYS or lowered.endswith("_path"):
            _validate_path(str(key), value)

    if risk_class == "pay":
        _validate_pay_arguments(normalized)
    if risk_class in {"send", "pay"}:
        _validate_recipient_arguments(tool_name, normalized)
    _validate_confirmation_nonce(normalized)
    return normalized


__all__ = [
    "ToolArgumentValidationError",
    "tool_schema_map",
    "validate_tool_arguments",
]
