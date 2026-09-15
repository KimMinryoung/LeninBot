"""Keyless local transport. Never retries through a direct provider endpoint."""
from __future__ import annotations

import httpx

from security_gateway.context import get_caller
from tool_gateway.results import ToolFailure

BASE_URL = "http://127.0.0.1:8111"
# Two 60s provider attempts plus validation; no automatic HTTP retry.
TIMEOUT = httpx.Timeout(150, connect=3)


class WebGatewayError(RuntimeError):
    pass


def identity() -> dict:
    from pathlib import Path
    import sys

    service = "cli:" + Path(sys.argv[0]).name
    try:
        for part in Path("/proc/self/cgroup").read_text().replace("\n", "/").split("/"):
            if part.endswith(".service"):
                service = part
                break
    except OSError:
        pass
    caller = get_caller()
    return {"service": service, "interface": caller.interface,
            "agent": caller.agent_name or "", "task_id": str(caller.task_id or ""),
            "scope_type": caller.scope_type or "", "scope_id": str(caller.scope_id or ""),
            "request_id": caller.request_id or ""}


def _payload(response: httpx.Response) -> dict:
    if response.status_code != 200:
        raise WebGatewayError(f"Web gateway unavailable (HTTP {response.status_code}); no direct-provider fallback.")
    result = response.json()
    if not isinstance(result, dict):
        raise WebGatewayError("Invalid web gateway response")
    return result


async def search(arguments: dict) -> str:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, trust_env=False) as client:
            response = await client.post(BASE_URL + "/search", json={"arguments": arguments, "caller": identity()})
        result = _payload(response)
        text = result["result"]
        if not isinstance(text, str) or not isinstance(result.get("error"), bool):
            raise WebGatewayError("Invalid search gateway response")
        return ToolFailure(text) if result["error"] else text
    except (httpx.HTTPError, ValueError, KeyError, WebGatewayError) as exc:
        return ToolFailure(f"Web gateway search unavailable ({type(exc).__name__}); "
                           "reuse saved evidence/free sources. No direct-provider fallback.")


def extract(url: str) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
            response = client.post(BASE_URL + "/extract", json={"url": url, "caller": identity()})
        payload = _payload(response)
        if payload.get("error"):
            raise WebGatewayError(str(payload.get("message", "Paid extraction unavailable")))
        return payload
    except (httpx.HTTPError, ValueError) as exc:
        raise WebGatewayError("Web gateway extraction unavailable; no direct-provider fallback.") from exc


def usage(days: int, by: str) -> dict:
    with httpx.Client(timeout=10, trust_env=False) as client:
        return _payload(client.get(BASE_URL + "/usage", params={"days": days, "by": by}))
