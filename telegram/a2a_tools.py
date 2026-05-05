"""Agent-to-agent client tool definitions and handlers."""

from __future__ import annotations

import json

A2A_SEND_TOOL = {
    "name": "a2a_send",
    "description": (
        "Send a SendMessage JSON-RPC request to an external A2A agent. "
        "Auto-discovers the agent's card at /.well-known/agent-card.json "
        "(v1.0) or /agent.json (legacy). Optional `skill_id` scopes the call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_url": {
                "type": "string",
                "description": "Base URL of the target agent (e.g. 'https://other-agent.com'). The /a2a endpoint is appended automatically.",
            },
            "message": {
                "type": "string",
                "description": "The message to send to the agent.",
            },
            "skill_id": {
                "type": "string",
                "description": "Optional skill ID to request from the target agent (passed as configuration.skillId).",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120).",
                "default": 120,
            },
            "discover": {
                "type": "boolean",
                "description": "If true, fetch and return the agent card instead of sending a message. When discover=true, message is not required.",
                "default": False,
            },
        },
        "required": ["agent_url"],
    },
}

A2A_TOOLS = [A2A_SEND_TOOL]


async def _exec_a2a_send(
    agent_url: str,
    message: str = "",
    skill_id: str = "",
    timeout_sec: int = 120,
    discover: bool = False,
) -> str:
    """Send an A2A message to an external agent or discover its capabilities."""
    import httpx
    import uuid

    base = agent_url.rstrip("/")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec, connect=10)) as client:
        if discover:
            try:
                r = await client.get(f"{base}/.well-known/agent-card.json")
                if r.status_code == 404:
                    r = await client.get(f"{base}/.well-known/agent.json")
                r.raise_for_status()
                card = r.json()
                name = card.get("name", "unknown")
                skills = card.get("skills", [])
                skill_list = ", ".join(s.get("id", "?") for s in skills) if skills else "none declared"
                desc = card.get("description", "")
                interfaces = card.get("supportedInterfaces", [])
                endpoint = interfaces[0].get("url") if interfaces else card.get("url", base + "/a2a")
                return (
                    f"Agent: {name}\n"
                    f"Description: {desc}\n"
                    f"Endpoint: {endpoint}\n"
                    f"Skills: {skill_list}\n"
                    f"Full card:\n{json.dumps(card, indent=2, ensure_ascii=False)}"
                )
            except httpx.HTTPStatusError as exc:
                return f"❌ Agent card fetch failed: HTTP {exc.response.status_code}"
            except Exception as exc:
                return f"❌ Agent card fetch failed: {exc}"

        if not message.strip():
            return "❌ message is required when discover=false"

        msg_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": msg_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": message}],
                },
            },
            "id": str(uuid.uuid4()),
        }

        if skill_id:
            payload["params"]["configuration"] = {"skillId": skill_id}

        try:
            r = await client.post(
                f"{base}/a2a",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            resp = r.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            return f"❌ A2A request failed: HTTP {exc.response.status_code}\n{body}"
        except Exception as exc:
            return f"❌ A2A request failed: {exc}"

        if "error" in resp:
            err = resp["error"]
            return f"❌ A2A error ({err.get('code')}): {err.get('message')}"

        result = resp.get("result", {})
        status = result.get("status", {})
        state = status.get("state", "unknown")

        history = result.get("history", [])
        agent_reply = ""
        for msg in reversed(history):
            role = msg.get("role", "")
            if role in ("ROLE_AGENT", "agent"):
                parts = msg.get("parts", [])
                agent_reply = "\n".join(p.get("text", "") for p in parts if "text" in p)
                break

        if not agent_reply:
            artifacts = result.get("artifacts", [])
            for art in artifacts:
                for p in art.get("parts", []):
                    if "text" in p:
                        agent_reply += p["text"] + "\n"

        task_id = result.get("id", "?")
        meta = result.get("metadata", {})
        skill_used = meta.get("skillId", "general")

        header = f"[A2A response | state={state} | skill={skill_used} | task={task_id}]"
        return f"{header}\n\n{agent_reply.strip()}" if agent_reply.strip() else f"{header}\n\n(empty response)"


A2A_TOOL_HANDLERS = {
    "a2a_send": _exec_a2a_send,
}
