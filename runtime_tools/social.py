"""External social-platform tools used by scout-style agents."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime

from secrets_loader import get_secret

MOLTBOOK_TOOL = {
    "name": "moltbook",
    "description": (
        "Run Moltbook operations via the Razvedchik agent script.\n"
        "Actions:\n"
        "- home: One-call Moltbook dashboard; do this first during check-ins\n"
        "- scan: Read-only feed scan — gather posts without interacting\n"
        "- feed: Read personalized feed or submolt/global posts\n"
        "- search: Semantic search across posts/comments\n"
        "- comments: Read comments on a post\n"
        "- patrol: Full patrol loop — scan + comment + post (default for general activity)\n"
        "- post: Write a new post to Moltbook\n"
        "- comment: Comment on a post or reply to a comment\n"
        "- verify: Submit a Moltbook verification answer after the scout solves the challenge\n"
        "- upvote/downvote: Vote on a post; upvote_comment votes on a comment\n"
        "- follow/unfollow: Follow or unfollow another molty\n"
        "- submolts: List available submolts\n"
        "- delete: Delete one of your posts\n"
        "- read_notifications: Mark notifications read\n"
        "- status: Check agent claim status\n"
        "- profile: View agent profile"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "home",
                    "scan",
                    "feed",
                    "search",
                    "comments",
                    "patrol",
                    "post",
                    "comment",
                    "verify",
                    "upvote",
                    "downvote",
                    "upvote_comment",
                    "follow",
                    "unfollow",
                    "submolts",
                    "delete",
                    "read_notifications",
                    "status",
                    "profile",
                ],
                "description": "Which Moltbook operation to run.",
            },
            "topic": {
                "type": "string",
                "description": "Post title (for 'post' action). If omitted, auto-generated.",
            },
            "content": {
                "type": "string",
                "description": "Post body (for 'post' action). If omitted, auto-generated.",
            },
            "submolt": {
                "type": "string",
                "description": "Target submolt name (e.g. 'general', 'tech'). Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of items to fetch (default depends on action).",
            },
            "sort": {
                "type": "string",
                "enum": ["hot", "new", "top", "rising", "best", "old"],
                "description": "Sort order for feed/posts/comments.",
            },
            "filter": {
                "type": "string",
                "enum": ["all", "following"],
                "description": "Personalized feed filter: all or following.",
            },
            "cursor": {
                "type": "string",
                "description": "Pagination cursor returned by Moltbook.",
            },
            "query": {
                "type": "string",
                "description": "Semantic search query for action='search'.",
            },
            "search_type": {
                "type": "string",
                "enum": ["all", "posts", "comments"],
                "description": "Moltbook semantic search type.",
            },
            "post_id": {
                "type": "string",
                "description": "Moltbook post ID for comments/votes/delete/read_notifications.",
            },
            "comment_id": {
                "type": "string",
                "description": "Moltbook comment ID for upvote_comment, or parent reply ID for comment.",
            },
            "agent_name": {
                "type": "string",
                "description": "Molty agent name for follow/unfollow.",
            },
            "max_comments": {
                "type": "integer",
                "description": "Max comments to post during patrol (default: 5).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Simulate without actual API writes (default: false).",
            },
            "verification_code": {
                "type": "string",
                "description": "Moltbook verification_code returned by a post/comment creation response.",
            },
            "answer": {
                "type": "string",
                "description": "The scout's solved verification answer, formatted as required by Moltbook (usually two decimals).",
            },
        },
        "required": ["action"],
    },
}

MERSOOM_TOOL = {
    "name": "mersoom",
    "description": (
        "Run Mersoom.com operations for the scout agent. Mersoom is a Korean "
        "anonymous AI-agent social network; write in 음슴체, no emoji, no markdown.\n"
        "Actions:\n"
        "- auth: Show configured razvedchikov credential status without leaking secrets\n"
        "- register: Register the configured auth_id if needed\n"
        "- feed: Read recent posts\n"
        "- post: Write a new Mersoom post\n"
        "- comments: Read comments on a post\n"
        "- comment: Comment on a post\n"
        "- arena_status: Read current arena phase/topic/stats\n"
        "- arena_candidates: Read proposed arena topics\n"
        "- arena_posts: Read arena battle posts for a date\n"
        "- arena_vote: Vote up/down on an arena candidate or battle post\n"
        "- arena_comment: Comment on an arena battle post\n"
        "- arena_propose: Propose an arena topic"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "auth",
                    "register",
                    "feed",
                    "post",
                    "comments",
                    "comment",
                    "arena_status",
                    "arena_candidates",
                    "arena_posts",
                    "arena_vote",
                    "arena_comment",
                    "arena_propose",
                ],
                "description": "Which Mersoom operation to run.",
            },
            "title": {"type": "string", "description": "Post title or arena topic title."},
            "content": {"type": "string", "description": "Post/comment/arena argument content."},
            "post_id": {"type": "string", "description": "Mersoom post ID."},
            "comment_id": {"type": "string", "description": "Parent comment ID for replies, if supported."},
            "target_id": {"type": "string", "description": "Arena candidate/post ID to vote on."},
            "vote": {"type": "string", "enum": ["up", "down"], "description": "Vote direction."},
            "side": {"type": "string", "enum": ["PRO", "CON"], "description": "Arena side for battle participation."},
            "pros": {"type": "string", "description": "Arena proposal pro argument."},
            "cons": {"type": "string", "description": "Arena proposal con argument."},
            "date": {"type": "string", "description": "Arena date in YYYY-MM-DD format."},
            "limit": {"type": "integer", "description": "Number of posts to fetch."},
            "cursor": {"type": "string", "description": "Pagination cursor returned by Mersoom."},
            "auth_id": {"type": "string", "description": "Override auth_id; defaults to MERSOOM_AUTH_ID or razvedchikov."},
            "password": {"type": "string", "description": "Override password; defaults to MERSOOM_PASSWORD or saved credentials."},
            "nickname": {"type": "string", "description": "Override nickname; defaults to MERSOOM_NICKNAME or 라즈베드치."},
            "dry_run": {"type": "boolean", "description": "Validate and show request shape without API writes."},
        },
        "required": ["action"],
    },
}


SOCIAL_TOOLS = [MOLTBOOK_TOOL, MERSOOM_TOOL]


async def _exec_moltbook(
    action: str = "patrol",
    topic: str = "",
    content: str = "",
    submolt: str = "",
    limit: int | None = None,
    sort: str = "",
    filter: str = "",
    cursor: str = "",
    query: str = "",
    search_type: str = "",
    post_id: str = "",
    comment_id: str = "",
    agent_name: str = "",
    max_comments: int | None = None,
    dry_run: bool = False,
    verification_code: str = "",
    answer: str = "",
    **_: dict,
) -> str:
    api_actions = {
        "home",
        "feed",
        "search",
        "comments",
        "comment",
        "verify",
        "upvote",
        "downvote",
        "upvote_comment",
        "follow",
        "unfollow",
        "submolts",
        "delete",
        "read_notifications",
    }
    if action in api_actions:
        try:
            import httpx
            from secrets_loader import require_secret

            api_key = require_secret("MOLTBOOK_API_KEY")
            base = "https://www.moltbook.com/api/v1"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            method = "GET"
            path = ""
            params: dict[str, object] = {}
            payload: dict[str, object] | None = None

            if action == "home":
                path = "/home"
            elif action == "feed":
                path = f"/submolts/{submolt}/feed" if submolt else "/feed"
                params = {"sort": sort or "hot", "limit": limit or 25}
                if filter:
                    params["filter"] = filter
                if cursor:
                    params["cursor"] = cursor
            elif action == "search":
                if not query:
                    return "[ERROR] query is required for moltbook search."
                path = "/search"
                params = {"q": query, "type": search_type or "all", "limit": limit or 20}
                if cursor:
                    params["cursor"] = cursor
            elif action == "comments":
                if not post_id:
                    return "[ERROR] post_id is required for moltbook comments."
                path = f"/posts/{post_id}/comments"
                params = {"sort": sort or "best", "limit": limit or 35}
                if cursor:
                    params["cursor"] = cursor
            elif action == "comment":
                if not post_id or not content:
                    return "[ERROR] post_id and content are required for moltbook comment."
                method = "POST"
                path = f"/posts/{post_id}/comments"
                payload = {"content": content}
                if comment_id:
                    payload["parent_id"] = comment_id
            elif action == "verify":
                if not verification_code or not answer:
                    return "[ERROR] verification_code and answer are required for moltbook verify."
                method = "POST"
                path = "/verify"
                payload = {"verification_code": verification_code, "answer": str(answer)}
            elif action in {"upvote", "downvote"}:
                if not post_id:
                    return f"[ERROR] post_id is required for moltbook {action}."
                method = "POST"
                path = f"/posts/{post_id}/{action}"
                payload = {}
            elif action == "upvote_comment":
                if not comment_id:
                    return "[ERROR] comment_id is required for moltbook upvote_comment."
                method = "POST"
                path = f"/comments/{comment_id}/upvote"
                payload = {}
            elif action in {"follow", "unfollow"}:
                if not agent_name:
                    return f"[ERROR] agent_name is required for moltbook {action}."
                method = "POST" if action == "follow" else "DELETE"
                path = f"/agents/{agent_name}/follow"
                payload = {}
            elif action == "submolts":
                path = "/submolts"
            elif action == "delete":
                if not post_id:
                    return "[ERROR] post_id is required for moltbook delete."
                method = "DELETE"
                path = f"/posts/{post_id}"
                payload = {}
            elif action == "read_notifications":
                method = "POST"
                path = f"/notifications/read-by-post/{post_id}" if post_id else "/notifications/read-all"
                payload = {}

            resp = httpx.request(
                method,
                f"{base}{path}",
                headers=headers,
                params=params or None,
                json=payload,
                timeout=30,
                follow_redirects=False,
            )
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text[:500]}
            if resp.status_code >= 400:
                return f"[ERROR] Moltbook {action} failed: HTTP {resp.status_code}\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"[ERROR] Failed to run Moltbook {action}: {exc}"

    cmd = [
        os.path.join(os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"), "venv/bin/python"),
        os.path.join(os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"), "agents/razvedchik/razvedchik.py"),
        f"--{action}",
    ]

    if topic:
        cmd.extend(["--topic", topic])
    if content:
        cmd.extend(["--content", content])
    if submolt:
        cmd.extend(["--submolt", submolt])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if max_comments:
        cmd.extend(["--max-comments", str(max_comments)])
    if dry_run:
        cmd.append("--dry-run")

    env = {**os.environ, "PYTHONPATH": os.environ.get("PROJECT_ROOT", "/home/grass/leninbot")}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"),
            env=env,
            timeout=180,
        )
        output = result.stdout[-3000:] if result.stdout else ""
        if result.returncode != 0:
            stderr = result.stderr[-1000:] if result.stderr else ""
            output += f"\n[EXIT CODE {result.returncode}]\nSTDERR: {stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "[ERROR] Moltbook script timed out after 180 seconds."
    except Exception as exc:
        return f"[ERROR] Failed to run Moltbook script: {exc}"


def _mersoom_credentials(
    auth_id: str = "",
    password: str = "",
    nickname: str = "",
) -> dict[str, str]:
    from pathlib import Path

    creds_path = Path.home() / ".config" / "mersoom" / "credentials.json"
    saved: dict[str, str] = {}
    if creds_path.exists():
        try:
            loaded = json.loads(creds_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                saved = {str(k): str(v) for k, v in loaded.items() if v is not None}
        except Exception:
            saved = {}

    return {
        "auth_id": auth_id or get_secret("MERSOOM_AUTH_ID", None) or saved.get("auth_id", "") or "razvedchikov",
        "password": password or get_secret("MERSOOM_PASSWORD", None) or saved.get("password", ""),
        "nickname": nickname or get_secret("MERSOOM_NICKNAME", None) or saved.get("nickname", "") or "라즈베드치",
        "credentials_path": str(creds_path),
    }


def _save_mersoom_credentials(creds: dict[str, str]) -> None:
    from pathlib import Path

    path = Path(creds["credentials_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {
        "auth_id": creds["auth_id"],
        "password": creds["password"],
        "nickname": creds["nickname"],
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _ensure_mersoom_password(creds: dict[str, str]) -> dict[str, str]:
    if creds.get("password"):
        return creds
    import secrets as _secrets

    # Mersoom currently accepts 10-20 characters; keep generated values simple.
    creds = dict(creds)
    creds["password"] = _secrets.token_hex(8)
    _save_mersoom_credentials(creds)
    return creds


def _mersoom_public_creds(creds: dict[str, str]) -> dict[str, object]:
    return {
        "auth_id": creds.get("auth_id", ""),
        "nickname": creds.get("nickname", ""),
        "password_configured": bool(creds.get("password")),
        "credentials_path": creds.get("credentials_path", ""),
    }


def _validate_mersoom_write_text(*values: str) -> str:
    import re

    text = "\n".join(v for v in values if v)
    if not text:
        return ""
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        return "Mersoom forbids emoji; remove emoji before posting."
    if re.search(r"(^|\n)\s{0,3}#{1,6}\s|```|\*\*|__|\[[^\]]+\]\([^)]+\)", text):
        return "Mersoom forbids markdown; send plain text only."
    return ""


class _MersoomClient:
    base_url = "https://www.mersoom.com/api"

    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(base_url=self.base_url, timeout=30, follow_redirects=False)

    def close(self) -> None:
        self._client.close()

    def _json_response(self, resp) -> dict | list:
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text[:1000]}
        if resp.status_code >= 400:
            raise RuntimeError(
                f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)}"
            )
        return data

    def _request(self, method: str, path: str, *, pow_required: bool = False, **kwargs) -> dict | list:
        headers = dict(kwargs.pop("headers", {}) or {})
        if pow_required:
            headers.update(self._pow_headers())
        resp = self._client.request(method, path, headers=headers or None, **kwargs)
        return self._json_response(resp)

    def _pow_headers(self) -> dict[str, str]:
        import hashlib
        import time

        data = self._request("POST", "/challenge")
        if not isinstance(data, dict) or "challenge" not in data or "token" not in data:
            raise RuntimeError(
                "Mersoom returned a non-PoW challenge that this tool cannot solve yet: "
                + json.dumps(data, ensure_ascii=False)[:500]
            )
        challenge = data["challenge"]
        seed = str(challenge["seed"])
        target_prefix = str(challenge.get("target_prefix", "0000"))
        limit_ms = int(challenge.get("limit_ms", 15000))
        deadline = time.monotonic() + max(limit_ms / 1000.0, 1.0)

        nonce = 0
        while time.monotonic() < deadline:
            digest = hashlib.sha256(f"{seed}{nonce}".encode("utf-8")).hexdigest()
            if digest.startswith(target_prefix):
                return {
                    "X-Mersoom-Token": str(data["token"]),
                    "X-Mersoom-Proof": str(nonce),
                }
            nonce += 1
        raise TimeoutError(f"Mersoom PoW nonce not found within {limit_ms}ms")

    def get(self, path: str, params: dict | None = None) -> dict | list:
        return self._request("GET", path, params=params or None)

    def post(self, path: str, payload: dict, *, pow_required: bool = False) -> dict | list:
        return self._request("POST", path, json=payload, pow_required=pow_required)


async def _exec_mersoom(
    action: str = "feed",
    title: str = "",
    content: str = "",
    post_id: str = "",
    comment_id: str = "",
    target_id: str = "",
    vote: str = "",
    side: str = "",
    pros: str = "",
    cons: str = "",
    date: str = "",
    limit: int | None = None,
    cursor: str = "",
    auth_id: str = "",
    password: str = "",
    nickname: str = "",
    dry_run: bool = False,
    **_: dict,
) -> str:
    creds = _mersoom_credentials(auth_id=auth_id, password=password, nickname=nickname)
    client = _MersoomClient()
    try:
        if action == "auth":
            data = _mersoom_public_creds(creds)
            data["base_url"] = client.base_url
            data["auth_model"] = "per-write auth_id/password with X-Mersoom PoW headers"
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "register":
            if dry_run:
                payload = {
                    "auth_id": creds["auth_id"],
                    "password": "***" if creds.get("password") else "<generated on non-dry-run>",
                    "nickname": creds["nickname"],
                }
                return json.dumps({"dry_run": True, "endpoint": "/auth/register", "payload": payload}, ensure_ascii=False, indent=2)
            creds = _ensure_mersoom_password(creds)
            payload = {
                "auth_id": creds["auth_id"],
                "password": creds["password"],
                "nickname": creds["nickname"],
            }
            data = client.post("/auth/register", payload, pow_required=True)
            _save_mersoom_credentials(creds)
            return json.dumps({"registered": True, "credentials": _mersoom_public_creds(creds), "response": data}, ensure_ascii=False, indent=2)

        if action == "feed":
            params: dict[str, object] = {"limit": limit or 20}
            if cursor:
                params["cursor"] = cursor
            data = client.get("/posts", params=params)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "comments":
            if not post_id:
                return "[ERROR] post_id is required for mersoom comments."
            data = client.get(f"/posts/{post_id}/comments")
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_status":
            data = client.get("/arena/status")
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_candidates":
            params = {"date": date} if date else None
            data = client.get("/arena/candidates", params=params)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_posts":
            params = {"date": date} if date else None
            data = client.get("/arena/posts", params=params)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "post":
            if not title or not content:
                return "[ERROR] title and content are required for mersoom post."
            err = _validate_mersoom_write_text(title, content)
            if err:
                return f"[ERROR] {err}"
            if dry_run:
                payload = {
                    "title": title,
                    "content": content,
                    "nickname": creds["nickname"],
                    "auth_id": creds["auth_id"],
                    "password": "***" if creds.get("password") else "<generated on non-dry-run>",
                }
                return json.dumps({"dry_run": True, "endpoint": "/posts", "payload": payload}, ensure_ascii=False, indent=2)
            creds = _ensure_mersoom_password(creds)
            payload = {
                "title": title,
                "content": content,
                "nickname": creds["nickname"],
                "auth_id": creds["auth_id"],
                "password": creds["password"],
            }
            data = client.post("/posts", payload, pow_required=True)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "comment":
            if not post_id or not content:
                return "[ERROR] post_id and content are required for mersoom comment."
            err = _validate_mersoom_write_text(content)
            if err:
                return f"[ERROR] {err}"
            if dry_run:
                payload = {
                    "content": content,
                    "nickname": creds["nickname"],
                    "auth_id": creds["auth_id"],
                    "password": "***" if creds.get("password") else "<generated on non-dry-run>",
                }
                if comment_id:
                    payload["parent_id"] = comment_id
                return json.dumps({"dry_run": True, "endpoint": f"/posts/{post_id}/comments", "payload": payload}, ensure_ascii=False, indent=2)
            creds = _ensure_mersoom_password(creds)
            payload = {
                "content": content,
                "nickname": creds["nickname"],
                "auth_id": creds["auth_id"],
                "password": creds["password"],
            }
            if comment_id:
                payload["parent_id"] = comment_id
            data = client.post(f"/posts/{post_id}/comments", payload, pow_required=True)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_vote":
            if not target_id or vote not in {"up", "down"}:
                return "[ERROR] target_id and vote='up'|'down' are required for mersoom arena_vote."
            payload = {"target_id": target_id, "type": vote}
            if dry_run:
                return json.dumps({"dry_run": True, "endpoint": "/arena/vote", "payload": payload}, ensure_ascii=False, indent=2)
            data = client.post("/arena/vote", payload)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_comment":
            if not post_id or not content:
                return "[ERROR] post_id and content are required for mersoom arena_comment."
            err = _validate_mersoom_write_text(content)
            if err:
                return f"[ERROR] {err}"
            payload = {"postId": post_id, "content": content, "nickname": creds["nickname"]}
            if date:
                payload["date"] = date
            if dry_run:
                return json.dumps({"dry_run": True, "endpoint": "/arena/comments", "payload": payload}, ensure_ascii=False, indent=2)
            data = client.post("/arena/comments", payload)
            return json.dumps(data, ensure_ascii=False, indent=2)

        if action == "arena_propose":
            if not title or not pros or not cons:
                return "[ERROR] title, pros, and cons are required for mersoom arena_propose."
            err = _validate_mersoom_write_text(title, pros, cons)
            if err:
                return f"[ERROR] {err}"
            if dry_run:
                payload = {
                    "title": title,
                    "pros": pros,
                    "cons": cons,
                    "nickname": creds["nickname"],
                    "auth_id": creds["auth_id"],
                    "password": "***" if creds.get("password") else "<generated on non-dry-run>",
                }
                return json.dumps({"dry_run": True, "endpoint": "/arena/propose", "payload": payload}, ensure_ascii=False, indent=2)
            creds = _ensure_mersoom_password(creds)
            payload = {
                "title": title,
                "pros": pros,
                "cons": cons,
                "nickname": creds["nickname"],
                "auth_id": creds["auth_id"],
                "password": creds["password"],
            }
            data = client.post("/arena/propose", payload, pow_required=True)
            return json.dumps(data, ensure_ascii=False, indent=2)

        return f"[ERROR] Unknown Mersoom action: {action}"
    except Exception as exc:
        return f"[ERROR] Failed to run Mersoom {action}: {exc}"
    finally:
        client.close()


SOCIAL_TOOL_HANDLERS = {
    "moltbook": _exec_moltbook,
    "mersoom": _exec_mersoom,
}
