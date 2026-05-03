"""X/Twitter API tools.

Read-only helpers that use X_BEARER_TOKEN from systemd credentials or .env.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from urllib.parse import urlparse

from secrets_loader import get_secret

logger = logging.getLogger(__name__)


X_POST_TOOL = {
    "name": "fetch_x_post",
    "description": (
        "Fetch public X/Twitter content using the X API. For a status URL or numeric post ID, "
        "fetches that single post. For a profile URL, @username, or user=<username>, fetches "
        "that user's latest public posts in reverse chronological order. Use this instead of "
        "fetch_url or web_search for x.com/twitter.com content, because normal web fetches hit "
        "login walls and search engines often return stale indexed status URLs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url_or_id": {
                "type": "string",
                "description": "An x.com/twitter.com status/profile URL, a numeric tweet/post ID, or an @username.",
            },
            "user": {
                "type": "string",
                "description": "Optional @username/plain username. When provided, fetch latest posts from this user.",
            },
            "max_results": {
                "type": "integer",
                "description": "For user timeline mode: number of posts to fetch. X API accepts 5-100; default 10.",
                "default": 10,
            },
            "exclude_replies": {
                "type": "boolean",
                "description": "For user timeline mode: exclude replies. Default true.",
                "default": True,
            },
            "exclude_retweets": {
                "type": "boolean",
                "description": "For user timeline mode: exclude retweets/reposts. Default true.",
                "default": True,
            },
            "include_raw": {
                "type": "boolean",
                "description": "Include compact raw API JSON for debugging. Default false.",
                "default": False,
            },
        },
        "anyOf": [
            {"required": ["url_or_id"]},
            {"required": ["user"]},
        ],
    },
}


_TWEET_ID_RE = re.compile(r"(?<!\d)(\d{1,19})(?!\d)")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_RESERVED_PROFILE_PATHS = {
    "home",
    "i",
    "intent",
    "messages",
    "notifications",
    "search",
    "settings",
    "share",
}


def extract_x_username(username_or_url: str) -> str | None:
    value = (username_or_url or "").strip()
    if not value:
        return None
    if value.startswith("@"):
        value = value[1:]
    if _USERNAME_RE.match(value):
        return value

    parsed = urlparse(value)
    if parsed.netloc:
        host = parsed.netloc.lower().removeprefix("www.").removeprefix("mobile.")
        if host not in {"x.com", "twitter.com"} and not host.endswith(".twitter.com"):
            return None
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None
        username = parts[0]
        if username.lower() in _RESERVED_PROFILE_PATHS:
            return None
        if _USERNAME_RE.match(username):
            return username
    return None


def extract_x_post_id(url_or_id: str) -> str | None:
    value = (url_or_id or "").strip()
    if not value:
        return None
    if value.isdigit() and 1 <= len(value) <= 19:
        return value

    parsed = urlparse(value)
    if parsed.netloc:
        host = parsed.netloc.lower().removeprefix("www.").removeprefix("mobile.")
        if host not in {"x.com", "twitter.com"} and not host.endswith(".twitter.com"):
            return None
        parts = [p for p in parsed.path.split("/") if p]
        for marker in ("status", "statuses"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    match = _TWEET_ID_RE.search(parts[idx + 1])
                    if match:
                        return match.group(1)
        return None

    match = _TWEET_ID_RE.search(value)
    return match.group(1) if match else None


def _user_by_id(payload: dict) -> dict[str, dict]:
    users = payload.get("includes", {}).get("users", [])
    return {str(u.get("id")): u for u in users if u.get("id")}


def _media_by_key(payload: dict) -> dict[str, dict]:
    media = payload.get("includes", {}).get("media", [])
    return {str(m.get("media_key")): m for m in media if m.get("media_key")}


def _tweet_by_id(payload: dict) -> dict[str, dict]:
    tweets = payload.get("includes", {}).get("tweets", [])
    return {str(t.get("id")): t for t in tweets if t.get("id")}


def _format_user(user: dict | None) -> str:
    if not user:
        return "unknown"
    username = user.get("username")
    name = user.get("name")
    verified = " verified" if user.get("verified") else ""
    if username and name:
        return f"{name} (@{username}){verified}"
    if username:
        return f"@{username}{verified}"
    return name or "unknown"


def _format_metrics(metrics: dict | None) -> str:
    if not metrics:
        return "unavailable"
    ordered = ["like_count", "retweet_count", "reply_count", "quote_count", "bookmark_count", "impression_count"]
    parts = []
    for key in ordered:
        if key in metrics:
            parts.append(f"{key.removesuffix('_count')}={metrics[key]}")
    return ", ".join(parts) if parts else "unavailable"


def _format_x_payload(tweet_id: str, payload: dict, include_raw: bool) -> str:
    data = payload.get("data") or {}
    users = _user_by_id(payload)
    media_map = _media_by_key(payload)
    included_tweets = _tweet_by_id(payload)
    author = users.get(str(data.get("author_id")))

    lines = [
        f"X post: {tweet_id}",
        f"URL: https://x.com/i/web/status/{tweet_id}",
        f"Author: {_format_user(author)}",
    ]
    if data.get("created_at"):
        lines.append(f"Created: {data['created_at']}")
    if data.get("lang"):
        lines.append(f"Language: {data['lang']}")
    if data.get("conversation_id"):
        lines.append(f"Conversation: {data['conversation_id']}")
    if data.get("possibly_sensitive") is not None:
        lines.append(f"Possibly sensitive: {data['possibly_sensitive']}")
    lines.append(f"Metrics: {_format_metrics(data.get('public_metrics'))}")
    lines.append("")
    note_tweet = data.get("note_tweet") or {}
    note_text = note_tweet.get("text") if isinstance(note_tweet, dict) else None
    lines.append("Text:")
    lines.append(note_text or data.get("text") or "")

    refs = data.get("referenced_tweets") or []
    if refs:
        lines.append("")
        lines.append("Referenced posts:")
        for ref in refs:
            ref_id = str(ref.get("id") or "")
            ref_tweet = included_tweets.get(ref_id)
            ref_author = users.get(str((ref_tweet or {}).get("author_id")))
            ref_text = (ref_tweet or {}).get("text", "")
            lines.append(f"- {ref.get('type', 'reference')}: {ref_id} by {_format_user(ref_author)}")
            if ref_text:
                lines.append(f"  {ref_text}")

    media_keys = (data.get("attachments") or {}).get("media_keys") or []
    media_lines = []
    for key in media_keys:
        item = media_map.get(str(key)) or {}
        url = item.get("url") or item.get("preview_image_url")
        if url:
            media_lines.append(f"- {item.get('type', 'media')}: {url}")
    if media_lines:
        lines.append("")
        lines.append("Media:")
        lines.extend(media_lines)

    if include_raw:
        import json

        lines.append("")
        lines.append("Raw JSON:")
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    return "\n".join(lines).strip()


def _tweet_text(tweet: dict) -> str:
    note_tweet = tweet.get("note_tweet") or {}
    note_text = note_tweet.get("text") if isinstance(note_tweet, dict) else None
    return note_text or tweet.get("text") or ""


def _tweet_media_lines(tweet: dict, media_map: dict[str, dict]) -> list[str]:
    media_keys = (tweet.get("attachments") or {}).get("media_keys") or []
    lines = []
    for key in media_keys:
        item = media_map.get(str(key)) or {}
        url = item.get("url") or item.get("preview_image_url")
        if url:
            lines.append(f"  media {item.get('type', 'media')}: {url}")
    return lines


def _format_x_user_posts_payload(
    username: str,
    user: dict | None,
    payload: dict,
    include_raw: bool,
    *,
    source: str,
) -> str:
    tweets = payload.get("data") or []
    media_map = _media_by_key(payload)
    included_tweets = _tweet_by_id(payload)
    users = _user_by_id(payload)
    meta = payload.get("meta") or {}

    lines = [
        f"X user posts: @{username}",
        f"URL: https://x.com/{username}",
        f"Source: {source}",
    ]
    if user:
        lines.append(f"User: {_format_user(user)}")
        metrics = user.get("public_metrics")
        if metrics:
            lines.append(f"User metrics: {_format_metrics(metrics)}")
    if meta:
        meta_bits = []
        for key in ("result_count", "newest_id", "oldest_id"):
            if key in meta:
                meta_bits.append(f"{key}={meta[key]}")
        if meta_bits:
            lines.append(f"Meta: {', '.join(meta_bits)}")

    if not tweets:
        lines.append("")
        lines.append("No public posts returned.")
    for idx, tweet in enumerate(tweets, 1):
        tweet_id = str(tweet.get("id") or "")
        author = users.get(str(tweet.get("author_id"))) or user
        lines.append("")
        lines.append(f"[{idx}] https://x.com/i/web/status/{tweet_id}")
        lines.append(f"Author: {_format_user(author)}")
        if tweet.get("created_at"):
            lines.append(f"Created: {tweet['created_at']}")
        lines.append(f"Metrics: {_format_metrics(tweet.get('public_metrics'))}")
        lines.append(_tweet_text(tweet))

        refs = tweet.get("referenced_tweets") or []
        if refs:
            for ref in refs:
                ref_id = str(ref.get("id") or "")
                ref_tweet = included_tweets.get(ref_id)
                ref_text = _tweet_text(ref_tweet or {})
                lines.append(f"  referenced {ref.get('type', 'reference')}: {ref_id}")
                if ref_text:
                    lines.append(f"  {ref_text}")

        lines.extend(_tweet_media_lines(tweet, media_map))

    if include_raw:
        import json

        lines.append("")
        lines.append("Raw JSON:")
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    return "\n".join(lines).strip()


def _strip_html(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _format_oembed_payload(tweet_id: str, payload: dict) -> str:
    lines = [
        f"X post: {tweet_id}",
        f"URL: https://x.com/i/web/status/{tweet_id}",
        "Source: publish.twitter.com oEmbed fallback",
    ]
    author_name = payload.get("author_name")
    author_url = payload.get("author_url")
    if author_name or author_url:
        lines.append(f"Author: {author_name or 'unknown'} ({author_url or 'no author URL'})")
    text = _strip_html(payload.get("html", ""))
    if text:
        lines.append("")
        lines.append("Text:")
        lines.append(text)
    return "\n".join(lines).strip()


def _tweet_fields() -> str:
    return ",".join(
        [
            "attachments",
            "author_id",
            "conversation_id",
            "created_at",
            "display_text_range",
            "edit_controls",
            "edit_history_tweet_ids",
            "entities",
            "lang",
            "note_tweet",
            "possibly_sensitive",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "source",
        ]
    )


def _x_common_params() -> dict[str, str]:
    return {
        "tweet.fields": _tweet_fields(),
        "expansions": "author_id,referenced_tweets.id,referenced_tweets.id.author_id,attachments.media_keys",
        "user.fields": "id,name,username,verified,verified_type,description,created_at,public_metrics",
        "media.fields": "media_key,type,url,preview_image_url,alt_text,width,height,public_metrics,variants",
    }


def _redacted_error(resp, token: str) -> str:
    body = (resp.text or "")[:500].replace(token, "[redacted]")
    return f"HTTP {resp.status_code}: {body}"


async def _exec_fetch_x_post(
    url_or_id: str = "",
    user: str = "",
    max_results: int = 10,
    exclude_replies: bool = True,
    exclude_retweets: bool = True,
    include_raw: bool = False,
) -> str:
    username = extract_x_username(user) if user else None
    tweet_id = extract_x_post_id(url_or_id)
    if not tweet_id and not username:
        username = extract_x_username(url_or_id)
    if not tweet_id and not username:
        return (
            "Could not find an X/Twitter post ID or username. "
            "Pass a status URL/post ID in url_or_id, or pass user='username' for latest posts."
        )

    token = (get_secret("X_BEARER_TOKEN", "") or "").strip()
    if not token:
        return "X_BEARER_TOKEN is not configured for this service."

    def _fallback_oembed(reason: str) -> str:
        import requests

        errors = []
        for canonical_url in (
            f"https://x.com/i/status/{tweet_id}",
            f"https://twitter.com/i/status/{tweet_id}",
        ):
            try:
                resp = requests.get(
                    "https://publish.twitter.com/oembed",
                    params={"url": canonical_url, "omit_script": "1", "dnt": "1"},
                    timeout=15,
                )
                if resp.status_code == 404:
                    errors.append(f"{canonical_url}: 404")
                    continue
                resp.raise_for_status()
                fallback = _format_oembed_payload(tweet_id, resp.json())
                return f"{reason}\n\n{fallback}" if fallback else f"{reason}\nFallback returned no content."
            except Exception as exc:
                errors.append(f"{canonical_url}: {exc}")
        return f"{reason}\nFallback failed: {'; '.join(errors)}"

    def _request_single_post() -> str:
        import requests

        resp = requests.get(
            f"https://api.x.com/2/tweets/{tweet_id}",
            params=_x_common_params(),
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if resp.status_code == 404:
            return _fallback_oembed(f"X API returned 404 for {tweet_id}: post not found or unavailable.")
        if resp.status_code in {401, 403}:
            reason = (
                f"X API authorization failed ({resp.status_code}). "
                "Check X_BEARER_TOKEN validity, app permissions, and endpoint access."
            )
            return _fallback_oembed(reason)
        if resp.status_code == 429:
            reset = resp.headers.get("x-rate-limit-reset")
            suffix = f" Rate limit resets at epoch {reset}." if reset else ""
            return _fallback_oembed(f"X API rate limit exceeded.{suffix}")
        try:
            resp.raise_for_status()
        except Exception as exc:
            body = resp.text[:500].replace(token, "[redacted]")
            return _fallback_oembed(f"X API request failed: {exc}\n{body}")

        payload = resp.json()
        return _format_x_payload(tweet_id, payload, include_raw=bool(include_raw))

    def _request_user_posts() -> str:
        import json
        import requests

        headers = {"Authorization": f"Bearer {token}"}
        user_resp = requests.get(
            f"https://api.x.com/2/users/by/username/{username}",
            params={"user.fields": "id,name,username,verified,verified_type,description,created_at,public_metrics"},
            headers=headers,
            timeout=20,
        )
        if user_resp.status_code == 404:
            return f"X user not found or unavailable: @{username}"
        if user_resp.status_code in {401, 403}:
            return (
                f"X API authorization failed during user lookup ({user_resp.status_code}). "
                "Check X_BEARER_TOKEN validity, app permissions, and endpoint access."
            )
        if user_resp.status_code == 429:
            reset = user_resp.headers.get("x-rate-limit-reset")
            suffix = f" Rate limit resets at epoch {reset}." if reset else ""
            return f"X API rate limit exceeded during user lookup.{suffix}"
        try:
            user_resp.raise_for_status()
        except Exception:
            return f"X API user lookup failed: {_redacted_error(user_resp, token)}"

        user_payload = user_resp.json()
        user_data = user_payload.get("data") or {}
        user_id = user_data.get("id")
        if not user_id:
            return f"X API user lookup returned no user ID for @{username}."

        try:
            count = int(max_results or 10)
        except (TypeError, ValueError):
            count = 10
        count = min(100, max(5, count))

        params = _x_common_params()
        params["max_results"] = str(count)
        exclude = []
        if exclude_replies:
            exclude.append("replies")
        if exclude_retweets:
            exclude.append("retweets")
        if exclude:
            params["exclude"] = ",".join(exclude)

        timeline_resp = requests.get(
            f"https://api.x.com/2/users/{user_id}/tweets",
            params=params,
            headers=headers,
            timeout=20,
        )
        if timeline_resp.status_code in {401, 403}:
            fallback = _request_recent_search(username, headers, count)
            reason = (
                f"X API user timeline authorization failed ({timeline_resp.status_code}). "
                "Falling back to recent search from:username, which only covers recent searchable posts."
            )
            return f"{reason}\n\n{fallback}"
        if timeline_resp.status_code == 429:
            reset = timeline_resp.headers.get("x-rate-limit-reset")
            suffix = f" Rate limit resets at epoch {reset}." if reset else ""
            return f"X API rate limit exceeded during user timeline lookup.{suffix}"
        try:
            timeline_resp.raise_for_status()
        except Exception:
            fallback = _request_recent_search(username, headers, count)
            return f"X API user timeline failed: {_redacted_error(timeline_resp, token)}\n\n{fallback}"

        payload = timeline_resp.json()
        if include_raw:
            payload = dict(payload)
            payload["_user_lookup"] = user_payload
        return _format_x_user_posts_payload(
            username,
            user_data,
            payload,
            include_raw=bool(include_raw),
            source="X API user timeline (/2/users/:id/tweets)",
        )

    def _request_recent_search(username: str, headers: dict[str, str], count: int) -> str:
        import requests

        params = _x_common_params()
        params["query"] = f"from:{username}"
        params["max_results"] = str(min(100, max(10, count)))
        resp = requests.get(
            "https://api.x.com/2/tweets/search/recent",
            params=params,
            headers=headers,
            timeout=20,
        )
        if resp.status_code in {401, 403}:
            return (
                f"Recent search fallback authorization failed ({resp.status_code}). "
                "Do not fall back to web_search for latest X posts; search indexes are stale."
            )
        if resp.status_code == 429:
            reset = resp.headers.get("x-rate-limit-reset")
            suffix = f" Rate limit resets at epoch {reset}." if reset else ""
            return f"Recent search fallback rate limit exceeded.{suffix}"
        try:
            resp.raise_for_status()
        except Exception:
            return f"Recent search fallback failed: {_redacted_error(resp, token)}"
        return _format_x_user_posts_payload(
            username,
            None,
            resp.json(),
            include_raw=bool(include_raw),
            source="X API recent search fallback (/2/tweets/search/recent query=from:username)",
        )

    try:
        from shared import _wrap_external

        if username:
            content = await asyncio.to_thread(_request_user_posts)
            return _wrap_external(content, f"x_user:{username}")
        content = await asyncio.to_thread(_request_single_post)
        return _wrap_external(content, f"x_post:{tweet_id}")
    except Exception as exc:
        logger.exception("fetch_x_post error")
        return f"X post fetch failed: {exc}"


X_TOOLS = [X_POST_TOOL]
X_TOOL_HANDLERS = {"fetch_x_post": _exec_fetch_x_post}
