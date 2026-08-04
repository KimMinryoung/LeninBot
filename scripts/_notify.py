"""Shared Telegram notify helper for timer/CLI scripts.

Consolidates the byte-identical `_notify_telegram` copies that lived in
check_kg_integrity, check_replication_health, manage_secrets and
commulingo_find_name_variants. Import cost is near zero: secrets_loader
is only imported when a notification is actually sent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def notify_telegram(message: str) -> bool:
    """Send `message` to the configured Telegram chat. Returns True on success."""
    import os
    import urllib.parse
    import urllib.request

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from secrets_loader import get_secret
    except Exception as e:
        print(f"WARNING: cannot import secrets_loader ({e}); skipping notify", file=sys.stderr)
        return False
    token = get_secret("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping notify",
              file=sys.stderr)
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"WARNING: telegram notify failed: {e}", file=sys.stderr)
        return False
