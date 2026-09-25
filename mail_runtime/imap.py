"""IMAP access for the Cyber-Lenin mailbox: connect and parse raw messages."""
import os

from secrets_loader import get_secret


def connect():
    """Create and return an authenticated IMAP connection."""
    import imaplib
    host = os.environ.get("EMAIL_IMAP_HOST", "")
    port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
    username = os.environ.get("EMAIL_IMAP_USERNAME", "")
    password = get_secret("EMAIL_IMAP_PASSWORD", "") or ""
    if not all([host, username, password]):
        return None
    conn = imaplib.IMAP4_SSL(host, port)
    conn.login(username, password)
    return conn


def parse_message(
    raw_bytes,
    *,
    include_body: bool = True,
    body_max_chars: int = 4000,
    body_offset: int = 0,
):
    """Parse a raw email and return dict with subject, from, date, links, and extracted body text."""
    import email as _email
    from email.header import decode_header
    from email.utils import parsedate_to_datetime
    from html import unescape
    import re

    msg = _email.message_from_bytes(raw_bytes)

    subj_parts = decode_header(msg.get("Subject", ""))
    subject = ""
    for part, enc in subj_parts:
        if isinstance(part, bytes):
            subject += part.decode(enc or "utf-8", errors="replace")
        else:
            subject += part

    sender = msg.get("From", "")
    date = msg.get("Date", "")
    try:
        date_sort_timestamp = parsedate_to_datetime(date).timestamp()
    except (TypeError, ValueError, OverflowError):
        date_sort_timestamp = 0.0

    def _decode_payload(part):
        payload = part.get_payload(decode=True)
        if payload is None:
            raw = part.get_payload()
            if isinstance(raw, str):
                return raw
            if isinstance(raw, bytes):
                payload = raw
            else:
                return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")

    def _html_to_text(html: str) -> str:
        text = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</\s*(p|div|li|tr|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    text_parts = []
    html_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = (part.get("Content-Disposition") or "").lower()
            if "attachment" in content_disposition:
                continue
            ct = (part.get_content_type() or "").lower()
            if ct == "text/plain":
                decoded = _decode_payload(part).strip()
                if decoded:
                    text_parts.append(decoded)
            elif ct == "text/html":
                decoded = _decode_payload(part).strip()
                if decoded:
                    html_parts.append(decoded)
    else:
        ct = (msg.get_content_type() or "").lower()
        decoded = _decode_payload(msg).strip()
        if ct == "text/html":
            html_parts.append(decoded)
        elif decoded:
            text_parts.append(decoded)

    raw_body_for_links = "\n\n".join([*html_parts, *text_parts])
    extracted_body = "\n\n".join(text_parts).strip()
    if not extracted_body and html_parts:
        extracted_body = "\n\n".join(_html_to_text(part) for part in html_parts if part.strip()).strip()
    body_chars = len(extracted_body)
    try:
        body_start = max(0, int(body_offset or 0))
    except (TypeError, ValueError):
        body_start = 0
    if body_start >= body_chars:
        sliced_body = ""
        body_end = body_start
    elif body_max_chars > 0:
        body_end = min(body_chars, body_start + body_max_chars)
        sliced_body = extracted_body[body_start:body_end]
    else:
        body_end = body_chars
        sliced_body = extracted_body[body_start:]

    links = re.findall(r'https?://[^\s<>")\']+', raw_body_for_links)
    seen = set()
    unique_links = []
    for lnk in links:
        cleaned = lnk.rstrip('.,);>\"\'')
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_links.append(cleaned)

    return {
        "subject": subject,
        "from": sender,
        "date": date,
        "date_sort_timestamp": date_sort_timestamp,
        "links": unique_links[:50],
        "body": sliced_body if include_body else "",
        "body_chars": body_chars,
        "body_start": body_start,
        "body_end": body_end,
        "body_truncated": include_body and body_end < body_chars,
    }
