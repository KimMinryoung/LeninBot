"""Agent-facing mail tools: send_email, check_inbox, allowlist_sender.

runtime_tools/registry.py registers these; the implementations live here with
the rest of the mail runtime."""
import asyncio
import logging
import os

from mail_runtime import imap
from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)


# ── Send Email Tool ──────────────────────────────────────────────────
def _load_email_signature_config() -> dict:
    """Load email signature config from config/email_signature.json."""
    sig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "email_signature.json")
    try:
        import json as _json
        with open(sig_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception as e:
        logger.warning("Failed to load email signature config: %s", e)
        return {}


def _signature_mode_from_config(cfg: dict) -> str:
    mode = str(cfg.get("insertion_mode", "html_only") or "html_only").strip().lower()
    if mode not in {"html_only", "plain_text_only", "both", "none"}:
        mode = "html_only"
    return mode


def _load_email_signature() -> dict | None:
    cfg = _load_email_signature_config()
    if not cfg:
        return None

    mode = _signature_mode_from_config(cfg)
    if mode == "none":
        return None

    enabled = cfg.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None

    name = str(cfg.get("name", "") or "").strip()
    email_addr = str(cfg.get("email", "") or "").strip()
    website_url = str(cfg.get("website_url", "") or "").strip()
    website_display = str(cfg.get("website_display", website_url) or website_url).strip()
    logo_url = str(cfg.get("logo_url") or "").strip()
    logo_width = int(cfg.get("logo_width", 200) or 200)

    text_lines = [line for line in [name, email_addr, website_display] if line]
    text_sig = "\n".join(text_lines)

    # Build text info column
    info_lines = []
    if name:
        info_lines.append(f'<td style="font-size:15px;font-weight:700;color:#111;padding:0 0 4px 0;">{name}</td>')
    if email_addr:
        info_lines.append(f'<td style="font-size:13px;color:#555;padding:0 0 3px 0;"><a href="mailto:{email_addr}" style="color:#555;text-decoration:none;">{email_addr}</a></td>')
    if website_url:
        info_lines.append(f'<td style="font-size:13px;color:#555;padding:0 0 3px 0;"><a href="{website_url}" style="color:#555;text-decoration:none;">{website_display}</a></td>')
    info_html = "".join(f"<tr>{line}</tr>" for line in info_lines)

    # Horizontal layout: logo left + text right, inside a bordered box
    # Gmail/Outlook strip border-radius and padding on <table>, so use
    # a wrapping <td> with explicit padding and inline border on each side.
    logo_td = ""
    if logo_url:
        logo_td = (
            f'<td valign="middle" width="{logo_width}" style="padding:12px 14px 12px 12px;">'
            f'<img src="{logo_url}" alt="{name}" width="{logo_width}" height="{logo_width}" '
            f'style="display:block;border:0;outline:none;text-decoration:none;"></td>'
        )
    html_sig = (
        '<br><br>'
        '<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;font-family:Arial,Helvetica,sans-serif;">'
        '<tr><td style="border:1px solid #dddddd;padding:0;">'
        '<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">'
        f'<tr>{logo_td}'
        '<td valign="middle" style="padding:12px 12px 12px 0;">'
        f'<table cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">{info_html}</table>'
        '</td></tr></table>'
        '</td></tr></table>'
    )

    return {
        "text": text_sig,
        "html": html_sig,
        "mode": mode,
        "config": cfg,
        "logo_url": logo_url,
    }


SEND_EMAIL_TOOL = {
    "name": "send_email",
    "description": (
        "Send an email as Cyber-Lenin via Resend API. "
        "Supports plain text and HTML body. Use html_body for rich formatting with images. "
        "Image URLs from upload_to_r2 can be embedded in html_body with <img> tags. "
        "All sent emails are recorded in the email_messages DB table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recipient email addresses.",
            },
            "subject": {"type": "string", "description": "Email subject line."},
            "body": {"type": "string", "description": "Plain text body."},
            "html_body": {"type": "string", "description": "Optional HTML body. If provided, this is sent as the primary content."},
            "reply_to_message_id": {"type": "integer", "description": "Optional: inbound email_messages.id to reply to. Sets In-Reply-To header and thread."},
        },
        "required": ["to", "subject", "body"],
    },
}


async def exec_send_email(
    to: list[str], subject: str, body: str, html_body: str = "", reply_to_message_id: int | None = None,
) -> str:
    from services.email_bridge import (
        CONFIG, email_sending_is_configured, get_email_message,
    )
    from db import execute as db_execute, query as db_query
    import json as _json

    if not email_sending_is_configured():
        return "Email sending not configured. Check RESEND_API_KEY and EMAIL_SMTP_FROM_EMAIL in .env."

    original_body = body or ""
    original_html_body = html_body or ""

    # Load email signature and append through a single config-controlled path.
    # The caller must provide pure body content only; all signature insertion happens here.
    # To prevent duplicate signatures in clients like Gmail, plain text stays pure body
    # unless the operator explicitly selects a text-inserting mode in config.
    sig = _load_email_signature()
    if sig:
        sig_mode = sig.get("mode", "html_only")
        sig_text = sig.get("text", "")
        sig_html = sig.get("html", "")

        body = original_body
        if sig_mode in {"plain_text_only", "both"} and sig_text:
            body = original_body.rstrip() + "\n\n--\n" + sig_text

        if sig_mode in {"html_only", "both"} and sig_html:
            if original_html_body:
                html_body = original_html_body + sig_html
            else:
                escaped_body = original_body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                html_body = f"<div style='font-family:sans-serif;font-size:14px;'>{escaped_body}</div>{sig_html}"
                if sig_mode == "html_only":
                    # Keep plain text fallback free of signature duplication.
                    body = original_body
        else:
            html_body = original_html_body
    else:
        body = original_body
        html_body = original_html_body

    # If replying, look up the inbound message for threading
    in_reply_to = None
    thread_id = None
    if reply_to_message_id:
        inbound = await asyncio.to_thread(get_email_message, reply_to_message_id)
        if inbound:
            in_reply_to = inbound.get("external_message_id")
            thread_id = inbound.get("thread_id")

    # Record outbound in DB
    from_addr = f"{CONFIG.smtp_from_name} <{CONFIG.smtp_from_email}>"
    rows = await asyncio.to_thread(
        db_query,
        "INSERT INTO email_messages ("
        "  thread_id, provider, direction, status, mailbox, in_reply_to,"
        "  sender_email, sender_name, recipient_emails, subject,"
        "  text_body, html_body, metadata, created_at, updated_at"
        ") VALUES ("
        "  %s, %s, 'outbound', 'sending', 'outbox', %s,"
        "  %s, %s, %s::jsonb, %s,"
        "  %s, %s, '{}'::jsonb, NOW(), NOW()"
        ") RETURNING id",
        (
            thread_id, CONFIG.provider, in_reply_to,
            CONFIG.smtp_from_email, CONFIG.smtp_from_name, _json.dumps(to), subject,
            body, html_body or None,
        ),
    )
    message_id = rows[0]["id"] if rows else None

    # Send via Resend
    import resend
    resend.api_key = CONFIG.resend_api_key

    send_params = {
        "from": from_addr,
        "to": to,
        "subject": subject,
        "text": body,
    }
    if html_body:
        send_params["html"] = html_body
    if in_reply_to:
        send_params["headers"] = {"In-Reply-To": in_reply_to, "References": in_reply_to}

    try:
        result = resend.Emails.send(send_params)
        resend_id = result.get("id") if isinstance(result, dict) else str(result)
    except Exception as e:
        if message_id:
            await asyncio.to_thread(
                db_execute,
                "UPDATE email_messages SET status = 'failed', metadata = jsonb_build_object('error', %s), updated_at = NOW() WHERE id = %s",
                (str(e)[:500], message_id),
            )
        return ToolFailure(f"Email send failed: {e}")

    if message_id:
        await asyncio.to_thread(
            db_execute,
            "UPDATE email_messages SET status = 'sent', sent_at = NOW(), external_message_id = %s, updated_at = NOW() WHERE id = %s",
            (resend_id, message_id),
        )

    return f"Email sent to {', '.join(to)}\nSubject: {subject}\nResend ID: {resend_id}"



# ── check_inbox Tool ────────────────────────────────────────────────
CHECK_INBOX_TOOL = {
    "name": "check_inbox",
    "description": (
        "Read lenin@cyber-lenin.com mail (INBOX and Junk unless folder is set) as JSON: "
        "subject, sender, date, folder, read status, body, links, delivery history. "
        "Delegated tasks default to unbriefed mail regardless of IMAP read flags."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mail_id": {"type": "integer", "description": "Stored mail ID: read cached content without IMAP. Follow next for body pagination."},
            "unbriefed_only": {"type": "boolean", "description": "Only mail within the new-mail window (new_mail_window_days) not yet delivered to this audience. Default true in delegated tasks unless unread_only=true; false browses history."},
            "sender_filter": {
                "type": "string",
                "description": "Sender address or domain, e.g. 'substack.com'.",
            },
            "subject_filter": {
                "type": "string",
                "description": "Subject keyword, e.g. 'verify'.",
            },
            "unread_only": {
                "type": "boolean",
                "description": "Only unread emails.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max emails to return (default 5, max 20); coverage reports remaining candidates.",
                "default": 5,
            },
            "include_body": {
                "type": "boolean",
                "description": "Include body text.",
                "default": True,
            },
            "body_max_chars": {
                "type": "integer",
                "description": "Body characters per email (max 12000); continue via mail_id.",
                "default": 12000,
            },
            "body_offset": {
                "type": "integer",
                "description": "Body offset for a single email (mail_id or uid).",
                "default": 0,
            },
            # The default must stay "" and not "INBOX": _apply_top_level_defaults
            # injects a declared default into the arguments, so a declared "INBOX"
            # arrives even when the caller omitted the field and there is then no
            # way to mean "both folders".
            "folder": {
                "type": "string",
                "description": (
                    "Which mailbox to read: 'INBOX', 'Junk', or omit for both. "
                    "Also picks the folder for a uid single-email read, where "
                    "omitting it means INBOX."
                ),
                "default": "",
            },
            "uid": {
                "type": "string",
                "description": "IMAP UID from a previous result: read that one email.",
            },
        },
        "required": [],
    },
}


async def exec_check_inbox(**kwargs) -> str:
    from mail_runtime.inbox import check_inbox
    return await check_inbox(**kwargs)


# ── allowlist_sender Tool ───────────────────────────────────────────
ALLOWLIST_SENDER_TOOL = {
    "name": "allowlist_sender",
    "description": (
        "Move emails from a sender out of Junk into INBOX, preventing future spam filtering. "
        "Use after check_inbox shows folder=Junk emails from a legitimate sender."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sender_filter": {
                "type": "string",
                "description": "Sender address or domain to rescue from Junk (e.g. 'substack.com', 'noreply@platformer.news').",
            },
        },
        "required": ["sender_filter"],
    },
}


async def exec_allowlist_sender(sender_filter: str) -> str:
    """Move all Junk emails matching sender_filter to INBOX."""
    def _move():
        conn = imap.connect()
        if conn is None:
            return "Error: IMAP credentials not configured"

        status, _ = conn.select("Junk")
        if status != "OK":
            conn.logout()
            return "Junk folder not found or empty."

        _, data = conn.search(None, "ALL")
        all_ids = data[0].split()
        if not all_ids:
            conn.logout()
            return "Junk folder is empty."

        import email as _email
        moved = 0
        for mid in all_ids:
            _, msg_data = conn.fetch(mid, "(RFC822.HEADER)")
            header_raw = msg_data[0][1]
            msg = _email.message_from_bytes(header_raw)
            sender = msg.get("From", "")
            if sender_filter.lower() not in sender.lower():
                continue
            # COPY to INBOX then flag for deletion in Junk
            conn.copy(mid, "INBOX")
            conn.store(mid, "+FLAGS", "(\\Deleted)")
            moved += 1

        conn.expunge()
        conn.logout()
        return f"Moved {moved} email(s) from Junk to INBOX matching '{sender_filter}'."

    try:
        return await asyncio.to_thread(_move)
    except Exception as e:
        return ToolFailure(f"IMAP error: {e}")
