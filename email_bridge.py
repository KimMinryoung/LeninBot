"""email_bridge.py — IMAP/SMTP email bridge MVP for Cyber-Lenin.

Flow:
1) poll IMAP inbox for unseen messages
2) normalize/store into PostgreSQL
3) generate reply draft context and notify approver via Telegram
4) approval happens through API or Telegram command
5) approved reply is sent via SMTP or saved to local draft file
"""

from __future__ import annotations

import email
import imaplib
import json
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import Message
from email.message import EmailMessage
from email.utils import getaddresses, make_msgid, parseaddr, parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any, Callable

from db import execute as db_execute, query as db_query, query_one as db_query_one
from secrets_loader import get_secret

logger = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class EmailBridgeConfig:
    enabled: bool
    provider: str
    imap_host: str
    imap_port: int
    imap_username: str
    imap_password: str
    imap_mailbox: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_from_email: str
    smtp_from_name: str
    polling_enabled: bool
    poll_interval_seconds: int
    runtime_secrets_path: str
    operations_chat_id: int
    approval_base_url: str
    draft_mode: str
    log_dir: Path
    default_approver_user_id: int
    approval_secret: str
    resend_api_key: str


def _apply_runtime_secrets_from_file(path_str: str | None) -> None:
    if not path_str:
        return
    path = Path(path_str).expanduser()
    if not path.is_file():
        logger.warning("email runtime secrets file not found: %s", path)
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
    except Exception as exc:
        logger.warning("failed to load email runtime secrets file %s: %s", path, exc)


def load_email_bridge_config() -> EmailBridgeConfig:
    root = Path(__file__).resolve().parent
    runtime_secrets_path = os.getenv("EMAIL_RUNTIME_SECRETS_FILE", "").strip()
    _apply_runtime_secrets_from_file(runtime_secrets_path)
    return EmailBridgeConfig(
        enabled=_truthy(os.getenv("EMAIL_BRIDGE_ENABLED"), False),
        provider=os.getenv("EMAIL_PROVIDER", "imap_smtp").strip() or "imap_smtp",
        imap_host=os.getenv("EMAIL_IMAP_HOST", "").strip(),
        imap_port=int(os.getenv("EMAIL_IMAP_PORT", "993") or "993"),
        imap_username=os.getenv("EMAIL_IMAP_USERNAME", "").strip(),
        imap_password=(get_secret("EMAIL_IMAP_PASSWORD", "") or "").strip(),
        imap_mailbox=os.getenv("EMAIL_IMAP_MAILBOX", "INBOX").strip() or "INBOX",
        smtp_host=os.getenv("EMAIL_SMTP_HOST", "").strip(),
        smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "465") or "465"),
        smtp_username=os.getenv("EMAIL_SMTP_USERNAME", "").strip(),
        smtp_password=(get_secret("EMAIL_SMTP_PASSWORD", "") or "").strip(),
        smtp_from_email=os.getenv("EMAIL_SMTP_FROM_EMAIL", "").strip(),
        smtp_from_name=os.getenv("EMAIL_SMTP_FROM_NAME", "Cyber-Lenin").strip() or "Cyber-Lenin",
        polling_enabled=_truthy(os.getenv("EMAIL_POLLING_ENABLED"), True),
        poll_interval_seconds=max(60, int(os.getenv("EMAIL_POLL_INTERVAL_SECONDS", "120") or "120")),
        runtime_secrets_path=runtime_secrets_path,
        operations_chat_id=int(os.getenv("EMAIL_OPERATIONS_CHAT_ID", os.getenv("EMAIL_DEFAULT_APPROVER_USER_ID", "0")) or "0"),
        approval_base_url=os.getenv("EMAIL_APPROVAL_BASE_URL", "").rstrip("/"),
        draft_mode=os.getenv("EMAIL_DRAFT_MODE", "jsonl").strip() or "jsonl",
        log_dir=Path(os.getenv("EMAIL_LOG_DIR", str(root / "logs" / "email_bridge"))),
        default_approver_user_id=int(os.getenv("EMAIL_DEFAULT_APPROVER_USER_ID", "0") or "0"),
        approval_secret=os.getenv("EMAIL_APPROVAL_SECRET", "").strip(),
        resend_api_key=(get_secret("RESEND_API_KEY", "") or "").strip(),
    )


CONFIG = load_email_bridge_config()


INBOUND_INTERNAL_STATUSES = {
    "approved_for_internal_input",
    "delivered_to_internal_input",
    "delivery_failed",
}


EMAIL_CLASSIFICATION_RULES: list[dict[str, Any]] = [
    {
        "label": "automated_sender",
        "field": "sender",
        "tokens": ["no-reply", "noreply", "mailer-daemon", "postmaster"],
        "score": 3,
    },
    {
        "label": "finance",
        "field": "subject",
        "tokens": ["invoice", "billing", "receipt", "payment"],
        "score": 2,
    },
    {
        "label": "urgent",
        "field": "subject",
        "tokens": ["urgent", "asap", "immediately"],
        "score": 2,
    },
    {
        "label": "bulk_like",
        "field": "body",
        "tokens": ["unsubscribe", "view in browser", "mailing list"],
        "score": 2,
    },
    {
        "label": "human_request",
        "field": "subject",
        "tokens": ["question", "inquiry", "request", "proposal", "interview", "press"],
        "score": 1,
    },
]


EMAIL_ROUTE_POLICIES: dict[str, dict[str, Any]] = {
    "manual_review": {
        "description": "기본 수동 검토",
        "internal_input_candidate": False,
        "auto_forward_allowed": False,
    },
    "finance_review": {
        "description": "재무/청구 관련 검토",
        "internal_input_candidate": False,
        "auto_forward_allowed": False,
    },
    "priority_review": {
        "description": "긴급 우선 검토",
        "internal_input_candidate": False,
        "auto_forward_allowed": False,
    },
    "candidate_internal_input": {
        "description": "승인 후 내부 입력 파이프라인 전달 후보",
        "internal_input_candidate": True,
        "auto_forward_allowed": True,
    },
    "archive_review": {
        "description": "자동 발신/대량 메일성 보관 검토",
        "internal_input_candidate": False,
        "auto_forward_allowed": False,
    },
}


def email_bridge_is_configured() -> bool:
    return bool(
        CONFIG.enabled
        and CONFIG.provider == "imap_smtp"
        and CONFIG.imap_host and CONFIG.imap_username and CONFIG.imap_password
    )


def email_sending_is_configured() -> bool:
    return bool(
        CONFIG.enabled
        and CONFIG.smtp_from_email
        and CONFIG.resend_api_key
    )


def _decode_mime(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _parse_address_list(*values: str | None) -> list[str]:
    addresses = getaddresses([v for v in values if v])
    result: list[str] = []
    for _, addr in addresses:
        if addr:
            normalized = addr.strip().lower()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def _extract_name_and_email(value: str | None) -> tuple[str, str]:
    name, addr = parseaddr(value or "")
    return _decode_mime(name), addr.strip().lower()


def _coerce_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _html_to_text(value: str) -> str:
    text = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    import re
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_bodies(msg: Message) -> tuple[str, str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict[str, Any]] = []
    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = (part.get("Content-Disposition") or "").lower()
            content_type = (part.get_content_type() or "").lower()
            filename = _decode_mime(part.get_filename())
            if filename or "attachment" in content_disposition:
                payload = part.get_payload(decode=True) or b""
                attachments.append({
                    "filename": filename or "attachment",
                    "content_type": content_type,
                    "size": len(payload),
                })
                continue
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                decoded = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain":
                text_parts.append(decoded)
            elif content_type == "text/html":
                html_parts.append(decoded)
    else:
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        decoded = payload.decode(charset, errors="replace")
        if msg.get_content_type() == "text/html":
            html_parts.append(decoded)
        else:
            text_parts.append(decoded)
    return "\n\n".join(p.strip() for p in text_parts if p.strip()), "\n\n".join(p.strip() for p in html_parts if p.strip()), attachments


def _message_to_record(uid: str, msg: Message) -> dict[str, Any]:
    sender_name, sender_email = _extract_name_and_email(msg.get("From"))
    header_message_id = (msg.get("Message-ID") or "").strip()
    message_id = header_message_id or f"<{uid}@imap.local>"
    subject = _decode_mime(msg.get("Subject"))
    text_body, html_body, attachments = _extract_bodies(msg)
    if not text_body and html_body:
        text_body = _html_to_text(html_body)
    recipients = _parse_address_list(msg.get("To"))
    cc_emails = _parse_address_list(msg.get("Cc"))
    return {
        "provider": CONFIG.provider,
        "external_message_id": message_id,
        "message_id_header": header_message_id,
        "external_thread_id": (msg.get("References") or msg.get("Thread-Index") or message_id)[:255],
        "in_reply_to": (msg.get("In-Reply-To") or "")[:255],
        "subject": subject,
        "sender_name": sender_name,
        "sender_email": sender_email,
        "recipient_emails": recipients,
        "cc_emails": cc_emails,
        "bcc_emails": [],
        "text_body": text_body,
        "html_body": html_body,
        "attachments": attachments,
        "mailbox": CONFIG.imap_mailbox,
        "received_at": _coerce_dt(msg.get("Date")) or datetime.now(timezone.utc),
        "raw_headers": {
            "message_id": message_id,
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "cc": msg.get("Cc", ""),
            "date": msg.get("Date", ""),
        },
        "metadata": {"imap_uid": uid, "message_id_header": header_message_id},
    }


def _ensure_thread(record: dict[str, Any]) -> int | None:
    rows = db_query(
        """
        INSERT INTO email_threads (provider, external_thread_id, subject, participants, metadata, updated_at)
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, NOW())
        ON CONFLICT (provider, external_thread_id)
        DO UPDATE SET subject = EXCLUDED.subject, participants = EXCLUDED.participants, updated_at = NOW()
        RETURNING id
        """,
        (
            record["provider"],
            record["external_thread_id"],
            record["subject"],
            json.dumps(sorted(set([record["sender_email"], *record["recipient_emails"], *record["cc_emails"]]))),
            json.dumps({"last_sender": record["sender_email"]}),
        ),
    )
    return rows[0]["id"] if rows else None


def _find_existing_inbound_email(record: dict[str, Any]) -> dict[str, Any] | None:
    existing = db_query_one(
        "SELECT id, status FROM email_messages WHERE provider = %s AND external_message_id = %s",
        (record["provider"], record["external_message_id"]),
    )
    if existing:
        return existing
    imap_uid = str((record.get("metadata") or {}).get("imap_uid") or "").strip()
    if imap_uid:
        return db_query_one(
            """
            SELECT id, status FROM email_messages
            WHERE provider = %s
              AND COALESCE(metadata->>'imap_uid', '') = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (record["provider"], imap_uid),
        )
    return None


def store_inbound_email(record: dict[str, Any]) -> dict[str, Any] | None:
    existing = _find_existing_inbound_email(record)
    if existing:
        existing["_is_duplicate"] = True
        return existing
    thread_id = _ensure_thread(record)
    rows = db_query(
        """
        INSERT INTO email_messages (
            thread_id, provider, direction, status, mailbox, external_message_id, in_reply_to,
            sender_email, sender_name, recipient_emails, cc_emails, bcc_emails, subject,
            text_body, html_body, raw_headers, attachments, metadata, received_at, created_at,
            updated_at, audit_log
        ) VALUES (
            %s, %s, 'inbound', 'received', %s, %s, %s,
            %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
            %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW(),
            NOW(), %s::jsonb
        ) RETURNING id, status
        """,
        (
            thread_id, record["provider"], record["mailbox"], record["external_message_id"], record["in_reply_to"],
            record["sender_email"], record["sender_name"], json.dumps(record["recipient_emails"]),
            json.dumps(record["cc_emails"]), json.dumps(record["bcc_emails"]), record["subject"],
            record["text_body"], record["html_body"], json.dumps(record["raw_headers"]),
            json.dumps(record["attachments"]), json.dumps(record["metadata"]), record["received_at"],
            json.dumps([{
                "at": datetime.now(timezone.utc).isoformat(),
                "event": "received",
                "actor": "email_bridge",
                "metadata": {"mailbox": record["mailbox"]},
            }]),
        ),
    )
    if not rows:
        return None
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (rows[0]["id"], "received", record["subject"], json.dumps({"sender": record["sender_email"]})),
    )
    return rows[0]


def classify_inbound_email(message_row: dict[str, Any]) -> dict[str, Any]:
    subject = (message_row.get("subject") or "").lower()
    sender = (message_row.get("sender_email") or "").lower()
    body = (message_row.get("text_body") or message_row.get("html_body") or "").lower()
    has_attachments = bool(message_row.get("attachments"))
    score = 0
    labels: list[str] = []

    field_map = {
        "subject": subject,
        "sender": sender,
        "body": body,
    }
    for rule in EMAIL_CLASSIFICATION_RULES:
        haystack = field_map.get(rule.get("field", ""), "")
        tokens = rule.get("tokens") or []
        if haystack and any(token in haystack for token in tokens):
            labels.append(str(rule.get("label") or "unknown"))
            score += int(rule.get("score") or 0)

    if has_attachments:
        labels.append("has_attachments")
    if message_row.get("sender_email") and sender.endswith("@cyber-lenin.com"):
        labels.append("internal_sender")

    route = "manual_review"
    if "finance" in labels:
        route = "finance_review"
    elif "urgent" in labels:
        route = "priority_review"
    elif "human_request" in labels and not has_attachments:
        route = "candidate_internal_input"
    elif "bulk_like" in labels or "automated_sender" in labels:
        route = "archive_review"

    policy = EMAIL_ROUTE_POLICIES.get(route, EMAIL_ROUTE_POLICIES["manual_review"])
    auto_forward_allowed = bool(policy.get("auto_forward_allowed")) and not has_attachments

    return {
        "labels": labels,
        "route": route,
        "route_description": policy.get("description", ""),
        "internal_input_candidate": bool(policy.get("internal_input_candidate")),
        "auto_forward_allowed": auto_forward_allowed,
        "contains_attachments": has_attachments,
        "confidence_score": score,
        "reason": ", ".join(labels) if labels else "fallback_manual_review",
    }


def build_reply_prompt_input(message_row: dict[str, Any]) -> dict[str, Any]:
    body = (message_row.get("text_body") or message_row.get("html_body") or "").strip()
    excerpt = body[:4000]
    return {
        "message_id": message_row["id"],
        "thread_id": message_row.get("thread_id"),
        "from": message_row.get("sender_email"),
        "subject": message_row.get("subject"),
        "classification": classify_inbound_email(message_row),
        "received_at": message_row.get("received_at").isoformat() if message_row.get("received_at") else None,
        "reply_instruction": (
            "아래 이메일에 대한 답변 초안을 작성하라. 사실관계가 불명확하면 추측하지 말고, "
            "필요시 추가 확인 질문을 제안하라. 답변은 외부 발신용 문체로 정리하라."
        ),
        "source_email": {
            "subject": message_row.get("subject"),
            "from": message_row.get("sender_email"),
            "body_excerpt": excerpt,
        },
    }


def list_pending_email_approvals(limit: int = 20) -> list[dict[str, Any]]:
    return db_query(
        """
        SELECT id, thread_id, direction, status, sender_email, recipient_emails, subject,
               LEFT(COALESCE(text_body, html_body, ''), 300) AS preview,
               created_at, approved_at, sent_at, metadata
        FROM email_messages
        WHERE direction = 'outbound' AND status IN ('draft_pending_approval', 'approved', 'draft_saved')
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_email_message(message_id: int) -> dict[str, Any] | None:
    return db_query_one("SELECT * FROM email_messages WHERE id = %s", (message_id,))


def queue_outbound_reply(
    inbound_message_id: int,
    draft_body: str,
    *,
    approver_user_id: int | None = None,
    subject: str | None = None,
    to_emails: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inbound = get_email_message(inbound_message_id)
    if not inbound:
        raise ValueError(f"inbound email not found: {inbound_message_id}")
    if inbound.get("direction") != "inbound":
        raise ValueError("draft target must be an inbound email")
    outbound_subject = subject or f"Re: {inbound.get('subject') or ''}".strip()
    recipients = to_emails or ([inbound.get("sender_email")] if inbound.get("sender_email") else [])
    approval_user = approver_user_id or CONFIG.default_approver_user_id or 0
    merged_metadata = {
        "source_message_id": inbound_message_id,
        "approval_user_id": approval_user,
        "approval_url": f"{CONFIG.approval_base_url}/email/pending" if CONFIG.approval_base_url else "",
    }
    if metadata:
        merged_metadata.update(metadata)
    rows = db_query(
        """
        INSERT INTO email_messages (
            thread_id, provider, direction, status, mailbox, external_message_id, in_reply_to,
            sender_email, sender_name, recipient_emails, cc_emails, bcc_emails, subject,
            text_body, html_body, raw_headers, attachments, metadata, created_at, updated_at, audit_log
        ) VALUES (
            %s, %s, 'outbound', 'draft_pending_approval', 'outbox', NULL, %s,
            %s, %s, %s::jsonb, '[]'::jsonb, '[]'::jsonb, %s,
            %s, NULL, '{}'::jsonb, '[]'::jsonb, %s::jsonb, NOW(), NOW(), %s::jsonb
        ) RETURNING id, status, subject, recipient_emails, metadata
        """,
        (
            inbound.get("thread_id"), inbound.get("provider") or CONFIG.provider, inbound.get("external_message_id"),
            CONFIG.smtp_from_email, CONFIG.smtp_from_name, json.dumps(recipients), outbound_subject,
            draft_body, json.dumps(merged_metadata),
            json.dumps([{
                "at": datetime.now(timezone.utc).isoformat(),
                "event": "draft_created",
                "actor": "email_bridge",
                "metadata": {"source_message_id": inbound_message_id, "approval_user_id": approval_user},
            }]),
        ),
    )
    if not rows:
        raise RuntimeError("failed to create outbound draft")
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (rows[0]["id"], "draft_created", outbound_subject, json.dumps({"source_message_id": inbound_message_id})),
    )
    return rows[0]


def save_draft_copy(message_row: dict[str, Any]) -> str:
    CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
    draft_path = CONFIG.log_dir / f"draft_{message_row['id']}.json"
    draft_path.write_text(json.dumps(message_row, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    return str(draft_path)


def list_inbound_messages(limit: int = 20, *, route: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    clauses = ["direction = 'inbound'"]
    params: list[Any] = []
    if route:
        clauses.append("COALESCE(metadata->'classification'->>'route', '') = %s")
        params.append(route)
    if status:
        clauses.append("status = %s")
        params.append(status)
    params.append(limit)
    where_sql = " AND ".join(clauses)
    return db_query(
        f"""
        SELECT id, status, sender_email, subject,
               LEFT(COALESCE(text_body, html_body, ''), 280) AS preview,
               metadata->'classification' AS classification,
               metadata->>'delivery_status' AS delivery_status,
               received_at, created_at
        FROM email_messages
        WHERE {where_sql}
        ORDER BY COALESCE(received_at, created_at) DESC
        LIMIT %s
        """,
        tuple(params),
    )


def mark_email_for_internal_delivery(message_id: int, *, approved_by: int | None = None, note: str = "") -> dict[str, Any]:
    row = get_email_message(message_id)
    if not row:
        raise ValueError(f"email message not found: {message_id}")
    if row.get("direction") != "inbound":
        raise ValueError("only inbound email can be approved for internal delivery")
    classification = classify_inbound_email(row)
    if classification.get("contains_attachments"):
        raise ValueError("messages with attachments require manual handling; auto processing is disabled")
    if not classification.get("internal_input_candidate"):
        raise ValueError("this message route is not approved for internal input delivery")
    metadata = dict(row.get("metadata") or {})
    metadata["classification"] = classification
    metadata["delivery_status"] = "approved_for_internal_input"
    metadata["delivery_approved_at"] = datetime.now(timezone.utc).isoformat()
    if note:
        metadata["delivery_note"] = note[:500]
    db_execute(
        """
        UPDATE email_messages
        SET status = 'approved_for_internal_input',
            approved_by = COALESCE(%s, approved_by),
            approved_at = COALESCE(approved_at, NOW()),
            approval_note = COALESCE(NULLIF(%s, ''), approval_note),
            metadata = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (approved_by, note, json.dumps(metadata), message_id),
    )
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (message_id, "internal_delivery_approved", note or None, json.dumps({"approved_by": approved_by, "route": classification.get("route")})),
    )
    return {"message_id": message_id, "status": "approved_for_internal_input", "classification": classification}


def list_messages_approved_for_internal_delivery(limit: int = 20) -> list[dict[str, Any]]:
    return db_query(
        """
        SELECT id, status, sender_email, subject,
               LEFT(COALESCE(text_body, html_body, ''), 280) AS preview,
               metadata->'classification' AS classification,
               metadata->>'delivery_status' AS delivery_status,
               metadata->>'delivered_to_internal_at' AS delivered_to_internal_at,
               received_at, approved_at, created_at
        FROM email_messages
        WHERE direction = 'inbound'
          AND status = ANY(%s)
        ORDER BY COALESCE(approved_at, received_at, created_at) ASC
        LIMIT %s
        """,
        (list(INBOUND_INTERNAL_STATUSES), limit),
    )


def deliver_inbound_email_to_internal_input(
    message_id: int,
    *,
    delivered_by: str = "email_bridge",
    note: str = "",
    target_handler: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    row = get_email_message(message_id)
    if not row:
        raise ValueError(f"email message not found: {message_id}")
    if row.get("direction") != "inbound":
        raise ValueError("only inbound email can be delivered to internal input")
    if row.get("status") != "approved_for_internal_input":
        raise ValueError("email message is not approved for internal input")

    classification = classify_inbound_email(row)
    metadata = dict(row.get("metadata") or {})
    metadata["classification"] = classification
    delivery_payload = {
        "source": "email_bridge",
        "message_id": row.get("id"),
        "thread_id": row.get("thread_id"),
        "sender_email": row.get("sender_email"),
        "subject": row.get("subject"),
        "text_body": row.get("text_body") or row.get("html_body") or "",
        "received_at": row.get("received_at").isoformat() if row.get("received_at") else None,
        "classification": classification,
        "attachments": row.get("attachments") or [],
        "metadata": metadata,
    }

    if target_handler is not None:
        try:
            target_handler(delivery_payload)
        except Exception as exc:
            metadata["delivery_status"] = "delivery_failed"
            metadata["delivery_error"] = str(exc)[:500]
            db_execute(
                "UPDATE email_messages SET status = 'delivery_failed', metadata = %s::jsonb, updated_at = NOW() WHERE id = %s",
                (json.dumps(metadata), message_id),
            )
            db_execute(
                "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
                (message_id, "internal_delivery_failed", str(exc)[:300], json.dumps({"delivered_by": delivered_by})),
            )
            raise

    metadata["delivery_status"] = "delivered_to_internal_input"
    metadata["delivered_to_internal_at"] = datetime.now(timezone.utc).isoformat()
    metadata["delivered_to_internal_by"] = delivered_by[:100]
    if note:
        metadata["delivery_note"] = note[:500]
    db_execute(
        "UPDATE email_messages SET status = 'delivered_to_internal_input', metadata = %s::jsonb, updated_at = NOW() WHERE id = %s",
        (json.dumps(metadata), message_id),
    )
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (message_id, "internal_delivered", note or None, json.dumps({"delivered_by": delivered_by, "route": classification.get("route")})),
    )
    return {
        "message_id": message_id,
        "status": "delivered_to_internal_input",
        "classification": classification,
        "payload": delivery_payload,
    }


def send_outbound_email(message_id: int, approve: bool = True, approval_note: str = "") -> dict[str, Any]:
    row = get_email_message(message_id)
    if not row:
        raise ValueError(f"email message not found: {message_id}")
    if row.get("direction") != "outbound":
        raise ValueError("only outbound draft can be sent")
    if approve:
        db_execute(
            "UPDATE email_messages SET status = 'approved', approved_at = NOW(), approval_note = %s, approved_by = COALESCE(approved_by, %s), updated_at = NOW() WHERE id = %s",
            (approval_note or None, CONFIG.default_approver_user_id or None, message_id),
        )
    if not email_sending_is_configured():
        path = save_draft_copy(row)
        db_execute(
            "UPDATE email_messages SET status = 'draft_saved', draft_saved_at = NOW(), updated_at = NOW() WHERE id = %s",
            (message_id,),
        )
        return {"status": "draft_saved", "path": path}

    import resend
    resend.api_key = CONFIG.resend_api_key

    recipients = row.get("recipient_emails") or []
    subject = row.get("subject") or "(no subject)"
    body = row.get("text_body") or ""
    from_addr = f"{CONFIG.smtp_from_name} <{CONFIG.smtp_from_email}>"

    html_body = row.get("html_body") or ""
    send_params: dict[str, Any] = {
        "from": from_addr,
        "to": recipients,
        "subject": subject,
    }
    if html_body:
        send_params["html"] = html_body
    if body:
        send_params["text"] = body
    if row.get("in_reply_to"):
        send_params["headers"] = {
            "In-Reply-To": row["in_reply_to"],
            "References": row["in_reply_to"],
        }

    result = resend.Emails.send(send_params)
    resend_id = result.get("id") if isinstance(result, dict) else str(result)

    db_execute(
        "UPDATE email_messages SET status = 'sent', sent_at = NOW(), updated_at = NOW(), external_message_id = %s WHERE id = %s",
        (resend_id, message_id),
    )
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (message_id, "sent", subject, json.dumps({"recipients": recipients, "resend_id": resend_id})),
    )
    return {"status": "sent", "message_id": resend_id, "recipients": recipients}


def reject_outbound_email(message_id: int, note: str = "") -> None:
    db_execute(
        "UPDATE email_messages SET status = 'rejected', approval_note = %s, updated_at = NOW() WHERE id = %s",
        (note or None, message_id),
    )
    db_execute(
        "INSERT INTO email_bridge_events (message_id, event_type, detail, metadata) VALUES (%s, %s, %s, %s::jsonb)",
        (message_id, "rejected", note or None, json.dumps({})),
    )


def poll_inbox_once(limit: int = 10) -> list[dict[str, Any]]:
    if not email_bridge_is_configured() or not CONFIG.polling_enabled:
        return []
    processed: list[dict[str, Any]] = []
    mailbox = None
    try:
        mailbox = imaplib.IMAP4_SSL(CONFIG.imap_host, CONFIG.imap_port)
        mailbox.login(CONFIG.imap_username, CONFIG.imap_password)
        mailbox.select(CONFIG.imap_mailbox)
        typ, data = mailbox.uid("search", None, "UNSEEN")
        if typ != "OK":
            return []
        uids = [u.decode() if isinstance(u, bytes) else str(u) for u in (data[0].split() if data and data[0] else [])][-limit:]
        for uid in uids:
            typ, msg_data = mailbox.uid("fetch", uid, "(RFC822)")
            if typ != "OK" or not msg_data:
                continue
            raw = b""
            for item in msg_data:
                if isinstance(item, tuple):
                    raw = item[1]
                    break
            if not raw:
                continue
            msg = email.message_from_bytes(raw)
            record = _message_to_record(uid, msg)
            classification = classify_inbound_email(record)
            record["metadata"] = {
                **(record.get("metadata") or {}),
                "classification": classification,
                "delivery_status": "pending_review",
                "attachments_auto_processing": "disabled",
            }
            stored = store_inbound_email(record)
            if stored:
                status = "duplicate" if stored.get("_is_duplicate") else "stored"
                processed.append({
                    "stored_message_id": stored.get("id"),
                    "subject": record.get("subject"),
                    "sender": record.get("sender_email"),
                    "classification": classification,
                    "processing_status": status,
                })
            mailbox.uid("store", uid, "+FLAGS", "(\\Seen)")
    finally:
        try:
            if mailbox is not None:
                mailbox.close()
                mailbox.logout()
        except Exception:
            pass
    return processed


def run_polling_cycle(limit: int = 10) -> dict[str, Any]:
    processed = poll_inbox_once(limit=limit)
    new_count = sum(1 for item in processed if item.get("processing_status") == "stored")
    duplicate_count = max(0, len(processed) - new_count)
    return {
        "processed": processed,
        "new_count": new_count,
        "duplicate_count": duplicate_count,
        "poll_interval_seconds": CONFIG.poll_interval_seconds,
        "operations_chat_id": CONFIG.operations_chat_id,
    }


def build_inbound_summary_notification(item: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    classification = (item.get("classification") or {}) if isinstance(item, dict) else {}
    route = classification.get("route") or "manual_review"
    labels = ", ".join(classification.get("labels") or []) or "-"
    preview_source = ""
    if row:
        preview_source = (row.get("text_body") or row.get("html_body") or "").strip()
    preview = preview_source.replace("```", "'''")[:500] if preview_source else ""
    lines = [
        "📨 새 메일 수신",
        f"inbound_id: `{item.get('stored_message_id') or '-'}`",
        f"from: {item.get('sender') or '-'}",
        f"subject: {item.get('subject') or '(no subject)'}",
        f"route: `{route}` / labels: {labels}",
        f"status: `{item.get('processing_status') or '-'}`",
    ]
    if classification.get("contains_attachments"):
        lines.append("attachments: 감지됨 — 자동처리 금지")
    if preview:
        lines.append(f"preview:\n{preview}")
    return "\n".join(lines)
