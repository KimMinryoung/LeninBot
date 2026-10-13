-- Visitor non-publication requests extracted from web chats (telegram/diary_privacy.py).
CREATE TABLE IF NOT EXISTS web_privacy_requests (
    id BIGSERIAL PRIMARY KEY,
    visitor TEXT NOT NULL,
    chat_log_id BIGINT NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    quote TEXT NOT NULL DEFAULT '',
    withheld JSONB NOT NULL DEFAULT '[]',
    revoked_at TIMESTAMPTZ,
    revoked_by_chat_log_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_web_privacy_requests_active
    ON web_privacy_requests (visitor, requested_at) WHERE revoked_at IS NULL;

-- How far each visitor's messages have been read, so a scan reads only new ones.
CREATE TABLE IF NOT EXISTS web_privacy_scans (
    visitor TEXT PRIMARY KEY,
    last_chat_log_id BIGINT NOT NULL,
    scanned_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
