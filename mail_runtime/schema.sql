CREATE TABLE IF NOT EXISTS mail_briefing_messages (
    id BIGSERIAL PRIMARY KEY,
    account TEXT NOT NULL,
    folder TEXT NOT NULL,
    uidvalidity TEXT NOT NULL,
    uid TEXT NOT NULL,
    raw BYTEA NOT NULL,
    parsed JSONB NOT NULL,
    imap_seen BOOLEAN,
    flags_observed_at TIMESTAMPTZ,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(account, folder, uidvalidity, uid)
);
CREATE TABLE IF NOT EXISTS mail_briefing_reads (
    task_id BIGINT NOT NULL,
    mail_id BIGINT NOT NULL REFERENCES mail_briefing_messages(id),
    ranges JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(task_id, mail_id)
);
CREATE TABLE IF NOT EXISTS mail_briefing_items (
    task_id BIGINT NOT NULL,
    mail_id BIGINT NOT NULL REFERENCES mail_briefing_messages(id),
    audience TEXT NOT NULL,
    summary TEXT NOT NULL,
    prepared_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sent_at TIMESTAMPTZ,
    telegram_message_id BIGINT,
    PRIMARY KEY(task_id, mail_id)
);
CREATE INDEX IF NOT EXISTS mail_briefing_delivered
ON mail_briefing_items(audience, mail_id) WHERE sent_at IS NOT NULL;
