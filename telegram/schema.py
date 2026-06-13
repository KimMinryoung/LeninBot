"""Telegram-owned PostgreSQL schema setup.

This module keeps DDL out of the Telegram runtime entrypoint. The functions are
still intentionally idempotent so existing startup paths can keep calling them
until the DDL is moved behind an explicit migration command.
"""

from __future__ import annotations

from collections.abc import MutableMapping

from db import execute as _execute, query as _query

_summary_table_ready = False


def ensure_telegram_tables() -> None:
    """Create and update tables owned by the Telegram service."""
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_tasks (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'pending',
            result      TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_chat_history (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            role        VARCHAR(10) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
        ON telegram_chat_history (user_id, id DESC)
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_system_events (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            event_type  VARCHAR(50) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_schedules (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            content     TEXT NOT NULL,
            cron_expr   VARCHAR(100) NOT NULL,
            enabled     BOOLEAN DEFAULT TRUE,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            last_run_at TIMESTAMPTZ,
            agent_type  VARCHAR(50)
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_error_log (
            id          SERIAL PRIMARY KEY,
            level       VARCHAR(10) NOT NULL DEFAULT 'error',
            source      VARCHAR(100) NOT NULL,
            message     TEXT NOT NULL,
            detail      TEXT,
            task_id     INTEGER,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_error_log_created
        ON telegram_error_log (created_at DESC)
    """)
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS parent_task_id INTEGER REFERENCES telegram_tasks(id)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS scratchpad TEXT DEFAULT ''")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS depth INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS agent_type VARCHAR(50)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS mission_id INTEGER")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS tool_log TEXT DEFAULT ''")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS metadata JSONB")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20) DEFAULT 'pending'")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_initiated BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_target_service VARCHAR(20)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_completed BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS post_restart_phase VARCHAR(50)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_attempt_count INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_requested_at TIMESTAMPTZ")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS resumed_after_restart BOOLEAN DEFAULT FALSE")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS restart_reentry_block_reason TEXT")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_details TEXT")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS verification_attempts INTEGER DEFAULT 0")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS last_verification_at TIMESTAMPTZ")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS plan_id INTEGER")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS plan_role VARCHAR(20)")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal'")
    _execute("ALTER TABLE telegram_tasks ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ DEFAULT NOW()")
    _execute("ALTER TABLE telegram_schedules ADD COLUMN IF NOT EXISTS agent_type VARCHAR(50)")
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_parent
        ON telegram_tasks(parent_task_id) WHERE parent_task_id IS NOT NULL
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_plan
        ON telegram_tasks(plan_id) WHERE plan_id IS NOT NULL
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_user
        ON telegram_tasks(user_id, agent_type, status) WHERE status = 'done'
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_pending_priority
        ON telegram_tasks(
            status,
            (CASE priority
                WHEN 'high' THEN 0
                WHEN 'normal' THEN 1
                WHEN 'low' THEN 2
                ELSE 1
            END),
            available_at,
            created_at
        ) WHERE status = 'pending'
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_missions (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            title       TEXT NOT NULL,
            status      VARCHAR(20) DEFAULT 'active',
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            closed_at   TIMESTAMPTZ
        )
    """)
    _execute("""
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY created_at DESC, id DESC
                ) AS rn
            FROM telegram_missions
            WHERE status = 'active'
        )
        UPDATE telegram_missions m
        SET status = 'done', closed_at = COALESCE(closed_at, NOW())
        FROM ranked r
        WHERE m.id = r.id AND r.rn > 1
    """)
    _execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_missions_one_active_per_user
        ON telegram_missions(user_id)
        WHERE status = 'active'
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS telegram_mission_events (
            id          SERIAL PRIMARY KEY,
            mission_id  INTEGER NOT NULL REFERENCES telegram_missions(id),
            source      TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_mission_events_timeline
        ON telegram_mission_events(mission_id, created_at)
    """)
    _execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                JOIN pg_attribute
                  ON pg_attribute.attrelid = pg_constraint.conrelid
                 AND pg_attribute.attnum = ANY(pg_constraint.conkey)
                WHERE pg_constraint.contype = 'f'
                  AND pg_constraint.conrelid = 'telegram_tasks'::regclass
                  AND pg_constraint.confrelid = 'telegram_missions'::regclass
                  AND pg_attribute.attname = 'mission_id'
            ) THEN
                ALTER TABLE telegram_tasks
                ADD CONSTRAINT telegram_tasks_mission_id_fkey
                FOREIGN KEY (mission_id)
                REFERENCES telegram_missions(id)
                NOT VALID;
            END IF;
        END $$;
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_threads (
            id SERIAL PRIMARY KEY,
            provider VARCHAR(50) NOT NULL DEFAULT 'imap_smtp',
            external_thread_id VARCHAR(255),
            subject TEXT,
            participants JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(provider, external_thread_id)
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_messages (
            id SERIAL PRIMARY KEY,
            thread_id INTEGER REFERENCES email_threads(id) ON DELETE SET NULL,
            provider VARCHAR(50) NOT NULL DEFAULT 'imap_smtp',
            direction VARCHAR(20) NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'received',
            mailbox VARCHAR(50),
            external_message_id VARCHAR(255),
            in_reply_to VARCHAR(255),
            sender_email TEXT,
            sender_name TEXT,
            recipient_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            cc_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            bcc_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
            subject TEXT,
            text_body TEXT,
            html_body TEXT,
            raw_headers JSONB NOT NULL DEFAULT '{}'::jsonb,
            attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            received_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            approved_by BIGINT,
            approved_at TIMESTAMPTZ,
            approval_note TEXT,
            sent_at TIMESTAMPTZ,
            draft_saved_at TIMESTAMPTZ,
            audit_log JSONB NOT NULL DEFAULT '[]'::jsonb,
            UNIQUE(provider, external_message_id)
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_status_created
        ON email_messages(status, created_at DESC)
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_thread_created
        ON email_messages(thread_id, created_at DESC)
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_email_messages_provider_imap_uid
        ON email_messages(provider, ((metadata->>'imap_uid')))
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_bridge_events (
            id SERIAL PRIMARY KEY,
            message_id INTEGER REFERENCES email_messages(id) ON DELETE CASCADE,
            event_type VARCHAR(50) NOT NULL,
            detail TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS email_bridge_state (
            provider VARCHAR(50) PRIMARY KEY,
            state JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS file_registry (
            id SERIAL PRIMARY KEY,
            local_path TEXT NOT NULL,
            public_url TEXT,
            filename TEXT NOT NULL,
            content_type VARCHAR(100),
            description TEXT,
            category VARCHAR(50) DEFAULT 'general',
            file_size BIGINT,
            created_by_task_id INTEGER,
            created_by_agent VARCHAR(50),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_file_registry_category
        ON file_registry(category, created_at DESC)
    """)


def ensure_roleplay_tables() -> None:
    """Create tables owned by the standalone roleplay bot (telegram/roleplay_bot.py).

    Kept isolated from the Cyber-Lenin chat tables so the two bots' sessions
    never share history. Applied via scripts/schema_migrations.py, not at
    service startup.
    """
    _execute("""
        CREATE TABLE IF NOT EXISTS roleplay_chat_history (
            id          SERIAL PRIMARY KEY,
            user_id     BIGINT NOT NULL,
            role        VARCHAR(10) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _execute("""
        CREATE INDEX IF NOT EXISTS idx_rp_history_user
        ON roleplay_chat_history (user_id, id DESC)
    """)
    _execute("""
        CREATE TABLE IF NOT EXISTS roleplay_clear_markers (
            user_id        BIGINT PRIMARY KEY,
            clear_after_id BIGINT NOT NULL DEFAULT 0
        )
    """)


def ensure_summary_tables(clear_after_id: MutableMapping[int, int]) -> None:
    """Create summary tables and hydrate persisted clear markers."""
    global _summary_table_ready
    if _summary_table_ready:
        return
    _execute(
        "CREATE TABLE IF NOT EXISTS chat_clear_markers ("
        "  user_id BIGINT PRIMARY KEY,"
        "  clear_after_id BIGINT NOT NULL DEFAULT 0"
        ")"
    )
    rows = _query("SELECT user_id, clear_after_id FROM chat_clear_markers")
    for row in rows:
        user_id = row["user_id"]
        current = clear_after_id.get(user_id, 0)
        clear_after_id[user_id] = max(current, row["clear_after_id"])
    _execute(
        "CREATE TABLE IF NOT EXISTS chat_history_summaries ("
        "  id SERIAL PRIMARY KEY,"
        "  user_id BIGINT NOT NULL,"
        "  chunk_start_id BIGINT NOT NULL,"
        "  chunk_end_id BIGINT NOT NULL,"
        "  summary TEXT NOT NULL,"
        "  msg_count INTEGER DEFAULT 0,"
        "  created_at TIMESTAMPTZ DEFAULT NOW()"
        ")"
    )
    _summary_table_ready = True


def hydrate_summary_state(clear_after_id: MutableMapping[int, int]) -> None:
    """Hydrate persisted clear markers without creating or altering tables."""
    rows = _query("SELECT user_id, clear_after_id FROM chat_clear_markers")
    for row in rows:
        user_id = row["user_id"]
        current = clear_after_id.get(user_id, 0)
        clear_after_id[user_id] = max(current, row["clear_after_id"])
