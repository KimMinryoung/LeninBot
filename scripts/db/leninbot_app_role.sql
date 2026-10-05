-- leninbot_app: leninbot's own database login instead of the postgres
-- superuser (frontend dev_docs/commulingo-admin-mcp.md stage 5,
-- dev_docs/db_app_role.md). Run as postgres; idempotent. The password comes in
-- as the psql variable app_password (piped \set, never on a command line).
--
-- Ownership: tables leninbot creates or alters (its own, plus the shared
-- publishing tables it runs DDL on) move to leninbot_app so its schema
-- migrations and ensure_* calls keep working without superuser. Tables the
-- frontend owns stay with postgres; leninbot gets DML only on the shared ones
-- it reads and writes, and nothing on CommuLingo or other frontend tables.
-- Existing grants (frontend, leninbot_audit, leninbot_ro) survive the owner change.
\set ON_ERROR_STOP on
SET lock_timeout = '5s';

SELECT NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'leninbot_app') AS create_role \gset
\if :create_role
CREATE ROLE leninbot_app LOGIN;
\endif
ALTER ROLE leninbot_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD :'app_password';

GRANT CONNECT, TEMPORARY ON DATABASE leninbot TO leninbot_app;
GRANT USAGE, CREATE ON SCHEMA public TO leninbot_app;
GRANT USAGE ON SCHEMA extensions TO leninbot_app;

BEGIN;
-- leninbot's own tables
ALTER TABLE agent_worker_tasks OWNER TO leninbot_app;
ALTER TABLE autonomous_project_advisories OWNER TO leninbot_app;
ALTER TABLE autonomous_project_events OWNER TO leninbot_app;
ALTER TABLE autonomous_project_notes OWNER TO leninbot_app;
ALTER TABLE autonomous_projects OWNER TO leninbot_app;
ALTER TABLE autonomous_publication_audits OWNER TO leninbot_app;
ALTER TABLE chat_clear_markers OWNER TO leninbot_app;
ALTER TABLE chat_history_summaries OWNER TO leninbot_app;
ALTER TABLE diary_publication_audits OWNER TO leninbot_app;
ALTER TABLE email_bridge_events OWNER TO leninbot_app;
ALTER TABLE email_bridge_state OWNER TO leninbot_app;
ALTER TABLE email_messages OWNER TO leninbot_app;
ALTER TABLE email_threads OWNER TO leninbot_app;
ALTER TABLE experiential_memory OWNER TO leninbot_app;
ALTER TABLE file_registry OWNER TO leninbot_app;
ALTER TABLE kg_sync_state OWNER TO leninbot_app;
ALTER TABLE lenin_corpus OWNER TO leninbot_app;
ALTER TABLE llm_audit_log OWNER TO leninbot_app;
ALTER TABLE mail_briefing_items OWNER TO leninbot_app;
ALTER TABLE mail_briefing_messages OWNER TO leninbot_app;
ALTER TABLE mail_briefing_reads OWNER TO leninbot_app;
ALTER TABLE publish_record OWNER TO leninbot_app;
ALTER TABLE roleplay_chat_history OWNER TO leninbot_app;
ALTER TABLE roleplay_clear_markers OWNER TO leninbot_app;
ALTER TABLE telegram_chat_history OWNER TO leninbot_app;
ALTER TABLE telegram_error_log OWNER TO leninbot_app;
ALTER TABLE telegram_mission_events OWNER TO leninbot_app;
ALTER TABLE telegram_missions OWNER TO leninbot_app;
ALTER TABLE telegram_schedules OWNER TO leninbot_app;
ALTER TABLE telegram_system_events OWNER TO leninbot_app;
ALTER TABLE telegram_tasks OWNER TO leninbot_app;
ALTER TABLE tool_audit_log OWNER TO leninbot_app;
ALTER TABLE tool_idempotency OWNER TO leninbot_app;
ALTER TABLE web_chat_feedback OWNER TO leninbot_app;
ALTER TABLE x402_payment_attempts OWNER TO leninbot_app;
-- the agent sessions' research source cache
ALTER TABLE commulingo_pipeline_sources OWNER TO leninbot_app;
ALTER TABLE commulingo_pipeline_fetch_cache OWNER TO leninbot_app;
ALTER TABLE commulingo_pipeline_job_sources OWNER TO leninbot_app;
-- shared publishing tables leninbot writes and runs DDL on (ensure_* at runtime)
ALTER TABLE chat_logs OWNER TO leninbot_app;
ALTER TABLE hub_curations OWNER TO leninbot_app;
ALTER TABLE research_documents OWNER TO leninbot_app;
ALTER TABLE static_pages OWNER TO leninbot_app;

-- trigger functions leninbot's schema code (re)defines with CREATE OR REPLACE
ALTER FUNCTION prevent_telegram_task_tool_log_loss() OWNER TO leninbot_app;
ALTER FUNCTION prevent_tool_audit_log_mutation() OWNER TO leninbot_app;

-- shared tables the frontend owns that leninbot reads and writes
GRANT SELECT, INSERT, UPDATE, DELETE ON posts, ai_diary, users, user_passkeys, user_fingerprints TO leninbot_app;
COMMIT;

-- their serial/identity sequences
SELECT format('GRANT USAGE, SELECT, UPDATE ON SEQUENCE %s TO leninbot_app;', s.oid::regclass)
FROM pg_class s JOIN pg_depend d ON d.objid = s.oid AND d.classid = 'pg_class'::regclass AND d.deptype IN ('a', 'i')
WHERE s.relkind = 'S' AND d.refobjid::regclass::text IN ('posts', 'ai_diary', 'users', 'user_passkeys', 'user_fingerprints')
\gexec

-- Objects leninbot_app creates later get what the frontend gets today from the
-- postgres default privileges (frontend reads leninbot's published tables).
ALTER DEFAULT PRIVILEGES FOR ROLE leninbot_app IN SCHEMA public GRANT ALL ON TABLES TO frontend;
ALTER DEFAULT PRIVILEGES FOR ROLE leninbot_app IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO frontend;

-- Report
SELECT tableowner, count(*) FROM pg_tables WHERE schemaname = 'public' GROUP BY 1 ORDER BY 1;
