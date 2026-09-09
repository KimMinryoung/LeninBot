CREATE TABLE IF NOT EXISTS commulingo_pipeline_jobs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('person','term')),
    action text NOT NULL CHECK (action IN ('create','update')),
    target text NOT NULL,
    topic text NOT NULL,
    baseline text NOT NULL DEFAULT '',
    reason text NOT NULL,
    priority integer NOT NULL DEFAULT 50,
    stage text NOT NULL DEFAULT 'research' CHECK (stage IN
        ('discover','research','judge','draft','validate','review','submit','complete')),
    status text NOT NULL DEFAULT 'ready' CHECK (status IN
        ('ready','running','deferred','complete','escalated','cancelled')),
    payload jsonb NOT NULL DEFAULT '{}',
    lease_token uuid,
    lease_until timestamptz,
    available_at timestamptz NOT NULL DEFAULT now(),
    attempts integer NOT NULL DEFAULT 0,
    last_error text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS commulingo_pipeline_active_target
ON commulingo_pipeline_jobs(kind,target,topic)
WHERE status IN ('ready','running','deferred','escalated');
ALTER TABLE commulingo_pipeline_jobs DROP CONSTRAINT IF EXISTS commulingo_pipeline_jobs_stage_check;
ALTER TABLE commulingo_pipeline_jobs ADD CONSTRAINT commulingo_pipeline_jobs_stage_check
CHECK (stage IN ('discover','research','judge','draft','validate','review','submit','complete'));
CREATE INDEX IF NOT EXISTS commulingo_pipeline_ready
ON commulingo_pipeline_jobs(priority,available_at,id)
WHERE status IN ('ready','running','deferred');
CREATE TABLE IF NOT EXISTS commulingo_pipeline_artifacts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id bigint NOT NULL REFERENCES commulingo_pipeline_jobs(id),
    stage text NOT NULL,
    value jsonb NOT NULL,
    metrics jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE commulingo_pipeline_artifacts ADD COLUMN IF NOT EXISTS metrics jsonb NOT NULL DEFAULT '{}';
CREATE TABLE IF NOT EXISTS commulingo_pipeline_sources (
    id text PRIMARY KEY,
    url text NOT NULL,
    content_hash text NOT NULL,
    fetched_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    body text,
    UNIQUE(url,content_hash)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_materials (
    material_id text PRIMARY KEY,
    content_hash text NOT NULL,
    processed_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_mentions (
    kind text NOT NULL, target text NOT NULL, material_id text NOT NULL,
    mention text NOT NULL, PRIMARY KEY(kind,target,material_id)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_job_sources (
    job_id bigint NOT NULL REFERENCES commulingo_pipeline_jobs(id),
    source_id text NOT NULL REFERENCES commulingo_pipeline_sources(id),
    PRIMARY KEY(job_id,source_id)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_fetch_cache (
    tool text NOT NULL, args_hash text NOT NULL,
    source_id text NOT NULL REFERENCES commulingo_pipeline_sources(id),
    PRIMARY KEY(tool,args_hash)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_budget (
    id uuid PRIMARY KEY,
    day date NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')::date,
    lane text NOT NULL,
    job_id bigint REFERENCES commulingo_pipeline_jobs(id),
    reserved numeric(12,6) NOT NULL CHECK (reserved >= 0),
    actual numeric(12,6) CHECK (actual >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    settled_at timestamptz
);
CREATE INDEX IF NOT EXISTS commulingo_pipeline_budget_day ON commulingo_pipeline_budget(day);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_scheduler (
    id integer PRIMARY KEY CHECK (id=1), cursor integer NOT NULL DEFAULT 0
);
INSERT INTO commulingo_pipeline_scheduler(id) VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS commulingo_pipeline_publications (
    job_id bigint PRIMARY KEY REFERENCES commulingo_pipeline_jobs(id),
    day date NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')::date,
    kind text NOT NULL, action text NOT NULL
);
