-- leninbot's research source cache for the CommuLingo agent sessions
-- (commulingo/pipeline/store.py). The enrichment queue tables
-- (commulingo_pipeline_jobs, artifacts, …) belong to the frontend since
-- 2026-10-05 (its migration 288); job_sources rows point at its jobs.
CREATE TABLE IF NOT EXISTS commulingo_pipeline_sources (
    id text PRIMARY KEY,
    url text NOT NULL,
    content_hash text NOT NULL,
    fetched_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    body text,
    UNIQUE(url,content_hash)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_fetch_cache (
    tool text NOT NULL, args_hash text NOT NULL,
    source_id text NOT NULL REFERENCES commulingo_pipeline_sources(id),
    PRIMARY KEY(tool,args_hash)
);
CREATE TABLE IF NOT EXISTS commulingo_pipeline_job_sources (
    job_id bigint NOT NULL REFERENCES commulingo_pipeline_jobs(id),
    source_id text NOT NULL REFERENCES commulingo_pipeline_sources(id),
    PRIMARY KEY(job_id,source_id)
);
