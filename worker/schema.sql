-- Agent worker tasks: work other services hand to leninbot's agents over the
-- worker MCP (dev_docs/agent_worker.md). The requester owns what the work is
-- for; this table only records the request, its execution and its result.
CREATE TABLE IF NOT EXISTS agent_worker_tasks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client text NOT NULL,
    idempotency_key text NOT NULL,
    request jsonb NOT NULL,
    request_hash text NOT NULL,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    attempts integer NOT NULL DEFAULT 0,
    lease_until timestamptz,
    result jsonb,
    sources jsonb NOT NULL DEFAULT '[]',
    usage jsonb NOT NULL DEFAULT '{}',
    rejections jsonb NOT NULL DEFAULT '[]',
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (client, idempotency_key)
);
CREATE INDEX IF NOT EXISTS agent_worker_tasks_open
    ON agent_worker_tasks (created_at) WHERE status IN ('queued', 'running');
