"""Target-scoped commissions, retaining the individual editorial topics."""


def topics(job):
    payload = job.get('payload') or {}
    return payload.get('remaining_topics', payload.get('topics', [job['topic']]))


def work_topics(job):
    pending = topics(job)
    # A section has its own save/review contract. Process it after the card,
    # from a new research snapshot, within the same durable job.
    if job['kind'] == 'person' and len(pending) > 1:
        return [topic for topic in pending if topic != 'sections']
    return pending
