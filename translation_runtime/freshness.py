"""Pure SQL predicates shared by scheduled translation and read-only diagnostics."""

TABLES = {'research': 'research_documents', 'posts': 'posts', 'diary': 'ai_diary', 'curation': 'hub_curations'}


def source_fields(kind):
    return ('title', 'source_title', 'selection_rationale', 'context') if kind == 'curation' else ('title', 'content')


def source_hash_sql(kind):
    fields = ', '.join(source_fields(kind))
    return f"encode(sha256(convert_to(jsonb_build_array({fields})::text, 'UTF8')), 'hex')"


def missing_translation_sql(kind):
    fields = ('markdown_en',) if kind == 'research' else (('title_en', 'selection_rationale_en', 'context_en') if kind == 'curation' else ('title_en', 'content_en'))
    return ' OR '.join(f"NULLIF(BTRIM(COALESCE({field}, '')), '') IS NULL" for field in fields)


def changed_source_sql(kind):
    if kind == 'research':
        return 'markdown_en_source_sha256 IS DISTINCT FROM content_sha256'
    return f'translation_source_sha256 IS DISTINCT FROM {source_hash_sql(kind)}'


def pending_translation_sql(kind):
    return f'({missing_translation_sql(kind)}) OR {changed_source_sql(kind)}'


def cooldown_fingerprint_sql(kind):
    if kind == 'research':
        return "encode(sha256(convert_to(markdown, 'UTF8')), 'hex')"
    return source_hash_sql(kind)
