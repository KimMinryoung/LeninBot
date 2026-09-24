"""Review risk flags and the tool-handler wrapper shared by the standalone
reviewer (``scripts/commulingo_person_reviewer.py``) and the staged pipeline."""
import re

from runtime_tools.commulingo_review_policy import (
    DECISION_TOOL, validate_decision, external_url, review_source, resolve_review_checks,
)


def review_risks(row, current):
    fields = row['patch_json'] or {}
    reasons = set(fields.get('reviewFlags') or [])
    reasons.update(r.strip() for r in (row.get('review_note') or '').split(',') if r.strip())
    if row['action'] == 'delete': reasons.add('deletion')
    if any(e.get('stance') == 'disputes' for e in fields.get('evidence') or []): reasons.add('source_conflict')
    before = current or {}
    checked_fields = ['definition','body'] if row['target_type']=='term' else ['bio']
    if row['target_type'] == 'person_section':
        before = next((s for s in before.get('sections', []) if s['slug'] == fields.get('slug')), {})
        checked_fields = ['body']
    for field in checked_fields:
        for lang in ('ko','en'):
            old = (before.get(field) or {}).get(lang) or ''
            proposed = fields.get(field)
            value = '' if field in fields and proposed is None else proposed.get(lang) if isinstance(proposed,dict) else proposed if lang=='ko' and isinstance(proposed,str) else None
            if isinstance(value,str) and len(old)>=120 and len(value)<len(old)*0.6: reasons.add('large_deletion')
    return sorted(reasons)


def make_handlers(read_handlers, proposal, snapshots, box, gate=None, triage=None):
    """``snapshots`` collects the slices this review retrieves (review_source);
    ``gate(value)`` (async) may annotate the resolved decision or raise
    ValueError to send it back to the reviewer before it is boxed;
    ``triage(text)`` (async) sees each rendered web_search result (shadow)."""
    from tool_gateway.results import ToolRejection
    from commulingo_pipeline.evidence import Passages
    from provenance.runtime import external_body
    handlers = {}
    passages = Passages()
    for name, handler in read_handlers.items():
        def wrap(tool_name, call):
            async def wrapped(**kwargs):
                result = await call(**kwargs)
                text = str(result)
                if tool_name == 'web_search' and triage is not None:
                    await triage(text)
                body = external_body(text)
                if tool_name in {'fetch_url','wiki_get'} and body and len(body[1])>20:
                    urls = [kwargs.get('url')] if tool_name=='fetch_url' else re.findall(r'https?://[^\s<>\]"\)]+', text[:1000])
                    for url in urls:
                        if isinstance(url,str) and external_url(url):
                            source_id, labelled = review_source(url, body[1], snapshots, passages, base=int(kwargs.get('offset') or 0))
                            return (f'Review source_id={source_id}; each paragraph below starts with its immutable '
                                    'passage label; a check cites the labels actually shown.\n'
                                    + text[:body.start(1)] + labelled + text[body.end(1):])
                return result
            return wrapped
        handlers[name] = wrap(name,handler)
    async def decide(**value):
        if box: raise ToolRejection('a decision has already been submitted')
        try:
            value = resolve_review_checks(value, proposal, snapshots, passages)
            validate_decision(value, proposal)
            if gate is not None:
                value = await gate(value)
        except ValueError as exc: raise ToolRejection(str(exc)) from exc
        box.update(value)
        return 'OK: review decision recorded; no dictionary write was made by this tool.'
    handlers[DECISION_TOOL['name']] = decide
    return handlers
