"""Reuse the existing persistent fetch cache; never concatenate source pages."""
import asyncio
import re
from datetime import datetime, timezone

from provenance.runtime import external_body
from runtime_tools.commulingo_review_policy import external_url
from .evidence import Passages, snapshot, MAX_SNAPSHOT_CHARS


class Sources:
    def __init__(self, store, job, usage, sources):
        self.store, self.job, self.usage = store, job, usage
        self.sources = sources
        self.passages = Passages()
        self.requests = []

    @classmethod
    async def load(cls, store, job, usage, checkpoint=None):
        sources = await asyncio.to_thread(store.job_sources, job['id'])
        session = cls(store, job, usage, sources)
        if checkpoint:
            session.passages.restore(checkpoint.get('passages', {}), sources)
            session.requests = checkpoint.get('source_requests', [])
        return session

    def display(self, source):
        if source['expires_at'] <= datetime.now(timezone.utc) or not source.get('body'):
            raise ValueError('source expired; retrieve the page again')
        if len(source['body']) > MAX_SNAPSHOT_CHARS:
            raise ValueError('source page is too large; request a smaller page')
        self.sources[source['id']] = source
        labelled = self.passages.show(source['id'], source['body'])
        return (f"URL: {source['url']}\nRetrieved: {source['fetched_at']}\n"
                'Cite the immutable passage labels shown before the paragraphs.\n'
                '<external source="pipeline-source">\n' + labelled + '\n</external>')

    def context(self):
        return {'available_pages': [{'source_id':s['id'],'url':s['url'],'fetched_at':s['fetched_at'],
                                     'expires_at':s['expires_at'],
                                     'passage_labels':[label for label,entry in self.passages.shown.items() if entry[0]==s['id']]}
                                    for s in self.sources.values() if s.get('body')],
                'previous_fetches': self.requests[-30:]}

    def cached_tool(self):
        async def read(passages):
            now = datetime.now(timezone.utc)
            output = []
            for label in passages:
                if label not in self.passages.shown:
                    raise ValueError(f'unknown cached passage: {label}')
                source_id, start, end = self.passages.shown[label]
                source = self.sources.get(source_id)
                if not source or not source.get('body') or source['expires_at'] <= now:
                    raise ValueError(f'{label} expired or unavailable; fetch its original again')
                self.passages.resolve([label],lambda sid:self.sources[sid]['body'])
                output.append(f"[{label}] URL: {source['url']}\n" + source['body'][start:end])
            return '<external source="pipeline-cache">\n'+'\n\n'.join(output)+'\n</external>'
        return ({'name':'commulingo_pipeline_cached_passages',
            'description':'Read original text for existing P-labels from the job cache without network access. Labels and retrieval timestamps are unchanged.',
            'input_schema':{'type':'object','additionalProperties':False,
                'properties':{'passages':{'type':'array','minItems':1,'maxItems':8,
                    'items':{'type':'string','pattern':'^P[1-9][0-9]*$'}}},'required':['passages']}},read,False)

    def wrap(self, name, call):
        async def fetched(**args):
            if name not in {'fetch_url','wiki_get'}:
                return await call(**args)
            cached = None if args.get('use_cache') is False else await asyncio.to_thread(self.store.cached_source, name, args)
            if cached:
                self.usage.tracker['pipeline_cache_hits'] = self.usage.tracker.get('pipeline_cache_hits', 0) + 1
                page = cached
            else:
                raw = await call(**args)
                text = str(raw)
                body = external_body(text)
                urls = [args.get('url')] if name == 'fetch_url' else re.findall(r'https?://[^\s<>\]"\)]+', text[:1000])
                url = next((u for u in urls if isinstance(u,str) and external_url(u)), None)
                if not body or not url:
                    return raw
                if len(body[1]) > MAX_SNAPSHOT_CHARS:
                    raise ValueError('source page is too large; request a smaller page')
                page = snapshot(url, body[1])
                await asyncio.to_thread(self.store.save_source, page)
                await asyncio.to_thread(self.store.cache_source, name, args, page['id'])
            await asyncio.to_thread(self.store.link_source, self.job['id'], page['id'])
            request = {'tool':name, 'args':args, 'source_id':page['id']}
            if request not in self.requests:
                self.requests.append(request)
            return self.display(page)
        return fetched
