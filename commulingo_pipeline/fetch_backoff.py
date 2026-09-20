"""Job-local, durable backoff for explicit fetch failures, never evidence."""
import asyncio
import hashlib
import re
import time
from urllib.parse import urldefrag
from tool_gateway.results import ToolFailure

TTL = {'http_forbidden':1800, 'http_rate_limited':300, 'origin_tcp_timeout':600,
       'origin_connection_timeout':600, 'origin_server_error':300,
       'tls_or_certificate_error':1800, 'dns_resolution_failed':600,
       'origin_connection_refused':600, 'anti_bot_challenge':1800}


class FetchBackoff:
    def __init__(self, store, job, usage, artifacts):
        self.store, self.job, self.usage = store, job, usage
        saved = next((a['value'].get('failures', {}) for a in reversed(artifacts)
                      if a['stage']=='fetch_failures'), {})
        self.failures = {k:v for k,v in saved.items() if v['until'] > time.time()}
        self.locks = {}
        self.save_lock = asyncio.Lock()

    async def save(self):
        async with self.save_lock:
            self.failures = dict(sorted(((k,v) for k,v in self.failures.items() if v['until'] > time.time()),
                                        key=lambda kv:kv[1]['until'], reverse=True)[:64])
            await asyncio.to_thread(self.store.save_fetch_failures, self.job,
                                    {'failures':dict(self.failures)})

    def wrap(self, name, call):
        if name != 'fetch_url':
            return call
        async def fetch(**args):
            key = hashlib.sha256(urldefrag(args['url'])[0].encode()).hexdigest()
            async with self.locks.setdefault(key, asyncio.Lock()):
                failure = self.failures.get(key)
                if failure and failure['until'] > time.time():
                    self.usage.tracker['fetch_backoff_hits'] = self.usage.tracker.get('fetch_backoff_hits',0)+1
                    return ToolFailure(f"Fetch temporarily deferred after {failure['reason']}; retry in "
                        f"{max(1,int(failure['until']-time.time()))} seconds or choose another source. "
                        'No page body was retrieved; this is not evidence. Changing offsets or use_cache does not bypass backoff.')
                result = await call(**args)
                match = re.search(r'Fetch diagnosis: ([a-z_]+)', str(result)) if isinstance(result,ToolFailure) else None
                reason = match.group(1) if match else None
                if reason in TTL:
                    self.failures[key] = {'reason':reason, 'until':time.time()+TTL[reason]}
                    self.usage.tracker['fetch_failures'] = self.usage.tracker.get('fetch_failures',0)+1
                    await self.save()
                elif key in self.failures:
                    del self.failures[key]
                    await self.save()
                return result
        return fetch
