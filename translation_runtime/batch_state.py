"""Disposable cooldown records for repeated validation failures only."""
from __future__ import annotations

import json
import time
from pathlib import Path

from translation_runtime import TranslationCallError, TranslationProviderError
from translation_runtime.storage import atomic_write, source_hash

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'output' / 'translation_failures'


class BatchState:
    def __init__(self, path=DEFAULT_PATH, *, now=time.time):
        self.path = Path(path)
        self.now = now

    def _file(self, key):
        return self.path / (source_hash(key) + '.json')

    def get(self, key):
        try:
            record = json.loads(self._file(key).read_text())
            return record if isinstance(record, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def deferred(self, key, source):
        record = self.get(key)
        return record.get('source') == source and self.now() < record.get('retry_after', 0)

    def failed(self, key, source, error):
        if not isinstance(error, TranslationCallError) or isinstance(error, TranslationProviderError):
            # Network/authentication/quota/runtime failures retry next scheduled run.
            self.succeeded(key, source)
            return
        previous = self.get(key)
        count = min(3, previous.get('count', 0) + 1) if previous.get('source') == source else 1
        atomic_write(self._file(key), json.dumps({
            'item': key, 'source': source, 'count': count,
            'retry_after': self.now() + 48 * 3600 if count >= 3 else 0,
        }))

    def succeeded(self, key, source):
        self._file(key).unlink(missing_ok=True)
