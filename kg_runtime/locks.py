"""Host-local serialization for sync runs and document replacement.

All KG consumers run as grass on this host. Locks live in ignored data/ so
systemd PrivateTmp and ad-hoc processes share the same lock namespace.
"""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import fcntl


@contextmanager
def kg_write_lock(key: str):
    directory = Path(__file__).resolve().parents[1] / 'data' / 'kg_locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (sha256(key.encode()).hexdigest() + '.lock')).open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
