"""Compatibility alias: the module lives in memory_store.redis_state.

Importing or patching `redis_state` reaches the same module object.
"""
import sys

from memory_store import redis_state as _module

sys.modules[__name__] = _module
