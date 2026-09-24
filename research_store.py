"""Compatibility alias: the module lives in runtime_tools.research_store.

Importing or patching `research_store` reaches the same module object.
"""
import sys

from runtime_tools import research_store as _module

sys.modules[__name__] = _module
