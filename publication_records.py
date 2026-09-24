"""Compatibility alias: the module lives in runtime_tools.publication_records.

Importing or patching `publication_records` reaches the same module object.
"""
import sys

from runtime_tools import publication_records as _module

sys.modules[__name__] = _module
