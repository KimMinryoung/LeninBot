"""Compatibility alias: the module lives in ops.audit_sink.

Importing or patching `audit_sink` reaches the same module object.
"""
import sys

from ops import audit_sink as _module

sys.modules[__name__] = _module
