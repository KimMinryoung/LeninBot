"""Compatibility alias: the module lives in runtime_tools.site_publishing.

Importing or patching `site_publishing` reaches the same module object.
"""
import sys

from runtime_tools import site_publishing as _module

sys.modules[__name__] = _module
