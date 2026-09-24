"""Compatibility alias: the module lives in llm.skills_loader.

Importing or patching `skills_loader` reaches the same module object.
"""
import sys

from llm import skills_loader as _module

sys.modules[__name__] = _module
