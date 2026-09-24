"""Compatibility alias: the module lives in llm.prompt_context.

Importing or patching `prompt_context` reaches the same module object.
"""
import sys

from llm import prompt_context as _module

sys.modules[__name__] = _module
