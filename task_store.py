"""Compatibility alias: the module lives in telegram.task_store.

Importing or patching `task_store` reaches the same module object.
"""
import sys

from telegram import task_store as _module

sys.modules[__name__] = _module
