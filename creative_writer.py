"""Compatibility shim for the personal fiction workspace.

The implementation moved to the ``writer`` package (writer/store.py,
writer/documents.py, writer/models.py, writer/prompts.py, writer/tools.py,
writer/runs.py, writer/stream.py). This module re-exports the public API so
existing imports (api.py routes, scripts/schema_migrations.py) keep working.
Import from ``writer`` directly in new code.
"""

from writer import *  # noqa: F401,F403
from writer import __all__  # noqa: F401
