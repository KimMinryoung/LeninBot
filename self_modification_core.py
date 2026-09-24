"""Compatibility shim: the implementation lives in self_runtime.self_modification_core."""

from self_runtime.self_modification_core import *  # noqa: F401,F403
from self_runtime.self_modification_core import (  # noqa: F401
    backup_file,
    git_backup_before_modification,
    git_reset_to_commit,
    restore_backup,
    run_sandbox_tests,
    self_modify_with_safety,
)
