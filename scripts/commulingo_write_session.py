"""Compatibility re-export; the implementation lives in ``commulingo.pipeline.write_session``."""
from commulingo.pipeline.write_session import draft_id, prepare_write, repair_schema

__all__ = ["draft_id", "prepare_write", "repair_schema"]
