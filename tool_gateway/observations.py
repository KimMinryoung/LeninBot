"""Optional, context-local observation of pre-handler argument rejections."""

from contextvars import ContextVar
import logging

argument_rejection_observer = ContextVar("argument_rejection_observer", default=None)


def observe_argument_rejection(name, args, error):
    observer = argument_rejection_observer.get()
    if observer is not None:
        try:
            observer(name, args, str(error))
        except Exception:
            logging.getLogger(__name__).warning("Argument rejection observer failed", exc_info=True)


# Completed tool calls (name, args, result text, is_error). A caller that wants
# a per-run record of tool outcomes sets a callable for the duration of its run;
# None (the default) disables observation, including for nested agent runs that
# reset it.
tool_result_observer = ContextVar("tool_result_observer", default=None)


def observe_tool_result(name, args, result, is_error):
    observer = tool_result_observer.get()
    if observer is not None:
        try:
            observer(name, args, result, bool(is_error))
        except Exception:
            logging.getLogger(__name__).warning("Tool result observer failed", exc_info=True)
