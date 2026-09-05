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
