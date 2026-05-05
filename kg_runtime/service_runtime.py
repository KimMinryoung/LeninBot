"""Knowledge graph service singleton and dedicated event-loop runtime."""

import asyncio
import logging
import threading
import time
from concurrent.futures import Future

logger = logging.getLogger(__name__)

# Neo4j driver binds Futures to the event loop that created it,
# so we must reuse a single persistent loop for all KG operations.
_kg_service = None
_kg_init_cooldown = 0.0  # monotonic timestamp when retry is allowed
_KG_RETRY_INTERVAL = 120  # seconds before retrying after init failure
_kg_lock = threading.Lock()
_kg_loop = None


_kg_run_lock = threading.Lock()
_kg_loop_thread = None
_kg_loop_ready = threading.Event()
_KG_LOOP_START_TIMEOUT = 10
_KG_TRANSIENT_KEYWORDS = (
    "defunct connection",
    "incompletecommit",
    "read timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection refused",
    "failed to read",
    "dns",
    "neo4j",
)


def _is_transient_kg_error(err) -> bool:
    s = str(err).lower()
    return any(k in s for k in _KG_TRANSIENT_KEYWORDS)


def _mark_kg_unhealthy(reason: str = ""):
    """Mark KG singleton unhealthy so next call re-initializes after cooldown."""
    global _kg_service, _kg_init_cooldown
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = time.monotonic() + _KG_RETRY_INTERVAL
    if reason:
        logger.warning("[KG] marked unhealthy (retry in %ds): %s", _KG_RETRY_INTERVAL, reason)


def _kg_loop_exception_handler(loop, context):
    """Handle background async errors from Graphiti/Neo4j tasks."""
    exc = context.get("exception")
    msg = context.get("message", "")
    combined = f"{msg} | {exc}" if exc else msg
    if _is_transient_kg_error(combined):
        logger.warning("[KG loop] transient async exception: %s", combined)
        _mark_kg_unhealthy(combined)
        return
    logger.error("[KG loop] unhandled async exception: %s", combined, exc_info=exc)


def _kg_loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    """Background worker that owns the persistent KG event loop."""
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_kg_loop_exception_handler)
    _kg_loop_ready.set()
    loop.run_forever()


def _ensure_kg_loop() -> asyncio.AbstractEventLoop:
    """Create/start the dedicated KG event loop thread once and return the loop."""
    global _kg_loop, _kg_loop_thread
    with _kg_run_lock:
        if _kg_loop is not None and not _kg_loop.is_closed() and _kg_loop_thread and _kg_loop_thread.is_alive():
            return _kg_loop

        if _kg_loop is not None and not _kg_loop.is_closed():
            try:
                _kg_loop.call_soon_threadsafe(_kg_loop.stop)
            except Exception:
                pass
            try:
                _kg_loop.close()
            except Exception:
                pass

        _kg_loop_ready.clear()
        _kg_loop = asyncio.new_event_loop()
        _kg_loop_thread = threading.Thread(
            target=_kg_loop_worker,
            args=(_kg_loop,),
            daemon=True,
            name="kg-event-loop",
        )
        _kg_loop_thread.start()

    if not _kg_loop_ready.wait(timeout=_KG_LOOP_START_TIMEOUT):
        raise RuntimeError("KG event loop thread failed to start")
    return _kg_loop


def _submit_to_kg_loop(coro) -> Future:
    """Submit a coroutine to the KG loop and return its Future (non-blocking).

    Low-level helper — prefer submit_kg_task() or run_kg_task() instead.
    """
    loop = _ensure_kg_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


def _wait_kg_future(future: Future):
    """Block until a KG Future resolves; mark unhealthy on transient errors."""
    try:
        return future.result()
    except Exception as e:
        if _is_transient_kg_error(e):
            _mark_kg_unhealthy(str(e))
        raise


async def _run_kg_task(async_fn, *args, **kwargs):
    """Internal KG-loop trampoline that creates and awaits the coroutine in-loop."""
    return await async_fn(*args, **kwargs)


def run_kg_task(async_fn, *args, **kwargs):
    """Create and run an async callable entirely on the dedicated KG loop (blocking).

    This prevents cross-event-loop contamination when the async callable uses
    Graphiti/Neo4j objects that were initialized on the KG loop thread.
    Blocks the calling thread until the result is ready.
    """
    future = _submit_to_kg_loop(_run_kg_task(async_fn, *args, **kwargs))
    return _wait_kg_future(future)


def submit_kg_task(async_fn, *args, **kwargs) -> Future:
    """Submit an async callable to the KG loop and return a Future (non-blocking).

    Same safety as run_kg_task (coroutine created on the KG loop), but the
    caller can collect multiple Futures and wait on them in parallel.

    Usage:
        futures = [submit_kg_task(svc.ingest_episode, ...) for art in articles]
        results = collect_kg_futures(futures)
    """
    return _submit_to_kg_loop(_run_kg_task(async_fn, *args, **kwargs))


def collect_kg_futures(futures: list[Future], timeout: float = 120) -> list[dict]:
    """Wait for multiple KG Futures and return results.

    Returns a list of dicts: {"ok": True, "result": ...} or {"ok": False, "error": ...}.
    Transient errors mark the KG service unhealthy.
    """
    results = []
    for f in futures:
        try:
            results.append({"ok": True, "result": _wait_kg_future(f)})
        except Exception as e:
            results.append({"ok": False, "error": str(e)})
    return results


# Keep run_kg_async as a thin wrapper for backward compat (internal use only)
def run_kg_async(coro):
    """Run a pre-built coroutine on the KG loop (blocking).

    WARNING: the coroutine must be created on the KG loop to avoid cross-loop
    errors. Prefer run_kg_task(async_fn, *args, **kwargs) for safety.
    """
    return _wait_kg_future(_submit_to_kg_loop(coro))


def get_kg_service():
    """Lazy singleton for GraphMemoryService (Neo4j/Graphiti).

    Retries after _KG_RETRY_INTERVAL seconds if init previously failed
    (e.g. AuraDB was paused and later resumed).
    """
    import time

    global _kg_service, _kg_init_cooldown
    if _kg_service is not None:
        return _kg_service
    if time.monotonic() < _kg_init_cooldown:
        return None
    with _kg_lock:
        if _kg_service is not None:
            return _kg_service
        if time.monotonic() < _kg_init_cooldown:
            return None
        try:
            def _build_service():
                from graph_memory.service import GraphMemoryService
                return GraphMemoryService()

            svc = _build_service()
            run_kg_task(svc.initialize)
            _kg_service = svc
            logger.info("[KG] init succeeded")
            return svc
        except Exception as e:
            _kg_init_cooldown = time.monotonic() + _KG_RETRY_INTERVAL
            logger.warning("[KG] init failed (retry in %ds): %s", _KG_RETRY_INTERVAL, e)
            return None


def reset_kg_service():
    """Reset the KG singleton so next get_kg_service() retries initialization.

    Call this when KG operations fail due to connection issues (e.g. AuraDB paused).
    """
    global _kg_service, _kg_init_cooldown, _kg_loop, _kg_loop_thread
    with _kg_lock:
        _kg_service = None
        _kg_init_cooldown = 0.0
    with _kg_run_lock:
        if _kg_loop is not None and not _kg_loop.is_closed():
            try:
                _kg_loop.call_soon_threadsafe(_kg_loop.stop)
            except Exception as e:
                logger.debug("[KG] loop stop skipped: %s", e)
            _kg_loop = None
        _kg_loop_thread = None
        _kg_loop_ready.clear()
    logger.info("[KG] service reset — will retry on next access")

# ── KG Health Check ──────────────────────────────────────────────────
_kg_healthcheck_started = False


def start_kg_healthcheck(interval: int = 300) -> None:
    """Start a background daemon thread that pings Neo4j every `interval` seconds.

    If the ping fails the KG singleton is marked unhealthy so the next
    get_kg_service() call triggers a fresh re-initialization.
    Called once from api.py / telegram_bot.py lifespan.
    """
    global _kg_healthcheck_started
    if _kg_healthcheck_started:
        return
    _kg_healthcheck_started = True

    def _checker():
        while True:
            time.sleep(interval)
            svc = _kg_service  # read without lock — snapshot
            if svc is None:
                # Already unhealthy; get_kg_service() will retry on next real request
                logger.debug("[KG healthcheck] service is None, skipping ping")
                continue
            try:
                run_kg_task(svc._graphiti.driver.execute_query, "RETURN 1")
                logger.debug("[KG healthcheck] ping OK")
            except Exception as e:
                logger.warning("[KG healthcheck] ping failed — marking unhealthy: %s", e)
                _mark_kg_unhealthy(str(e))

    t = threading.Thread(target=_checker, daemon=True, name="kg-healthcheck")
    t.start()
    logger.info("[KG healthcheck] started (interval=%ds)", interval)



