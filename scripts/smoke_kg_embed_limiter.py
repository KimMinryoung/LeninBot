"""Smoke test — Phase 6 KG embedding resilience (client limiter + key separation).

Hermetic: no network, no Neo4j, no real Gemini key. Exercises
graph_memory.service._EmbedRateLimiter pacing, the retry wrapper's
limiter integration, and _resolve_kg_gemini_key fallback order.

Run: venv/bin/python scripts/smoke_kg_embed_limiter.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def main():
    from graph_memory import service as svc

    # ── 1) Limiter pacing ────────────────────────────────────────────
    async def run_n(limiter, n):
        t0 = time.monotonic()
        waits = [await limiter.acquire() for _ in range(n)]
        return time.monotonic() - t0, waits

    os.environ["KG_EMBED_MAX_RPS"] = "10"
    elapsed, waits = asyncio.run(run_n(svc._EmbedRateLimiter(), 5))
    # 5 requests at 10 rps: slots at 0, .1, .2, .3, .4 → ~0.4s total
    check("pacing: 5 acquires @10rps take ~0.4s", 0.35 <= elapsed <= 0.7, f"elapsed={elapsed:.3f}s")
    check("pacing: first acquire immediate", waits[0] == 0.0, f"waits={['%.3f' % w for w in waits]}")
    check("pacing: later acquires waited", all(w > 0 for w in waits[1:]))

    os.environ["KG_EMBED_MAX_RPS"] = "0"
    elapsed, waits = asyncio.run(run_n(svc._EmbedRateLimiter(), 20))
    check("disabled: KG_EMBED_MAX_RPS=0 never sleeps", elapsed < 0.05 and all(w == 0.0 for w in waits),
          f"elapsed={elapsed:.4f}s")

    os.environ["KG_EMBED_MAX_RPS"] = "not-a-number"
    check("bad env value falls back to default 2/s", svc._EmbedRateLimiter._rate() == 2.0)

    # Rate change picked up without recreating the limiter (no restart needed)
    limiter = svc._EmbedRateLimiter()
    os.environ["KG_EMBED_MAX_RPS"] = "0"
    asyncio.run(limiter.acquire())
    os.environ["KG_EMBED_MAX_RPS"] = "50"
    elapsed, _ = asyncio.run(run_n(limiter, 3))
    check("rate re-read per acquire (env hot-tune)", 0.02 <= elapsed <= 0.3, f"elapsed={elapsed:.3f}s")

    # ── 2) Retry wrapper integrates the limiter per attempt ─────────
    os.environ["KG_EMBED_MAX_RPS"] = "20"
    os.environ["KG_EMBED_RETRY_DELAYS"] = "0.05,0.05"

    class _Probe:
        pass

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return "ok"

    result = asyncio.run(svc.RetryingGeminiEmbedder._with_retry(_Probe(), "create", flaky))
    check("retry wrapper: retryable 429s retried then succeed", result == "ok" and calls["n"] == 3,
          f"attempts={calls['n']}")

    calls["n"] = 0

    async def fatal():
        calls["n"] += 1
        raise RuntimeError("400 INVALID_ARGUMENT")

    try:
        asyncio.run(svc.RetryingGeminiEmbedder._with_retry(_Probe(), "create", fatal))
        check("retry wrapper: non-retryable error raises immediately", False)
    except RuntimeError:
        check("retry wrapper: non-retryable error raises immediately", calls["n"] == 1,
              f"attempts={calls['n']}")

    # ── 3) Key resolution: KG_GEMINI_API_KEY > GEMINI_API_KEY ───────
    import secrets_loader

    saved = {k: os.environ.get(k) for k in ("KG_GEMINI_API_KEY", "GEMINI_API_KEY", "CREDENTIALS_DIRECTORY")}
    try:
        os.environ.pop("CREDENTIALS_DIRECTORY", None)  # force env-var path in secrets_loader

        def resolve_fresh():
            # get_secret is lru_cached for the process lifetime (rotation
            # implies restart in production); clear it so each scenario
            # below sees its own env state.
            secrets_loader.get_secret.cache_clear()
            return svc._resolve_kg_gemini_key()

        os.environ["GEMINI_API_KEY"] = "main-key"
        os.environ.pop("KG_GEMINI_API_KEY", None)
        check("key: falls back to GEMINI_API_KEY", resolve_fresh() == "main-key")

        os.environ["KG_GEMINI_API_KEY"] = "kg-key"
        check("key: KG_GEMINI_API_KEY wins when set", resolve_fresh() == "kg-key")

        os.environ["KG_GEMINI_API_KEY"] = "   "
        check("key: blank KG key treated as unset", resolve_fresh() == "main-key")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print()
    print("=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
