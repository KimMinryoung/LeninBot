"""finance_data.py — Real-time financial market data via yfinance.

On-demand fetching with 10-minute in-memory cache.
No separate collector service needed — the bot process is long-running.

Integration in telegram_tools.py:
    from finance_data import FINANCE_TOOL, FINANCE_TOOL_HANDLER
    TOOLS.append(FINANCE_TOOL)
    TOOL_HANDLERS["get_finance_data"] = FINANCE_TOOL_HANDLER
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "wti": "CL=F",
    "brent": "BZ=F",
    "sp500": "^GSPC",
    "us10y": "^TNX",
}

LABELS = {
    "gold": "금(USD/oz)",
    "silver": "은(USD/oz)",
    "dxy": "달러지수(DXY)",
    "wti": "WTI유(USD/bbl)",
    "brent": "브렌트유(USD/bbl)",
    "sp500": "S&P 500",
    "us10y": "미국채10Y(%)",
}

CACHE_TTL = 600  # 10 minutes

_cache: dict = {"data": {}, "fetched_at": 0.0}


def fetch_finance_data(symbols: list[str] | None = None) -> dict:
    """Fetch market data via yfinance. Returns cached data if fresh enough.

    Args:
        symbols: List of asset keys (e.g. ["gold", "dxy"]). None = all.

    Returns:
        Dict mapping symbol name to {price, change_pct, ticker, fetched_at}.
    """
    now = time.time()
    if _cache["data"] and (now - _cache["fetched_at"]) < CACHE_TTL:
        data = _cache["data"]
        if symbols:
            return {k: v for k, v in data.items() if k in symbols}
        return dict(data)

    # Lazy import — yfinance is slow to import (~1-2s)
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}

    ticker_str = " ".join(TICKERS.values())
    try:
        tickers = yf.Tickers(ticker_str)
    except Exception as e:
        logger.error("yfinance Tickers init failed: %s", e)
        if _cache["data"]:
            logger.warning("Returning stale cache (age: %.0fs)", now - _cache["fetched_at"])
            data = _cache["data"]
            if symbols:
                return {k: v for k, v in data.items() if k in symbols}
            return dict(data)
        return {"error": f"yfinance failed: {e}"}

    from datetime import datetime, timezone
    result = {}
    for name, ticker_symbol in TICKERS.items():
        try:
            t = tickers.tickers[ticker_symbol]
            info = t.fast_info
            price = info["last_price"]
            prev_close = info.get("previous_close") or info.get("regularMarketPreviousClose")
            change_pct = None
            if prev_close and prev_close > 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)
            result[name] = {
                "price": round(price, 2),
                "change_pct": change_pct,
                "ticker": ticker_symbol,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.warning("Failed to fetch %s (%s): %s", name, ticker_symbol, e)
            result[name] = {"price": None, "error": str(e)}

    _cache["data"] = result
    _cache["fetched_at"] = time.time()
    logger.info("Finance data refreshed: %s", {k: v.get("price") for k, v in result.items()})

    if symbols:
        return {k: v for k, v in result.items() if k in symbols}
    return result


def finance_summary() -> str:
    """One-line summary string for prompt injection."""
    data = fetch_finance_data()
    if "error" in data:
        return f"금융 데이터 조회 실패: {data['error']}"
    if not data:
        return "금융 데이터 없음"

    parts = []
    for key, label in LABELS.items():
        entry = data.get(key, {})
        price = entry.get("price")
        if price is None:
            continue
        change = entry.get("change_pct")
        if change is not None:
            sign = "+" if change >= 0 else ""
            parts.append(f"{label}: {price:,.2f} ({sign}{change}%)")
        else:
            parts.append(f"{label}: {price:,.2f}")

    fetched = data.get("gold", {}).get("fetched_at", "")
    ts = fetched[:16].replace("T", " ") + " UTC" if fetched else "?"
    return ", ".join(parts) + f"  [{ts}]"


# ── Tool Definition (Anthropic API format) ────────────────────────────

FINANCE_TOOL = {
    "name": "get_finance_data",
    "description": (
        "Get real-time financial market data: gold, silver, DXY, WTI/Brent oil, "
        "S&P 500, US 10Y yield. Cached for 10 min. Use for market analysis, "
        "economic discussions, or any task requiring current prices."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(TICKERS.keys()),
                },
                "description": "Specific assets to fetch. Omit for all.",
            },
        },
        "required": [],
    },
}


async def _exec_finance_data(symbols: list[str] | None = None) -> str:
    """Async handler for the get_finance_data tool."""
    data = await asyncio.to_thread(fetch_finance_data, symbols)

    if "error" in data:
        return f"Error: {data['error']}"

    lines = []
    for key, label in LABELS.items():
        entry = data.get(key)
        if not entry:
            continue
        price = entry.get("price")
        if price is None:
            lines.append(f"  {label}: N/A ({entry.get('error', '?')})")
            continue
        change = entry.get("change_pct")
        if change is not None:
            sign = "+" if change >= 0 else ""
            lines.append(f"  {label}: {price:,.2f} ({sign}{change}%)")
        else:
            lines.append(f"  {label}: {price:,.2f}")

    fetched = next((v.get("fetched_at", "") for v in data.values() if v.get("fetched_at")), "")
    ts = fetched[:16].replace("T", " ") + " UTC" if fetched else "?"
    header = f"📊 Market Data [{ts}]"
    return header + "\n" + "\n".join(lines)


FINANCE_TOOL_HANDLER = _exec_finance_data
