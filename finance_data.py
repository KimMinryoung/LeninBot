"""finance_data.py — Financial market data via yfinance.

Provides get_finance_data tool with:
- Preset assets (gold, silver, DXY, oil, S&P500, US10Y, KOSPI)
- Arbitrary yfinance tickers (stocks, crypto, forex, etc.)
- Historical data with period selection
- News headlines for queried assets

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
    "kospi": "^KS11",
}

LABELS = {
    "gold": "금(USD/oz)",
    "silver": "은(USD/oz)",
    "dxy": "달러지수(DXY)",
    "wti": "WTI유(USD/bbl)",
    "brent": "브렌트유(USD/bbl)",
    "sp500": "S&P 500",
    "us10y": "미국채10Y(%)",
    "kospi": "KOSPI",
}

CACHE_TTL = 600  # 10 minutes

_cache: dict = {"data": {}, "fetched_at": 0.0}


def _get_yf():
    """Lazy import yfinance."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None


def fetch_finance_data(symbols: list[str] | None = None) -> dict:
    """Fetch current prices for preset assets. Returns cached data if fresh.

    Args:
        symbols: List of preset keys (e.g. ["gold", "dxy"]). None = all.

    Returns:
        Dict mapping symbol name to {price, change_pct, ticker, fetched_at}.
    """
    now = time.time()
    if _cache["data"] and (now - _cache["fetched_at"]) < CACHE_TTL:
        data = _cache["data"]
        if symbols:
            return {k: v for k, v in data.items() if k in symbols}
        return dict(data)

    yf = _get_yf()
    if not yf:
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


def fetch_custom_tickers(ticker_symbols: list[str]) -> dict:
    """Fetch current prices for arbitrary yfinance tickers.

    Args:
        ticker_symbols: List of yfinance ticker strings (e.g. ["AAPL", "BTC-USD"]).

    Returns:
        Dict mapping ticker to {price, change_pct, fetched_at}.
    """
    yf = _get_yf()
    if not yf:
        return {"error": "yfinance not installed"}

    from datetime import datetime, timezone
    result = {}
    for sym in ticker_symbols[:10]:  # cap at 10 to avoid abuse
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            price = info["last_price"]
            prev_close = info.get("previous_close") or info.get("regularMarketPreviousClose")
            change_pct = None
            if prev_close and prev_close > 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)
            result[sym] = {
                "price": round(price, 4),
                "change_pct": change_pct,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.warning("Failed to fetch custom ticker %s: %s", sym, e)
            result[sym] = {"price": None, "error": str(e)}
    return result


def fetch_history(ticker_symbol: str, period: str) -> dict:
    """Fetch historical OHLCV data for a single ticker.

    Args:
        ticker_symbol: yfinance ticker string.
        period: One of 1d, 5d, 1mo, 3mo, 6mo, 1y.

    Returns:
        Dict with summary stats and recent data points.
    """
    yf = _get_yf()
    if not yf:
        return {"error": "yfinance not installed"}

    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(period=period)
        if df.empty:
            return {"error": f"No data for {ticker_symbol} (period={period})"}

        first_close = df["Close"].iloc[0]
        last_close = df["Close"].iloc[-1]
        period_return = round((last_close - first_close) / first_close * 100, 2)

        summary = {
            "ticker": ticker_symbol,
            "period": period,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "open": round(float(df["Close"].iloc[0]), 2),
            "close": round(float(last_close), 2),
            "high": round(float(df["High"].max()), 2),
            "low": round(float(df["Low"].min()), 2),
            "period_return_pct": period_return,
            "data_points": len(df),
        }

        # Include last 5 data points for detail
        tail = df.tail(5)
        recent = []
        for idx, row in tail.iterrows():
            recent.append({
                "date": str(idx.date()),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if row["Volume"] > 0 else None,
            })
        summary["recent"] = recent
        return summary
    except Exception as e:
        logger.warning("History fetch failed for %s: %s", ticker_symbol, e)
        return {"error": str(e)}


def fetch_news(ticker_symbol: str, max_items: int = 5) -> list[dict]:
    """Fetch recent news for a ticker.

    Returns:
        List of dicts with title, publisher, link.
    """
    yf = _get_yf()
    if not yf:
        return [{"error": "yfinance not installed"}]

    try:
        t = yf.Ticker(ticker_symbol)
        news = t.news
        if not news:
            return []
        items = []
        for item in news[:max_items]:
            items.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
            })
        return items
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", ticker_symbol, e)
        return [{"error": str(e)}]


def search_ticker(query: str, max_results: int = 5) -> list[dict]:
    """Search for yfinance ticker symbols by name.

    Args:
        query: Search term (e.g. "Samsung Electronics", "bitcoin", "Tesla").

    Returns:
        List of dicts with symbol, name, exchange, type.
    """
    yf = _get_yf()
    if not yf:
        return [{"error": "yfinance not installed"}]

    try:
        result = yf.Search(query)
        items = []
        for q in result.quotes[:max_results]:
            items.append({
                "symbol": q.get("symbol", ""),
                "name": q.get("longname") or q.get("shortname", ""),
                "exchange": q.get("exchDisp", ""),
                "type": q.get("typeDisp", ""),
            })
        return items
    except Exception as e:
        logger.warning("Ticker search failed for %r: %s", query, e)
        return [{"error": str(e)}]


# ── Tool Definition (Anthropic API format) ────────────────────────────

FINANCE_TOOL = {
    "name": "get_finance_data",
    "description": (
        "Get financial market data. Preset assets (gold, silver, DXY, WTI/Brent oil, "
        "S&P 500, US 10Y yield, KOSPI) or any yfinance ticker (stocks, crypto, forex). "
        "Use 'query' to search by name when you don't know the ticker symbol. "
        "Supports historical period data and news headlines."
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
                "description": "Preset asset keys. Omit for all presets.",
            },
            "custom_tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Any yfinance ticker symbols (e.g. AAPL, BTC-USD, EURUSD=X, 005930.KS). "
                    "Max 10."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Search by company/asset name (e.g. 'Samsung Electronics', 'bitcoin'). "
                    "Resolves to ticker automatically and fetches data for the top match."
                ),
            },
            "period": {
                "type": "string",
                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                "description": "Historical period. Omit for current price only.",
            },
            "include_news": {
                "type": "boolean",
                "description": "Include recent news headlines for queried assets. Default false.",
            },
        },
        "required": [],
    },
}


async def _exec_finance_data(
    symbols: list[str] | None = None,
    custom_tickers: list[str] | None = None,
    query: str | None = None,
    period: str | None = None,
    include_news: bool = False,
) -> str:
    """Async handler for the get_finance_data tool."""
    sections = []

    # --- Resolve query to ticker via search ---
    resolved_from_query: list[tuple[str, str]] = []  # (display_name, yf_symbol)
    if query:
        results = await asyncio.to_thread(search_ticker, query)
        if results and "error" not in results[0]:
            # Show search results for transparency
            search_lines = ["🔍 Search results for: " + query]
            for r in results:
                search_lines.append(
                    f"  {r['symbol']} — {r['name']} ({r['exchange']}, {r['type']})"
                )
            sections.append("\n".join(search_lines))
            # Use top match
            top = results[0]
            resolved_from_query.append((
                f"{top['name']} ({top['symbol']})",
                top["symbol"],
            ))
        else:
            err = results[0].get("error", "no results") if results else "no results"
            sections.append(f"🔍 Search for '{query}': {err}")

    # --- Resolve which tickers to work with ---
    all_tickers: list[tuple[str, str]] = []  # (display_name, yf_symbol)

    # Preset assets
    has_specific = custom_tickers or query
    if has_specific:
        # If specific tickers given, skip presets unless symbols explicitly set
        if symbols:
            for key in symbols:
                if key in TICKERS:
                    all_tickers.append((LABELS.get(key, key), TICKERS[key]))
    else:
        # Default: fetch preset assets
        preset_keys = symbols or list(TICKERS.keys())
        for key in preset_keys:
            if key in TICKERS:
                all_tickers.append((LABELS.get(key, key), TICKERS[key]))

    # Custom tickers
    if custom_tickers:
        for sym in custom_tickers[:10]:
            all_tickers.append((sym, sym))

    # Resolved from query
    all_tickers.extend(resolved_from_query)

    if not all_tickers:
        return "Error: No valid tickers specified."

    # --- Current prices ---
    if not period:
        # Split all_tickers into presets (use cached batch) vs custom (individual)
        preset_keys = [k for k in (symbols or list(TICKERS.keys()))
                       if k in TICKERS and not has_specific or (symbols and k in symbols)]
        custom_syms = [sym for _, sym in all_tickers if sym not in TICKERS.values()]

        lines = []
        # Preset batch fetch (cached)
        if preset_keys and not has_specific:
            data = await asyncio.to_thread(fetch_finance_data, preset_keys if symbols else None)
            if "error" not in data:
                for key in preset_keys:
                    entry = data.get(key, {})
                    label = LABELS.get(key, key)
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
        elif preset_keys and symbols:
            data = await asyncio.to_thread(fetch_finance_data, symbols)
            if "error" not in data:
                for key in symbols:
                    entry = data.get(key, {})
                    label = LABELS.get(key, key)
                    price = entry.get("price")
                    if price is None:
                        continue
                    change = entry.get("change_pct")
                    if change is not None:
                        sign = "+" if change >= 0 else ""
                        lines.append(f"  {label}: {price:,.2f} ({sign}{change}%)")
                    else:
                        lines.append(f"  {label}: {price:,.2f}")

        # Custom/query tickers (individual fetch)
        if custom_syms:
            custom_data = await asyncio.to_thread(fetch_custom_tickers, custom_syms)
            for sym in custom_syms:
                entry = custom_data.get(sym, {})
                if isinstance(entry, dict) and entry.get("price") is not None:
                    price = entry["price"]
                    change = entry.get("change_pct")
                    # Find display name from all_tickers
                    display = next((d for d, s in all_tickers if s == sym), sym)
                    if change is not None:
                        sign = "+" if change >= 0 else ""
                        lines.append(f"  {display}: {price:,.2f} ({sign}{change}%)")
                    else:
                        lines.append(f"  {display}: {price:,.2f}")
                else:
                    display = next((d for d, s in all_tickers if s == sym), sym)
                    err = entry.get("error", "?") if isinstance(entry, dict) else "?"
                    lines.append(f"  {display}: N/A ({err})")

        if lines:
            sections.append("📊 Market Data\n" + "\n".join(lines))

    # --- Historical data ---
    if period:
        history_lines = []
        for display_name, yf_sym in all_tickers:
            hist = await asyncio.to_thread(fetch_history, yf_sym, period)
            if "error" in hist:
                history_lines.append(f"\n  {display_name}: {hist['error']}")
                continue
            ret = hist["period_return_pct"]
            sign = "+" if ret >= 0 else ""
            history_lines.append(
                f"\n  {display_name} ({hist['start_date']} → {hist['end_date']})"
                f"\n    시가: {hist['open']:,.2f}  종가: {hist['close']:,.2f}"
                f"\n    고가: {hist['high']:,.2f}  저가: {hist['low']:,.2f}"
                f"\n    기간수익률: {sign}{ret}%  ({hist['data_points']} data points)"
            )
            # Recent data points
            for pt in hist.get("recent", []):
                vol = f"  vol={pt['volume']:,}" if pt.get("volume") else ""
                history_lines.append(
                    f"    {pt['date']}: O={pt['open']:,.2f} H={pt['high']:,.2f} "
                    f"L={pt['low']:,.2f} C={pt['close']:,.2f}{vol}"
                )
        sections.append(f"📈 Historical ({period})" + "\n".join(history_lines))

    # --- News ---
    if include_news:
        news_lines = []
        seen_titles = set()
        for display_name, yf_sym in all_tickers:
            items = await asyncio.to_thread(fetch_news, yf_sym)
            for item in items:
                if "error" in item:
                    continue
                title = item.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                publisher = item.get("publisher", "")
                link = item.get("link", "")
                source = f" ({publisher})" if publisher else ""
                news_lines.append(f"  • {title}{source}")
                if link:
                    news_lines.append(f"    {link}")
        if news_lines:
            sections.append("📰 News\n" + "\n".join(news_lines))
        else:
            sections.append("📰 News: No recent articles found.")

    return "\n\n".join(sections)


FINANCE_TOOL_HANDLER = _exec_finance_data
