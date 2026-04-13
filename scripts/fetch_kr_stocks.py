"""Fetch Korean stock listings from NAVER Finance and save to data/kr_stocks.json.

Run periodically (e.g. weekly) to keep the mapping fresh:
    python scripts/fetch_kr_stocks.py

Output format (data/kr_stocks.json):
{
  "fetched_at": "2026-04-13T...",
  "count": 4254,
  "stocks": {
    "삼성전자":   {"code": "005930", "market": "KOSPI"},
    "SK하이닉스": {"code": "000660", "market": "KOSPI"},
    ...
  }
}
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests

API_BASE = "https://m.stock.naver.com/api/stocks/marketValue"
PAGE_SIZE = 100
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "kr_stocks.json")


def fetch_market(market: str) -> list[dict]:
    """Fetch all stocks for a market (KOSPI or KOSDAQ) via pagination."""
    stocks = []
    page = 1
    while True:
        url = f"{API_BASE}/{market}?page={page}&pageSize={PAGE_SIZE}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("stocks", [])
        if not batch:
            break
        for s in batch:
            name = s.get("stockName", "").strip()
            code = s.get("itemCode", "").strip()
            if name and code:
                stocks.append({"name": name, "code": code, "market": market})
        total = data.get("totalCount", 0)
        if page * PAGE_SIZE >= total:
            break
        page += 1
    return stocks


def main():
    print("Fetching KOSPI listings...")
    kospi = fetch_market("KOSPI")
    print(f"  KOSPI: {len(kospi)} stocks")

    print("Fetching KOSDAQ listings...")
    kosdaq = fetch_market("KOSDAQ")
    print(f"  KOSDAQ: {len(kosdaq)} stocks")

    # Build name → {code, market} mapping
    # On duplicate names, prefer by market cap order (NAVER returns by market cap desc)
    mapping = {}
    for s in kospi + kosdaq:
        name = s["name"]
        if name not in mapping:
            mapping[name] = {"code": s["code"], "market": s["market"]}

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(mapping),
        "stocks": mapping,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    print(f"\nSaved {len(mapping)} stocks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
