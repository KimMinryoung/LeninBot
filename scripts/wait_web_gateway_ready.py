#!/usr/bin/env python3
"""Wait for key and budget-store readiness; never call a paid provider."""
import json
import sys
import time
from urllib.request import ProxyHandler, build_opener


def main():
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with opener.open("http://127.0.0.1:8111/health", timeout=1) as response:
                if response.status == 200 and json.load(response).get("status") == "ok":
                    return 0
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    print("Web gateway readiness failed (credentials, policy or store unavailable)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
