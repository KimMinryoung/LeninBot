#!/usr/bin/env python3
"""Wait until the local LLM proxy has credentials and accepts traffic."""

from __future__ import annotations

import json
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    deadline = time.monotonic() + 20.0
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urlopen("http://127.0.0.1:8110/health", timeout=1.0) as response:
                payload = json.load(response)
            if response.status == 200 and payload.get("status") == "ok":
                return 0
            last_error = str(payload)
        except (OSError, URLError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    print(f"LLM proxy did not become ready: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
