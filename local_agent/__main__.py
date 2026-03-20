"""Entry point: python -m local_agent"""

import asyncio
import sys
import os

# Ensure project root is on sys.path so we can import shared, db, self_tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
else:
    print(
        "[local_agent] Optional dependency `python-dotenv` not installed; skipping .env load.",
        file=sys.stderr,
    )

from local_agent.cli import main

if __name__ == "__main__":
    asyncio.run(main())
