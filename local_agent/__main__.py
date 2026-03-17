"""Entry point: python -m local_agent"""

import asyncio
import sys
import os

# Ensure project root is on sys.path so we can import shared, db, self_tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from local_agent.cli import main

if __name__ == "__main__":
    asyncio.run(main())
