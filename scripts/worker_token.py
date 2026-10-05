#!/usr/bin/env python3
"""Make a worker MCP client token (dev_docs/agent_worker.md).

Usage: scripts/worker_token.py <client>
Prints the token once on stdout (give it to the client, e.g. the frontend's
.env COMMULINGO_WORKER_TOKEN) and the WORKER_MCP_CLIENTS entry on stderr
(add it to leninbot .env; only the hash is kept here).
"""
import hashlib
import re
import secrets
import sys

if len(sys.argv) != 2 or not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", sys.argv[1]):
    sys.exit("usage: scripts/worker_token.py <client>  (lowercase name)")
token = secrets.token_urlsafe(32)
print(f"WORKER_MCP_CLIENTS entry: {sys.argv[1]}:{hashlib.sha256(token.encode()).hexdigest()}", file=sys.stderr)
print(token)
