#!/usr/bin/env python3
"""Show durable audit rows awaiting delivery to the DB sink."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.audit_sink import spool_stats

print(json.dumps(spool_stats(), sort_keys=True))
