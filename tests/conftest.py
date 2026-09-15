"""Keep pytest's synthetic LLM usage out of the production audit ledger."""

import os

# Apply before test collection imports clients; focused pytest runs must be as
# isolated as scripts/run_unit_tests.sh. Sink tests explicitly mock their writes.
os.environ["LENINBOT_LLM_AUDIT_DB"] = "0"
