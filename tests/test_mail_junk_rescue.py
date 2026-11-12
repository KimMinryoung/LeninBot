"""allowlist_sender deletes a Junk message only after its copy to INBOX succeeded."""
import asyncio
import unittest
from unittest.mock import patch

from mail_runtime import tools


class FakeImap:
    def __init__(self, copy_status):
        self.copy_status = copy_status
        self.stored, self.expunged, self.logged_out = [], False, False

    def select(self, folder):
        return "OK", [b"2"]

    def search(self, charset, criterion):
        return "OK", [b"1 2"]

    def fetch(self, mid, spec):
        sender = b"friend@example.org" if mid == b"1" else b"spam@example.com"
        return "OK", [(b"1 (RFC822.HEADER)", b"From: " + sender + b"\r\n\r\n")]

    def copy(self, mid, folder):
        return self.copy_status, [b""]

    def store(self, mid, op, flags):
        self.stored.append(mid)

    def expunge(self):
        self.expunged = True

    def logout(self):
        self.logged_out = True


class JunkRescueTests(unittest.TestCase):
    def run_tool(self, conn):
        with patch.object(tools.imap, "connect", return_value=conn):
            return str(asyncio.run(tools.exec_allowlist_sender("friend@")))

    def test_successful_copy_moves_only_matching_mail(self):
        conn = FakeImap("OK")
        out = self.run_tool(conn)
        self.assertIn("Moved 1", out)
        self.assertEqual(conn.stored, [b"1"])
        self.assertTrue(conn.expunged and conn.logged_out)

    def test_failed_copy_keeps_the_message(self):
        conn = FakeImap("NO")
        out = self.run_tool(conn)
        self.assertIn("Moved 0", out)
        self.assertIn("left in Junk", out)
        self.assertEqual(conn.stored, [])
        self.assertFalse(conn.expunged)
        self.assertTrue(conn.logged_out)


if __name__ == "__main__":
    unittest.main()
