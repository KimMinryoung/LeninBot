#!/usr/bin/env python3
"""Smoke test for check_inbox UID reads and body pagination."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_tools.registry as registry


SOURCE = "A" * 1000 + "B" * 1000 + "C" * 1000
RAW_EMAIL = (
    "From: sender@example.test\n"
    "Subject: Long body\n"
    "Date: Mon, 01 Jan 2024 00:00:00 +0000\n"
    "Content-Type: text/plain; charset=utf-8\n"
    "\n"
    + SOURCE
).encode("utf-8")

OLDER_EMAIL = (
    "From: older@example.test\n"
    "Subject: Older lexical trap\n"
    "Date: Wed, 8 Jul 2026 18:11:40 +0000\n"
    "Content-Type: text/plain; charset=utf-8\n\nolder"
).encode("utf-8")

NEWER_EMAIL = (
    "From: newer@example.test\n"
    "Subject: Newer chronological message\n"
    "Date: Wed, 15 Jul 2026 09:03:31 +0000\n"
    "Content-Type: text/plain; charset=utf-8\n\nnewer"
).encode("utf-8")

# Newer than anything in the fake INBOX, so a listing that drops it is a listing
# that never reached Junk.
JUNK_EMAIL = (
    "From: spam@example.test\n"
    "Subject: Junk folder message\n"
    "Date: Mon, 20 Jul 2026 11:00:00 +0000\n"
    "Content-Type: text/plain; charset=utf-8\n\njunk"
).encode("utf-8")


class FakeImap:
    """One-message INBOX. Junk exists and is empty, as a real mailbox may be.

    The fakes here are folder-aware on purpose: the earlier versions served the
    same messages whatever was selected, which is not how IMAP works and which
    hid the folder bugs this test now covers.
    """

    MESSAGES = {"INBOX": {b"216": RAW_EMAIL}, "Junk": {}}

    def __init__(self):
        self.selected = None

    def select(self, folder, readonly=True):
        assert folder in self.MESSAGES, f"selected an unknown folder: {folder}"
        self.selected = folder
        return "OK", [b""]

    def _box(self):
        return self.MESSAGES[self.selected]

    def uid(self, command, *args):
        box = self._box()
        if command == "search":
            assert args in ((None, "ALL"), (None, "UNSEEN")), args
            return "OK", [b" ".join(box) if box else b""]
        if command == "fetch":
            uid, query = args
            assert query == "(FLAGS BODY.PEEK[])", query
            key = uid if isinstance(uid, bytes) else str(uid).encode()
            if key not in box:
                return "NO", [None]
            n = key.decode()
            header = f"{n} (UID {n} FLAGS (\\Seen) BODY[] {{3000}}".encode()
            return "OK", [(header, box[key])]
        raise AssertionError(f"unexpected uid command: {command}")

    def logout(self):
        return "BYE", [b""]


class FakeChronologicalImap(FakeImap):
    MESSAGES = {"INBOX": {b"244": OLDER_EMAIL, b"259": NEWER_EMAIL}, "Junk": {}}


class FakeFolderedImap(FakeImap):
    """INBOX and Junk holding different mail, the newest of it in Junk."""

    MESSAGES = {
        "INBOX": {b"244": OLDER_EMAIL, b"259": NEWER_EMAIL},
        "Junk": {b"17": JUNK_EMAIL},
    }


class FakeBrokenImap(FakeImap):
    def select(self, folder, readonly=True):
        return "NO", [b"mailbox unavailable"]


async def _main() -> None:
    original_imap_connect = registry._imap_connect
    registry._imap_connect = lambda: FakeImap()
    try:
        inbox_list = await registry._exec_check_inbox(limit=1, body_max_chars=1000)
        assert "Folder: INBOX | UID: 216" in inbox_list, inbox_list
        assert "returned_chars=0:1000" in inbox_list, inbox_list
        assert "next: check_inbox(folder='INBOX', uid='216', body_offset=1000" in inbox_list, inbox_list

        inbox_page = await registry._exec_check_inbox(
            folder="INBOX", uid="216", body_offset=1000, body_max_chars=1000
        )
        assert "Folder: INBOX | UID: 216" in inbox_page, inbox_page
        assert "returned_chars=1000:2000" in inbox_page, inbox_page
        assert "B" * 80 in inbox_page, inbox_page
        assert "A" * 80 not in inbox_page, inbox_page

        registry._imap_connect = lambda: FakeChronologicalImap()
        chronological = await registry._exec_check_inbox(limit=2, include_body=False)
        newer_position = chronological.index("Newer chronological message")
        older_position = chronological.index("Older lexical trap")
        assert newer_position < older_position, chronological

        # `folder` used to be read only by the uid branch, so a listing call
        # asking for Junk was answered with INBOX and the caller could not tell.
        registry._imap_connect = lambda: FakeFolderedImap()
        junk_only = await registry._exec_check_inbox(folder="Junk", limit=5, include_body=False)
        assert "Folder: Junk | UID: 17" in junk_only, junk_only
        assert "Folder: INBOX" not in junk_only, junk_only
        assert "[JUNK]" in junk_only, junk_only

        inbox_only = await registry._exec_check_inbox(folder="INBOX", limit=5, include_body=False)
        assert "Folder: INBOX | UID: 259" in inbox_only, inbox_only
        assert "Folder: Junk" not in inbox_only, inbox_only

        both = await registry._exec_check_inbox(limit=5, include_body=False)
        assert "Folder: INBOX | UID: 259" in both, both
        assert "Folder: Junk | UID: 17" in both, both

        # The limit is a per-folder budget. When it was shared, INBOX was walked
        # first, filled it, and Junk broke out on its first candidate — so the
        # newest message on the account was unreachable whenever INBOX held
        # `limit` of its own.
        budgeted = await registry._exec_check_inbox(limit=1, include_body=False)
        assert "Folder: Junk | UID: 17" in budgeted, budgeted
        assert "Folder: INBOX" not in budgeted, budgeted

        # An unrecognized folder is an error, not a silent INBOX listing.
        unknown = await registry._exec_check_inbox(folder="Spam", limit=1)
        assert unknown.startswith("Error: unknown folder 'Spam'"), unknown

        registry._imap_connect = lambda: FakeBrokenImap()
        broken = await registry._exec_check_inbox(limit=2, include_body=False)
        assert broken.startswith("Error: IMAP connected, but mailbox checks failed"), broken
    finally:
        registry._imap_connect = original_imap_connect

    print("ok")


if __name__ == "__main__":
    asyncio.run(_main())
