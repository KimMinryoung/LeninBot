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


class FakeImap:
    def __init__(self):
        self.selected = None

    def select(self, folder, readonly=True):
        self.selected = folder
        return "OK", [b""]

    def uid(self, command, *args):
        if command == "search":
            assert args == (None, "ALL"), args
            return "OK", [b"216"]
        if command == "fetch":
            uid, query = args
            assert uid in (b"216", "216"), uid
            assert query == "(FLAGS BODY.PEEK[])", query
            header = b"216 (UID 216 FLAGS (\\Seen) BODY[] {3000}"
            return "OK", [(header, RAW_EMAIL)]
        raise AssertionError(f"unexpected uid command: {command}")

    def logout(self):
        return "BYE", [b""]


class FakeChronologicalImap(FakeImap):
    def uid(self, command, *args):
        if command == "search":
            return "OK", [b"244 259"]
        if command == "fetch":
            uid, query = args
            assert query == "(FLAGS BODY.PEEK[])", query
            if uid == b"244":
                return "OK", [(b"244 (UID 244 FLAGS () BODY[] {6}", OLDER_EMAIL)]
            if uid == b"259":
                return "OK", [(b"259 (UID 259 FLAGS () BODY[] {6}", NEWER_EMAIL)]
        raise AssertionError(f"unexpected uid command: {command} {args}")


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

        registry._imap_connect = lambda: FakeBrokenImap()
        broken = await registry._exec_check_inbox(limit=2, include_body=False)
        assert broken.startswith("Error: IMAP connected, but mailbox checks failed"), broken
    finally:
        registry._imap_connect = original_imap_connect

    print("ok")


if __name__ == "__main__":
    asyncio.run(_main())
