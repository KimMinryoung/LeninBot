"""Scheduled tasks survive a restart near their cron time without duplicates."""

import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from shared import KST
from telegram import tasks


class ScheduleTests(unittest.TestCase):
    def test_recent_missed_fire_is_due_but_old_fire_is_not(self):
        now = datetime(2026, 9, 24, 14, 20, tzinfo=KST)
        sched = {"cron_expr": "0 2,14 * * *", "last_run_at": now - timedelta(days=1),
                 "created_at": now - timedelta(days=30)}
        self.assertEqual(tasks._schedule_due_fire(sched, now).hour, 14)
        self.assertIsNone(tasks._schedule_due_fire(sched, now + timedelta(hours=3)))
        sched["last_run_at"] = now
        self.assertIsNone(tasks._schedule_due_fire(sched, now))

    def test_worker_creates_scheduled_task_and_records_occurrence(self):
        now = datetime.now(KST)
        sched = {"id": 9, "user_id": 7, "agent_type": "diary",
                 "content": "[diary] Write a periodic diary entry", "cron_expr": "* * * * *",
                 "last_run_at": now - timedelta(minutes=3),
                 "created_at": now - timedelta(days=1)}
        created = []
        updates = []
        sleeps = []

        async def inline(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def sleep(_):
            sleeps.append(1)
            if len(sleeps) > 1:
                raise asyncio.CancelledError

        async def send_message(self, **_):
            return None

        def query(sql, *args):
            return [sched]

        def execute(sql, params):
            updates.append(params)

        def create(*args, **kwargs):
            created.append(kwargs)
            return {"status": "ok", "task_id": 42}

        with patch.object(tasks.asyncio, "to_thread", inline), \
             patch.object(tasks.asyncio, "sleep", sleep), \
             patch.object(tasks, "_query", query), \
             patch.object(tasks, "_query_one", return_value=None), \
             patch.object(tasks, "_execute", execute), \
             patch.object(tasks, "create_task_in_db", create):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(tasks.schedule_worker(type("Bot", (), {"send_message": send_message})(),
                                                  allowed_user_ids={7}))
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["metadata"]["schedule_id"], 9)
        self.assertIn("scheduled_for", created[0]["metadata"])
        self.assertEqual(len(updates), 1)
