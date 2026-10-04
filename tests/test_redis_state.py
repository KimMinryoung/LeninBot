"""Hermetic tests for the Redis connection guard and the PG mission board."""

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from redis.exceptions import ConnectionError as RedisConnectionError

from memory_store import redis_state
from telegram import mission


class RedisCircuitTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, redis_state, "_redis_client", None)
        self.addCleanup(setattr, redis_state, "_down_until", 0.0)
        redis_state._redis_client = None
        redis_state._down_until = 0.0

    def test_client_built_once_without_per_call_ping(self):
        client = Mock()
        with patch.dict(os.environ), \
                patch.object(redis_state.redis.Redis, "from_url", return_value=client) as from_url:
            os.environ.pop("REDIS_URL", None)
            first = redis_state.get_redis()
            second = redis_state.get_redis()
        self.assertIs(first, second)
        from_url.assert_called_once()
        self.assertEqual(client.ping.call_count, 1)
        self.assertTrue(from_url.call_args.args[0].endswith("/1"))  # database 0 is the frontend's

    def test_connect_failure_skips_redis_during_cooldown(self):
        with patch.object(redis_state.redis.Redis, "from_url", side_effect=RedisConnectionError("refused")) as from_url:
            self.assertIsNone(redis_state.get_redis())
            self.assertIsNone(redis_state.get_redis())
            self.assertFalse(redis_state.redis_available())
        from_url.assert_called_once()

    def test_command_error_opens_circuit(self):
        client = Mock()
        client.rpush.side_effect = RedisConnectionError("gone")
        with patch.object(redis_state.redis.Redis, "from_url", return_value=client):
            self.assertFalse(redis_state.push_owner_alert("x"))
            self.assertIsNone(redis_state.get_redis())

    def test_cooldown_expiry_reuses_client(self):
        client = Mock()
        with patch.object(redis_state.redis.Redis, "from_url", return_value=client) as from_url:
            r = redis_state.get_redis()
            redis_state._open_circuit(RuntimeError("blip"))
            self.assertIsNone(redis_state.get_redis())
            redis_state._down_until = 0.0
            self.assertIs(redis_state.get_redis(), r)
        from_url.assert_called_once()


class MissionBoardTests(unittest.TestCase):
    def test_post_reports_closed_mission(self):
        with patch.object(mission, "_execute_rowcount", return_value=0):
            self.assertFalse(mission.post_agent_message(3, 9, "scout", "hi"))
        with patch.object(mission, "_execute_rowcount", return_value=1) as ex:
            self.assertTrue(mission.post_agent_message(3, 9, "scout", "x" * 5000))
        params = ex.call_args.args[1]
        self.assertEqual(params[0], "scout#9")
        self.assertEqual(len(params[2]), 2000)

    def test_read_parses_source_oldest_first(self):
        t = datetime(2026, 10, 4, tzinfo=timezone.utc)
        rows = [
            {"source": "analyst#12", "content": "second", "created_at": t},
            {"source": "scout#9", "content": "first", "created_at": t},
        ]
        with patch.object(mission, "_query", return_value=rows):
            messages = mission.read_agent_messages(3)
        self.assertEqual([m["message"] for m in messages], ["first", "second"])
        self.assertEqual((messages[0]["agent"], messages[0]["task_id"]), ("scout", "9"))
        self.assertEqual(messages[0]["ts"], t.timestamp())

    def test_timeline_excludes_board_messages(self):
        with patch.object(mission, "_query", return_value=[]) as q:
            mission.get_mission_events(3)
        self.assertIn("event_type <>", q.call_args.args[0])
        self.assertIn("agent_message", q.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
