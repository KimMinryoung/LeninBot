"""A slug taken between _unique_slug's read and the INSERT gets the next suffix."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import psycopg2.errors

from publishing import site_publishing as sp


class HubSlugRaceTests(unittest.IsolatedAsyncioTestCase):
    async def publish(self, insert_side_effect, taken):
        def lookup(sql, params):
            if sql.lstrip().startswith("SELECT 1"):
                return {"x": 1} if params[0] in taken else None
            return insert_side_effect(params)

        with patch.object(sp, "_ensure_hub_table"), \
             patch.object(sp, "db_query_one", side_effect=lookup), \
             patch.object(sp, "db_execute"), \
             patch.object(sp, "is_autonomous_publication_context", return_value=False), \
             patch.object(sp, "check_autonomous_publication_allowed", return_value=(True, "")), \
             patch.object(sp, "review_autonomous_publication", new=AsyncMock(return_value="")), \
             patch.object(sp, "_wrap_external_fetch", new=AsyncMock(return_value=None)), \
             patch.object(sp, "maybe_broadcast_autonomous_publication",
                          new=AsyncMock(return_value=SimpleNamespace(ok=False))), \
             patch.object(sp, "record_autonomous_publication"):
            return await sp._exec_publish_hub_curation(
                title="T", source_url="https://example.org/a", selection_rationale="r",
                context="c", slug="foo")

    async def test_collision_at_insert_moves_to_next_suffix(self):
        taken = set()

        def insert(params):
            if params[0] == "foo" and "foo" not in taken:
                taken.add("foo")  # a concurrent publish committed "foo" first
                raise psycopg2.errors.UniqueViolation()
            return {"id": 7, "published_at": "now"}

        result = await self.publish(insert, taken)
        self.assertIn("Slug: foo-2", result)
        self.assertIn("https://cyber-lenin.com/hub/foo-2", result)

    async def test_persistent_collision_returns_an_error(self):
        def insert(params):
            raise psycopg2.errors.UniqueViolation()

        result = await self.publish(insert, set())
        self.assertTrue(result.startswith("Error: slug"))


if __name__ == "__main__":
    unittest.main()
