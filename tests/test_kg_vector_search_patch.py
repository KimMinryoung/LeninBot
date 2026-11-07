"""Semantic KG search uses the Neo4j vector indexes instead of a full cosine scan."""
import unittest
from types import SimpleNamespace

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

from graph_memory import graphiti_patches as gp

gp.apply_graphiti_patches()
from graphiti_core.search import search as search_mod  # noqa: E402  (patched bindings)


class _Driver:
    search_interface = None
    provider = GraphProvider.NEO4J

    def __init__(self):
        self.calls = []

    async def execute_query(self, query, **params):
        self.calls.append((query, params))
        return [], None, None


class VectorSearchPatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_edge_search_queries_index_and_keeps_filters(self):
        driver = _Driver()
        filters = SearchFilters(expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]])
        await search_mod.edge_similarity_search(driver, [0.1], None, None, filters, ["documents"], 10, 0.6)
        query, params = driver.calls[0]
        self.assertIn("db.index.vector.queryRelationships('edge_fact_embedding'", query)
        self.assertIn("e.expired_at IS NULL", query)
        self.assertIn("e.group_id IN $group_ids", query)
        self.assertNotIn("vector.similarity.cosine", query)
        self.assertEqual(params["k"], 200)
        self.assertAlmostEqual(params["index_min_score"], 0.8)

    async def test_node_search_queries_index_with_label_filter(self):
        driver = _Driver()
        await search_mod.node_similarity_search(driver, [0.1], SearchFilters(node_labels=["Person"]), None, 30, 0.6)
        query, params = driver.calls[0]
        self.assertIn("db.index.vector.queryNodes('entity_name_embedding'", query)
        self.assertIn("n:Person", query)
        self.assertEqual(params["k"], 600)

    async def test_endpoint_anchored_edge_search_keeps_exact_scan(self):
        driver = _Driver()
        await search_mod.edge_similarity_search(driver, [0.1], "src", None, SearchFilters(), ["g"], 10, 0.6)
        self.assertNotIn("db.index.vector", driver.calls[0][0])

    def test_k_is_bounded(self):
        self.assertEqual(gp._vector_k(1), gp.VECTOR_SEARCH_MIN_K)
        self.assertEqual(gp._vector_k(10_000), gp.VECTOR_SEARCH_MAX_K)


if __name__ == "__main__":
    unittest.main()
