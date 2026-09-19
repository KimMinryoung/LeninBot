"""The proxy forwards client headers minus hop-by-hop, auth and internal ones."""
import unittest

from llm_proxy.app import build_forward_headers


class ForwardHeaderTests(unittest.TestCase):
    def test_internal_caller_and_auth_headers_never_reach_the_provider(self):
        incoming = {"authorization": "Bearer via-llm-proxy", "x-llm-caller": "commulingo_person_classification",
                    "content-type": "application/json", "host": "127.0.0.1:8110", "x-api-key": "old"}
        out = build_forward_headers(incoming, {"auth": ("bearer",)}, "real-key")
        self.assertEqual(out["authorization"], "Bearer real-key")
        self.assertEqual(out["content-type"], "application/json")
        for name in ("x-llm-caller", "host", "x-api-key"):
            self.assertNotIn(name, out)


if __name__ == "__main__":
    unittest.main()
