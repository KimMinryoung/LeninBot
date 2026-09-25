"""The vendored CommuLingo contracts must match the frontend originals."""
import unittest

from ops.paths import CONTRACT_FILES, FRONTEND_DIR, VENDORED_CONTRACTS_DIR
from scripts.sync_commulingo_contracts import drifted


class ContractCopyTest(unittest.TestCase):
    def test_every_contract_has_a_vendored_copy(self):
        for name in CONTRACT_FILES:
            self.assertTrue((VENDORED_CONTRACTS_DIR / name).is_file(), name)

    @unittest.skipUnless(FRONTEND_DIR.exists(), "frontend checkout not present")
    def test_copies_match_frontend(self):
        self.assertEqual(drifted(), [], "run scripts/sync_commulingo_contracts.py and commit")


if __name__ == "__main__":
    unittest.main()
