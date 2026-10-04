import unittest

from services.web_chat_store import chat_identity_clause


class ChatIdentityClauseTest(unittest.TestCase):
    def test_account_also_owns_unstamped_rows_of_its_fingerprints(self):
        clause, params = chat_identity_clause(16, ["fp-a", "", "fp-b"])
        self.assertEqual(clause, "(user_id = %s OR (user_id IS NULL AND fingerprint = ANY(%s)))")
        self.assertEqual(params, [16, ["fp-a", "fp-b"]])

    def test_account_without_fingerprints_and_column_prefix(self):
        self.assertEqual(chat_identity_clause(16, []), ("user_id = %s", [16]))
        clause, _ = chat_identity_clause(16, ["fp"], "l.")
        self.assertIn("l.user_id IS NULL AND l.fingerprint", clause)

    def test_anonymous_owns_its_fingerprints(self):
        self.assertEqual(chat_identity_clause(None, ["fp"]), ("fingerprint = ANY(%s)", [["fp"]]))


if __name__ == "__main__":
    unittest.main()
