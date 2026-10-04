"""db.py falls back to the read-only login only outside a service."""

import os
import tempfile
import unittest
from unittest.mock import patch

import db

_ENV = {"DB_HOST": "127.0.0.1", "DB_USER": "postgres", "DB_NAME": "leninbot"}


class ReadonlyLoginTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, db, "_pool", None)
        db._pool = None
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        tmp.write("ro-secret\n")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        self.pw_file = tmp.name

    def _pool_kwargs(self, env: dict, password: str | None):
        env = {**_ENV, "DB_RO_PASSWORD_FILE": self.pw_file, **env}
        with patch.dict(os.environ, env), \
                patch.object(db, "get_secret", return_value=password), \
                patch.object(db.pool, "ThreadedConnectionPool") as tcp:
            for key in ("INVOCATION_ID", "LENINBOT_SERVICE", "LENINBOT_ALLOW_WRITE"):
                if key not in env:
                    os.environ.pop(key, None)
            db._get_pool()
        return tcp.call_args.kwargs

    def test_outside_service_without_password_uses_readonly_role(self):
        kw = self._pool_kwargs({}, None)
        self.assertEqual((kw["user"], kw["password"]), ("leninbot_ro", "ro-secret"))
        self.assertIn("default_transaction_read_only=on", kw["options"])

    def test_service_never_uses_readonly_role(self):
        with self.assertRaisesRegex(RuntimeError, "DB_PASSWORD"):
            self._pool_kwargs({"INVOCATION_ID": "x"}, None)

    def test_explicit_password_wins(self):
        kw = self._pool_kwargs({}, "main-secret")
        self.assertEqual((kw["user"], kw["password"]), ("postgres", "main-secret"))

    def test_missing_file_keeps_clear_error(self):
        os.unlink(self.pw_file)
        open(self.pw_file, "w").close()  # recreate empty for cleanup
        with self.assertRaisesRegex(RuntimeError, "leninbot_ro"):
            self._pool_kwargs({}, None)


if __name__ == "__main__":
    unittest.main()
