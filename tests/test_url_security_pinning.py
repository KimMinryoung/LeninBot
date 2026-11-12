"""safe_requests_get dials the addresses it validated; DNS is never asked twice."""
import http.server
import socket
import threading
import unittest
from unittest.mock import Mock, patch

import requests

from content_fetch import url_security
from content_fetch.url_security import UnsafeUrlError, _pinned_get, safe_requests_get


def resolver_for(answers):
    calls = []

    def resolve(host, port, **kwargs):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answers[host], port))]
    return resolve, calls


class PinningTests(unittest.TestCase):
    def test_connects_to_the_validated_address_without_a_second_lookup(self):
        resolve, calls = resolver_for({"example.org": "93.184.216.34"})
        dialed = []

        def refuse(address, *args, **kwargs):
            dialed.append(address)
            raise OSError("refused in test")

        def no_dns(*args, **kwargs):
            raise AssertionError("requests resolved the hostname again")

        with patch("urllib3.util.connection.create_connection", refuse), \
                patch("socket.getaddrinfo", no_dns):
            with self.assertRaises(requests.ConnectionError):
                safe_requests_get("https://example.org/a", resolver=resolve, timeout=5)
        self.assertEqual(calls, ["example.org"])
        self.assertEqual(dialed, [("93.184.216.34", 443)])

    def test_private_answer_is_refused_before_any_connection(self):
        resolve, _ = resolver_for({"rebind.example": "10.0.0.5"})
        with patch("urllib3.util.connection.create_connection") as dial:
            with self.assertRaises(UnsafeUrlError):
                safe_requests_get("http://rebind.example/", resolver=resolve)
        dial.assert_not_called()

    def test_redirect_hop_is_resolved_and_pinned_again(self):
        resolve, calls = resolver_for({"a.example": "93.184.216.34", "b.example": "10.1.2.3"})
        redirect = Mock(status_code=302, headers={"Location": "http://b.example/"})
        with patch.object(url_security, "_pinned_get", return_value=redirect) as get:
            with self.assertRaises(UnsafeUrlError):
                safe_requests_get("http://a.example/", resolver=resolve)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.args[1:3], ("a.example", ["93.184.216.34"]))
        self.assertEqual(calls, ["a.example", "b.example"])
        redirect.close.assert_called_once()

    def test_pinned_request_keeps_the_hostname_in_the_host_header(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["host"] = self.headers["Host"]
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with _pinned_get(f"http://site.example:{port}/", "site.example", ["127.0.0.1"], timeout=5) as response:
                self.assertEqual(response.text, "ok")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(seen["host"], f"site.example:{port}")

    def test_a_host_outside_the_pin_is_refused(self):
        with self.assertRaises(requests.ConnectionError):
            _pinned_get("http://other.example/", "site.example", ["93.184.216.34"], timeout=5)


if __name__ == "__main__":
    unittest.main()
