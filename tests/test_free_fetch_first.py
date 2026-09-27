"""Free retrieval must precede the keyless gateway; no real HTTP requests."""
import unittest
from unittest.mock import Mock, patch

from content_fetch import urls
from content_fetch.url_security import UnsafeUrlError
from web_gateway import client


class FreeFetchTests(unittest.TestCase):
    def setUp(self):
        validator = patch.object(urls, "validate_public_http_url", side_effect=lambda url: url)
        validator.start()
        self.addCleanup(validator.stop)
        self.body = "This is a substantive historical source paragraph with documented facts. " * 30
        self.response = Mock(text=f"<article>{self.body}</article>", apparent_encoding="utf-8")

    def test_free_http_success_never_calls_gateway(self):
        with patch.object(urls, "safe_requests_get", return_value=self.response), patch.object(client, "extract") as extract:
            self.assertIn("substantive", urls._fetch_url_fallbacks("https://example.com"))
        extract.assert_not_called()

    def test_gateway_only_after_free_failure(self):
        order = []
        def failed(*args, **kwargs):
            order.append("free")
            raise TimeoutError()
        def paid(url):
            order.append("gateway")
            return {"results": [{"raw_content": self.body}]}
        with patch.object(urls, "safe_requests_get", side_effect=failed), patch.object(client, "extract", side_effect=paid):
            self.assertIn("substantive", urls._fetch_url_fallbacks("https://example.com"))
        self.assertEqual(order, ["free", "gateway"])

    def test_gateway_failure_has_no_provider_fallback(self):
        with patch.object(urls, "safe_requests_get", side_effect=TimeoutError), \
             patch.object(client, "extract", side_effect=client.WebGatewayError("budget exhausted")) as extract, \
             patch("tavily.TavilyClient") as sdk:
            self.assertIsNone(urls._fetch_url_fallbacks("https://example.com"))
        extract.assert_called_once()
        sdk.assert_not_called()

    def test_pdf_response_is_converted_not_parsed_as_html(self):
        pdf = Mock(headers={"Content-Type": "application/octet-stream"}, content=b"%PDF-1.4\n\x00\xff binary")
        with patch.object(urls, "safe_requests_get", return_value=pdf), \
             patch("content_fetch.documents.convert_document", return_value=self.body) as convert, \
             patch.object(client, "extract") as extract:
            text = urls._fetch_url_fallbacks("https://example.com/paper")
        self.assertIn("substantive", text)
        self.assertTrue(convert.call_args.args[0].endswith(".pdf"))
        extract.assert_not_called()

    def test_textless_pdf_never_returns_raw_bytes(self):
        pdf = Mock(headers={"Content-Type": "application/pdf"}, content=b"%PDF-1.7\n" + b"stream x y z " * 100)
        with patch.object(urls, "safe_requests_get", return_value=pdf), \
             patch("content_fetch.documents.convert_document", return_value=None), \
             patch.object(client, "extract", return_value={"results": []}) as extract:
            self.assertIsNone(urls._fetch_url_fallbacks("https://example.com/scan.pdf"))
        extract.assert_called_once()

    def test_unsafe_destination_blocked_before_any_fetch(self):
        with patch.object(urls, "validate_public_http_url", side_effect=UnsafeUrlError("private")), \
             patch.object(urls, "safe_requests_get") as http, patch.object(client, "extract") as extract:
            with self.assertRaises(UnsafeUrlError):
                urls._fetch_url_fallbacks("http://127.0.0.1")
        http.assert_not_called()
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
