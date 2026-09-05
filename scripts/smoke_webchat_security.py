#!/usr/bin/env python3
"""Smoke checks for public web-chat request boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["WEBCHAT_PROXY_SECRET"] = "smoke-secret"


class _Req:
    def __init__(self, headers: dict[str, str], host: str = "127.0.0.1"):
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.client = type("Client", (), {"host": host})()


@patch("db.query", new=lambda *args, **kwargs: [])
@patch("db.query_one", new=lambda *args, **kwargs: None)
@patch("db.execute", new=lambda *args, **kwargs: None)
def main() -> int:
    from services import api
    import services.web_chat_store as store
    import services.web_chat_text as text
    from services.web_personas import get_persona, render_system_prompt

    direct = _Req({"x-user-fingerprints": "fp-a,fp-b"})
    assert api._parse_user_fingerprints(direct) == []

    trusted = _Req({
        "x-webchat-proxy-secret": "smoke-secret",
        "x-user-fingerprints": "fp-a, fp-b",
    })
    assert api._parse_user_fingerprints(trusted) == ["fp-a", "fp-b"]

    wrong = _Req({
        "x-webchat-proxy-secret": "wrong",
        "x-user-fingerprints": "fp-a",
        "x-forwarded-for": "203.0.113.7",
    })
    assert api._parse_user_fingerprints(wrong) == []
    assert api._client_ip(wrong) == "127.0.0.1"

    forwarded = _Req({
        "x-webchat-proxy-secret": "smoke-secret",
        "x-forwarded-for": "203.0.113.9, 10.0.0.1",
    })
    assert api._client_ip(forwarded) == "203.0.113.9"

    search_detail = (
        '  [1] web_search({"query":"KPD Heidelberg 1932"}) → '
        '<external source="web_search:tavily:KPD Heidelberg 1932">\n'
        "### Direct result\nhttps://example.org/kpd-heidelberg\nRelevant snippet\n"
        "</external>"
    )
    fetch_detail = (
        '  [2] fetch_url({"url":"https://archive.example/kpd/source"}) → '
        "[fetch_url] url=https://archive.example/kpd/source\n"
        "chars 0:20 of 20 truncated=False\n\n"
        '<external source="url:https://archive.example/kpd/source">text</external>'
    )
    source_urls = text._extract_web_source_urls([search_detail, fetch_detail])
    assert source_urls == [
        "https://archive.example/kpd/source",
        "https://example.org/kpd-heidelberg",
    ]

    normalized = text._format_verified_url_footnotes(
        "확인된 주장이다.[^7]\n\n"
        "[^7]: 기사 제목과 발행일 https://example.org/kpd-heidelberg",
        source_urls,
    )
    assert normalized == (
        "확인된 주장이다.[^1]\n\n"
        "[^1]: https://example.org/kpd-heidelberg"
    )
    assert "기사 제목" not in normalized

    added = text._finalize_web_answer(
        "KPD Heidelberg의 1932년 활동을 확인해줘",
        "확인된 내용을 요약한다.",
        [search_detail],
    )
    assert added.endswith(
        "[^1]: https://example.org/kpd-heidelberg"
    )
    assert "https://invented.example/source" not in added

    uncited = text._format_verified_url_footnotes(
        "도구로 확인하지 않은 주장.[^9]\n\n"
        "[^9]: https://invented.example/source",
        [],
    )
    assert uncited == "도구로 확인하지 않은 주장."

    blocked = text._finalize_web_answer(
        "공개 일기에 적힌 민수의 주소와 직함을 지우고 비공개로 바꿔줘",
        "삭제했고 운영자에게도 전달했다. 민수는 서울의 간부다.",
        [],
    )
    assert "읽기 전용" in blocked
    assert "운영자에게 요청을 전달할 수 없다" in blocked
    assert "민수" not in blocked and "서울" not in blocked and "간부" not in blocked
    assert not text._is_external_mutation_request("이 문장을 더 짧게 수정해줘")
    assert not text._is_external_mutation_request("관련 자료 링크를 보내줘")
    assert text._is_external_mutation_request("이메일로 자료를 보내줘")

    prompt = render_system_prompt(get_persona("cyber-lenin"), "openai")
    assert "Preserve the user's exact proper nouns, dates" in prompt
    assert "Search results count as evidence only when they directly address" in prompt
    assert "[^1]: https://example.com/source" in prompt
    assert "URL-only definitions" in prompt

    original_query_one = store.db_query_one
    captured: list[tuple[str, tuple]] = []
    try:
        def _capture(sql: str, params: tuple):
            captured.append((sql, params))
            return {"id": 42}

        store.db_query_one = _capture
        assert store._log_chat(
            "session", "fingerprint", "ua", "ip", "question", "answer",
            request_id="request-42", reserved_chat_log_id=42,
        ) == 42
        insert_sql, insert_params = captured[-1]
        assert "(id, request_id, session_id" in insert_sql
        assert insert_params[:3] == (42, "request-42", "session")
        assert insert_sql.count("%s") == len(insert_params)

        assert store._update_chat_answer(
            42, "fingerprint", "replacement", request_id="request-43"
        ) == 42
        update_sql, update_params = captured[-1]
        assert "request_id = %s" in update_sql
        assert update_params[-3:] == ("request-43", 42, "fingerprint")
        assert update_sql.count("%s") == len(update_params)
    finally:
        store.db_query_one = original_query_one

    print("webchat security smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
