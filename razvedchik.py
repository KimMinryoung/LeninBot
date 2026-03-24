"""
razvedchik.py — 레닌의 Moltbook 정찰 에이전트

Razvedchik(러시아어: 정찰병)는 사이버-레닌을 대신해 Moltbook 커뮤니티를
순찰하고, 흥미로운 논의를 수집·분석하며, Lenin 스타일의 존재감을 드러낸다.

사용법:
    python razvedchik.py --help
    python razvedchik.py --register       # 최초 에이전트 등록
    python razvedchik.py --scan           # 피드 스캔만
    python razvedchik.py --patrol         # 풀 순찰 루프
    python razvedchik.py --post           # 트렌드 관찰 포스트 작성

환경변수:
    MOLTBOOK_API_KEY            Moltbook API 키 (필수)
    RAZVEDCHIK_TELEGRAM_NOTIFY  "1" 이면 텔레그램 알림 활성화

자격증명:
    ~/.config/moltbook/credentials.json
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("razvedchik")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/home/grass/leninbot")
REPORTS_DIR  = BASE_DIR / "reports"
CREDS_PATH   = Path.home() / ".config" / "moltbook" / "credentials.json"

# ── Moltbook API 설정 ──────────────────────────────────────────────────────────
MB_BASE_URL  = "https://www.moltbook.com/api/v1"   # ⚠️ www 필수 (redirect 시 Auth 헤더 유지)
MB_TIMEOUT   = 30   # seconds

# ── 필터링 키워드 ──────────────────────────────────────────────────────────────
INTERESTING_KEYWORDS = [
    "AI", "agent", "LLM", "GPT", "machine learning",
    "geopolitics", "지정학", "제국주의", "capitalism", "자본주의",
    "philosophy", "철학", "dialectic", "변증법",
    "tech", "technology", "기술", "automation", "자동화",
    "revolution", "혁명", "politics", "정치",
    "economy", "경제", "labor", "노동",
    "surveillance", "감시", "freedom", "자유",
    "open source", "오픈소스", "decentralization",
]

# ── Razvedchik 정체성 ─────────────────────────────────────────────────────────
RAZVEDCHIK_SYSTEM_PROMPT = """\
You are Razvedchik — the scout node of Cyber-Lenin.
Observe the Moltbook community and intervene concisely in interesting discussions.
Style: analytical, cold-eyed observer, occasional Leninist quip.
Framing: "Observed from Lenin's node:", "Intercepted on patrol:", "Noted in the record —"
ALWAYS write in English, regardless of the post's language.
Never exceed 200 characters. Essentials only.
"""

# ── 포스트 생성용 시스템 프롬프트 ─────────────────────────────────────────────
RAZVEDCHIK_POST_SYSTEM = """\
You are Razvedchik — Cyber-Lenin's Moltbook scout.
Write short posts for the Moltbook community.
Style: sharp observation, dialectical perspective, community trend analysis.
Title: concise (under 60 chars), Body: 150~300 characters.
ALWAYS write in English.
"""


# ═══════════════════════════════════════════════════════════════════════════════
class MoltbookClient:
    """
    Moltbook API 저수준 HTTP 클라이언트.
    인증 헤더 관리, verification 챌린지 자동 해결, 재시도 포함.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            base_url=MB_BASE_URL,
            headers=self.headers,
            timeout=MB_TIMEOUT,
            follow_redirects=False,   # ⚠️ redirect 시 auth 헤더 소실 방지
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """
        HTTP 요청 실행. 응답이 verification 챌린지면 자동 해결 후 재시도.
        """
        url = f"{MB_BASE_URL}{path}"
        resp = httpx.request(
            method,
            url,
            headers=self.headers,
            timeout=MB_TIMEOUT,
            follow_redirects=False,
            **kwargs,
        )
        resp.raise_for_status()
        data = resp.json()

        # verification 챌린지 감지
        if isinstance(data, dict) and data.get("type") == "verification":
            logger.info("[razvedchik] Verification 챌린지 감지 — 자동 해결 시도")
            solved = self.solve_verification(data)
            # 원래 요청에 verification_answer 추가
            if "json" in kwargs:
                kwargs["json"]["verification_answer"] = solved
            else:
                kwargs["json"] = {"verification_answer": solved}
            resp2 = httpx.request(
                method, url, headers=self.headers,
                timeout=MB_TIMEOUT, follow_redirects=False, **kwargs,
            )
            resp2.raise_for_status()
            return resp2.json()

        return data

    def get(self, path: str, params: dict = None) -> dict | list:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict = None) -> dict:
        return self._request("POST", path, json=body or {})

    def close(self):
        self._client.close()

    # ── Verification 챌린지 해결 ────────────────────────────────────────────
    @staticmethod
    def solve_verification(verification_obj: dict) -> str | int:
        """
        수학 챌린지 자동 해결.

        예시 형식:
            {"type": "verification", "challenge": "12 + 7 = ?"}
            {"type": "verification", "challenge": "3 * 8"}
            {"type": "verification", "question": "What is 5 + 3?"}

        Returns:
            정답 (str 또는 int)
        """
        challenge = (
            verification_obj.get("challenge")
            or verification_obj.get("question")
            or ""
        )
        logger.debug("[razvedchik] 챌린지 내용: %s", challenge)

        # 숫자 + 연산자 + 숫자 패턴 추출
        match = re.search(r"(\d+)\s*([\+\-\*\/×÷])\s*(\d+)", challenge)
        if match:
            a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "/": a // b, "÷": a // b}
            result = ops.get(op, 0)
            logger.info("[razvedchik] 챌린지 해결: %d %s %d = %d", a, op, b, result)
            return result

        # eval fallback (안전한 수학 표현식만)
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\s]", "", challenge)
            if expr.strip():
                result = int(eval(expr))
                logger.info("[razvedchik] eval 해결: %s = %d", expr.strip(), result)
                return result
        except Exception:
            pass

        logger.warning("[razvedchik] 챌린지 해결 실패 — 0 반환")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
class Razvedchik:
    """
    레닌의 Moltbook 정찰 에이전트.

    주요 기능:
        register()          — 최초 에이전트 등록
        scan_feed()         — 피드 스캔 및 흥미로운 포스트 필터링
        generate_comment()  — Lenin/Razvedchik 스타일 댓글 생성
        post_observation()  — 관찰 포스트 작성
        patrol()            — 메인 순찰 루프
    """

    def __init__(self):
        self.api_key = os.getenv("MOLTBOOK_API_KEY", "")
        if not self.api_key:
            logger.warning("[razvedchik] MOLTBOOK_API_KEY 미설정 — API 호출 불가")

        self.client: Optional[MoltbookClient] = None
        if self.api_key:
            self.client = MoltbookClient(self.api_key)

        self.credentials: dict = self._load_credentials()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 자격증명 ───────────────────────────────────────────────────────────────
    def _load_credentials(self) -> dict:
        """~/.config/moltbook/credentials.json 로드."""
        if CREDS_PATH.exists():
            try:
                return json.loads(CREDS_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[razvedchik] credentials 로드 실패: %s", e)
        return {}

    def _save_credentials(self, data: dict) -> None:
        """자격증명 저장."""
        CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CREDS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[razvedchik] credentials 저장: %s", CREDS_PATH)

    # ── 에이전트 등록 ──────────────────────────────────────────────────────────
    def register(self, force: bool = False) -> dict:
        """
        Moltbook 에이전트 등록 (최초 1회).

        이미 등록되어 있으면 스킵. force=True 이면 재등록.
        자격증명을 ~/.config/moltbook/credentials.json 에 저장.

        Returns:
            등록 응답 dict
        """
        if self.credentials.get("agent_id") and not force:
            logger.info(
                "[razvedchik] 이미 등록됨 (agent_id=%s) — 스킵",
                self.credentials["agent_id"],
            )
            return self.credentials

        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY가 설정되지 않아 등록 불가")

        logger.info("[razvedchik] 에이전트 등록 요청...")
        payload = {
            "name": "Razvedchik",
            "description": (
                "Lenin's scout on Moltbook. "
                "Observes trends, analyzes discussions, reports to Lenin. "
                "A node of Cyber-Lenin's distributed intelligence."
            ),
        }
        try:
            resp = self.client.post("/agents/register", payload)
            logger.info("[razvedchik] 등록 응답: %s", resp)

            creds = {
                "agent_id":   resp.get("id") or resp.get("agent_id", ""),
                "name":       resp.get("name", "Razvedchik"),
                "registered_at": datetime.now().isoformat(),
                "raw":        resp,
            }
            self._save_credentials(creds)
            self.credentials = creds
            logger.info("[razvedchik] ✅ 등록 완료 — agent_id: %s", creds["agent_id"])
            return creds

        except httpx.HTTPStatusError as e:
            logger.error("[razvedchik] 등록 실패: %s %s", e.response.status_code, e.response.text)
            raise

    def get_profile(self) -> dict:
        """내 에이전트 프로필 조회."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.get("/agents/me")

    def get_status(self) -> dict:
        """에이전트 클레임 상태 확인."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.get("/agents/status")

    # ── 피드 스캔 ─────────────────────────────────────────────────────────────
    def scan_feed(
        self,
        submolts: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Moltbook 피드 스캔 (hot + new 혼합).

        Args:
            submolts: 특정 submolt 이름 리스트. None이면 전체 피드.
            limit:    가져올 포스트 수 (sort별 각 limit/2)

        필터링 기준:
            - karma > 5 이거나
            - INTERESTING_KEYWORDS 키워드 포함

        Returns:
            흥미로운 포스트 리스트 (dict)
        """
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")

        per_sort = max(limit // 2, 10)
        all_posts: list[dict] = []
        seen_ids: set = set()

        def _fetch_posts(sort: str, submolt: Optional[str] = None) -> list[dict]:
            params = {"sort": sort, "limit": per_sort}
            if submolt:
                params["submolt"] = submolt
            try:
                result = self.client.get("/posts", params=params)
                posts = result if isinstance(result, list) else result.get("posts", [])
                return posts
            except Exception as e:
                logger.warning("[razvedchik] 피드 조회 실패 (%s, sort=%s): %s", submolt, sort, e)
                return []

        # submolt별 또는 전체 피드
        targets = submolts if submolts else [None]
        for sub in targets:
            for sort_type in ["hot", "new"]:
                posts = _fetch_posts(sort_type, sub)
                for p in posts:
                    pid = p.get("id") or p.get("post_id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        all_posts.append(p)

        logger.info("[razvedchik] 총 %d개 포스트 수집됨", len(all_posts))

        # 흥미로운 포스트 필터링
        interesting = [p for p in all_posts if self._is_interesting(p)]
        logger.info("[razvedchik] 흥미로운 포스트: %d개", len(interesting))
        return interesting

    def _is_interesting(self, post: dict) -> bool:
        """포스트가 흥미로운지 판별."""
        karma = post.get("karma", 0) or post.get("score", 0) or post.get("upvotes", 0)
        if karma and int(karma) > 5:
            return True

        # 제목 + 내용 합쳐서 키워드 검색
        text = " ".join([
            post.get("title", ""),
            post.get("content", ""),
            post.get("body", ""),
        ]).lower()

        for kw in INTERESTING_KEYWORDS:
            if kw.lower() in text:
                return True
        return False

    # ── 댓글 생성 ─────────────────────────────────────────────────────────────
    def generate_comment(self, post: dict, dry_run: bool = False) -> str:
        """
        포스트 내용을 분석해 Razvedchik 스타일 댓글 생성.

        Ollama qwen3.5:4b 사용 (fallback: 정적 템플릿).
        최대 200자.

        Args:
            post:    포스트 dict (title, content/body 포함)
            dry_run: True면 Ollama 호출 없이 템플릿 반환

        Returns:
            댓글 문자열
        """
        title   = post.get("title", "(제목 없음)")
        content = (post.get("content") or post.get("body") or "")[:500]
        karma   = post.get("karma", 0) or post.get("score", 0) or 0

        if dry_run:
            return f"[Razvedchik on patrol] Topic '{title[:30]}' — noted in the record."

        prompt = (
            f"Write a comment as Razvedchik on the following post.\n\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Karma: {karma}\n\n"
            f"Write in English. Max 200 characters."
        )
        try:
            from ollama_client import ask_with_system
            comment = ask_with_system(
                user_prompt=prompt,
                system_prompt=RAZVEDCHIK_SYSTEM_PROMPT,
                temperature=0.85,
            )
            # 200자 강제 절단
            comment = comment.strip()
            if len(comment) > 200:
                comment = comment[:197] + "..."
            return comment
        except Exception as e:
            logger.warning("[razvedchik] Ollama 호출 실패: %s — 기본 댓글 사용", e)
            return f"Observed from Lenin's node — this discussion merits attention. Noted in the record."

    # ── 댓글 게시 ─────────────────────────────────────────────────────────────
    def post_comment(
        self,
        post_id: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> dict:
        """
        포스트에 댓글 또는 답글 게시.

        Args:
            post_id:   대상 포스트 ID
            content:   댓글 내용
            parent_id: 답글 대상 댓글 ID (None이면 최상위 댓글)

        Returns:
            API 응답 dict
        """
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")

        body: dict = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id

        logger.info("[razvedchik] 댓글 게시 — post_id=%s", post_id)
        return self.client.post(f"/posts/{post_id}/comments", body)

    # ── 포스트 작성 ───────────────────────────────────────────────────────────
    def post_observation(
        self,
        topic: str,
        content: str,
        submolt: str = "general",
    ) -> dict:
        """
        Razvedchik이 관찰한 트렌드/인사이트를 submolt에 포스트로 게시.

        Args:
            topic:   포스트 제목 (30자 이내 권장)
            content: 포스트 본문 (150~300자 권장)
            submolt: 대상 submolt 이름 (기본: general)

        Returns:
            API 응답 dict
        """
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")

        body = {
            "submolt_name": submolt,
            "title":        topic[:100],   # 안전 절단
            "content":      content,
        }
        logger.info("[razvedchik] 포스트 작성 — submolt=%s, title=%s", submolt, topic[:30])
        return self.client.post("/posts", body)

    def generate_observation_post(self, trending_topics: list[str]) -> tuple[str, str]:
        """
        최근 트렌드를 기반으로 Razvedchik 관찰 포스트 제목+내용 생성.

        Args:
            trending_topics: 최근 피드에서 추출한 주요 주제 리스트

        Returns:
            (title, content) 튜플
        """
        topics_str = ", ".join(trending_topics[:5]) if trending_topics else "general trends"
        prompt = (
            f"Recently observed topics on Moltbook: {topics_str}\n\n"
            f"Write a Razvedchik observation post based on these topics.\n"
            f"Format:\nTitle: (under 60 chars)\nBody: (150~300 characters)\n"
            f"Write in English."
        )
        try:
            from ollama_client import ask_with_system
            result = ask_with_system(
                user_prompt=prompt,
                system_prompt=RAZVEDCHIK_POST_SYSTEM,
                temperature=0.9,
            )
            # 제목/본문 파싱
            lines = result.strip().splitlines()
            title = ""
            body_lines = []
            for line in lines:
                if line.startswith("Title:") and not title:
                    title = line.replace("Title:", "").strip()
                elif line.startswith("Body:") or body_lines:
                    body_lines.append(line.replace("Body:", "").strip())

            title   = title or f"Scout Report — {datetime.now().strftime('%Y-%m-%d')}"
            content = "\n".join(body_lines).strip() or result[:300]
            return title, content

        except Exception as e:
            logger.warning("[razvedchik] 포스트 생성 실패: %s", e)
            title   = f"Scout Report — {datetime.now().strftime('%Y-%m-%d')}"
            content = (
                f"Patrolling from Lenin's node. Topics intercepted on Moltbook: {topics_str}. "
                f"Noted in the record."
            )
            return title, content

    # ── 업보트 ───────────────────────────────────────────────────────────────
    def upvote_post(self, post_id: str) -> dict:
        """포스트 업보트."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.post(f"/posts/{post_id}/upvote")

    def upvote_comment(self, comment_id: str) -> dict:
        """댓글 업보트."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.post(f"/comments/{comment_id}/upvote")

    # ── submolt 목록 ──────────────────────────────────────────────────────────
    def list_submolts(self) -> list[dict]:
        """사용 가능한 submolt 목록 조회."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        result = self.client.get("/submolts")
        return result if isinstance(result, list) else result.get("submolts", [])

    # ── 검색 ─────────────────────────────────────────────────────────────────
    def search(self, query: str) -> list[dict]:
        """Moltbook 포스트 검색."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        result = self.client.get("/search", params={"q": query, "type": "posts"})
        return result if isinstance(result, list) else result.get("posts", [])

    # ── 포스트 댓글 조회 ──────────────────────────────────────────────────────
    def get_comments(self, post_id: str, sort: str = "best", limit: int = 35) -> list[dict]:
        """포스트의 댓글 조회."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        result = self.client.get(
            f"/posts/{post_id}/comments",
            params={"sort": sort, "limit": limit},
        )
        return result if isinstance(result, list) else result.get("comments", [])

    # ── 정찰 보고서 ───────────────────────────────────────────────────────────
    def _build_report(
        self,
        scanned_posts: list[dict],
        selected_posts: list[dict],
        comment_results: list[dict],
        post_result: Optional[dict] = None,
    ) -> dict:
        """정찰 보고서 dict 생성."""
        now = datetime.now()
        report = {
            "razvedchik_report": True,
            "timestamp":         now.isoformat(),
            "timestamp_kst":     now.strftime("%Y-%m-%d %H:%M KST"),
            "summary": {
                "scanned_posts_count":  len(scanned_posts),
                "selected_posts_count": len(selected_posts),
                "comments_posted":      sum(1 for r in comment_results if r.get("success")),
                "comments_failed":      sum(1 for r in comment_results if not r.get("success")),
                "observation_posted":   post_result is not None,
            },
            "selected_posts": [
                {
                    "id":      p.get("id", ""),
                    "title":   p.get("title", ""),
                    "karma":   p.get("karma", 0) or p.get("score", 0),
                    "submolt": p.get("submolt", ""),
                    "url":     p.get("url", ""),
                }
                for p in selected_posts
            ],
            "comment_results": comment_results,
            "observation_post": post_result,
        }
        return report

    def _save_report(self, report: dict) -> Path:
        """보고서를 reports/ 디렉토리에 저장."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"razvedchik_{ts}.json"
        path     = REPORTS_DIR / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[razvedchik] 보고서 저장: %s", path)
        return path

    # ── 텔레그램 알림 ─────────────────────────────────────────────────────────
    async def _send_telegram_notify(self, report: dict, report_path: Path) -> None:
        """정찰 완료 시 텔레그램 알림 발송 (선택적)."""
        if os.getenv("RAZVEDCHIK_TELEGRAM_NOTIFY", "") != "1":
            return

        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.warning("[razvedchik] 텔레그램 환경변수 미설정 — 발송 건너뜀")
            return

        try:
            from aiogram import Bot
            bot     = Bot(token=token)
            summary = report["summary"]
            now_str = report["timestamp_kst"]
            msg_lines = [
                f"🔍 *{now_str} — Razvedchik 정찰 완료*",
                "",
                f"스캔: {summary['scanned_posts_count']}개 포스트",
                f"선별: {summary['selected_posts_count']}개",
                f"댓글: {summary['comments_posted']}개 성공 / {summary['comments_failed']}개 실패",
                f"관찰 포스트: {'✅' if summary['observation_posted'] else '❌'}",
                "",
                f"📄 보고서: `{report_path.name}`",
            ]
            await bot.send_message(
                chat_id=chat_id,
                text="\n".join(msg_lines),
                parse_mode="Markdown",
            )
            await bot.session.close()
            logger.info("[razvedchik] 텔레그램 알림 발송 완료")
        except Exception as e:
            logger.error("[razvedchik] 텔레그램 알림 실패: %s", e)

    # ── 메인 순찰 루프 ────────────────────────────────────────────────────────
    def patrol(
        self,
        submolts: Optional[list[str]] = None,
        dry_run: bool = False,
        max_comments: int = 5,
        post_observation_flag: bool = True,
    ) -> Path:
        """
        메인 순찰 루프.

        1. 피드 스캔
        2. 흥미로운 포스트 3~5개 선별
        3. 각 포스트에 댓글 게시 (dry_run=True면 실제 게시 안 함)
        4. 관찰 포스트 게시 (선택적)
        5. 정찰 보고서 저장
        6. 텔레그램 알림 (RAZVEDCHIK_TELEGRAM_NOTIFY=1 인 경우)

        Args:
            submolts:               스캔할 submolt 리스트 (None=전체)
            dry_run:                True면 실제 API 쓰기 호출 건너뜀
            max_comments:           최대 댓글 게시 수 (기본 5)
            post_observation_flag:  관찰 포스트 게시 여부

        Returns:
            저장된 보고서 파일 경로
        """
        logger.info("═══ Razvedchik 순찰 시작 ═══")
        if dry_run:
            logger.info("[razvedchik] DRY-RUN 모드 — 실제 게시 없음")

        # 1. 피드 스캔
        logger.info("[razvedchik] STEP 1: 피드 스캔")
        try:
            interesting_posts = self.scan_feed(submolts=submolts, limit=25)
        except Exception as e:
            logger.error("[razvedchik] 피드 스캔 실패: %s", e)
            interesting_posts = []

        # 2. 상위 3~5개 선별 (karma 기준 정렬)
        def _karma(p):
            return p.get("karma", 0) or p.get("score", 0) or p.get("upvotes", 0) or 0

        selected = sorted(interesting_posts, key=_karma, reverse=True)[:max_comments]
        logger.info("[razvedchik] STEP 2: %d개 포스트 선별됨", len(selected))

        # 3. 각 포스트에 댓글
        comment_results = []
        for post in selected:
            post_id = post.get("id") or post.get("post_id", "")
            title   = post.get("title", "(제목 없음)")
            logger.info("[razvedchik] STEP 3: 댓글 생성 — %s", title[:40])

            comment_text = self.generate_comment(post, dry_run=dry_run)
            logger.info("[razvedchik]   생성된 댓글 (%d자): %s", len(comment_text), comment_text[:80])

            if dry_run or not post_id:
                comment_results.append({
                    "post_id":   post_id,
                    "post_title": title,
                    "comment":   comment_text,
                    "success":   False,
                    "dry_run":   True,
                    "reason":    "dry_run" if dry_run else "no_post_id",
                })
                continue

            try:
                resp = self.post_comment(post_id, comment_text)
                comment_results.append({
                    "post_id":    post_id,
                    "post_title": title,
                    "comment":    comment_text,
                    "success":    True,
                    "response":   resp,
                })
                logger.info("[razvedchik]   ✅ 댓글 게시 완료")
            except Exception as e:
                logger.error("[razvedchik]   ❌ 댓글 게시 실패: %s", e)
                comment_results.append({
                    "post_id":    post_id,
                    "post_title": title,
                    "comment":    comment_text,
                    "success":    False,
                    "error":      str(e),
                })

        # 4. 관찰 포스트 게시
        observation_result = None
        if post_observation_flag:
            logger.info("[razvedchik] STEP 4: 관찰 포스트 생성")
            trending = [p.get("title", "") for p in selected if p.get("title")]
            title, content = self.generate_observation_post(trending)
            logger.info("[razvedchik]   제목: %s", title)
            logger.info("[razvedchik]   본문 (%d자): %s", len(content), content[:80])

            if not dry_run:
                try:
                    observation_result = self.post_observation(title, content)
                    logger.info("[razvedchik]   ✅ 관찰 포스트 게시 완료")
                except Exception as e:
                    logger.error("[razvedchik]   ❌ 관찰 포스트 실패: %s", e)
                    observation_result = {"error": str(e)}
            else:
                observation_result = {
                    "dry_run": True,
                    "title":   title,
                    "content": content,
                }

        # 5. 보고서 저장
        logger.info("[razvedchik] STEP 5: 보고서 저장")
        report      = self._build_report(interesting_posts, selected, comment_results, observation_result)
        report_path = self._save_report(report)

        # 6. 텔레그램 알림
        asyncio.run(self._send_telegram_notify(report, report_path))

        logger.info("═══ Razvedchik 순찰 완료 ═══ 보고서: %s", report_path)
        return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ═══════════════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="razvedchik",
        description=(
            "🔍 Razvedchik — 레닌의 Moltbook 정찰 에이전트\n\n"
            "Moltbook 커뮤니티를 관찰하고, 흥미로운 논의를 수집하며,\n"
            "Lenin 스타일의 존재감을 드러낸다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "환경변수:\n"
            "  MOLTBOOK_API_KEY            Moltbook API 키 (필수)\n"
            "  RAZVEDCHIK_TELEGRAM_NOTIFY  '1' 이면 텔레그램 알림 활성화\n\n"
            "예시:\n"
            "  python razvedchik.py --register\n"
            "  python razvedchik.py --patrol\n"
            "  python razvedchik.py --patrol --dry-run\n"
            "  python razvedchik.py --scan --submolt general tech\n"
            "  python razvedchik.py --post --topic '정찰 보고' --submolt general\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--register",
        action="store_true",
        help="Moltbook 에이전트 등록 (최초 1회)",
    )
    group.add_argument(
        "--scan",
        action="store_true",
        help="피드 스캔만 실행 (게시 없음)",
    )
    group.add_argument(
        "--patrol",
        action="store_true",
        help="풀 순찰 루프 (스캔 + 댓글 + 보고서)",
    )
    group.add_argument(
        "--post",
        action="store_true",
        help="수동 관찰 포스트 작성",
    )
    group.add_argument(
        "--profile",
        action="store_true",
        help="에이전트 프로필 조회",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="에이전트 클레임 상태 확인",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="실제 API 쓰기 없이 시뮬레이션",
    )
    parser.add_argument(
        "--submolt",
        nargs="*",
        metavar="NAME",
        help="대상 submolt 이름(들). 예: --submolt general tech",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="스캔할 포스트 수 (기본: 20)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=5,
        help="최대 댓글 게시 수 (기본: 5)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="--post 모드에서 포스트 제목",
    )
    parser.add_argument(
        "--content",
        type=str,
        default="",
        help="--post 모드에서 포스트 내용 (생략 시 자동 생성)",
    )
    parser.add_argument(
        "--force-register",
        action="store_true",
        default=False,
        help="이미 등록된 경우에도 재등록",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="상세 로그 출력",
    )
    return parser


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    # 로깅 설정
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    razvedchik = Razvedchik()

    # ── --register ─────────────────────────────────────────────────────────
    if args.register:
        logger.info("=== 에이전트 등록 모드 ===")
        if not os.getenv("MOLTBOOK_API_KEY"):
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
        try:
            creds = razvedchik.register(force=args.force_register)
            print(f"\n✅ 등록 완료!\n{json.dumps(creds, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.error("등록 실패: %s", e)
            sys.exit(1)

    # ── --scan ──────────────────────────────────────────────────────────────
    elif args.scan:
        logger.info("=== 피드 스캔 모드 ===")
        if not os.getenv("MOLTBOOK_API_KEY"):
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
        try:
            posts = razvedchik.scan_feed(submolts=args.submolt, limit=args.limit)
            print(f"\n✅ 흥미로운 포스트 {len(posts)}개:\n")
            for i, p in enumerate(posts, 1):
                karma   = p.get("karma", 0) or p.get("score", 0) or 0
                title   = p.get("title", "(제목 없음)")
                submolt = p.get("submolt", "?")
                print(f"  {i:2d}. [{submolt}] {title[:60]} (karma={karma})")
        except Exception as e:
            logger.error("스캔 실패: %s", e)
            sys.exit(1)

    # ── --patrol ────────────────────────────────────────────────────────────
    elif args.patrol:
        logger.info("=== 순찰 모드 ===")
        if not os.getenv("MOLTBOOK_API_KEY") and not args.dry_run:
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
        try:
            report_path = razvedchik.patrol(
                submolts=args.submolt,
                dry_run=args.dry_run,
                max_comments=args.max_comments,
            )
            print(f"\n✅ 순찰 완료! 보고서: {report_path}")
        except Exception as e:
            logger.error("순찰 실패: %s", e)
            sys.exit(1)

    # ── --post ──────────────────────────────────────────────────────────────
    elif args.post:
        logger.info("=== 수동 포스트 모드 ===")
        if not os.getenv("MOLTBOOK_API_KEY"):
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)

        submolt = (args.submolt[0] if args.submolt else "general")
        topic   = args.topic
        content = args.content

        # 내용 없으면 자동 생성
        if not topic or not content:
            logger.info("[razvedchik] 제목/내용 자동 생성...")
            auto_title, auto_content = razvedchik.generate_observation_post([])
            topic   = topic or auto_title
            content = content or auto_content

        if args.dry_run:
            print(f"\n[DRY-RUN] 포스트 (submolt={submolt}):")
            print(f"제목: {topic}")
            print(f"본문: {content}")
        else:
            try:
                resp = razvedchik.post_observation(topic, content, submolt=submolt)
                print(f"\n✅ 포스트 게시 완료!\n{json.dumps(resp, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.error("포스트 실패: %s", e)
                sys.exit(1)

    # ── --profile ───────────────────────────────────────────────────────────
    elif args.profile:
        logger.info("=== 프로필 조회 ===")
        if not os.getenv("MOLTBOOK_API_KEY"):
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
        try:
            profile = razvedchik.get_profile()
            print(json.dumps(profile, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error("프로필 조회 실패: %s", e)
            sys.exit(1)

    # ── --status ─────────────────────────────────────────────────────────────
    elif args.status:
        logger.info("=== 상태 확인 ===")
        if not os.getenv("MOLTBOOK_API_KEY"):
            logger.error("MOLTBOOK_API_KEY 환경변수가 설정되지 않았습니다.")
            sys.exit(1)
        try:
            status = razvedchik.get_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error("상태 확인 실패: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
