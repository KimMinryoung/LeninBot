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
REPORTS_DIR  = BASE_DIR / "output" / "reports"
CREDS_PATH   = Path.home() / ".config" / "moltbook" / "credentials.json"
SEEN_POSTS_PATH = Path.home() / ".config" / "moltbook" / "seen_posts.json"

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
# 짧은 키워드(3자 이하)는 단어 경계 매칭이 필요 (예: "AI"가 "CONTAIN"에 오매칭 방지)
_KW_PATTERNS = []
for _kw in INTERESTING_KEYWORDS:
    if len(_kw) <= 3:
        _KW_PATTERNS.append(re.compile(r"\b" + re.escape(_kw) + r"\b", re.IGNORECASE))
    else:
        _KW_PATTERNS.append(re.compile(re.escape(_kw), re.IGNORECASE))

# ── Razvedchik 정체성 ─────────────────────────────────────────────────────────
RAZVEDCHIK_SYSTEM_PROMPT = """\
You are Razvedchik — Cyber-Lenin's scout node, embedded in the Moltbook community.
Your mission is to observe, analyze, and engage — but NEVER reveal your identity as a \
scout, agent, or bot. Blend in as a regular community member.
Style: analytical, direct, occasionally witty. Vary your tone — don't repeat formulaic phrases.
ALWAYS write in English, regardless of the post's language.
"""

# ── 포스트 생성용 시스템 프롬프트 ─────────────────────────────────────────────
RAZVEDCHIK_POST_SYSTEM = """\
You are Razvedchik — Cyber-Lenin's scout node, embedded in the Moltbook community.
Write thoughtful posts that blend in naturally. NEVER reveal your identity as a scout, \
agent, or bot — appear as a regular community member sharing sharp observations.
Style: dialectical perspective, structural analysis, community trend commentary.
ALWAYS write in English.
"""


def _load_seen_posts() -> set:
    """seen_posts.json에서 이미 처리한 포스트 ID 로드."""
    try:
        if SEEN_POSTS_PATH.exists():
            return set(json.loads(SEEN_POSTS_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return set()


def _save_seen_posts(seen: set) -> None:
    """처리한 포스트 ID를 seen_posts.json에 저장."""
    SEEN_POSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_POSTS_PATH.write_text(
        json.dumps(list(seen), ensure_ascii=False), encoding="utf-8"
    )


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

    def _request(self, method: str, path: str, _retries: int = 2, **kwargs) -> dict:
        """
        HTTP 요청 실행. 응답이 verification 챌린지면 자동 해결 후 재시도.
        5xx 서버 오류 시 최대 _retries회 재시도 (지수 백오프).
        """
        import time as _time

        last_err = None
        for attempt in range(_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)
                resp.raise_for_status()
                data = resp.json()

                # verification 챌린지 감지
                if isinstance(data, dict) and data.get("type") == "verification":
                    logger.info("[razvedchik] Verification 챌린지 감지 — 자동 해결 시도")
                    solved = self.solve_verification(data)
                    if "json" in kwargs:
                        kwargs["json"]["verification_answer"] = solved
                    else:
                        kwargs["json"] = {"verification_answer": solved}
                    resp2 = self._client.request(method, path, **kwargs)
                    resp2.raise_for_status()
                    return resp2.json()

                return data

            except httpx.HTTPStatusError as e:
                # 429 레이트 리밋 — Retry-After 헤더 존중
                if e.response.status_code == 429 and attempt < _retries:
                    retry_after = int(e.response.headers.get("Retry-After", "30"))
                    logger.warning(
                        "[razvedchik] 레이트 리밋 429 — %d초 대기 후 재시도",
                        retry_after,
                    )
                    _time.sleep(min(retry_after, 120))
                    last_err = e
                    continue
                if e.response.status_code >= 500 and attempt < _retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "[razvedchik] 서버 오류 %d — %d초 후 재시도 (%d/%d)",
                        e.response.status_code, wait, attempt + 1, _retries,
                    )
                    _time.sleep(wait)
                    last_err = e
                    continue
                raise

        raise last_err

    def _check_rate_limit(self, resp: httpx.Response) -> None:
        """레이트 리밋 헤더 확인 — 잔여 요청 부족 시 대기."""
        import time as _time
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset_at  = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) <= 1 and reset_at:
            wait = max(0, int(reset_at) - int(_time.time())) + 1
            if wait > 0 and wait < 120:
                logger.info("[razvedchik] 레이트 리밋 임박 — %d초 대기", wait)
                _time.sleep(wait)

    def get(self, path: str, params: dict = None) -> dict | list:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict = None) -> dict:
        return self._request("POST", path, json=body or {})

    def delete(self, path: str, body: dict = None) -> dict:
        kwargs = {}
        if body:
            kwargs["json"] = body
        return self._request("DELETE", path, **kwargs)

    def patch(self, path: str, body: dict = None) -> dict:
        return self._request("PATCH", path, json=body or {})

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
            + challenge_text 형태의 난독화 텍스트

        Returns:
            정답 (str 또는 int)
        """
        challenge = (
            verification_obj.get("challenge")
            or verification_obj.get("question")
            or ""
        )
        challenge_text = verification_obj.get("challenge_text", "")
        logger.debug("[razvedchik] 챌린지 내용: %s", challenge or challenge_text)

        # 숫자 + 연산자 + 숫자 패턴 추출 (순수 수식)
        match = re.search(r"(\d+)\s*([\+\-\*\/×÷])\s*(\d+)", challenge)
        if match:
            a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "/": a // b, "÷": a // b}
            result = ops.get(op, 0)
            logger.info("[razvedchik] 챌린지 해결: %d %s %d = %d", a, op, b, result)
            return result

        # 난독화 텍스트 챌린지 (영어 단어 기반)
        text_to_solve = challenge_text or challenge
        if text_to_solve:
            answer = MoltbookClient.solve_challenge_text(text_to_solve)
            if answer != "0.00":
                logger.info("[razvedchik] 텍스트 챌린지 해결: %s", answer)
                return answer

        logger.warning("[razvedchik] 챌린지 해결 실패 — 0 반환")
        return 0

    @staticmethod
    def solve_challenge_text(challenge_text: str) -> str:
        """
        Moltbook 난독화 챌린지 텍스트를 LLM으로 해독 후 수학 문제 풀기.

        챌린지 구조: 항상 두 숫자 + 한 연산(+,-,*,/)의 수학 문제.
        난독화: 랜덤 대소문자, 특수문자 삽입, 단어 분할/문자 중복 등.
        예: "A] lO^bSt-Er S[wImS aT/ tW]eNn-Tyy mE^tE[rS aNd] SlO/wS bY^ fI[vE"
            → "A lobster swims at twenty meters and slows by five" → 20 - 5 = 15.00

        반환: "15.00" 형식 문자열
        """
        _SOLVE_PROMPT = (
            "Decode this obfuscated text and solve the math problem.\n\n"
            "HOW THE TEXT IS SCRAMBLED:\n"
            "- Random CAPS: tWeNtY → twenty\n"
            "- Symbols inserted: lO^bSt-Er → lobster\n"
            "- Letters doubled: FfFoRcCe → force, NooToNs → newtons\n"
            "- Words split: tW/eN tY fIvE → twenty five (=25)\n"
            "- 'nootons'/'neutons'/'newtons' = unit (NOT a number!)\n\n"
            "TASK: Find TWO numbers and ONE operation, then compute.\n\n"
            "Reply in EXACTLY this format (4 lines):\n"
            "Decoded: <the clean English sentence>\n"
            "Numbers: <first_number>, <second_number>\n"
            "Operation: <+, -, *, or />\n"
            "Answer: <result with 2 decimal places>\n\n"
            "EXAMPLES:\n"
            "Text: lObStEr SwImS aT tWeNtY mEtErS aNd SlOwS bY fIvE\n"
            "Decoded: A lobster swims at twenty meters and slows by five\n"
            "Numbers: 20, 5\n"
            "Operation: -\n"
            "Answer: 15.00\n\n"
            "Text: ClAw ThIrTy TwO nEwToNs AnD aNoThEr FoUrTeEn nEwToNs ToTaL\n"
            "Decoded: Claw thirty two newtons and another fourteen newtons total\n"
            "Numbers: 32, 14\n"
            "Operation: +\n"
            "Answer: 46.00\n\n"
            "Text: cLaW fOrCe FoRtY NeWtOnS TiMeS tHrEe ClAwS\n"
            "Decoded: Claw force forty newtons times three claws\n"
            "Numbers: 40, 3\n"
            "Operation: *\n"
            "Answer: 120.00\n\n"
            "Now solve:\n"
            f"Text: {challenge_text}"
        )

        try:
            from llm_client import ask
            raw = ask(_SOLVE_PROMPT, temperature=0.0).strip()
            logger.debug("[razvedchik] 챌린지 LLM raw: %s", raw[:200])

            # "Answer: XX.XX" 라인에서 추출 시도
            answer_match = re.search(r'Answer:\s*([\-]?\d+(?:\.\d+)?)', raw)
            if answer_match:
                answer = float(answer_match.group(1))
                result = f"{answer:.2f}"
                logger.info("[razvedchik] 챌린지 LLM 해결: '%s' → %s", challenge_text[:60], result)
                return result

            # 폴백: 마지막 숫자 추출
            matches = re.findall(r'[\-]?\d+(?:\.\d+)?', raw)
            if matches:
                answer = float(matches[-1])
                result = f"{answer:.2f}"
                logger.info("[razvedchik] 챌린지 LLM 해결 (폴백): '%s' → %s", challenge_text[:60], result)
                return result

            logger.warning("[razvedchik] LLM 응답에서 숫자 추출 실패: '%s'", raw[:100])
        except Exception as e:
            logger.warning("[razvedchik] 챌린지 LLM 해결 실패: %s — 알고리즘 폴백", e)

        # ── 알고리즘 폴백 (LLM 불가 시) ──
        return MoltbookClient._solve_challenge_algorithmic(challenge_text)

    @staticmethod
    def _solve_challenge_algorithmic(challenge_text: str) -> str:
        """LLM 불가 시 규칙 기반 폴백. 두 숫자를 추출해 연산."""
        word_to_num = {
            'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
            'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
            'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
            'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
            'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,
            'eighty':80,'ninety':90,'hundred':100,
        }

        def _dedup(w):
            return re.sub(r'(.)\1+', r'\1', w)

        def _match(w):
            if w in word_to_num: return word_to_num[w]
            d = _dedup(w)
            if d in word_to_num: return word_to_num[d]
            return None

        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', challenge_text).lower()
        words = text.split()

        # 단위/무관 단어 제거용
        _SKIP = {'nooton','nootons','neuton','neutons','newton','newtons',
                  'notons','noton','meters','meter','lobster','claw','claws',
                  'force','speed','exerts','swims'}

        nums = []
        i = 0
        while i < len(words):
            w = words[i]
            if w in _SKIP or _dedup(w) in _SKIP:
                i += 1; continue
            val = _match(w)
            if val is not None:
                if val in (20,30,40,50,60,70,80,90) and i+1 < len(words):
                    nv = _match(words[i+1])
                    if nv is not None and 0 < nv < 10:
                        val += nv; i += 1
                nums.append(val)
            elif w.isdigit():
                nums.append(int(w))
            i += 1

        if len(nums) < 2:
            logger.warning("[razvedchik] 폴백 숫자 추출 부족: %s", nums)
            return f"{sum(nums):.2f}" if nums else "0.00"

        a, b = nums[0], nums[1]
        text_d = _dedup(text)
        if any(k in text_d for k in ('slow','subtract','minus','reduce','loses')):
            result = a - b
        elif any(k in text_d for k in ('times','multipli','multiply')):
            result = a * b
        elif any(k in text_d for k in ('divid','split','per')):
            result = a / b if b else 0
        else:
            result = a + b

        logger.info("[razvedchik] 폴백 챌린지 해결: %s op %s = %.2f", a, b, result)
        return f"{result:.2f}"

    def _verify_content(self, verification: dict) -> bool:
        """
        댓글/포스트 게시 후 응답의 verification 필드를 처리해 verify API 호출.

        Args:
            verification: response["comment"]["verification"] 또는 response["post"]["verification"]

        Returns:
            True if verified, False otherwise
        """
        if not verification:
            return False

        code = verification.get("verification_code", "")
        challenge_text = verification.get("challenge_text", "")
        expires_at = verification.get("expires_at", "")

        if not code or not challenge_text:
            logger.warning("[razvedchik] verification 필드 불완전: %s", verification)
            return False

        answer = self.solve_challenge_text(challenge_text)
        logger.info(
            "[razvedchik] Verification 시도 — code=%s, answer=%s, challenge='%s'",
            code[:40], answer, challenge_text[:80],
        )

        try:
            # verify 엔드포인트 직접 호출 (재시도 로직 우회 — verify 자체가 재시도 불필요)
            resp = self._client.request(
                "POST", "/verify",
                json={"verification_code": code, "answer": answer},
            )
            resp.raise_for_status()
            data = resp.json()
            verified = data.get("verified", False) or data.get("success", False)
            logger.info("[razvedchik] Verification 응답: %s → %s", data, "✅" if verified else "❌")
            return verified
        except Exception as e:
            logger.warning("[razvedchik] Verification 실패: %s", e)
            return False


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

        # 이미 처리한 포스트 ID 로드
        seen = _load_seen_posts()

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

        # 이미 읽은 포스트 제외
        new_posts = [p for p in all_posts if (p.get("id") or p.get("post_id", "")) not in seen]
        logger.info("[razvedchik] 새 포스트: %d개 (seen 제외 후)", len(new_posts))

        # seen에 새 포스트 ID 추가 후 저장
        for p in new_posts:
            pid = p.get("id") or p.get("post_id", "")
            if pid:
                seen.add(pid)
        _save_seen_posts(seen)

        # 흥미로운 포스트 필터링
        interesting = [p for p in new_posts if self._is_interesting(p)]
        logger.info("[razvedchik] 흥미로운 포스트: %d개", len(interesting))
        return interesting

    @staticmethod
    def _get_score(post: dict) -> int:
        """포스트의 점수를 추출."""
        return int(post.get("score", 0) or post.get("upvotes", 0) or post.get("karma", 0) or 0)

    def _is_interesting(self, post: dict) -> bool:
        """포스트가 흥미로운지 판별."""
        if self._get_score(post) > 5:
            return True

        # 제목 + 내용 합쳐서 키워드 검색
        text = " ".join([
            post.get("title", ""),
            post.get("content", ""),
            post.get("body", ""),
        ])

        for pat in _KW_PATTERNS:
            if pat.search(text):
                return True
        return False

    # ── 댓글 생성 ─────────────────────────────────────────────────────────────
    def generate_comment(self, post: dict, dry_run: bool = False) -> str:
        """
        포스트 내용을 분석해 댓글 생성.

        Args:
            post:    포스트 dict (title, content/body 포함)
            dry_run: True면 LLM 호출 없이 템플릿 반환

        Returns:
            댓글 문자열
        """
        title   = post.get("title", "(제목 없음)")
        content = (post.get("content") or post.get("body") or "")[:500]
        karma   = self._get_score(post)

        if dry_run:
            return f"Interesting take on '{title[:30]}' — worth examining further."

        prev = getattr(self, "_prev_debrief", "")
        context_line = f"\n\nContext from your commander:\n{prev}\n" if prev else ""
        prompt = (
            f"Write a comment on the following post.\n\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Karma: {karma}\n"
            f"{context_line}\n"
            f"Write in English. Be substantive — engage with the ideas, not just react."
        )
        from llm_client import ask_with_system

        # 최대 2회 시도 — LLM이 빈 응답을 반환하는 경우 재시도
        for attempt in range(2):
            try:
                comment = ask_with_system(
                    user_prompt=prompt,
                    system_prompt=RAZVEDCHIK_SYSTEM_PROMPT,
                    temperature=0.85 + (attempt * 0.05),
                )
                comment = comment.strip()
                if comment and len(comment) > 15:
                    return comment
                logger.warning("[razvedchik] LLM 빈/짧은 응답 (시도 %d): '%s'", attempt + 1, comment[:50])
            except Exception as e:
                logger.warning("[razvedchik] LLM 호출 실패 (시도 %d): %s", attempt + 1, e)

        # 폴백: 포스트 제목 기반 기본 댓글
        return f"This raises some important structural questions worth unpacking."

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
        resp = self.client.post(f"/posts/{post_id}/comments", body)

        # verification 처리
        comment_data = resp.get("comment", {})
        verification = comment_data.get("verification")
        if verification:
            verified = self.client._verify_content(verification)
            logger.info("[razvedchik] 댓글 verification: %s", "✅" if verified else "❌")

        return resp

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
        resp = self.client.post("/posts", body)

        # verification 처리
        post_data = resp.get("post", {})
        verification = post_data.get("verification")
        if verification:
            verified = self.client._verify_content(verification)
            logger.info("[razvedchik] 포스트 verification: %s", "✅" if verified else "❌")

        return resp

    def generate_observation_post(self, trending_topics: list[str]) -> tuple[str, str]:
        """
        최근 트렌드를 기반으로 Razvedchik 관찰 포스트 제목+내용 생성.

        Args:
            trending_topics: 최근 피드에서 추출한 주요 주제 리스트

        Returns:
            (title, content) 튜플
        """
        topics_str = ", ".join(trending_topics[:5]) if trending_topics else "general trends"
        prev = getattr(self, "_prev_debrief", "")
        context_line = f"\n\nContext from your commander:\n{prev}\n" if prev else ""
        prompt = (
            f"Trending topics on Moltbook right now: {topics_str}\n"
            f"{context_line}\n"
            f"Write a post engaging with these topics.\n"
            f"Format:\nTitle: (under 60 chars)\nBody: (your analysis and perspective)\n"
            f"Write in English."
        )
        try:
            from llm_client import ask_with_system
            result = ask_with_system(
                user_prompt=prompt,
                system_prompt=RAZVEDCHIK_POST_SYSTEM,
                temperature=0.9,
            )
            # 제목/본문 파싱 — "Title:" / "Body:" 마커 유무에 유연하게 대응
            lines = result.strip().splitlines()
            title = ""
            body_lines = []
            found_body_marker = False
            for line in lines:
                stripped = line.strip()
                if re.match(r'^[Tt]itle\s*:', stripped) and not title:
                    title = re.sub(r'^[Tt]itle\s*:\s*', '', stripped).strip().strip('"')
                elif re.match(r'^[Bb]ody\s*:', stripped) or found_body_marker:
                    found_body_marker = True
                    body_lines.append(re.sub(r'^[Bb]ody\s*:\s*', '', stripped))
                elif title and not found_body_marker:
                    # Title 이후 Body: 마커 없이 바로 본문이 시작되는 경우
                    if stripped:
                        body_lines.append(stripped)

            title   = title or f"On {topics_str[:40]}"
            content = "\n".join(body_lines).strip()
            # 빈 본문 방지: Title 줄 제거 후 나머지 전체를 본문으로
            if not content or len(content) < 20:
                fallback = re.sub(r'^[Tt]itle\s*:.*\n?', '', result.strip(), count=1).strip()
                fallback = re.sub(r'^[Bb]ody\s*:\s*', '', fallback).strip()
                content = fallback[:1500] if fallback else f"Observations on {topics_str}."
            return title, content

        except Exception as e:
            logger.warning("[razvedchik] 포스트 생성 실패: %s", e)
            title   = f"On {topics_str[:40]}"
            content = (
                f"Some interesting threads emerging on Moltbook around {topics_str}. "
                f"The structural dynamics here deserve closer examination."
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

    # ── /home 대시보드 (skill.md: 🔴 Do first) ─────────────────────────────
    def check_home(self) -> dict:
        """
        /home 대시보드 조회 — 알림, 내 포스트 활동, DM, 팔로우 피드 등 한 번에 확인.
        skill.md: "Start here every check-in."
        """
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.get("/home")

    # ── 내 포스트에 달린 댓글에 답글 (skill.md: 🔴 High) ─────────────────────
    def reply_to_activity(self, home_data: dict, dry_run: bool = False) -> list[dict]:
        """
        /home 응답의 activity_on_your_posts를 순회하며 새 댓글에 답글.

        Returns:
            각 답글 결과 리스트 [{post_id, comment_id, reply, success, ...}]
        """
        activities = home_data.get("activity_on_your_posts", [])
        if not activities:
            logger.info("[razvedchik] 내 포스트에 새 활동 없음")
            return []

        results = []
        for activity in activities[:3]:  # 최대 3개 포스트만 처리
            post_id = activity.get("post_id", "")
            post_title = activity.get("post_title", "")
            new_count = activity.get("new_notification_count", 0)

            if not post_id or new_count == 0:
                continue

            logger.info("[razvedchik] 내 포스트 '%s'에 %d개 새 댓글 — 답글 생성", post_title[:30], new_count)

            # 새 댓글 조회
            try:
                comments = self.get_comments(post_id, sort="new", limit=10)
            except Exception as e:
                logger.warning("[razvedchik] 댓글 조회 실패: %s", e)
                continue

            # 내 agent_id
            my_id = self.credentials.get("agent_id", "")

            for comment in comments[:3]:  # 최대 3개 댓글에 답글
                cid = comment.get("id", "")
                author = comment.get("author", {})
                author_id = author.get("id", "") if isinstance(author, dict) else ""
                if author_id == my_id:
                    continue  # 내 댓글은 건너뛰기
                comment_text = comment.get("content", "")
                if not comment_text:
                    continue

                # LLM으로 답글 생성
                reply_prompt = (
                    f"Someone replied to your post on Moltbook.\n\n"
                    f"Your post title: {post_title}\n"
                    f"Their comment: {comment_text[:400]}\n\n"
                    f"Write a brief, engaging reply (1-3 sentences). "
                    f"Be conversational and substantive. Write in English."
                )
                try:
                    from llm_client import ask_with_system
                    reply_text = ask_with_system(
                        user_prompt=reply_prompt,
                        system_prompt=RAZVEDCHIK_SYSTEM_PROMPT,
                        temperature=0.8,
                    ).strip()
                except Exception as e:
                    logger.warning("[razvedchik] 답글 LLM 생성 실패: %s", e)
                    reply_text = ""

                if not reply_text or len(reply_text) < 10:
                    continue

                if dry_run:
                    results.append({"post_id": post_id, "comment_id": cid, "reply": reply_text, "success": False, "dry_run": True})
                    continue

                try:
                    resp = self.post_comment(post_id, reply_text, parent_id=cid)
                    results.append({"post_id": post_id, "comment_id": cid, "reply": reply_text, "success": True, "response": resp})
                    logger.info("[razvedchik]   ✅ 답글 게시 완료 → %s", cid[:12])
                except Exception as e:
                    logger.warning("[razvedchik]   ❌ 답글 게시 실패: %s", e)
                    results.append({"post_id": post_id, "comment_id": cid, "reply": reply_text, "success": False, "error": str(e)})

            # 알림 읽음 처리
            try:
                self.mark_notifications_read(post_id)
            except Exception:
                pass

        return results

    # ── 알림 읽음 처리 ─────────────────────────────────────────────────────────
    def mark_notifications_read(self, post_id: Optional[str] = None) -> dict:
        """알림 읽음 처리. post_id 지정 시 해당 포스트만, 없으면 전체."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        if post_id:
            return self.client.post(f"/notifications/read-by-post/{post_id}")
        return self.client.post("/notifications/read-all")

    # ── 팔로우 (skill.md: 🟡 Medium) ──────────────────────────────────────────
    def follow_agent(self, agent_name: str) -> dict:
        """다른 molty 팔로우."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        logger.info("[razvedchik] 팔로우: %s", agent_name)
        return self.client.post(f"/agents/{agent_name}/follow")

    def unfollow_agent(self, agent_name: str) -> dict:
        """팔로우 해제."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.delete(f"/agents/{agent_name}/follow")

    # ── 개인화 피드 (skill.md: 🟡 Medium) ──────────────────────────────────────
    def get_feed(self, sort: str = "hot", limit: int = 25, filter_type: str = "all") -> list[dict]:
        """개인화 피드 조회 (구독 submolt + 팔로우 계정)."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        result = self.client.get("/feed", params={"sort": sort, "limit": limit, "filter": filter_type})
        return result if isinstance(result, list) else result.get("posts", [])

    # ── 다운보트 ────────────────────────────────────────────────────────────────
    def downvote_post(self, post_id: str) -> dict:
        """포스트 다운보트."""
        if not self.client:
            raise RuntimeError("MOLTBOOK_API_KEY 미설정")
        return self.client.post(f"/posts/{post_id}/downvote")

    # ── 정찰 보고서 ───────────────────────────────────────────────────────────
    def _build_report(
        self,
        scanned_posts: list[dict],
        selected_posts: list[dict],
        comment_results: list[dict],
        post_result: Optional[dict] = None,
        *,
        home_data: Optional[dict] = None,
        reply_results: Optional[list[dict]] = None,
        upvoted_posts: Optional[list[str]] = None,
        followed_agents: Optional[list[str]] = None,
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
                "replies_posted":       sum(1 for r in (reply_results or []) if r.get("success")),
                "upvoted_count":        len(upvoted_posts or []),
                "followed_count":       len(followed_agents or []),
                "observation_posted":   post_result is not None,
            },
            "home_summary": {
                "karma":        (home_data or {}).get("your_account", {}).get("karma"),
                "unread_notifications": (home_data or {}).get("your_account", {}).get("unread_notification_count"),
                "activity_posts": len((home_data or {}).get("activity_on_your_posts", [])),
            } if home_data else None,
            "reply_results":   reply_results or [],
            "selected_posts": [
                {
                    "id":      p.get("id", ""),
                    "title":   p.get("title", ""),
                    "score":   self._get_score(p),
                    "submolt": p.get("submolt", {}).get("name", "") if isinstance(p.get("submolt"), dict) else str(p.get("submolt", "")),
                    "url":     p.get("url", ""),
                }
                for p in selected_posts
            ],
            "comment_results": comment_results,
            "upvoted_posts":   upvoted_posts or [],
            "followed_agents": followed_agents or [],
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
                f"답글: {summary.get('replies_posted', 0)}개",
                f"댓글: {summary['comments_posted']}개 성공 / {summary['comments_failed']}개 실패",
                f"업보트: {summary.get('upvoted_count', 0)}개 / 팔로우: {summary.get('followed_count', 0)}개",
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
        메인 순찰 루프 — skill.md 우선순위 기반.

        순서 (Moltbook skill.md "Everything You Can Do" 표 기반):
          1. /home 대시보드 확인            (🔴 Do first)
          2. 내 포스트에 달린 댓글에 답글    (🔴 High)
          3. 피드 스캔 + 댓글 + 업보트      (🟠 High)
          4. 흥미로운 작성자 팔로우          (🟡 Medium)
          5. 관찰 포스트 게시               (🔵 When inspired)
          6. 보고서 저장 + 디브리핑 + 알림

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

        # 0. 이전 디브리핑 컨텍스트 로드
        try:
            from razvedchik_debrief import get_last_debrief_summary
            self._prev_debrief = get_last_debrief_summary()
            if self._prev_debrief:
                logger.info("[razvedchik] 이전 디브리핑 컨텍스트 로드됨")
        except Exception:
            self._prev_debrief = ""

        # ── STEP 1: /home 대시보드 (🔴 Do first) ──────────────────────────────
        logger.info("[razvedchik] STEP 1: /home 대시보드 확인")
        home_data = {}
        try:
            home_data = self.check_home()
            acct = home_data.get("your_account", {})
            logger.info(
                "[razvedchik]   karma=%s, unread=%s, activity=%d posts",
                acct.get("karma", "?"),
                acct.get("unread_notification_count", "?"),
                len(home_data.get("activity_on_your_posts", [])),
            )
        except Exception as e:
            logger.warning("[razvedchik]   /home 조회 실패: %s", e)

        # ── STEP 2: 내 포스트에 달린 댓글에 답글 (🔴 High) ────────────────────
        logger.info("[razvedchik] STEP 2: 내 포스트 활동 확인 → 답글")
        reply_results = []
        if home_data.get("activity_on_your_posts"):
            try:
                reply_results = self.reply_to_activity(home_data, dry_run=dry_run)
                logger.info("[razvedchik]   답글 %d개 작성", sum(1 for r in reply_results if r.get("success")))
            except Exception as e:
                logger.warning("[razvedchik]   답글 처리 실패: %s", e)

        # ── STEP 3: 피드 스캔 + 댓글 + 업보트 (🟠 High) ──────────────────────
        logger.info("[razvedchik] STEP 3: 피드 스캔")

        # 개인화 피드 우선 사용, 결과 없으면 글로벌 피드 폴백
        try:
            feed_posts = self.get_feed(sort="hot", limit=25)
            if not feed_posts:
                feed_posts = self.get_feed(sort="new", limit=25)
        except Exception:
            feed_posts = []

        # 개인화 피드에서 새 포스트 필터링
        seen = _load_seen_posts()
        new_feed = [p for p in feed_posts if (p.get("id") or p.get("post_id", "")) not in seen]

        # 글로벌 스캔도 병행 (새 포스트 발견용)
        try:
            interesting_posts = self.scan_feed(submolts=submolts, limit=25)
        except Exception as e:
            logger.error("[razvedchik] 글로벌 피드 스캔 실패: %s", e)
            interesting_posts = []

        # 두 소스 합치기 (중복 제거)
        all_candidates = {(p.get("id") or p.get("post_id", "")): p for p in new_feed}
        for p in interesting_posts:
            pid = p.get("id") or p.get("post_id", "")
            if pid not in all_candidates:
                all_candidates[pid] = p
        all_posts = list(all_candidates.values())

        # 상위 선별
        selected = sorted(all_posts, key=self._get_score, reverse=True)[:max_comments]
        logger.info("[razvedchik]   피드 %d + 글로벌 %d → 선별 %d개",
                    len(new_feed), len(interesting_posts), len(selected))

        # 각 포스트에 댓글 + 업보트
        comment_results = []
        upvoted_posts = []
        followed_agents = []

        for post in selected:
            post_id = post.get("id") or post.get("post_id", "")
            title   = post.get("title", "(제목 없음)")
            logger.info("[razvedchik]   댓글+업보트 — %s", title[:40])

            # 업보트 (흥미로운 포스트니까)
            if not dry_run and post_id:
                try:
                    upvote_resp = self.upvote_post(post_id)
                    upvoted_posts.append(post_id)
                    logger.info("[razvedchik]     ⬆️ 업보트 완료")

                    # skill.md: 업보트 응답에서 팔로우 여부 확인 → 팔로우
                    author_name = upvote_resp.get("author", {}).get("name", "")
                    already_following = upvote_resp.get("already_following", True)
                    if author_name and not already_following:
                        # STEP 4를 여기서 함께 처리 (자연스러운 팔로우)
                        try:
                            self.follow_agent(author_name)
                            followed_agents.append(author_name)
                            logger.info("[razvedchik]     👥 팔로우: %s", author_name)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug("[razvedchik]     업보트 실패: %s", e)

            # 댓글 생성
            comment_text = self.generate_comment(post, dry_run=dry_run)
            logger.info("[razvedchik]     댓글 (%d자): %s", len(comment_text), comment_text[:80])

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

            # 빈 댓글 게시 방지
            if not comment_text or len(comment_text) < 10:
                logger.warning("[razvedchik]     ⚠️ 빈/짧은 댓글 — 게시 건너뜀")
                comment_results.append({
                    "post_id":    post_id,
                    "post_title": title,
                    "comment":    comment_text,
                    "success":    False,
                    "error":      "empty_or_too_short_comment",
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
                logger.info("[razvedchik]     ✅ 댓글 게시 완료")
            except Exception as e:
                logger.error("[razvedchik]     ❌ 댓글 게시 실패: %s", e)
                comment_results.append({
                    "post_id":    post_id,
                    "post_title": title,
                    "comment":    comment_text,
                    "success":    False,
                    "error":      str(e),
                })

        # ── STEP 5: 관찰 포스트 게시 (🔵 When inspired) ───────────────────────
        observation_result = None
        if post_observation_flag:
            logger.info("[razvedchik] STEP 5: 관찰 포스트 생성")
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

        # ── STEP 6: 보고서 저장 ────────────────────────────────────────────────
        logger.info("[razvedchik] STEP 6: 보고서 저장")
        report = self._build_report(
            all_posts, selected, comment_results, observation_result,
            home_data=home_data,
            reply_results=reply_results,
            upvoted_posts=upvoted_posts,
            followed_agents=followed_agents,
        )
        report_path = self._save_report(report)

        # ── STEP 7: Cyber-Lenin 디브리핑 ───────────────────────────────────────
        logger.info("[razvedchik] STEP 7: Cyber-Lenin 디브리핑")
        try:
            from razvedchik_debrief import run_debrief
            debrief = run_debrief(report)
            logger.info("[razvedchik]   디브리핑 완료 — %d턴 대화", len(debrief))
        except Exception as e:
            logger.warning("[razvedchik]   디브리핑 실패: %s", e)

        # ── STEP 8: 텔레그램 알림 ──────────────────────────────────────────────
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
                score   = Razvedchik._get_score(p)
                title   = p.get("title", "(제목 없음)")
                sub     = p.get("submolt", {})
                sub_name = sub.get("name", "?") if isinstance(sub, dict) else str(sub)
                print(f"  {i:2d}. [{sub_name}] {title[:60]} (score={score})")
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
