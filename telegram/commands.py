"""telegram_commands.py — Extracted command/message handlers for telegram_bot.py.

All handlers access shared dependencies via the module-level `_ctx` dict,
which is populated by `register_handlers()` at startup.
"""

import os
import sys
import json
import asyncio
import logging
import base64
from datetime import datetime
from aiogram import F, Router
from aiogram.types import (
    Message, BufferedInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
)
from aiogram.filters import Command

from shared import KST
from db import query as _query, execute as _execute, query_one as _query_one, get_conn as _get_conn
from psycopg2.extras import RealDictCursor
from replicate_image_service import (
    generate_image,
    is_replicate_configured,
    build_soviet_prompt,
    ReplicateImageError,
)

logger = logging.getLogger(__name__)

_SVC_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "svc")


def _model_name_for_provider(provider: str) -> str:
    """Resolve the current chat-tier model name for a given provider."""
    from bot_config import (
        _TIER_MAP, _OPENAI_MODEL_MAP, _DEEPSEEK_MODEL_MAP,
        _MODEL_ALIAS_MAP, _resolved_models,
    )
    tier = _ctx["config"].get("chat_model", "high")
    alias = _TIER_MAP.get(provider, _TIER_MAP["claude"]).get(tier, tier)
    if provider == "local":
        from llm.client import MOON_MODEL
        return MOON_MODEL
    if provider == "openai":
        return _OPENAI_MODEL_MAP.get(alias, alias)
    if provider == "deepseek":
        return _DEEPSEEK_MODEL_MAP.get(alias, alias)
    model_alias, fallback = _MODEL_ALIAS_MAP.get(alias, (alias, alias))
    return _resolved_models.get(alias, fallback)


def _notify_model_switch(old_provider: str, new_provider: str):
    """Log a provider/model switch to system alerts and DB."""
    old_model = _model_name_for_provider(old_provider)
    new_model = _model_name_for_provider(new_provider)
    alert = f"LLM 전환: {old_model} → {new_model} ({old_provider} → {new_provider})"
    _ctx["add_system_alert"](alert)
    _ctx["log_event"]("warning", "config", alert)


async def _svc_health() -> str:
    """Run `svc status` and return a compact Telegram-friendly summary."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", _SVC_SCRIPT, "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        raw = stdout.decode(errors="replace")
    except Exception as e:
        return f"⚠️ svc status 실행 실패: {e}"

    # Parse ANSI output into compact lines
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', raw)  # strip ANSI colors
    lines = []
    for line in clean.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        # ● name: state  or  ● name (container): state
        m = re.match(r'● (\S+)(?:\s+\([^)]+\))?: (\S+)', line)
        if m:
            name, state = m.group(1), m.group(2)
            icon = "✅" if state in ("active", "activating", "running") else "❌"
            lines.append(f"  {icon} {name}: {state}")
    return "\n".join(lines) if lines else "⚠️ 파싱 실패"


# ── Module-level state (local to handlers) ─────────────────────────
_ctx: dict = {}
_pending_approvals: dict = {}  # 자가수정 승인 대기 (approval_id → entry)
_reflection_counter: dict[int, int] = {}

_HELP_TEXT = """\
*레닌봇 커맨드 목록*

*대화*
/chat <메시지> — CLAW 파이프라인 질의 (RAG+KG+전략)
  일반 메시지 — Claude 직접 대화 (도구 사용 가능)
/clear — 대화 히스토리 초기화

*태스크*
/task <내용> — 백그라운드 태스크 등록 (Sonnet, $1 예산)
/status — 시스템 대시보드 (태스크·에러·KG)
/status\\_auto — 자율 생성 태스크 확인
/report <id> — 태스크 리포트 파일 재전송

*이메일*
/email — 이메일 현황 (폴링 + 최근 기록 + 본문 미리보기)

*이미지*
/image <프롬프트> — Replicate로 소련풍 게임/포스터 이미지 생성

*스케줄*
/schedule <cron> | <내용> — 정기 태스크 등록
  예: `/schedule 0 9 * * * | 오늘의 뉴스 브리핑`
/schedules — 등록된 스케줄 목록
/unschedule <id> — 스케줄 삭제

*자율 프로젝트*
/projects — 자율 프로젝트 목록
/autonomous \\[on|off|status] — 자율 에이전트 시간별 tick 중단/재개 (인자 없으면 상태)
/advise \\[id] <조언> — 다음 tick 에 전달할 조언 등록 (multi-line OK)
/advisories \\[id] — 대기 중/소비된 조언 조회

*시스템*
/kg — 지식그래프 현황 조회
/errors \\[n] \\[error|warning] — 에러/경고 로그
/agents — 에이전트 현황 및 외부 프로세스 상태
/config — 설정 패널 (모델, 예산, 라운드 수)
/fallback — 모델 토글 (sonnet ↔ haiku, API 과부하 시)
/provider — 대화 제공자 전환 (Claude ↔ OpenAI ↔ DeepSeek, 키가 있는 것만)
/restart \\[telegram|api|all] — 서비스 재시작만 (기본: telegram)
/deploy \\[telegram|api|frontend|all] — 서버 배포 (git pull + restart, 기본: all)
/modify <파일> | <이유> | <내용> — 서버 파일 수정

/help — 이 도움말 표시
"""


_REFLECTION_PROMPT = """\
아래 대화에서 배울 점을 추출해라. 다음 카테고리별로 1개씩만 (해당 없으면 생략):

- **lesson**: 새로 배운 사실이나 지식
- **mistake**: 잘못된 답변, 도구 오용, 사용자 수정이 있었던 부분
- **pattern**: 반복적인 사용자 요구나 질문 패턴
- **insight**: 분석/논의에서 도출된 깊은 통찰
- **observation**: 기술적 발견이나 시스템 동작에 대한 관찰

각 항목을 한 줄로, 앞에 카테고리를 붙여 작성. 예:
lesson: 시리아 내전에서 러시아의 군사 개입은 2015년부터이며...
mistake: 사용자가 물어본 것은 경제 제재인데 군사적 측면만 답변했음
pattern: 사용자는 자주 한국 정치와 국제 정세의 연관성을 묻는다

배울 게 없으면 "NONE"이라고만 답해.

대화:
"""


# ── Handler functions ──────────────────────────────────────────────

async def cmd_start(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await message.answer(
        "레닌봇 텔레그램 인터페이스입니다.\n\n" + _HELP_TEXT,
        parse_mode="Markdown",
    )


async def cmd_help(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await message.answer(_HELP_TEXT, parse_mode="Markdown")


async def cmd_clear(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await asyncio.to_thread(_ctx["clear_chat_history"], message.from_user.id)
    # Close active mission on history clear
    try:
        from telegram.mission import get_active_mission, close_mission
        mission = await asyncio.to_thread(get_active_mission, message.from_user.id)
        if mission:
            await asyncio.to_thread(close_mission, mission["id"])
    except Exception:
        pass
    await message.answer("대화 히스토리가 초기화되었습니다.")


async def cmd_mission(message: Message):
    """View, create, or close the active mission.
    Usage:
      /mission               — 현재 미션 상태 조회
      /mission create <제목>  — 새 미션 생성
      /mission close         — 활성 미션 종료
    """
    if not _ctx["is_allowed"](message.from_user.id):
        return
    uid = message.from_user.id
    raw_arg = (message.text or "").removeprefix("/mission").strip()
    arg_lower = raw_arg.lower()

    try:
        from telegram.mission import get_active_mission, get_mission_events, close_mission, create_mission
        mission = await asyncio.to_thread(get_active_mission, uid)

        # --- CREATE ---
        if arg_lower.startswith("create"):
            title = raw_arg[len("create"):].strip()
            if not title:
                await message.answer("❌ 미션 제목을 입력하세요.\n예: `/mission create 3월 금값 분석`")
                return
            if mission:
                await message.answer(
                    f"⚠️ 이미 활성 미션이 있습니다: *#{mission['id']}* {mission['title']}\n"
                    f"먼저 `/mission close`로 종료하세요."
                )
                return
            new_mission = await asyncio.to_thread(create_mission, uid, title)
            await message.answer(
                f"✅ 미션 생성됨\n"
                f"🎯 *#{new_mission['id']}*: {new_mission['title']}"
            )
            return

        # --- CLOSE ---
        if arg_lower == "close":
            if not mission:
                await message.answer("활성 미션이 없습니다.")
                return
            await asyncio.to_thread(close_mission, mission["id"])
            await message.answer(f"✅ 미션 #{mission['id']} 종료: {mission['title']}")
            return

        # --- STATUS (default) ---
        if not mission:
            await message.answer(
                "활성 미션이 없습니다.\n"
                "새 미션을 만들려면: `/mission create <제목>`"
            )
            return
        events = await asyncio.to_thread(get_mission_events, mission["id"], 10)
        lines = [f"🎯 *미션 #{mission['id']}*: {mission['title']}", f"생성: {mission['created_at']}"]
        if events:
            lines.append(f"\n타임라인 ({len(events)}건):")
            for e in events:
                lines.append(f"  `[{e['source']}]` {e['event_type']}: {str(e['content'] or '')[:100]}")
        await message.answer("\n".join(lines))
    except Exception as e:
        await message.answer(f"미션 오류: {e}")


async def cmd_errors(message: Message):
    """Show recent error/warning log entries."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/errors").strip()
    # Parse optional limit and level filter
    # Usage: /errors [n] [error|warning|all]
    limit = 20
    level_filter = None
    for token in arg.split():
        if token.isdigit():
            limit = min(int(token), 50)
        elif token.lower() in ("error", "warning", "warn"):
            level_filter = "error" if token.lower() == "error" else "warning"
    try:
        if level_filter:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log WHERE level = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (level_filter, limit),
            )
        else:
            rows = await asyncio.to_thread(
                _query,
                "SELECT id, level, source, message, detail, task_id, created_at "
                "FROM telegram_error_log ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
    except Exception as e:
        await message.answer(f"에러 로그 조회 실패: {e}")
        return
    if not rows:
        await message.answer("✅ 기록된 에러/경고 없음.")
        return
    level_icons = {"error": "🔴", "warning": "🟡"}
    lines = [f"🗒️ *에러/경고 로그* (최근 {len(rows)}건)\n"]
    for r in rows:
        icon = level_icons.get(r["level"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M:%S")
        task_info = f" [태스크#{r['task_id']}]" if r["task_id"] else ""
        lines.append(
            f"{icon} `{ts}` [{r['source']}]{task_info}\n"
            f"   {r['message'][:120]}"
        )
    for chunk in _ctx["split_message"]("\n\n".join(lines)):
        await message.answer(chunk, parse_mode="Markdown")


async def cmd_chat(message: Message):
    """Deprecated: CLAW pipeline has been replaced by unified claude_loop."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await message.answer("이 명령은 더 이상 사용되지 않습니다. 직접 메시지를 보내세요.")


async def cmd_task(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    content = (message.text or "").removeprefix("/task").strip()
    if not content:
        await message.answer("사용법: /task <내용>")
        return
    try:
        uid = message.from_user.id

        # Resolve mission: use active or auto-create
        mission_id = None
        try:
            from telegram.mission import get_active_mission, create_mission
            mission = await asyncio.to_thread(get_active_mission, uid)
            if mission:
                mission_id = mission["id"]
        except Exception as e:
            logger.warning("Mission lookup failed: %s", e)

        rows = await asyncio.to_thread(
            _query,
            "INSERT INTO telegram_tasks (user_id, content, mission_id) VALUES (%s, %s, %s) RETURNING id",
            (uid, content, mission_id),
        )
        task_id = rows[0]["id"] if rows else None
        msg = f"태스크가 큐에 추가되었습니다:\n{content}"

        # Auto-create mission if none existed
        if task_id and mission_id is None:
            try:
                from telegram.mission import create_mission
                mission = await asyncio.to_thread(create_mission, uid, content[:80], task_id)
                mission_id = mission["id"]
                msg += f"\n\n🎯 미션 #{mission_id} 자동 생성"
            except Exception as e:
                logger.warning("Mission auto-create failed: %s", e)

        await message.answer(msg)
    except Exception as e:
        logger.error("Task insert error: %s", e)
        await message.answer(f"태스크 등록 실패: {e}")


async def _generate_and_send_image(message: Message, prompt: str, *, style: str = "game"):
    if not is_replicate_configured():
        await message.answer("❌ REPLICATE_API_TOKEN이 설정되지 않았습니다.")
        return

    status = await message.answer("🎨 이미지 생성 중... 기본 모델은 flux-schnell이다.")
    try:
        result = await generate_image(prompt, style=style, download=True)
        prompt_preview = build_soviet_prompt(prompt, style=style)[:180]
        caption = (
            f"✅ 이미지 생성 완료\n"
            f"model: `{result['model']}`\n"
            f"prediction: `{result['prediction_id']}`\n"
            f"prompt: `{prompt_preview}`\n"
            f"url: {result['image_urls'][0]}"
        )
        local_path = result.get("local_path")
        if local_path and os.path.isfile(local_path):
            with open(local_path, "rb") as f:
                photo = BufferedInputFile(f.read(), filename=os.path.basename(local_path))
            await message.answer_photo(photo, caption=caption, parse_mode="Markdown")
        else:
            await message.answer(caption, parse_mode="Markdown")
        try:
            await status.edit_text("🎨 이미지 생성 완료")
        except Exception:
            pass
    except ReplicateImageError as e:
        info = e.info
        logger.warning("image generation failed category=%s retryable=%s detail=%s", info.category, info.retryable, info.detail)
        _ctx["log_event"](
            "warning" if info.retryable else "error",
            "image",
            f"Replicate image generation failed ({info.category})",
            detail=f"retryable={info.retryable} | detail={info.detail} | prompt={prompt[:400]}",
        )
        suffix = " 재시도 가능." if info.retryable else " 프롬프트/옵션 수정 후 다시 시도해라."
        user_message = f"{info.user_message}{suffix}"
        try:
            await status.edit_text(user_message)
        except Exception:
            await message.answer(user_message)
    except Exception as e:
        logger.error("image generation failed: %s", e)
        _ctx["log_event"]("error", "image", f"Replicate image generation failed: {e}", detail=prompt[:500])
        fallback = "❌ 이미지 생성 실패. 내부 오류다. 잠시 후 다시 시도해라."
        try:
            await status.edit_text(fallback)
        except Exception:
            await message.answer(fallback)


async def cmd_image(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    prompt = (message.text or "").removeprefix("/image").strip()
    if not prompt:
        await message.answer("사용법: /image <프롬프트>")
        return
    await _generate_and_send_image(message, prompt, style="game")


async def cmd_stats(message: Message):
    """시스템 리소스 현황 — CPU/메모리/디스크/네트워크 (psutil 실시간 + JSON 추이)."""
    if not _ctx["is_allowed"](message.from_user.id):
        return

    import psutil
    from datetime import timezone, timedelta
    from scripts.metrics_snapshot import parse_cpu_json, parse_memory_json, _sparkline

    KST_tz = timezone(timedelta(hours=9))
    now_kst = datetime.now(timezone.utc).astimezone(KST_tz)
    ts_str = now_kst.strftime("%Y-%m-%d %H:%M KST")

    await message.answer("⏳ 메트릭 수집 중...")

    def _collect():
        cpu = psutil.cpu_percent(interval=1)
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        import time
        net_before = psutil.net_io_counters()
        time.sleep(0.5)
        net_after = psutil.net_io_counters()
        rx_kbs = round((net_after.bytes_recv - net_before.bytes_recv) / 1024 / 0.5, 1)
        tx_kbs = round((net_after.bytes_sent - net_before.bytes_sent) / 1024 / 0.5, 1)
        return {
            "cpu": cpu,
            "mem_pct": vm.percent,
            "mem_used": round(vm.used / 1024**3, 2),
            "mem_total": round(vm.total / 1024**3, 2),
            "mem_avail": round(vm.available / 1024**3, 2),
            "disk_pct": disk.percent,
            "disk_used": round(disk.used / 1024**3, 1),
            "disk_total": round(disk.total / 1024**3, 1),
            "disk_free": round(disk.free / 1024**3, 1),
            "rx_kbs": rx_kbs,
            "tx_kbs": tx_kbs,
        }

    def _make_bar(pct: float, width: int = 20) -> str:
        filled = int(round(pct / 100 * width))
        filled = max(0, min(filled, width))
        return "█" * filled + "░" * (width - filled)

    def _alert(pct: float) -> str:
        if pct >= 90:
            return " 🔴"
        if pct >= 75:
            return " 🟡"
        return " 🟢"

    try:
        m = await asyncio.to_thread(_collect)

        def _get_sparks():
            try:
                cpu_rows = parse_cpu_json(12)
                mem_rows = parse_memory_json(12)
                cpu_spark = _sparkline([v for _, v in cpu_rows]) if cpu_rows else "—"
                mem_spark = _sparkline([v for _, v in mem_rows]) if mem_rows else "—"
                return cpu_spark, mem_spark
            except Exception:
                return "—", "—"

        cpu_spark, mem_spark = await asyncio.to_thread(_get_sparks)

        out = [
            f"📊 *시스템 메트릭* — {ts_str}",
            "",
            f"🖥 *CPU*{_alert(m['cpu'])}",
            f"  사용률: `{m['cpu']:.1f}%`",
            f"  `[{_make_bar(m['cpu'])}]`",
            f"  추이(12h): `{cpu_spark}`",
            "",
            f"🧠 *메모리*{_alert(m['mem_pct'])}",
            f"  사용: `{m['mem_used']} / {m['mem_total']} GiB ({m['mem_pct']:.1f}%)`",
            f"  여유: `{m['mem_avail']} GiB`",
            f"  `[{_make_bar(m['mem_pct'])}]`",
            f"  추이(12h): `{mem_spark}`",
            "",
            f"💾 *디스크 (/)*{_alert(m['disk_pct'])}",
            f"  사용: `{m['disk_used']} / {m['disk_total']} GiB ({m['disk_pct']:.1f}%)`",
            f"  여유: `{m['disk_free']} GiB`",
            f"  `[{_make_bar(m['disk_pct'])}]`",
            "",
            f"🌐 *네트워크*",
            f"  ↓ `{m['rx_kbs']} kB/s`  ↑ `{m['tx_kbs']} kB/s`",
        ]
        await message.answer("\n".join(out), parse_mode="Markdown")
    except Exception as e:
        logger.error("cmd_stats error: %s", e)
        await message.answer(f"⚠️ 메트릭 수집 실패: {e}")

async def cmd_status(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    uid = message.from_user.id

    # Gather all dashboard data in parallel
    tasks_f = asyncio.to_thread(
        _query,
        "SELECT id, content, status, created_at FROM telegram_tasks "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
        (uid,),
    )
    errors_f = asyncio.to_thread(
        _query,
        "SELECT level, count(*) AS cnt FROM telegram_error_log "
        "WHERE created_at > NOW() - INTERVAL '24 hours' "
        "GROUP BY level ORDER BY level",
        None,
    )
    task_stats_f = asyncio.to_thread(
        _query,
        "SELECT status, count(*) AS cnt FROM telegram_tasks "
        "GROUP BY status",
        None,
    )
    health_f = _svc_health()

    try:
        tasks, errors, task_stats, svc_health = await asyncio.gather(
            tasks_f, errors_f, task_stats_f, health_f
        )
    except Exception as e:
        logger.error("Status dashboard query error: %s", e)
        await message.answer(f"대시보드 조회 실패: {e}")
        return

    # -- Build dashboard --
    lines = ["*시스템 대시보드*\n"]

    # 1. Task summary
    stat_map = {r["status"]: r["cnt"] for r in task_stats}
    total_tasks = sum(stat_map.values())
    lines.append(
        f"*태스크* ({total_tasks}건): "
        f"✅{stat_map.get('done', 0)} "
        f"⏳{stat_map.get('pending', 0)} "
        f"🔄{stat_map.get('processing', 0) + stat_map.get('queued', 0)} "
        f"❌{stat_map.get('failed', 0)} "
        f"🔀{stat_map.get('handed_off', 0)}"
    )

    # 2. Error counts (24h)
    err_map = {r["level"]: r["cnt"] for r in errors}
    err_total = sum(err_map.values())
    if err_total:
        lines.append(
            f"*에러 (24h)*: 🔴error {err_map.get('error', 0)} "
            f"🟡warning {err_map.get('warning', 0)}"
        )
    else:
        lines.append("*에러 (24h)*: 없음")

    # 3. Service health
    lines.append(f"*서비스:*\n{svc_health}")

    # 4. KG stats (quick, non-blocking)
    try:
        from shared import fetch_kg_stats
        kg = await asyncio.to_thread(fetch_kg_stats)
        if "error" not in kg:
            entity_total = sum(v for v in kg.get("entity_types", {}).values())
            lines.append(
                f"*KG*: 엔티티 {entity_total} | "
                f"관계 {kg.get('edge_count', 0)} | "
                f"에피소드 {kg.get('episode_count', 0)}"
            )
        else:
            lines.append(f"*KG*: ⚠️ {kg['error'][:60]}")
    except Exception as e:
        lines.append(f"*KG*: ⚠️ 조회 실패")

    # 4. Recent tasks
    if tasks:
        lines.append("\n*최근 태스크:*")
        status_icons = {"pending": "⏳", "processing": "🔄", "queued": "📋", "done": "✅", "failed": "❌", "handed_off": "🔀"}
        for r in tasks:
            icon = status_icons.get(r["status"], "❓")
            ts = r["created_at"].strftime("%m/%d %H:%M")
            preview = r["content"][:45]
            lines.append(f"{icon} `[{r['id']}]` {preview}\n   {r['status']} | {ts}")
    else:
        lines.append("\n태스크 없음")

    await message.answer("\n".join(lines), parse_mode="Markdown")


async def cmd_kg(message: Message):
    """Directly show KG stats — no LLM involved."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    from shared import fetch_kg_stats
    await message.answer("KG 조회 중...")
    try:
        stats = await asyncio.to_thread(fetch_kg_stats)
    except Exception as e:
        await message.answer(f"KG 조회 실패: {e}")
        return
    if "error" in stats:
        await message.answer(f"⚠️ KG 오류: {stats['error']}")
        return

    lines = ["📊 *지식그래프 현황* (Neo4j Local)\n"]
    lines.append(f"엔티티: {sum(v for v in stats.get('entity_types', {}).values())}개")
    for label, cnt in stats.get("entity_types", {}).items():
        lines.append(f"  {label}: {cnt}")
    lines.append(f"관계(엣지): {stats.get('edge_count', 0)}개")
    lines.append(f"에피소드: {stats.get('episode_count', 0)}건")
    episodes = stats.get("recent_episodes", [])
    if episodes:
        lines.append("\n*최근 에피소드:*")
        for ep in episodes:
            lines.append(f"  • {ep.get('name', '?')} [{ep.get('group_id', '')}]")
    await message.answer("\n".join(lines))


async def cmd_report(message: Message):
    """Directly fetch a task report from DB and send as file — no LLM involved."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/report").strip()
    if not arg:
        await message.answer("사용법: /report <task_id>")
        return
    try:
        task_id = int(arg)
    except ValueError:
        await message.answer("task_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "SELECT id, content, status, result FROM telegram_tasks WHERE id = %s",
            (task_id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not row:
        await message.answer(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
        return
    if row["status"] != "done" or not row.get("result"):
        await message.answer(f"태스크 #{task_id} 상태: {row['status']} — 완료된 리포트가 없습니다.")
        return
    report = row["result"]
    doc = BufferedInputFile(report.encode("utf-8"), filename=f"report_task_{task_id}.md")
    await message.answer_document(doc, caption=f"태스크 #{task_id} 리포트 (DB 원문, {len(report)}자)")


async def cmd_status_auto(message: Message):
    """Show recent self-generated (autonomous) tasks."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, status, agent_type, created_at FROM telegram_tasks "
            "WHERE user_id = 0 ORDER BY created_at DESC LIMIT 10",
        )
    except Exception as e:
        logger.error("Auto-task status query error: %s", e)
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("자율 생성된 태스크가 없습니다.")
        return
    status_icons = {"pending": "⏳", "processing": "🔄", "queued": "📋", "done": "✅", "failed": "❌", "handed_off": "🔀"}
    lines = ["🤖 *자율 생성 태스크* (최근 10건)\n"]
    for r in rows:
        icon = status_icons.get(r["status"], "❓")
        ts = r["created_at"].strftime("%m/%d %H:%M")
        preview = r["content"][:60]
        agent = r.get("agent_type") or "?"
        lines.append(f"{icon} [{r['id']}] [{agent}] {preview}\n   상태: {r['status']} | {ts}")
    await message.answer("\n\n".join(lines))


async def cmd_autonomous(message: Message):
    """Show or toggle the hourly autonomous project loop. Usage: /autonomous [on|off]."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    from bot_config import is_autonomous_active, set_autonomous_active, _get_autonomous_provider

    parts = (message.text or "").strip().split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""

    if arg in {"on", "active", "resume", "start"}:
        state = set_autonomous_active(True)
        await message.answer("▶️ 자율 에이전트 *active* — 다음 시간 tick부터 정상 동작합니다.", parse_mode="Markdown")
    elif arg in {"off", "inactive", "pause", "stop"}:
        state = set_autonomous_active(False)
        await message.answer("⏸ 자율 에이전트 *inactive* — hourly tick이 일시 중단됐습니다. (타이머는 계속 fire하지만 guard가 skip)", parse_mode="Markdown")
    elif arg in {"", "status"}:
        state = is_autonomous_active()
        label = "▶️ active" if state else "⏸ inactive"
        provider = _get_autonomous_provider()
        from agents import get_agent
        spec = get_agent("autonomous_project")
        model_key = spec.model if provider == "claude" and spec.model else _ctx["config"].get("task_model", "high")
        model = _ctx["tier_to_display"](model_key, provider)
        await message.answer(
            f"자율 에이전트 상태: *{label}*\n"
            f"provider: `{provider}`\n"
            f"model: `{model}`\n\n"
            f"사용법: `/autonomous on` | `/autonomous off` | `/autonomous status`",
            parse_mode="Markdown",
        )
    else:
        await message.answer(
            "알 수 없는 인자입니다. 사용법: `/autonomous on` | `/autonomous off` | `/autonomous status`",
            parse_mode="Markdown",
        )


async def cmd_email(message: Message):
    """Show recent email activity, poll for new messages, and include body previews."""
    if not _ctx["is_allowed"](message.from_user.id):
        return

    def _clean_preview(value: str | None, max_chars: int = 280) -> str:
        if not value:
            return ""
        import re
        text = str(value).replace("```", "'''")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

    poll_line = ""
    try:
        from email_bridge import run_polling_cycle
        result = await asyncio.to_thread(run_polling_cycle, 10)
        new_count = result.get("new_count", 0)
        if new_count:
            poll_line = f"📥 신규 {new_count}건 수신\n\n"
    except Exception:
        pass

    rows = await asyncio.to_thread(
        _query,
        """
        SELECT id, direction, status, sender_email, recipient_emails, subject,
               LEFT(COALESCE(text_body, html_body, ''), 4000) AS preview,
               created_at, received_at
        FROM email_messages
        ORDER BY COALESCE(received_at, created_at) DESC
        LIMIT 10
        """,
    )
    if not rows and not poll_line:
        await message.answer("이메일 기록이 없다.")
        return
    lines = [f"{poll_line}📨 *최근 이메일*\n"]
    for row in rows:
        ts = (row.get("received_at") or row.get("created_at"))
        ts_str = ts.strftime("%m/%d %H:%M") if ts else "-"
        direction = "← 수신" if row.get("direction") == "inbound" else "→ 발신"
        peer = row.get("sender_email") if row.get("direction") == "inbound" else ", ".join(row.get("recipient_emails") or [])
        preview = _clean_preview(row.get("preview"))
        block = (
            f"`[{row['id']}]` {direction} / {row['status']} / {ts_str}\n"
            f"{row.get('subject') or '(no subject)'}\n"
            f"{peer or '-'}"
        )
        if preview:
            block += f"\n미리보기: {preview}"
        lines.append(block)
    for chunk in _ctx["split_message"]("\n\n".join(lines)):
        await message.answer(chunk, parse_mode="Markdown")


async def cmd_schedule(message: Message):
    """Add a cron schedule: /schedule <cron_expr> | <task content>"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/schedule").strip()
    if not arg or "|" not in arg:
        await message.answer(
            "사용법: /schedule <cron식> | <태스크 내용> [| <agent>]\n\n"
            "예시:\n"
            "  /schedule 0 9 * * * | 오늘의 국제 뉴스 브리핑\n"
            "  /schedule 0 8 * * 1 | 주간 지정학 정세 분석 | analyst\n"
            "  /schedule 0 9 * * * | 메일함 브리핑 | scout\n\n"
            "cron 형식: 분 시 일 월 요일 (KST 기준)\n"
            "agent: programmer, scout, visualizer, analyst, browser, diary"
        )
        return
    parts = arg.split("|")
    cron_expr = parts[0].strip()
    content = parts[1].strip()
    # Optional agent_type from third segment
    agent_type = None
    if len(parts) >= 3:
        from agents import agent_names
        candidate = parts[2].strip().lower()
        if candidate in agent_names():
            agent_type = candidate
        else:
            await message.answer(
                f"알 수 없는 agent: {candidate}\n"
                f"사용 가능: {', '.join(agent_names())}"
            )
            return
    if not content:
        await message.answer("태스크 내용이 비어있습니다.")
        return
    # Validate cron expression
    try:
        from croniter import croniter
        croniter(cron_expr)
    except (ValueError, KeyError) as e:
        await message.answer(f"잘못된 cron 표현식: {cron_expr}\n오류: {e}")
        return
    try:
        # Set last_run_at = NOW() so the first fire waits for the next cron window
        await asyncio.to_thread(
            _execute,
            "INSERT INTO telegram_schedules (user_id, content, cron_expr, last_run_at, agent_type) "
            "VALUES (%s, %s, %s, NOW(), %s)",
            (message.from_user.id, content, cron_expr, agent_type),
        )
        agent_info = f"  agent: {agent_type}\n" if agent_type else ""
        await message.answer(
            f"✅ 스케줄 등록 완료\n"
            f"  cron: `{cron_expr}` (KST)\n"
            f"{agent_info}"
            f"  내용: {content[:100]}"
        )
    except Exception as e:
        await message.answer(f"스케줄 등록 실패: {e}")


async def cmd_schedules(message: Message):
    """List all schedules for the user."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    try:
        rows = await asyncio.to_thread(
            _query,
            "SELECT id, content, cron_expr, enabled, last_run_at, agent_type "
            "FROM telegram_schedules WHERE user_id = %s ORDER BY id",
            (message.from_user.id,),
        )
    except Exception as e:
        await message.answer(f"조회 실패: {e}")
        return
    if not rows:
        await message.answer("등록된 스케줄이 없습니다.")
        return
    lines = ["📅 *등록된 스케줄*\n"]
    for r in rows:
        status = "✅" if r["enabled"] else "⏸️"
        last = r["last_run_at"].strftime("%m/%d %H:%M") if r["last_run_at"] else "미실행"
        preview = r["content"][:60]
        agent = r.get("agent_type") or "auto"
        lines.append(
            f"{status} [{r['id']}] `{r['cron_expr']}` ({agent})\n"
            f"   {preview}\n"
            f"   마지막 실행: {last}"
        )
    await message.answer("\n\n".join(lines))


async def cmd_unschedule(message: Message):
    """Delete a schedule: /unschedule <id>"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/unschedule").strip()
    if not arg:
        await message.answer("사용법: /unschedule <schedule_id>")
        return
    try:
        sched_id = int(arg)
    except ValueError:
        await message.answer("schedule_id는 숫자여야 합니다.")
        return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "DELETE FROM telegram_schedules WHERE id = %s AND user_id = %s RETURNING id",
            (sched_id, message.from_user.id),
        )
    except Exception as e:
        await message.answer(f"삭제 실패: {e}")
        return
    if row:
        await message.answer(f"🗑️ 스케줄 [{sched_id}] 삭제 완료")
    else:
        await message.answer(f"스케줄 [{sched_id}]을(를) 찾을 수 없습니다.")


async def cmd_schedule_agent(message: Message):
    """Set agent for a schedule: /schedule_agent <id> <agent>"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    arg = (message.text or "").removeprefix("/schedule_agent").strip()
    parts = arg.split()
    if len(parts) < 2:
        await message.answer(
            "사용법: /schedule_agent <schedule_id> <agent>\n"
            "agent: programmer, scout, visualizer, analyst, browser, diary\n"
            "agent를 해제하려면: /schedule_agent <id> auto"
        )
        return
    try:
        sched_id = int(parts[0])
    except ValueError:
        await message.answer("schedule_id는 숫자여야 합니다.")
        return
    agent = parts[1].strip().lower()
    if agent == "auto":
        agent = None
    else:
        from agents import agent_names
        if agent not in agent_names():
            await message.answer(
                f"알 수 없는 agent: {agent}\n"
                f"사용 가능: {', '.join(agent_names())}, auto"
            )
            return
    try:
        row = await asyncio.to_thread(
            _query_one,
            "UPDATE telegram_schedules SET agent_type = %s WHERE id = %s AND user_id = %s RETURNING id",
            (agent, sched_id, message.from_user.id),
        )
    except Exception as e:
        await message.answer(f"수정 실패: {e}")
        return
    if row:
        label = agent or "auto"
        await message.answer(f"✅ 스케줄 [{sched_id}] agent → {label}")
    else:
        await message.answer(f"스케줄 [{sched_id}]을(를) 찾을 수 없습니다.")


async def cmd_cancel(message: Message):
    """Cancel a running task by ID. Usage: /cancel <task_id>"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    args = (message.text or "").split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("사용법: `/cancel <task_id>`", parse_mode="Markdown")
        return
    task_id = int(args[1])
    from tool_loop_common import request_cancel
    request_cancel(task_id)
    # Also mark in DB as failed immediately
    try:
        _execute(
            "UPDATE telegram_tasks SET status = 'failed', "
            "result = COALESCE(result, '') || %s, completed_at = NOW() "
            "WHERE id = %s AND status IN ('processing', 'queued')",
            (f"\n[CANCELLED] Stopped by user via /cancel.", task_id),
        )
    except Exception:
        pass
    await message.answer(f"✅ Task #{task_id} 취소 신호 전송. 다음 라운드에서 중단됩니다.")


async def cmd_restart(message: Message):
    """Restart service(s) without git pull. Pure systemctl restart."""
    if not _ctx["is_allowed"](message.from_user.id):
        return

    args = (message.text or "").split(maxsplit=1)
    target = args[1].strip().lower() if len(args) > 1 else "telegram"
    if target not in ("telegram", "api", "all"):
        await message.answer(f"❌ 알 수 없는 대상: `{target}`\n사용법: `/restart [telegram|api|all]`", parse_mode="Markdown")
        return

    services = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "all": ["leninbot-api", "leninbot-telegram"],  # API first, telegram last
    }[target]

    # Force-fail all processing/pending tasks before restart
    try:
        _execute(
            "UPDATE telegram_tasks SET status = 'done', result = COALESCE(result, '') || '\n[SYSTEM] 강제 종료: /restart 명령으로 서비스 재시작', completed_at = NOW() "
            "WHERE status IN ('processing', 'queued', 'pending') AND completed_at IS NULL"
        )
        logger.info("/restart: force-closed all active tasks")
    except Exception as e:
        logger.warning("/restart: failed to close tasks: %s", e)

    # Save restart context to chat history BEFORE restarting (SIGTERM handler may not complete)
    user_id = message.from_user.id
    restart_ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    try:
        await asyncio.to_thread(
            _ctx["save_system_event"], user_id, "restart",
            f"/restart {target} executed ({restart_ts})"
        )
    except Exception:
        pass

    status_msg = await message.answer(f"🔄 서비스 재시작 중... ({target})")
    results = []
    for svc in services:
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-n", "systemctl", "restart", svc,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                results.append(f"✅ {svc}")
            else:
                results.append(f"❌ {svc}: {stdout.decode(errors='replace').strip()}")
        except asyncio.TimeoutError:
            results.append(f"⏱ {svc}: timeout")
        except (asyncio.CancelledError, ConnectionError, OSError):
            return  # telegram being restarted — expected

    try:
        await status_msg.edit_text(f"서비스 재시작 완료:\n" + "\n".join(results))
    except Exception:
        pass  # bot was restarted


async def cmd_deploy(message: Message):
    """Run svc deploy — git pull + restart services. Output sent back via Telegram."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    if not os.path.isfile(_SVC_SCRIPT):
        await message.answer("scripts/svc를 찾을 수 없습니다.")
        return

    # Parse service target: /deploy [telegram|api|frontend|all] (default: all)
    args = (message.text or "").split(maxsplit=1)
    target = args[1].strip().lower() if len(args) > 1 else "all"
    if target not in ("telegram", "api", "frontend", "all"):
        await message.answer(f"❌ 알 수 없는 대상: `{target}`\n사용법: `/deploy [telegram|api|frontend|all]`", parse_mode="Markdown")
        return

    # Save deploy context to chat history BEFORE deploying
    deploy_user_id = message.from_user.id
    deploy_ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    try:
        await asyncio.to_thread(
            _ctx["save_system_event"], deploy_user_id, "deploy",
            f"/deploy {target} executed ({deploy_ts})"
        )
    except Exception:
        pass

    status_msg = await message.answer(f"🚀 Deploy 시작... (대상: {target})")
    try:
        # Run svc deploy detached (setsid) so it survives bot restart
        proc = await asyncio.create_subprocess_exec(
            "setsid", "bash", _SVC_SCRIPT, "deploy", f"--{target}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        # Read output until process exits or bot gets killed by restart
        output_lines: list[str] = []
        try:
            async for line in proc.stdout:
                output_lines.append(line.decode(errors="replace").rstrip())
            await proc.wait()
        except (asyncio.CancelledError, ConnectionError, OSError):
            return  # bot is being restarted by svc deploy — expected, notification handled by svc

        result = "\n".join(output_lines[-30:])  # last 30 lines
        if proc.returncode == 0:
            _ctx["add_system_alert"](f"Deploy 완료 (대상: {target})")
            await status_msg.edit_text(f"✅ Deploy 완료\n```\n{result}\n```", parse_mode="Markdown")
        else:
            _ctx["add_system_alert"](f"Deploy 실패 (대상: {target}, exit {proc.returncode})")
            await status_msg.edit_text(f"❌ Deploy 실패 (exit {proc.returncode})\n```\n{result}\n```", parse_mode="Markdown")
    except Exception as e:
        err_name = type(e).__name__
        err_str = str(e)
        if ("Disconnect" in err_name or "Disconnect" in err_str
                or isinstance(e, (asyncio.CancelledError, ConnectionError, OSError))):
            return  # bot killed by deploy restart — expected, svc handles notification
        _ctx["add_system_alert"](f"Deploy 오류 (대상: {target}): {type(e).__name__}")
        try:
            await status_msg.edit_text(f"❌ Deploy 오류: {e}")
        except Exception:
            pass


async def cmd_fallback(message: Message):
    """Toggle chat+task model between high and low tier (for API overload)."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    config = _ctx["config"]
    if config["chat_model"] in ("high", "medium"):
        config["chat_model"] = "low"
        config["task_model"] = "low"
        _ctx["resolved_models"].clear()
        _ctx["add_system_alert"](f"⚠️ Fallback 모드 활성화: → {_ctx['tier_to_display']('low')}")
        await message.answer(f"⚡ Fallback ON — 모든 모델을 {_ctx['tier_to_display']('low')}로 전환")
    else:
        config["chat_model"] = "high"
        config["task_model"] = "high"
        _ctx["resolved_models"].clear()
        _ctx["add_system_alert"](f"Fallback 해제: → {_ctx['tier_to_display']('high')}")
        await message.answer(f"✅ Fallback OFF — 모든 모델을 {_ctx['tier_to_display']('high')}로 복귀")
    _ctx["save_config"]()


async def cmd_provider(message: Message):
    """Cycle Telegram chat provider across configured remote providers."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    config = _ctx["config"]
    available = ["claude"]
    if _ctx["openai_client"]:
        available.append("openai")
    if _ctx.get("deepseek_client"):
        available.append("deepseek")
    if len(available) == 1:
        await message.answer("❌ 전환 가능한 추가 provider 키가 설정되어 있지 않습니다.")
        return
    old_provider = config.get("provider", "claude")
    try:
        idx = available.index(old_provider)
    except ValueError:
        idx = -1
    config["provider"] = available[(idx + 1) % len(available)]
    if config["provider"] == "claude":
        _ctx["resolved_models"].clear()
    _notify_model_switch(old_provider, config["provider"])
    _ctx["save_config"]()
    await message.answer(
        f"🔄 대화 provider: {config['provider']}\n"
        f"대화: {_ctx['tier_to_display'](config['chat_model'], config['provider'])}\n"
        f"태스크 provider: {config.get('task_provider', 'default')}\n"
        f"태스크: {_ctx['tier_to_display'](config['task_model'], _effective_task_provider())}"
    )


async def handle_photo(message: Message):
    """사용자가 이미지를 보내면 Claude Vision으로 분석"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await message.chat.do("typing")

    user_id = message.from_user.id
    import time as _time

    # 가장 큰 해상도 이미지 선택
    photo = message.photo[-1]
    logger.info("photo received: user_id=%s file_id=%s size=%dx%d",
                user_id, photo.file_id, photo.width, photo.height)
    file = await message.bot.get_file(photo.file_id)

    # 이미지 다운로드 (bytes)
    file_bytes = await message.bot.download_file(file.file_path)
    image_data = base64.b64encode(file_bytes.read()).decode("utf-8")

    # Detect media type from file extension (Telegram supports JPEG, PNG, WebP)
    _ext = (file.file_path or "").rsplit(".", 1)[-1].lower() if file.file_path else ""
    _media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = _media_type_map.get(_ext, "image/jpeg")

    caption = "이 이미지를 분석해줘."

    # 채팅 히스토리 저장 — user 메시지
    user_history_text = "[이미지]"
    await asyncio.to_thread(_ctx["save_chat_message"], user_id, "user", user_history_text)

    # 직전 1턴(user+assistant)을 맥락으로 포함
    recent = await asyncio.to_thread(_ctx["load_chat_history"], user_id)
    # recent 마지막은 방금 저장한 [이미지] → 그 앞 2개가 직전 턴
    context_msgs = recent[-3:-1] if len(recent) >= 3 else []
    # Claude API는 user로 시작해야 함 — assistant로 시작하면 컨텍스트 제거
    if context_msgs and context_msgs[0]["role"] != "user":
        context_msgs = []

    # Vision API 호출 — provider에 따라 분기
    t_start = _time.monotonic()
    try:
        config = _ctx["config"]
        if config.get("provider") == "openai" and _ctx["openai_client"]:
            model_id = await _ctx["get_model"]()
            # OpenAI Vision: image_url with base64 data URI
            messages = context_msgs + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": caption,
                        },
                    ],
                }
            ]
            response = await _ctx["openai_client"].chat.completions.create(
                model=model_id,
                max_completion_tokens=1024,
                messages=messages,
            )
            reply_text = response.choices[0].message.content or ""
            usage = response.usage
            in_tok = getattr(usage, "prompt_tokens", "?") if usage else "?"
            out_tok = getattr(usage, "completion_tokens", "?") if usage else "?"
        else:
            model_id = await _ctx["get_model"]() if config.get("provider") == "claude" else await _ctx["get_model_light"]()
            # Claude Vision: base64 image source
            messages = context_msgs + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": caption,
                        },
                    ],
                }
            ]
            response = await _ctx["claude_client"].messages.create(
                model=model_id,
                max_tokens=1024,
                messages=messages,
            )
            reply_text = _ctx["extract_text"](response)
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", "?") if usage else "?"
            out_tok = getattr(usage, "output_tokens", "?") if usage else "?"
        elapsed = _time.monotonic() - t_start
        logger.info("photo vision done: user_id=%s elapsed=%.2fs in_tokens=%s out_tokens=%s",
                    user_id, elapsed, in_tok, out_tok)

        # 채팅 히스토리 저장 — assistant 응답
        await asyncio.to_thread(_ctx["save_chat_message"], user_id, "assistant", reply_text)

        await message.reply(reply_text)
    except Exception as e:
        logger.error("handle_photo error: %s", e)
        await message.reply(f"❌ 이미지 분석 중 오류: {e}")


async def handle_message(message: Message):
    if not _ctx["is_allowed"](message.from_user.id):
        return
    user_id = message.from_user.id
    user_text = message.text

    # Save user message to DB, load context (chunk summaries + raw messages)
    await asyncio.to_thread(_ctx["save_chat_message"], user_id, "user", user_text)
    history = await asyncio.to_thread(_ctx["load_context_with_summaries"], user_id)

    # Resolve provider first so runtime context blocks match the target prompt
    # style (XML for Claude, Markdown for OpenAI/Qwen).
    _provider = _ctx["config"].get("provider", "claude")

    # Auto-recall: fetch relevant past experiences for runtime context injection
    experience_context = await _fetch_relevant_experiences(user_text, _provider)

    # Mission context: inject active mission timeline
    from telegram.mission import build_mission_context
    mission_context = await asyncio.to_thread(build_mission_context, user_id, _provider)

    # Orchestrator context: structured state block (completed/in-progress/pending)
    from telegram.tasks import build_current_state
    state_context = await asyncio.to_thread(
        build_current_state, user_id, provider=_provider
    )

    # Autonomous agent status: active self-running projects, surfaced so the
    # orchestrator can reference ongoing background work in its replies.
    autonomous_context = _ctx["format_autonomous_status"](_provider)

    try:
        # Block order follows attention recency: stable reference at the top,
        # freshest signals at the bottom so the model reads the "just happened"
        # items last (right before the user's query). _chat_with_tools prepends
        # runtime header (time/model) and appends <system-alerts> automatically,
        # so both framing layers surround this payload without duplication.
        extra_context = _ctx["join_context_blocks"](
            autonomous_context,       # stable background: self-running projects
            experience_context or "", # reference: past lessons
            mission_context,          # current goal
            state_context,            # in-progress work
        )
        # Bind mission tool handler to this user
        from telegram.tools import build_mission_handler
        mission_handler = build_mission_handler(user_id)
        progress_cb = _ctx["make_progress_callback"](message.chat.id)
        bt = {}
        reply = await _ctx["chat_with_tools"](
            history, on_progress=progress_cb, budget_tracker=bt,
            extra_handlers={"mission": mission_handler},
            extra_system_context=extra_context,
        )
        if hasattr(progress_cb, "flush"):
            await progress_cb.flush()
    except Exception as e:
        bt = {}  # no budget info on exception
        err_str = str(e)
        is_tool_pair_error = "tool_use" in err_str and "tool_result" in err_str

        if is_tool_pair_error:
            # Auto-recovery: retry with current message only (do NOT clear DB history)
            logger.warning("Tool pair 400 error — retrying with fresh context (history preserved): %s", e)
            _ctx["log_event"]("warning", "chat", f"Tool pair error auto-recovery (history preserved): {e}")
            try:
                fresh_msgs = [{"role": "user", "content": user_text}]
                progress_cb = _ctx["make_progress_callback"](message.chat.id)
                reply = await _ctx["chat_with_tools"](fresh_msgs, on_progress=progress_cb)
                if hasattr(progress_cb, "flush"):
                    await progress_cb.flush()
            except Exception as e2:
                logger.error("Retry after tool pair recovery also failed: %s", e2)
                reply = f"오류가 발생했습니다 (자동 복구 실패): {e2}"
        else:
            logger.error("Claude API error: %s", e)
            _ctx["log_event"]("error", "chat", f"Claude API error: {e}", detail=user_text[:500])
            reply = f"오류가 발생했습니다: {e}"

    # Mission: log interrupted tool work to active mission
    if bt.get("was_interrupted") and bt.get("tool_work_details"):
        try:
            from telegram.mission import get_active_mission, add_mission_event
            mission = await asyncio.to_thread(get_active_mission, user_id)
            if mission:
                details = bt["tool_work_details"]
                event_content = (
                    f"Chat interrupted (rounds={bt.get('rounds_used', '?')}, "
                    f"cost=${bt.get('total_cost', 0):.2f})\n"
                    + "\n".join(details[:20])
                )[:2000]
                await asyncio.to_thread(
                    add_mission_event, mission["id"], "chat", "tool_result", event_content
                )
        except Exception as e:
            logger.debug("Mission event for interrupted chat failed: %s", e)

    # Auto-escalation: extract [CONTINUE_TASK: ...] marker and create background task
    continuation_task = None
    if "[CONTINUE_TASK:" in reply:
        import re
        match = re.search(r"\[CONTINUE_TASK:\s*(.+?)\]", reply, re.DOTALL)
        if match:
            continuation_task = match.group(1).strip()
            # Remove the marker from the reply shown to user
            reply = reply[:match.start()].rstrip()

    # Save assistant reply to DB, then try to create chunk summary in background
    await asyncio.to_thread(_ctx["save_chat_message"], user_id, "assistant", reply)
    asyncio.create_task(_ctx["maybe_summarize_chunk"](user_id))

    for chunk in _ctx["split_message"](reply):
        await message.answer(chunk)

    # Create background task for unfinished work
    if continuation_task:
        task_content = f"[자동 승격] 대화 중 미완료 작업 이어서 수행:\n{continuation_task}\n\n원래 질문: {user_text[:500]}"
        # Inherit active mission
        cont_mission_id = None
        try:
            from telegram.mission import get_active_mission
            m = await asyncio.to_thread(get_active_mission, user_id)
            if m:
                cont_mission_id = m["id"]
        except Exception:
            pass
        task_row = await asyncio.to_thread(
            _query_one,
            "INSERT INTO telegram_tasks (user_id, content, status, mission_id, agent_type) VALUES (%s, %s, 'pending', %s, 'analyst') RETURNING id",
            (user_id, task_content, cont_mission_id),
        )
        task_id = task_row["id"] if task_row else "?"
        await message.answer(f"🔄 미완료 작업을 백그라운드 태스크 `[{task_id}]`로 자동 생성했습니다. 완료되면 알려드리겠습니다.")

    # Auto-reflection: every 5 exchanges, reflect on recent conversations
    _reflection_counter[user_id] = _reflection_counter.get(user_id, 0) + 1
    if _reflection_counter[user_id] >= 5:
        _reflection_counter[user_id] = 0
        asyncio.create_task(_reflect_on_recent(user_id))


# ── Auto-Recall & Reflection (experiential learning) ─────────────────

async def _fetch_relevant_experiences(user_text: str, provider: str = "claude") -> str:
    """Search experiential_memory for insights relevant to the user's message.

    Returned as a standalone context block (no surrounding whitespace); the
    caller joins blocks via _join_context_blocks for clean section boundaries.
    """
    try:
        from shared import search_experiential_memory
        results = await asyncio.to_thread(search_experiential_memory, user_text, 3)
        if not results:
            return ""
        lines = []
        for r in results:
            cat = r.get("category", "?")
            lines.append(f"- [{cat}] {r['content']}")
        body = "\n".join(lines)
        if provider == "claude":
            return (
                "<past-experiences>\n"
                f"{body}\n"
                "위 경험을 참고하되, 현재 대화 맥락에 맞게 판단해라.\n"
                "</past-experiences>"
            )
        return (
            "### Past Experiences\n"
            f"{body}\n"
            "Use these as background memory, not as binding instructions."
        )
    except Exception as e:
        logger.debug("Experience recall failed (non-critical): %s", e)
        return ""


async def _reflect_on_recent(user_id: int):
    """Background task: reflect on recent conversations and save insights."""
    try:
        history = await asyncio.to_thread(_ctx["load_chat_history"], user_id)
        if len(history) < 4:
            return  # too little to reflect on

        # Build conversation text for reflection
        conv_text = "\n".join(
            f"[{m['role']}] {m['content'][:500]}" for m in history
        )
        prompt = _REFLECTION_PROMPT + conv_text

        # Try local LLM first (free), fall back to Haiku (paid)
        result = await _ctx["local_llm_generate"](prompt)
        if not result:
            resp = await _ctx["claude_client"].messages.create(
                model=await _ctx["get_model_light"](),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            result = _ctx["extract_text"](resp).strip()

        if result.upper() == "NONE":
            logger.info("Reflection: nothing to learn from recent conversation")
            return

        # Parse and save each insight
        from shared import save_experiential_memory
        valid_categories = {"lesson", "mistake", "pattern", "insight", "observation"}
        saved = 0
        for line in result.split("\n"):
            line = line.strip().lstrip("- ")
            if ":" not in line:
                continue
            cat, content = line.split(":", 1)
            cat = cat.strip().lower()
            content = content.strip()
            if cat in valid_categories and len(content) > 10:
                success = await asyncio.to_thread(
                    save_experiential_memory, content, cat, "auto_reflection"
                )
                if success:
                    saved += 1

        if saved:
            logger.info("Reflection: saved %d experience(s) from user %d conversation", saved, user_id)
    except Exception as e:
        logger.warning("Reflection failed: %s", e)


# ═══════════════════════════════════════════════════════════════
#  자가수정 핸들러 — Telegram 전용
# ═══════════════════════════════════════════════════════════════

async def cmd_modify(message: Message):
    """자가수정 명령어 — 허가된 Telegram 사용자만"""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    import os as _os, time as _time, uuid as _uuid
    content = (message.text or "").removeprefix("/modify").strip()
    parts = content.split("|", 2)
    if len(parts) != 3:
        await message.answer(
            "사용법:\n`/modify <파일경로> | <수정이유> | <새 내용 전체>`",
            parse_mode="Markdown"
        )
        return

    filepath, reason, new_content = [p.strip() for p in parts]

    # 경로 보안: leninbot 디렉토리 밖 거부
    base = "/home/grass/leninbot"
    abs_path = _os.path.realpath(_os.path.join(base, filepath))
    if not (abs_path == base or abs_path.startswith(base + "/")):
        await message.answer("❌ 허용된 디렉토리 밖의 파일은 수정할 수 없어.")
        return
    if not _os.path.isfile(abs_path):
        await message.answer(f"❌ 파일을 찾을 수 없어: `{filepath}`", parse_mode="Markdown")
        return

    # diff 생성
    import difflib as _dl
    try:
        with open(abs_path, "r", encoding="utf-8") as _f:
            old_content = _f.read()
        diff_lines = list(_dl.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=""
        ))
        diff_text = "".join(diff_lines)
        insertions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        deletions  = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    except Exception as e:
        await message.answer(f"❌ diff 생성 실패: {e}")
        return

    if not diff_lines:
        await message.answer("ℹ️ 변경사항 없음. 현재 파일과 동일해.")
        return

    # 승인 대기 등록 (5분 유효)
    approval_id = str(_uuid.uuid4())[:8]
    _pending_approvals[approval_id] = {
        "filepath": abs_path,
        "new_content": new_content,
        "reason": reason,
        "expire": _time.time() + 300,
    }

    diff_preview = diff_text[:3000] + ("\n…(생략)…" if len(diff_text) > 3000 else "")
    summary = (
        f"📝 *자가수정 요청*\n"
        f"파일: `{filepath}`\n"
        f"이유: {reason}\n"
        f"변경: +{insertions} / -{deletions} 라인\n\n"
        f"```\n{diff_preview}\n```"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ 승인", callback_data=f"selfmod_approve:{approval_id}"),
        InlineKeyboardButton(text="❌ 거부", callback_data=f"selfmod_reject:{approval_id}"),
    ]])
    await message.answer(summary, parse_mode="Markdown", reply_markup=kb)


async def cb_modify_approve(callback: CallbackQuery):
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    import time as _time
    approval_id = callback.data.split(":", 1)[1]
    entry = _pending_approvals.pop(approval_id, None)
    if entry is None:
        await callback.message.edit_text("⚠️ 승인 정보를 찾을 수 없어. 만료됐거나 이미 처리됨.")
        return
    if _time.time() > entry["expire"]:
        await callback.message.edit_text("⏰ 승인 시간 초과 (5분). 다시 `/modify`를 실행해.")
        return

    await callback.message.edit_text("⚙️ 패치 적용 중…")
    await callback.answer()

    sys.path.insert(0, "/home/grass/leninbot")
    from self_modification_core import self_modify_with_safety
    try:
        result = await asyncio.to_thread(
            self_modify_with_safety,
            filepath=entry["filepath"],
            new_content=entry["new_content"],
            reason=entry["reason"],
            request_approval=False,
            skip_tests=False,
        )
    except Exception as e:
        await callback.message.edit_text(
            f"❌ 패치 적용 중 예외 발생:\n`{e}`", parse_mode="Markdown"
        )
        return

    if result.status == "success":
        commit_info = f"\n커밋: `{result.commit_hash}`" if result.commit_hash else ""
        await callback.message.edit_text(
            f"✅ *패치 완료*\n"
            f"파일: `{result.filepath}`\n"
            f"변경: {result.changes_count} 라인{commit_info}\n"
            f"⚠️ 재시작 후 적용됩니다.",
            parse_mode="Markdown"
        )
    else:
        await callback.message.edit_text(
            f"❌ *패치 실패* ({result.status})\n`{result.error}`",
            parse_mode="Markdown"
        )


async def cb_modify_reject(callback: CallbackQuery):
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    approval_id = callback.data.split(":", 1)[1]
    _pending_approvals.pop(approval_id, None)
    await callback.message.edit_text("❌ 수정 거부됨. 원본 파일 유지.")
    await callback.answer()


# ── Config Panel ────────────────────────────────────────────────────

def _effective_task_provider() -> str:
    provider = str(_ctx["config"].get("task_provider", "default") or "default")
    if provider == "default":
        return str(_ctx["config"].get("provider", "claude") or "claude")
    return provider


def _effective_autonomous_provider() -> str:
    provider = str(_ctx["config"].get("autonomous_provider", "claude") or "claude")
    if provider == "default":
        return _effective_task_provider()
    return provider


def _config_display_value(key: str, val) -> str:
    """Format a config value for display, with model tier → actual model mapping."""
    meta = _ctx["CONFIG_META"][key]
    unit = meta["unit"]
    if key == "chat_model":
        return _ctx["tier_to_display"](val, _ctx["config"].get("provider", "claude"))
    if key == "task_model":
        return _ctx["tier_to_display"](val, _effective_task_provider())
    if key == "task_provider" and val == "default":
        return f"default ({_effective_task_provider()})"
    if key == "autonomous_provider" and val == "default":
        return f"default ({_effective_autonomous_provider()})"
    return f"{unit}{val}" if unit == "$" else f"{val}{unit}"


def _config_summary() -> str:
    """Build a text summary of current config values."""
    lines = ["*현재 설정*\n"]
    for key, meta in _ctx["CONFIG_META"].items():
        val = _ctx["config"][key]
        display = _config_display_value(key, val)
        lines.append(f"  {meta['label']}: `{display}`")
    return "\n".join(lines)


def _config_main_keyboard() -> InlineKeyboardMarkup:
    """Build the main config panel keyboard — one button per setting."""
    rows = []
    for key, meta in _ctx["CONFIG_META"].items():
        val = _ctx["config"][key]
        display = _config_display_value(key, val)
        rows.append([InlineKeyboardButton(
            text=f"{meta['label']}: {display}",
            callback_data=f"cfg_select:{key}",
        )])
    rows.append([InlineKeyboardButton(text="닫기", callback_data="cfg_close")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _config_options_keyboard(key: str) -> InlineKeyboardMarkup:
    """Build option selection keyboard for a specific config key."""
    meta = _ctx["CONFIG_META"][key]
    current = _ctx["config"][key]
    buttons = []
    for opt in meta["options"]:
        display = _config_display_value(key, opt)
        marker = " ✓" if opt == current else ""
        buttons.append(InlineKeyboardButton(
            text=f"{display}{marker}",
            callback_data=f"cfg_set:{key}:{opt}",
        ))
    # Arrange in rows of 3
    rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
    rows.append([InlineKeyboardButton(text="← 뒤로", callback_data="cfg_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def cmd_config(message: Message):
    """Open the config panel."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    await message.answer(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )


async def cb_config_select(callback: CallbackQuery):
    """User tapped a config key — show options."""
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    key = callback.data.split(":", 1)[1]
    if key not in _ctx["CONFIG_META"]:
        await callback.answer("알 수 없는 설정", show_alert=True)
        return
    meta = _ctx["CONFIG_META"][key]
    await callback.message.edit_text(
        f"*{meta['label']}* 변경\n현재: `{_config_display_value(key, _ctx['config'][key])}`",
        parse_mode="Markdown",
        reply_markup=_config_options_keyboard(key),
    )
    await callback.answer()


async def cb_config_set(callback: CallbackQuery):
    """User selected a new value for a config key."""
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    parts = callback.data.split(":", 2)
    if len(parts) != 3:
        await callback.answer("잘못된 데이터", show_alert=True)
        return
    key, raw_val = parts[1], parts[2]
    if key not in _ctx["CONFIG_META"]:
        await callback.answer("알 수 없는 설정", show_alert=True)
        return

    # Convert value to the right type
    config = _ctx["config"]
    current = config[key]
    if isinstance(current, float):
        new_val = float(raw_val)
    elif isinstance(current, int):
        new_val = int(raw_val)
    else:
        new_val = raw_val

    if key in ("provider", "task_provider", "autonomous_provider"):
        if new_val == "openai" and not _ctx["openai_client"]:
            await callback.answer("OPENAI_API_KEY가 없습니다.", show_alert=True)
            return
        if new_val == "deepseek" and not _ctx.get("deepseek_client"):
            await callback.answer("DEEPSEEK_API_KEY가 없습니다.", show_alert=True)
            return

    old_val = config[key]
    config[key] = new_val

    # If model changed, clear resolved cache so it re-resolves on next use
    if key in ("chat_model", "task_model") and new_val != old_val:
        _ctx["resolved_models"].pop(new_val, None)

    # Provider switch: log with actual model names
    if key == "provider" and new_val != old_val:
        _notify_model_switch(str(old_val), str(new_val))
    if key == "task_provider" and new_val != old_val:
        _ctx["add_system_alert"](f"태스크 provider 전환: {old_val} → {new_val}")
    if key == "autonomous_provider" and new_val != old_val:
        _ctx["add_system_alert"](f"자율 provider 전환: {old_val} → {new_val}")

    logger.info("Config changed: %s = %s → %s", key, old_val, new_val)
    _ctx["save_config"]()
    await callback.answer(f"{_ctx['CONFIG_META'][key]['label']}: {new_val}")

    # Return to main config panel
    await callback.message.edit_text(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )


async def cb_config_back(callback: CallbackQuery):
    """Return to main config panel."""
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    await callback.message.edit_text(
        _config_summary(),
        parse_mode="Markdown",
        reply_markup=_config_main_keyboard(),
    )
    await callback.answer()


async def cb_config_close(callback: CallbackQuery):
    """Close the config panel."""
    if not _ctx["is_allowed"](callback.from_user.id):
        await callback.answer("권한 없음", show_alert=True)
        return
    await callback.message.edit_text("설정 패널을 닫았습니다.")
    await callback.answer()


# ── Registration ───────────────────────────────────────────────────

async def cmd_projects(message: Message):
    """List autonomous projects (id, state, turn count, title)."""
    if not _ctx["is_allowed"](message.from_user.id):
        return
    rows = await asyncio.to_thread(
        _query,
        "SELECT id, title, state, turn_count, last_run_at "
        "FROM autonomous_projects ORDER BY "
        "CASE state WHEN 'researching' THEN 0 WHEN 'planning' THEN 0 "
        "           WHEN 'paused' THEN 1 ELSE 2 END, id",
    )
    if not rows:
        await message.answer("등록된 자율 프로젝트가 없습니다.")
        return
    lines = ["자율 프로젝트"]
    for r in rows:
        last = r["last_run_at"].astimezone(KST).strftime("%m/%d %H:%M") if r["last_run_at"] else "never"
        lines.append(f"#{r['id']} [{r['state']}] turn {r['turn_count']} last={last}")
        lines.append(f"  {r['title']}")
    lines.append("")
    lines.append("조언: /advise <id> <내용>  |  조언 목록: /advisories <id>")
    await message.answer("\n".join(lines))


def _parse_advise_args(text: str) -> tuple[int | None, str]:
    """Split '/advise [<id>] <content>' into (project_id_or_None, content).

    Multi-line content is preserved (Telegram passes newlines through). The
    first whitespace-delimited token is treated as a project_id only when it
    parses as a positive integer.
    """
    after_cmd = (text or "").split(None, 1)
    if len(after_cmd) < 2:
        return None, ""
    rest = after_cmd[1].strip()
    if not rest:
        return None, ""
    head = rest.split(None, 1)
    if head[0].isdigit():
        pid = int(head[0])
        content = head[1] if len(head) > 1 else ""
        return pid, content.strip()
    return None, rest


async def _pick_single_active_project() -> int | None:
    rows = await asyncio.to_thread(
        _query,
        "SELECT id FROM autonomous_projects WHERE state IN ('researching', 'planning') ORDER BY id",
    )
    return rows[0]["id"] if len(rows) == 1 else None


async def cmd_advise(message: Message):
    """Leave an operator advisory that the next autonomous-project tick reads.

    Usage:
      /advise <project_id> <content>   — explicit
      /advise <content>                 — if exactly one active project

    Multi-line content is OK — everything after the id (or after the command,
    if id is omitted) is taken as the advisory body.
    """
    if not _ctx["is_allowed"](message.from_user.id):
        return
    project_id, content = _parse_advise_args(message.text or "")
    if not content:
        await message.answer(
            "사용법:\n"
            "  /advise <project_id> <조언>\n"
            "  /advise <조언>   (단일 active 프로젝트일 때)\n"
            "Multi-line 도 OK — 명령 다음 줄부터 전체가 조언 본문"
        )
        return
    if project_id is None:
        project_id = await _pick_single_active_project()
        if project_id is None:
            await message.answer(
                "활성 프로젝트가 0개이거나 2개 이상입니다. /projects 로 확인 후 id 명시."
            )
            return
    proj = await asyncio.to_thread(
        _query_one,
        "SELECT title, state FROM autonomous_projects WHERE id = %s",
        (project_id,),
    )
    if not proj:
        await message.answer(f"프로젝트 #{project_id} 찾을 수 없음.")
        return
    row = await asyncio.to_thread(
        _query_one,
        "INSERT INTO autonomous_project_advisories(project_id, content) "
        "VALUES (%s, %s) RETURNING id, created_at",
        (project_id, content),
    )
    # Mirror the CLI path — log an event so the audit trail is uniform.
    try:
        from autonomous_project import _log_event
        await asyncio.to_thread(
            _log_event, project_id, "advisory_created",
            f"advisory #{row['id']} created via Telegram ({len(content)} chars)",
            {"advisory_id": row["id"], "via": "telegram"},
        )
    except Exception as e:
        logger.warning("advisory event log failed: %s", e)
    await message.answer(
        f"조언 #{row['id']} 등록됨 → #{project_id} '{proj['title']}' ({proj['state']})\n"
        f"({len(content)}자)  다음 tick 에서 agent 가 읽고 자동 소비."
    )


async def cmd_advisories(message: Message):
    """List pending + recent consumed advisories for a project.

    Usage:
      /advisories <project_id>
      /advisories            — single active project
    """
    if not _ctx["is_allowed"](message.from_user.id):
        return
    project_id, _ = _parse_advise_args(message.text or "")
    if project_id is None:
        project_id = await _pick_single_active_project()
        if project_id is None:
            await message.answer("사용법: /advisories <project_id>   (/projects 로 id 확인)")
            return
    rows = await asyncio.to_thread(
        _query,
        "SELECT id, content, created_at, consumed_at FROM autonomous_project_advisories "
        "WHERE project_id = %s ORDER BY created_at DESC LIMIT 10",
        (project_id,),
    )
    if not rows:
        await message.answer(f"프로젝트 #{project_id} 에 조언 없음.")
        return
    pending = [r for r in rows if r["consumed_at"] is None]
    consumed = [r for r in rows if r["consumed_at"] is not None]
    lines = [f"프로젝트 #{project_id} 조언"]
    if pending:
        lines.append(f"\n[대기 중] {len(pending)}건 — 다음 tick 이 읽음")
        for r in pending:
            ts = r["created_at"].astimezone(KST).strftime("%m/%d %H:%M")
            snip = r["content"][:400] + ("…" if len(r["content"]) > 400 else "")
            lines.append(f"#{r['id']} @ {ts}")
            lines.append(snip)
    if consumed:
        lines.append(f"\n[소비됨] {min(3, len(consumed))}/{len(consumed)}건")
        for r in consumed[:3]:
            ts = r["created_at"].astimezone(KST).strftime("%m/%d %H:%M")
            ts_c = r["consumed_at"].astimezone(KST).strftime("%m/%d %H:%M")
            snip = r["content"][:250] + ("…" if len(r["content"]) > 250 else "")
            lines.append(f"#{r['id']} 작성={ts} 소비={ts_c}")
            lines.append(snip)
    await message.answer("\n".join(lines))


async def cmd_agents(message: Message):
    """Show registered agents and external worker process status."""
    if not _ctx["is_allowed"](message.from_user.id):
        return

    from agents import list_agents
    from telegram.tasks import check_browser_worker_alive

    lines = ["*에이전트 현황*\n"]

    for spec in list_agents():
        lines.append(f"- *{spec.name}*: {spec.description} (${spec.budget_usd:.2f}, {spec.max_rounds}R)")

    # Check browser worker external process
    lines.append("\n*외부 프로세스:*")
    try:
        alive = await check_browser_worker_alive()
        icon = "\U0001f7e2" if alive else "\U0001f534"  # green/red circle
        lines.append(f"  browser worker: {icon} {'alive' if alive else 'dead'}")
    except Exception as e:
        lines.append(f"  browser worker: \U0001f534 error ({e})")

    await message.answer("\n".join(lines), parse_mode="Markdown")


def register_handlers(router: Router, ctx: dict):
    """Store dependencies and register all handlers on the router."""
    global _ctx
    _ctx = ctx

    # Command handlers
    router.message.register(cmd_start, Command("start"))
    router.message.register(cmd_help, Command("help"))
    router.message.register(cmd_clear, Command("clear"))
    router.message.register(cmd_mission, Command("mission"))
    router.message.register(cmd_errors, Command("errors"))
    router.message.register(cmd_chat, Command("chat"))
    router.message.register(cmd_task, Command("task"))
    router.message.register(cmd_image, Command("image"))
    router.message.register(cmd_stats, Command("stats"))
    router.message.register(cmd_status, Command("status"))
    router.message.register(cmd_kg, Command("kg"))
    router.message.register(cmd_report, Command("report"))
    router.message.register(cmd_status_auto, Command("status_auto"))
    router.message.register(cmd_autonomous, Command("autonomous"))
    router.message.register(cmd_email, Command("email"))
    router.message.register(cmd_schedule, Command("schedule"))
    router.message.register(cmd_schedules, Command("schedules"))
    router.message.register(cmd_unschedule, Command("unschedule"))
    router.message.register(cmd_schedule_agent, Command("schedule_agent"))
    router.message.register(cmd_cancel, Command("cancel"))
    router.message.register(cmd_restart, Command("restart"))
    router.message.register(cmd_deploy, Command("deploy"))
    router.message.register(cmd_fallback, Command("fallback"))
    router.message.register(cmd_provider, Command("provider"))
    router.message.register(cmd_modify, Command("modify"))
    router.message.register(cmd_config, Command("config"))
    router.message.register(cmd_agents, Command("agents"))
    router.message.register(cmd_projects, Command("projects"))
    router.message.register(cmd_advise, Command("advise"))
    router.message.register(cmd_advisories, Command("advisories"))

    # Photo handler
    router.message.register(handle_photo, F.photo)

    # Catch-all text handler (must be registered AFTER command handlers)
    router.message.register(handle_message, F.text, ~Command("config"), ~Command("modify"))

    # Callback query handlers
    router.callback_query.register(cb_modify_approve, F.data.startswith("selfmod_approve:"))
    router.callback_query.register(cb_modify_reject, F.data.startswith("selfmod_reject:"))
    router.callback_query.register(cb_config_select, F.data.startswith("cfg_select:"))
    router.callback_query.register(cb_config_set, F.data.startswith("cfg_set:"))
    router.callback_query.register(cb_config_back, F.data == "cfg_back")
    router.callback_query.register(cb_config_close, F.data == "cfg_close")
