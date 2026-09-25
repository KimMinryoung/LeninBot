"""restart_service tool: restart a leninbot service after syntax and import pre-flight checks."""
import asyncio
import logging
import os
import sys

from tool_gateway.results import ToolFailure

logger = logging.getLogger(__name__)


RESTART_PREFLIGHT_ENTRY_POINTS = {
    "telegram": "telegram.bot",
    "api": "services.api",
    "browser": "browser.worker",
}


RESTART_SERVICE_TOOL = {
    "name": "restart_service",
    "description": (
        "Restart a leninbot service with pre-flight syntax + import checks. "
        "Use instead of execute_python+subprocess. "
        "File→service mapping (and detailed procedure) lives in the programmer agent prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["telegram", "api", "browser", "all"],
                "description": "telegram=bot+agents, api=web+a2a, browser=browser worker, all=multi-service code. Default: telegram.",
            },
        },
        "required": [],
    },
}


async def restart_service(service: str = "telegram") -> str:
    """Safely restart service with pre-flight validation."""
    import ast
    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        from llm.runtime_context import current_task_ctx
        from telegram.tasks import persist_task_restart_state
        ctx = current_task_ctx.get()
        current_task_id = ctx["task_id"] if ctx else None
    except Exception:
        current_task_id = None
        persist_task_restart_state = None

    if service not in ("telegram", "api", "browser", "all"):
        return f"❌ Unknown service: {service}. Use: telegram, api, browser, all"

    # 1. Find .py files with uncommitted changes (staged + unstaged)
    try:
        diff_result = await asyncio.to_thread(
            subprocess.run,
            ["git", "diff", "--name-only", "HEAD", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        # Also include untracked .py files that might be new
        untracked = await asyncio.to_thread(
            subprocess.run,
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        changed_files = set()
        for line in (diff_result.stdout + "\n" + untracked.stdout).strip().split("\n"):
            line = line.strip()
            if line.endswith(".py"):
                changed_files.add(line)
    except Exception as e:
        return ToolFailure(f"❌ Failed to detect changed files: {e}")

    errors = []

    # 2. Syntax check all changed .py files
    for rel_path in sorted(changed_files):
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            errors.append(f"SyntaxError in {rel_path}:{e.lineno} — {e.msg}")

    if errors:
        return "❌ Restart blocked — syntax errors found:\n" + "\n".join(errors)

    # 3. Import-level validation: try importing the entry points in a subprocess
    targets = ["telegram", "api", "browser"] if service == "all" else [service]

    for target in targets:
        module = RESTART_PREFLIGHT_ENTRY_POINTS[target]
        module_path = os.path.join(project_root, module.replace(".", os.sep) + ".py")
        if not os.path.isfile(module_path):
            # A missing entry file means the map rotted (this exact guard
            # silently skipped telegram/browser for months when the flat
            # telegram_bot.py/browser_worker.py modules became packages).
            logger.warning(
                "restart_service preflight: entry module %s not found at %s — import check skipped",
                module, module_path,
            )
            continue
        try:
            # Run a quick import check in isolated subprocess
            check_code = (
                f"import sys; sys.path.insert(0, {project_root!r}); "
                f"import importlib; importlib.import_module({module!r})"
            )
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c", check_code],
                capture_output=True, text=True, timeout=30,
                cwd=project_root,
                env={**os.environ, "PREFLIGHT_CHECK": "1"},
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Extract the last meaningful error line
                err_lines = [l for l in stderr.split("\n") if l.strip()]
                last_err = err_lines[-1] if err_lines else "unknown error"
                errors.append(f"Import check failed for {module}.py: {last_err}")
        except subprocess.TimeoutExpired:
            errors.append(f"Import check timed out for {module}.py (>30s)")
        except Exception as e:
            errors.append(f"Import check error for {module}.py: {e}")

    if errors:
        return "❌ Restart blocked — import errors found:\n" + "\n".join(errors)

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="requested",
                mark_completed=False,
            )
        except Exception as e:
            return ToolFailure(f"❌ Restart blocked — failed to persist durable restart state: {e}")

    # 4. All checks passed — daemon-reload (picks up any unit file changes), then restart
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["sudo", "-n", "systemctl", "daemon-reload"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass  # non-fatal: restart will still use previous unit config

    svc_map = {
        "telegram": ["leninbot-telegram"],
        "api": ["leninbot-api"],
        "browser": ["leninbot-browser"],
        "all": ["leninbot-api", "leninbot-browser", "leninbot-telegram"],  # API first, browser second, telegram last
    }
    results = []
    restart_failed = False
    for svc in svc_map[service]:
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["sudo", "-n", "systemctl", "restart", svc],
                capture_output=True, text=True, timeout=15,
                start_new_session=True,
            )
            if proc.returncode == 0:
                results.append(f"✅ {svc}: restarted")
            else:
                restart_failed = True
                results.append(f"❌ {svc}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired:
            restart_failed = True
            results.append(f"⏱ {svc}: timeout")
        except Exception as e:
            restart_failed = True
            results.append(f"❌ {svc}: {e}")

    if current_task_id and persist_task_restart_state:
        try:
            persist_task_restart_state(
                current_task_id,
                service=service,
                phase="verification" if not restart_failed else "requested",
                mark_completed=not restart_failed,
                resumed_after_restart=not restart_failed,
                reentry_reason=(
                    "restart completed; next step is post-restart verification"
                    if not restart_failed
                    else "restart command failed; restart branch may retry after fix"
                ),
            )
        except Exception as e:
            results.append(f"⚠️ durable restart completion state update failed: {e}")

    checked_files = ", ".join(sorted(changed_files)[:10]) if changed_files else "(none)"
    return (
        f"Pre-flight checks passed (syntax + import OK, changed: {checked_files})\n"
        + "\n".join(results)
    )
