#!/usr/bin/env python3
"""Which long-running backend services must restart for a code change.

Usage:
    scripts/affected_services.py BASE [HEAD]

Prints one ``scripts/svc`` service name per line (in no particular order), or
the single line ``ALL`` when the change cannot be attributed safely. Prints
nothing when no running service is affected (for example a docs-only change).
Reasons go to stderr for the deploy log.

Rules, in order:

* ignored paths (``dev_docs/``, ``tests/``, top-level Markdown, the deploy
  scripts themselves) never restart anything;
* ``requirements*``, ``systemd/``, a deleted or renamed ``.py`` file (a stale
  process can still import the old path lazily), and any other non-Python file
  (config, prompts, skills, assets read at startup) mean ``ALL``;
* a changed ``.py`` file restarts every service whose static import closure
  contains it. The closure follows every ``import``/``from`` statement in the
  repository, including function-level imports. A file outside every closure
  (standalone scripts, timer jobs that start a fresh process) restarts nothing.

Timers and on-demand scripts are not listed: each run starts a new process.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# svc name -> entry module, as started by systemd/<unit> ExecStart.
SERVICE_ENTRYPOINTS = {
    "llm-proxy": "llm_proxy.app",
    "web-gateway": "web_gateway.app",
    "embedding": "services.embedding_server",
    "api": "services.api",
    "a2a-api": "services.a2a_api",
    "email-api": "services.email_api",
    "novel-writer": "services.novel_writer_api",
    "roleplay": "roleplay.bot",
    "worker": "worker.service",
    "browser": "browser.worker",
    "telegram": "telegram.bot",
}

IGNORED_PREFIXES = ("dev_docs/", "tests/", ".github/", ".claude/")
IGNORED_FILES = {
    "deploy.sh", "scripts/svc", "scripts/affected_services.py",
    "migrations/README.md", ".gitignore",
}
_SKIP_DIRS = {"venv", ".git", "node_modules", "tests", "__pycache__", "data", "temp_dev"}


def repo_modules(root: str = ROOT) -> dict[str, str]:
    modules = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            module = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            modules[module] = path
    return modules


_IMPORT_CACHE: dict[tuple[str, str], frozenset[str]] = {}


def _imports(module: str, modules: dict[str, str]) -> frozenset[str]:
    """Repository modules imported anywhere in ``module`` (cached per run)."""
    key = (module, modules[module])
    if key not in _IMPORT_CACHE:
        _IMPORT_CACHE[key] = frozenset(_parse_imports(module, modules))
    return _IMPORT_CACHE[key]


def _parse_imports(module: str, modules: dict[str, str]) -> set[str]:
    path = modules[module]
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError, ValueError):
        return set()
    package = module if path.endswith("__init__.py") else module.rpartition(".")[0]
    found = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                if node.level > 1:
                    parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        for name in names:
            # "from a.b import c" may name module a.b.c or attribute c of a.b;
            # walking up the dotted path also records the parent packages
            # whose __init__ runs on import.
            while name:
                if name in modules:
                    found.add(name)
                name = name.rpartition(".")[0]
    return found


def import_closure(entry: str, modules: dict[str, str]) -> set[str]:
    seen: set[str] = set()
    stack = [entry]
    while stack:
        module = stack.pop()
        if module in seen or module not in modules:
            continue
        seen.add(module)
        stack.extend(_imports(module, modules))
    return seen


def _module_name(path: str) -> str:
    module = path[:-3].replace("/", ".")
    return module[: -len(".__init__")] if module.endswith(".__init__") else module


def _is_ignored(path: str) -> bool:
    if path in IGNORED_FILES or path.startswith(IGNORED_PREFIXES):
        return True
    # Top-level Markdown and example files (config.json.example, .env.example)
    # are never loaded by a running service.
    return "/" not in path and (path.endswith(".md") or path.endswith(".example"))


def affected(changes: list[tuple[str, str]], modules: dict[str, str]) -> tuple[set[str] | None, list[str]]:
    """Return (services, reasons); services is None for ALL.

    ``changes`` are (git status letter, path) pairs from ``git diff --name-status``.
    """
    reasons: list[str] = []
    changed_modules: set[str] = set()
    for status, path in changes:
        if _is_ignored(path):
            continue
        if path.startswith("requirements") or path.startswith("systemd/"):
            return None, [f"{path}: dependencies or unit files changed"]
        if not path.endswith(".py"):
            return None, [f"{path}: non-Python runtime file; cannot attribute"]
        if status[:1] in ("D", "R"):
            return None, [f"{path}: deleted or renamed module; stale processes may import it"]
        changed_modules.add(_module_name(path))
    if not changed_modules:
        return set(), ["no runtime file changed"]
    services = set()
    for service, entry in SERVICE_ENTRYPOINTS.items():
        hit = sorted(import_closure(entry, modules) & changed_modules)
        if hit:
            services.add(service)
            reasons.append(f"{service}: imports {', '.join(hit[:3])}{' …' if len(hit) > 3 else ''}")
    if not services:
        reasons.append("changed modules are not imported by any long-running service")
    return services, reasons


def git_changes(base: str, head: str, root: str = ROOT) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "-C", root, "diff", "--name-status", "-M", base, head],
        check=True, capture_output=True, text=True,
    ).stdout
    changes = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            # A rename lists old and new paths; the old path is what stale
            # processes may still import.
            changes.append((parts[0], parts[1]))
    return changes


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    base, head = argv[1], argv[2] if len(argv) == 3 else "HEAD"
    try:
        changes = git_changes(base, head)
    except subprocess.CalledProcessError as exc:
        print(f"affected_services: git diff failed ({exc.stderr.strip()}); restarting all", file=sys.stderr)
        print("ALL")
        return 0
    services, reasons = affected(changes, repo_modules())
    for reason in reasons:
        print(f"affected_services: {reason}", file=sys.stderr)
    if services is None:
        print("ALL")
    else:
        for service in sorted(services):
            print(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
