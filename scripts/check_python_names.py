#!/usr/bin/env python3
"""Fail on undefined Python names, including names in delayed worker paths.

Pyflakes also reports unused imports throughout the existing codebase. This
gate focuses on names that can raise NameError at runtime and syntax errors.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tokenize
from pathlib import Path

from pyflakes.checker import Checker
from pyflakes.messages import UndefinedExport, UndefinedLocal, UndefinedName

ROOT = Path(__file__).resolve().parents[1]
NAME_ERRORS = (UndefinedName, UndefinedLocal, UndefinedExport)


def _project_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / name.decode() for name in result.stdout.split(b"\0") if name]


def check_file(path: Path) -> list[str]:
    try:
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=str(path))
        checker = Checker(tree, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return [f"{path}: {exc}"]
    return [str(message) for message in checker.messages if isinstance(message, NAME_ERRORS)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Python files; defaults to all project files")
    args = parser.parse_args()
    paths = args.paths or _project_files()
    errors = [error for path in paths for error in check_file(path)]
    if errors:
        print("\n".join(sorted(errors)), file=sys.stderr)
        return 1
    print(f"Python name check passed: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
