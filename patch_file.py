#!/usr/bin/env python3
"""
patch_file.py — 토큰 효율적인 파일 패치 유틸리티
=====================================================
전체 파일을 LLM에 넣지 않고, 최소한의 diff만 적용.

사용법:
  from patch_file import patch_file, replace_lines, insert_after

전략:
  1. replace_block  : old_str → new_str 문자열 교체 (difflib 기반 검증)
  2. replace_lines  : 특정 라인 범위 교체
  3. insert_after   : 특정 패턴 다음 줄에 삽입
  4. apply_unified_diff : unified diff 문자열 직접 적용
"""

import difflib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


# ─── 공통 헬퍼 ──────────────────────────────────────────────────────────────

def _backup(path: Path) -> Path:
    """수정 전 .bak 백업 생성."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ─── 1. replace_block ───────────────────────────────────────────────────────

def replace_block(
    filepath: str,
    old_str: str,
    new_str: str,
    backup: bool = True,
    min_similarity: float = 0.6,
) -> dict:
    """
    파일에서 old_str을 찾아 new_str로 교체.

    - 정확한 매치 우선
    - 없으면 유사도 기반 fuzzy 매치 (min_similarity 이상)
    - backup=True면 .bak 저장

    Returns:
        {"ok": bool, "message": str, "diff": str}
    """
    path = Path(filepath)
    if not path.exists():
        return {"ok": False, "message": f"파일 없음: {filepath}", "diff": ""}

    original = path.read_text(encoding="utf-8")

    # 정확한 매치
    if old_str in original:
        modified = original.replace(old_str, new_str, 1)
    else:
        # fuzzy: 라인 단위로 가장 유사한 블록 찾기
        orig_lines = original.splitlines()
        old_lines = old_str.splitlines()
        n = len(old_lines)
        best_score, best_idx = 0.0, -1
        for i in range(len(orig_lines) - n + 1):
            chunk = "\n".join(orig_lines[i:i+n])
            score = _similarity(chunk, old_str)
            if score > best_score:
                best_score, best_idx = score, i
        if best_score < min_similarity:
            return {
                "ok": False,
                "message": f"매치 실패 (최대 유사도 {best_score:.2f} < {min_similarity}). old_str을 확인해.",
                "diff": "",
            }
        new_lines = orig_lines[:best_idx] + new_str.splitlines() + orig_lines[best_idx+n:]
        modified = "\n".join(new_lines)

    if backup:
        _backup(path)

    path.write_text(modified, encoding="utf-8")

    diff = "\n".join(difflib.unified_diff(
        original.splitlines(), modified.splitlines(),
        fromfile=f"{path.name} (before)", tofile=f"{path.name} (after)",
        lineterm="",
    ))
    return {"ok": True, "message": "교체 완료", "diff": diff}


# ─── 2. replace_lines ───────────────────────────────────────────────────────

def replace_lines(
    filepath: str,
    line_start: int,
    line_end: int,
    new_content: str,
    backup: bool = True,
) -> dict:
    """
    특정 라인 범위(1-based, inclusive)를 new_content로 교체.

    Returns:
        {"ok": bool, "message": str, "diff": str}
    """
    path = Path(filepath)
    if not path.exists():
        return {"ok": False, "message": f"파일 없음: {filepath}", "diff": ""}

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    if line_start < 1 or line_end > len(lines) or line_start > line_end:
        return {
            "ok": False,
            "message": f"라인 범위 오류: {line_start}~{line_end} (총 {len(lines)}줄)",
            "diff": "",
        }

    if backup:
        _backup(path)

    new_lines = (
        lines[:line_start - 1]
        + [new_content if new_content.endswith("\n") else new_content + "\n"]
        + lines[line_end:]
    )
    modified = "".join(new_lines)
    path.write_text(modified, encoding="utf-8")

    diff = "\n".join(difflib.unified_diff(
        original.splitlines(), modified.splitlines(),
        fromfile=f"{path.name} (before)", tofile=f"{path.name} (after)",
        lineterm="",
    ))
    return {"ok": True, "message": f"라인 {line_start}~{line_end} 교체 완료", "diff": diff}


# ─── 3. insert_after ────────────────────────────────────────────────────────

def insert_after(
    filepath: str,
    pattern: str,
    new_lines: str,
    occurrence: int = 1,
    backup: bool = True,
) -> dict:
    """
    pattern이 포함된 줄 다음에 new_lines 삽입.
    occurrence: 몇 번째 매치에 삽입할지 (기본 1번째).

    Returns:
        {"ok": bool, "message": str, "diff": str}
    """
    path = Path(filepath)
    if not path.exists():
        return {"ok": False, "message": f"파일 없음: {filepath}", "diff": ""}

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    count = 0
    insert_idx = -1
    for i, line in enumerate(lines):
        if pattern in line:
            count += 1
            if count == occurrence:
                insert_idx = i + 1
                break

    if insert_idx == -1:
        return {
            "ok": False,
            "message": f"패턴 '{pattern}' {occurrence}번째 매치 없음",
            "diff": "",
        }

    if backup:
        _backup(path)

    insertion = [l + "\n" for l in new_lines.splitlines()]
    new_file_lines = lines[:insert_idx] + insertion + lines[insert_idx:]
    modified = "".join(new_file_lines)
    path.write_text(modified, encoding="utf-8")

    diff = "\n".join(difflib.unified_diff(
        original.splitlines(), modified.splitlines(),
        fromfile=f"{path.name} (before)", tofile=f"{path.name} (after)",
        lineterm="",
    ))
    return {"ok": True, "message": f"라인 {insert_idx} 다음에 삽입 완료", "diff": diff}


# ─── 4. apply_unified_diff ──────────────────────────────────────────────────

def apply_unified_diff(filepath: str, diff_str: str, backup: bool = True) -> dict:
    """
    unified diff 문자열을 직접 파일에 적용 (patch 명령어 사용).

    Returns:
        {"ok": bool, "message": str, "stdout": str, "stderr": str}
    """
    path = Path(filepath)
    if not path.exists():
        return {"ok": False, "message": f"파일 없음: {filepath}", "stdout": "", "stderr": ""}

    if backup:
        _backup(path)

    result = subprocess.run(
        ["patch", str(path)],
        input=diff_str.encode(),
        capture_output=True,
    )
    ok = result.returncode == 0
    return {
        "ok": ok,
        "message": "패치 적용 완료" if ok else "패치 실패",
        "stdout": result.stdout.decode(),
        "stderr": result.stderr.decode(),
    }


# ─── 5. show_diff (dry-run) ─────────────────────────────────────────────────

def show_diff(filepath: str, old_str: str, new_str: str) -> str:
    """
    실제 수정 없이 diff만 보여주기 (dry-run).
    """
    path = Path(filepath)
    if not path.exists():
        return f"파일 없음: {filepath}"
    original = path.read_text(encoding="utf-8")
    modified = original.replace(old_str, new_str, 1)
    return "\n".join(difflib.unified_diff(
        original.splitlines(), modified.splitlines(),
        fromfile=f"{path.name} (before)", tofile=f"{path.name} (after)",
        lineterm="",
    ))


# ─── CLI 사용 예시 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    print("patch_file.py 유틸리티 로드됨.")
    print("함수 목록: replace_block, replace_lines, insert_after, apply_unified_diff, show_diff")
