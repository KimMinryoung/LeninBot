#!/usr/bin/env python3
"""
import_skill.py — 외부 agentskills.io 표준 스킬을 leninbot skills/로 가져오기.

사용법:
    python scripts/import_skill.py github anthropics/skills/frontend-design
    python scripts/import_skill.py local /tmp/my-skill
    python scripts/import_skill.py list
    python scripts/import_skill.py search <query>       # skills.sh 검색
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# ─── 경로 설정 ───────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

# ─── allowed-tools 매핑 ─────────────────────────────────────
# agentskills.io (Claude Code 등) → leninbot 내부 tool 이름
TOOL_MAP = {
    "bash": "execute_python",
    "read": "read_file",
    "write": "write_file",
    "edit": "write_file",
    "glob": "list_directory",
    "grep": "list_directory",
    "websearch": "web_search",
    "webfetch": "fetch_url",
}


def map_allowed_tools(tools_str: str) -> str:
    """외부 allowed-tools 문자열을 leninbot tool 이름으로 변환."""
    if not tools_str.strip():
        return ""
    mapped = set()
    for token in tools_str.split():
        # "Bash(git:*)" → "bash"
        base = re.sub(r"\(.*\)", "", token).strip().lower()
        leninbot_tool = TOOL_MAP.get(base, base)
        mapped.add(leninbot_tool)
    return " ".join(sorted(mapped))


# ─── SKILL.md 파싱 ──────────────────────────────────────────
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """frontmatter + body 분리. dict와 body 문자열 반환."""
    fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not fm_match:
        raise ValueError("SKILL.md에 frontmatter(--- ... ---)가 없습니다.")

    meta = {}
    for line in fm_match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')

    return meta, fm_match.group(2)


def rebuild_frontmatter(meta: dict, body: str) -> str:
    """meta dict + body → SKILL.md 문자열 재구성."""
    lines = ["---"]

    # 필수 필드 먼저
    for key in ("name", "description"):
        if key in meta:
            lines.append(f"{key}: {meta[key]}")

    # 선택 필드
    for key in ("license", "compatibility"):
        if key in meta:
            lines.append(f"{key}: {meta[key]}")

    # metadata 블록
    meta_fields = {}
    for k, v in meta.items():
        if k.startswith("metadata."):
            meta_fields[k.replace("metadata.", "", 1)] = v
    # 원본 metadata 필드도 보존
    if "metadata" in meta and isinstance(meta["metadata"], str):
        pass  # 단순 문자열이면 무시
    if meta_fields:
        lines.append("metadata:")
        for mk, mv in meta_fields.items():
            lines.append(f"  {mk}: {mv}")

    # allowed-tools
    if meta.get("allowed-tools"):
        lines.append(f"allowed-tools: {meta['allowed-tools']}")

    lines.append("---")
    return "\n".join(lines) + "\n" + body


# ─── GitHub 다운로드 ────────────────────────────────────────
def download_github(spec: str, dest: Path) -> Path:
    """
    spec: 'owner/repo/path/to/skill' 또는 'owner/repo' (루트가 스킬인 경우)
    dest: 임시 디렉토리
    """
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"GitHub 경로는 최소 owner/repo 형식: {spec}")

    owner, repo = parts[0], parts[1]
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

    # GitHub API로 디렉토리 내용 가져오기
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{subpath}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    # GH token 있으면 사용
    gh_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if gh_token:
        headers["Authorization"] = f"token {gh_token}"

    print(f"[*] GitHub API 요청: {api_url}")
    try:
        req = Request(api_url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read())
    except URLError as e:
        raise RuntimeError(f"GitHub API 실패: {e}") from e

    if isinstance(items, dict) and items.get("message"):
        raise RuntimeError(f"GitHub API 에러: {items['message']}")

    # 단일 파일이면 items가 dict
    if isinstance(items, dict):
        items = [items]

    skill_dir = dest / (parts[-1] if subpath else repo)
    skill_dir.mkdir(parents=True, exist_ok=True)

    def _download_recursive(api_items: list, target_dir: Path):
        for item in api_items:
            if item["type"] == "file":
                file_url = item["download_url"]
                file_path = target_dir / item["name"]
                print(f"  ↓ {item['path']}")
                req2 = Request(file_url, headers=headers)
                with urlopen(req2, timeout=30) as r:
                    file_path.write_bytes(r.read())
            elif item["type"] == "dir":
                sub_dir = target_dir / item["name"]
                sub_dir.mkdir(exist_ok=True)
                req3 = Request(item["url"], headers=headers)
                with urlopen(req3, timeout=30) as r:
                    sub_items = json.loads(r.read())
                _download_recursive(sub_items, sub_dir)

    _download_recursive(items, skill_dir)
    return skill_dir


# ─── 로컬 복사 ─────────────────────────────────────────────
def copy_local(source: str, dest: Path) -> Path:
    """로컬 디렉토리를 dest로 복사."""
    src = Path(source).resolve()
    if not src.is_dir():
        raise ValueError(f"디렉토리 아님: {src}")
    if not (src / "SKILL.md").exists():
        raise ValueError(f"SKILL.md 없음: {src}")

    skill_dir = dest / src.name
    shutil.copytree(src, skill_dir)
    return skill_dir


# ─── skills.sh 검색 ────────────────────────────────────────
def search_skills_sh(query: str) -> list[dict]:
    """skills.sh에서 스킬 검색. GitHub repo 정보 반환."""
    url = f"https://skills.sh/api/skills?q={query}"
    print(f"[*] skills.sh 검색: {query}")
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except URLError:
        # API가 없거나 다른 형태일 수 있음 — 웹 스크래핑 대안
        print("[!] skills.sh API 접근 실패. 웹사이트에서 직접 검색하세요:")
        print(f"    https://skills.sh/?q={query}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "skills" in data:
        return data["skills"]
    return []


# ─── 변환 및 설치 ──────────────────────────────────────────
def convert_and_install(source_dir: Path, force: bool = False) -> str:
    """
    source_dir의 SKILL.md를 읽어 변환 후 skills/에 설치.
    설치된 스킬 이름 반환.
    """
    skill_md = source_dir / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"SKILL.md 없음: {source_dir}")

    text = skill_md.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)

    # 필수 필드 확인
    name = meta.get("name", source_dir.name)
    if not name:
        raise ValueError("name 필드가 비어 있습니다.")
    if not meta.get("description"):
        raise ValueError("description 필드가 비어 있습니다.")

    # 기존 스킬 충돌 확인
    target_dir = SKILLS_DIR / name
    if target_dir.exists() and not force:
        raise FileExistsError(
            f"스킬 '{name}'이 이미 존재합니다. --force로 덮어쓸 수 있습니다."
        )

    # allowed-tools 변환
    original_tools = meta.get("allowed-tools", "")
    mapped_tools = map_allowed_tools(original_tools)
    if original_tools and mapped_tools != original_tools:
        print(f"[*] allowed-tools 변환: {original_tools} → {mapped_tools}")
    meta["allowed-tools"] = mapped_tools

    # metadata에 import 정보 추가
    meta["metadata.source"] = str(source_dir)
    meta["metadata.imported_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if "metadata.author" not in meta:
        meta["metadata.author"] = "imported"

    # license 경고
    if meta.get("license"):
        print(f"[!] 라이선스: {meta['license']} — 사용 전 확인 필요")

    # 본문에 외부 tool 이름이 있으면 경고
    external_tools = re.findall(r'\b(Bash|Read|Write|Edit|Glob|Grep|WebSearch|WebFetch)\b', body)
    if external_tools:
        unique = sorted(set(external_tools))
        print(f"[!] 본문에 외부 tool 참조 발견: {', '.join(unique)}")
        print("    leninbot에서는 execute_python, read_file 등으로 대체 필요. 수동 확인 권장.")

    # 변환된 SKILL.md 생성
    converted = rebuild_frontmatter(meta, body)

    # 설치
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # SKILL.md 저장
    (target_dir / "SKILL.md").write_text(converted, encoding="utf-8")

    # 하위 디렉토리 복사 (scripts/, references/, assets/ 등)
    for item in source_dir.iterdir():
        if item.name == "SKILL.md":
            continue
        dest_item = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item)
        else:
            shutil.copy2(item, dest_item)

    print(f"[+] 설치 완료: skills/{name}/")
    return name


# ─── 검증 ──────────────────────────────────────────────────
def verify_installed(name: str) -> bool:
    """skills_loader로 설치된 스킬이 정상 로드되는지 확인."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import skills_loader
        skills_loader._skills_loaded = False
        skills = skills_loader.load_skills()
        names = [s["name"] for s in skills]
        if name in names:
            print(f"[+] 검증 완료: '{name}' 스킬이 정상 로드됩니다.")
            return True
        else:
            print(f"[-] 검증 실패: '{name}'이 skills_loader에서 로드되지 않습니다.")
            return False
    except Exception as e:
        print(f"[-] 검증 중 에러: {e}")
        return False


# ─── list 명령 ──────────────────────────────────────────────
def list_skills():
    """현재 설치된 스킬 목록 출력."""
    if not SKILLS_DIR.exists():
        print("skills/ 디렉토리가 없습니다.")
        return

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            print(f"  {skill_dir.name}/ — SKILL.md 없음")
            continue
        try:
            meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            source = meta.get("metadata.source", "local")
            desc = meta.get("description", "")[:60]
            print(f"  {meta.get('name', skill_dir.name):30s} {desc}")
            if "imported" in source or "github" in source.lower():
                print(f"  {'':30s} (imported: {source})")
        except Exception:
            print(f"  {skill_dir.name:30s} (파싱 실패)")


# ─── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="외부 agentskills.io 스킬을 leninbot으로 가져오기"
    )
    sub = parser.add_subparsers(dest="command", help="명령")

    # github
    gh = sub.add_parser("github", help="GitHub repo에서 스킬 가져오기")
    gh.add_argument("spec", help="owner/repo/path (예: anthropics/skills/frontend-design)")
    gh.add_argument("--force", action="store_true", help="기존 스킬 덮어쓰기")

    # local
    loc = sub.add_parser("local", help="로컬 디렉토리에서 스킬 가져오기")
    loc.add_argument("path", help="스킬 디렉토리 경로")
    loc.add_argument("--force", action="store_true", help="기존 스킬 덮어쓰기")

    # list
    sub.add_parser("list", help="설치된 스킬 목록")

    # search
    srch = sub.add_parser("search", help="skills.sh에서 검색")
    srch.add_argument("query", help="검색어")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        list_skills()
        return

    if args.command == "search":
        results = search_skills_sh(args.query)
        if results:
            for r in results[:10]:
                name = r.get("name", "?")
                desc = r.get("description", "")[:60]
                repo = r.get("repo", r.get("github", ""))
                print(f"  {name:30s} {desc}")
                if repo:
                    print(f"  {'':30s} → github {repo}")
        return

    # github / local → 공통 흐름
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if args.command == "github":
            source_dir = download_github(args.spec, tmp)
        elif args.command == "local":
            source_dir = copy_local(args.path, tmp)
        else:
            parser.print_help()
            sys.exit(1)

        name = convert_and_install(source_dir, force=args.force)
        verify_installed(name)


if __name__ == "__main__":
    main()
