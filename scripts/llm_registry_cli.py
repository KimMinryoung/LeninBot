#!/usr/bin/env python
"""llm_registry_cli.py — LLM 호출 지점 통합 조회/수정 CLI.

Usage:
  python scripts/llm_registry_cli.py list             # 전체 기능(원샷 + 에이전트 루프) 유효 설정
  python scripts/llm_registry_cli.py show <feature>   # 한 기능의 JSON 원본 + 유효값
  python scripts/llm_registry_cli.py set <feature> <key> <value>
                                                      # config/llm_call_sites.json 수정 (핫리로드 반영)
  python scripts/llm_registry_cli.py add <feature> --provider P --model M [--temperature T] ...

`set`/`add`는 원샷 레지스트리(config/llm_call_sites.json)만 수정한다.
에이전트 루프(config/agent_runtime.json)는 기존처럼 해당 파일을 직접 편집.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CALL_SITES_PATH = ROOT / "config" / "llm_call_sites.json"
AGENT_RUNTIME_PATH = ROOT / "config" / "agent_runtime.json"

_NUMERIC_KEYS = {"temperature", "max_tokens", "timeout"}
_BOOL_KEYS = {"json_mode"}
_KNOWN_KEYS = {"provider", "model", "temperature", "max_tokens", "timeout",
               "json_mode", "note", "managed", "env"}


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_call_sites(data: dict) -> None:
    with open(CALL_SITES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def cmd_list() -> None:
    from llm.call_registry import resolve

    call_sites = _load(CALL_SITES_PATH)
    print("== 원샷 호출 (config/llm_call_sites.json — 핫리로드, set으로 수정) ==")
    header = f"{'feature':28} {'provider':9} {'model':26} {'temp':5} {'max_tok':7} {'mode':10} note"
    print(header)
    print("-" * len(header))
    for feature in sorted(call_sites):
        p = resolve(feature)
        model = p.model + (f"  [env:{p.model_env_override}]" if p.model_env_override else "")
        print(f"{feature:28} {p.provider:9} {model:26} {p.temperature:<5.2f} "
              f"{p.max_tokens:<7d} {p.managed:10} {p.note[:52]}")

    agents = _load(AGENT_RUNTIME_PATH)
    print()
    print("== 에이전트 루프 (config/agent_runtime.json — 직접 편집, 서비스 재시작 필요) ==")
    header2 = f"{'agent':20} {'provider':9} {'model':22} {'budget':7} {'rounds':6}"
    print(header2)
    print("-" * len(header2))
    for agent in sorted(agents):
        cfg = agents[agent] or {}
        provider = str(cfg.get("provider") or "(config.json 기본)")
        model = str(cfg.get("model") or "(티어 기본)")
        budget = cfg.get("budget_usd", "-")
        rounds = cfg.get("max_rounds", "-")
        print(f"{agent:20} {provider:9} {model:22} {budget!s:7} {rounds!s:6}")


def cmd_show(feature: str) -> None:
    from llm.call_registry import resolve

    call_sites = _load(CALL_SITES_PATH)
    if feature not in call_sites:
        print(f"'{feature}' 미등록. 등록된 기능: {', '.join(sorted(call_sites))}")
        sys.exit(1)
    print("[JSON 원본]")
    print(json.dumps(call_sites[feature], ensure_ascii=False, indent=2))
    p = resolve(feature)
    print("\n[유효값 (env 오버라이드 반영)]")
    print(f"  provider={p.provider} model={p.model} temperature={p.temperature} "
          f"max_tokens={p.max_tokens} timeout={p.timeout} json_mode={p.json_mode}")
    if p.model_env_override:
        print(f"  ⚠ model이 env {p.model_env_override}로 오버라이드됨")
    generic = f"LLM_SITE_{feature.upper()}_MODEL"
    print(f"  (env 오버라이드 채널: {', '.join((call_sites[feature].get('env') or []) + [generic])})")


def _coerce(key: str, value: str):
    if key in _NUMERIC_KEYS:
        return int(value) if key == "max_tokens" else float(value)
    if key in _BOOL_KEYS:
        return value.lower() in ("1", "true", "yes", "on")
    if key == "env":
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def cmd_set(feature: str, key: str, value: str) -> None:
    if key not in _KNOWN_KEYS:
        print(f"알 수 없는 키 '{key}'. 허용: {', '.join(sorted(_KNOWN_KEYS))}")
        sys.exit(1)
    data = _load(CALL_SITES_PATH)
    if feature not in data:
        print(f"'{feature}' 미등록. 'add'로 먼저 등록하세요. 등록된 기능: {', '.join(sorted(data))}")
        sys.exit(1)
    old = data[feature].get(key)
    data[feature][key] = _coerce(key, value)
    _save_call_sites(data)
    print(f"{feature}.{key}: {old!r} → {data[feature][key]!r}  (핫리로드 — 재시작 불필요)")


def cmd_add(args) -> None:
    data = _load(CALL_SITES_PATH)
    if args.feature in data:
        print(f"'{args.feature}'는 이미 등록돼 있습니다. 'set'으로 수정하세요.")
        sys.exit(1)
    entry = {"provider": args.provider, "model": args.model}
    for key in ("temperature", "max_tokens", "timeout"):
        val = getattr(args, key)
        if val is not None:
            entry[key] = val
    if args.json_mode:
        entry["json_mode"] = True
    if args.managed:
        entry["managed"] = args.managed
    entry["note"] = args.note or ""
    data[args.feature] = entry
    _save_call_sites(data)
    print(f"등록 완료: {args.feature}")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print("\n호출부 연결: from llm.call_registry import generate; "
          f"await generate({args.feature!r}, prompt)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 호출 레지스트리 조회/수정")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_show = sub.add_parser("show")
    p_show.add_argument("feature")
    p_set = sub.add_parser("set")
    p_set.add_argument("feature")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_add = sub.add_parser("add")
    p_add.add_argument("feature")
    p_add.add_argument("--provider", required=True, choices=["gemini", "deepseek", "openai", "claude"])
    p_add.add_argument("--model", required=True)
    p_add.add_argument("--temperature", type=float)
    p_add.add_argument("--max-tokens", dest="max_tokens", type=int)
    p_add.add_argument("--timeout", type=float)
    p_add.add_argument("--json-mode", dest="json_mode", action="store_true")
    p_add.add_argument("--managed", choices=["executor", "model-only", "external"])
    p_add.add_argument("--note")
    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.feature)
    elif args.cmd == "set":
        cmd_set(args.feature, args.key, args.value)
    elif args.cmd == "add":
        cmd_add(args)


if __name__ == "__main__":
    main()
