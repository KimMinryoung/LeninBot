#!/usr/bin/env python
"""llm_registry_cli.py — LLM 호출 지점 통합 조회/수정 CLI.

Usage:
  python scripts/llm_registry_cli.py list             # 전체 기능(원샷 + 에이전트 루프) 유효 설정
  python scripts/llm_registry_cli.py show <feature>   # 한 기능의 JSON 원본 + 유효값
  python scripts/llm_registry_cli.py set <feature> <key> <value>
                                                      # config/llm_call_sites.json 수정 (핫리로드 반영)
  python scripts/llm_registry_cli.py add <feature> --provider P --model M [--temperature T] ...

  python scripts/llm_registry_cli.py agent-show <agent>
  python scripts/llm_registry_cli.py agent-set <agent> <key> <value>
                                                      # config/agent_runtime.json 수정 (핫리로드 반영)
                                                      # key: provider | model | budget_usd | max_rounds
                                                      # value "null" → 기본값 상속(provider/model)

`set`/`add`는 원샷 레지스트리(config/llm_call_sites.json),
`agent-set`은 에이전트 루프(config/agent_runtime.json)를 수정한다.
둘 다 핫리로드 — 서비스 재시작 불필요.
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


# ── 에이전트 루프 (config/agent_runtime.json) ─────────────────────────

_AGENT_PROVIDERS = {"claude", "openai", "deepseek", "kimi", "local", "moon", "codex"}
_AGENT_KEYS = {"provider", "model", "budget_usd", "max_rounds"}
# provider별 모델 별칭 힌트 (bot_config 티어/별칭 맵 기준; 오타 경고용, 차단 아님)
_AGENT_MODEL_HINTS = {
    "claude": {"haiku", "sonnet", "opus"},
    "openai": {"gpt54", "gpt54mini", "gpt54nano"},
    "deepseek": {"deepseek_pro", "deepseek_flash"},
    "kimi": {"kimi_k3"},
}


def _save_agent_runtime(data: dict) -> None:
    with open(AGENT_RUNTIME_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def cmd_agent_show(agent: str) -> None:
    agents = _load(AGENT_RUNTIME_PATH)
    if agent not in agents:
        print(f"'{agent}' 미등록. 등록된 에이전트: {', '.join(sorted(agents))}")
        sys.exit(1)
    print(json.dumps(agents[agent], ensure_ascii=False, indent=2))


def cmd_agent_set(agent: str, key: str, value: str) -> None:
    if key not in _AGENT_KEYS:
        print(f"알 수 없는 키 '{key}'. 허용: {', '.join(sorted(_AGENT_KEYS))} "
              "(terminal_tools 등 구조 필드는 파일을 직접 편집)")
        sys.exit(1)
    agents = _load(AGENT_RUNTIME_PATH)
    if agent not in agents:
        print(f"'{agent}' 미등록. 등록된 에이전트: {', '.join(sorted(agents))}")
        sys.exit(1)

    entry = agents[agent]
    if key in ("provider", "model") and value.lower() in ("null", "none", ""):
        new_value = None
    elif key == "provider":
        if value not in _AGENT_PROVIDERS:
            print(f"provider는 {sorted(_AGENT_PROVIDERS)} 또는 null이어야 합니다.")
            sys.exit(1)
        new_value = value
    elif key == "model":
        new_value = value
        provider = entry.get("provider")
        hints = _AGENT_MODEL_HINTS.get(provider or "", set())
        if hints and value not in hints:
            print(f"⚠ 경고: '{value}'는 provider '{provider}'의 알려진 별칭 {sorted(hints)}에 없습니다. "
                  "런타임 티어 해석에 실패하면 기본 티어로 폴백합니다.")
    elif key == "budget_usd":
        new_value = float(value)
    else:  # max_rounds
        new_value = int(value)

    old_value = entry.get(key)
    entry[key] = new_value
    _save_agent_runtime(agents)

    # 저장본을 런타임 로더로 재검증 — 실패 시 원복 (런타임 리로더도 fail-safe지만 이중 방어)
    try:
        from agents.runtime_config import _load_runtime_config
        _load_runtime_config()
    except Exception as e:
        entry[key] = old_value
        _save_agent_runtime(agents)
        print(f"검증 실패로 원복했습니다: {e}")
        sys.exit(1)

    print(f"{agent}.{key}: {old_value!r} → {new_value!r}  (핫리로드 — 다음 태스크부터 적용)")


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
    p_add.add_argument("--provider", required=True, choices=["gemini", "deepseek", "kimi", "openai", "claude"])
    p_add.add_argument("--model", required=True)
    p_add.add_argument("--temperature", type=float)
    p_add.add_argument("--max-tokens", dest="max_tokens", type=int)
    p_add.add_argument("--timeout", type=float)
    p_add.add_argument("--json-mode", dest="json_mode", action="store_true")
    p_add.add_argument("--managed", choices=["executor", "model-only", "external"])
    p_add.add_argument("--note")
    p_ashow = sub.add_parser("agent-show")
    p_ashow.add_argument("agent")
    p_aset = sub.add_parser("agent-set")
    p_aset.add_argument("agent")
    p_aset.add_argument("key")
    p_aset.add_argument("value")
    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.feature)
    elif args.cmd == "set":
        cmd_set(args.feature, args.key, args.value)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "agent-show":
        cmd_agent_show(args.agent)
    elif args.cmd == "agent-set":
        cmd_agent_set(args.agent, args.key, args.value)


if __name__ == "__main__":
    main()
