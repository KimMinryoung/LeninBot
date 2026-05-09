#!/usr/bin/env python3
"""Print effective provider/model/budget policy for runtime surfaces and agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _selection(kind: str, provider_override: str | None = None) -> dict[str, Any]:
    from bot_config import get_current_model_selection

    return get_current_model_selection(kind=kind, provider_override=provider_override)


def _agent_rows() -> list[dict[str, Any]]:
    from agents import list_agents
    from bot_config import (
        _DEEPSEEK_MODEL_MAP,
        _MODEL_ALIAS_MAP,
        _OPENAI_MODEL_MAP,
        _display_name_for_model_id,
        _resolved_models,
        get_current_model_selection,
    )

    task_selection = get_current_model_selection(kind="task")
    config_task_provider = task_selection["provider"]
    rows = []
    for spec in sorted(list_agents(), key=lambda item: item.name):
        raw_provider = spec.provider
        effective_provider = raw_provider or config_task_provider
        render_provider = spec.effective_provider(config_task_provider)
        if raw_provider in {"codex", "moon"}:
            model_selection = {
                "provider": raw_provider,
                "tier": None,
                "alias": spec.model,
                "model_id": spec.model,
                "display_name": spec.model or raw_provider,
                "resolved": True,
            }
        elif spec.model:
            if effective_provider == "openai":
                model_id = _OPENAI_MODEL_MAP.get(spec.model, spec.model)
            elif effective_provider == "deepseek":
                model_id = _DEEPSEEK_MODEL_MAP.get(spec.model, spec.model)
            elif effective_provider == "claude":
                _model_alias, fallback = _MODEL_ALIAS_MAP.get(spec.model, (spec.model, spec.model))
                model_id = _resolved_models.get(spec.model, fallback)
            else:
                model_id = spec.model
            model_selection = {
                "provider": effective_provider,
                "tier": "override",
                "alias": spec.model,
                "model_id": model_id,
                "display_name": _display_name_for_model_id(model_id),
                "resolved": True,
            }
        else:
            model_selection = get_current_model_selection(kind="task", provider_override=effective_provider)
        rows.append(
            {
                "agent": spec.name,
                "provider_config": raw_provider,
                "provider_effective": effective_provider,
                "prompt_render_provider": render_provider,
                "model_config": spec.model,
                "model_id": model_selection.get("model_id"),
                "display_name": model_selection.get("display_name"),
                "budget_usd": spec.budget_usd,
                "max_rounds": spec.max_rounds,
                "tools": len(spec.tools),
                "finalization_tools": list(spec.finalization_tools),
                "terminal_tools": list(spec.terminal_tools),
                "skip_orchestrator_report": spec.skip_orchestrator_report,
            }
        )
    return rows


def build_snapshot() -> dict[str, Any]:
    from bot_config import _config

    return {
        "runtime_config": dict(_config),
        "surfaces": {
            "telegram_chat": _selection("chat"),
            "delegated_task": _selection("task"),
            "autonomous": _selection("autonomous"),
            "webchat": _selection("webchat"),
        },
        "agents": _agent_rows(),
    }


def _print_table(snapshot: dict[str, Any]) -> None:
    print("Runtime surfaces")
    print("surface           provider   tier      model_id")
    for name, row in snapshot["surfaces"].items():
        print(f"{name:<17} {row.get('provider', ''):<10} {str(row.get('tier', '')):<9} {row.get('model_id', '')}")

    print("\nAgents")
    print("agent              provider      model                    budget  rounds  tools")
    for row in snapshot["agents"]:
        provider = row["provider_effective"]
        model = row["model_id"] or ""
        print(
            f"{row['agent']:<18} {provider:<13} {model:<24} "
            f"{row['budget_usd']:<7.2f} {row['max_rounds']:<7} {row['tools']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit effective LeninBot model/provider runtime policy.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON snapshot.")
    args = parser.parse_args()

    snapshot = build_snapshot()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))
    else:
        _print_table(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
