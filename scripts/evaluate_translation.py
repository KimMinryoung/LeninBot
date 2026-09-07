#!/usr/bin/env python3
"""Frozen translation regression set. Default/list/scoring make no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_tools.archival_translation import core
from translation_runtime.structure import markdown_problems, semantic_review

FIXTURE = ROOT / "tests" / "fixtures" / "translation_eval.json"


def evaluate(samples, candidates):
    results = []
    for sample in samples:
        if hashlib.sha256(sample["source"].encode()).hexdigest() != sample["sourceHash"]:
            raise ValueError(f"source hash mismatch: {sample['id']}")
        candidate = candidates.get(sample["id"])
        if candidate is None:
            results.append({"id": sample["id"], "missing": True})
            continue
        text = candidate["text"] if isinstance(candidate, dict) else candidate
        if sample["kind"] == "archival":
            problems = core.validate(sample["blocks"], core.parse_response(text), core.LANGUAGES[sample["lang"]])
        else:
            problems = markdown_problems(sample["source"], text)
        if isinstance(candidate, dict):
            if candidate.get("error"):
                problems.append("provider error: " + str(candidate["error"]))
            if candidate.get("truncated"):
                problems.append("provider output was truncated")
        results.append({"id": sample["id"], "problems": problems,
                        "review": semantic_review(sample["source"], text),
                        "source": sample["source"], "target": text,
                        "usage": candidate.get("usage", {}) if isinstance(candidate, dict) else {},
                        "humanScores": {key: None for key in (
                            "meaning_omission", "subject_object", "negation_conditions", "terminology", "fluency")}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--candidates", type=Path, help="JSON: sample ID → text or {text, usage}")
    parser.add_argument("--generate", action="store_true", help="Explicitly call current registered models (billable)")
    parser.add_argument("--context-chars", type=int, default=0, choices=(0, 600, 1200))
    parser.add_argument("--id", action="append", help="Only evaluate selected frozen IDs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = json.loads(args.fixture.read_text())["samples"]
    if args.id:
        unknown = set(args.id) - {s["id"] for s in samples}
        if unknown:
            parser.error(f"unknown IDs: {sorted(unknown)}")
        samples = [s for s in samples if s["id"] in args.id]
    if args.generate and args.candidates:
        parser.error("choose either --generate or --candidates")
    if not args.generate and not args.candidates:
        for sample in samples:
            print(sample["id"], sample["lang"], len(sample["source"]), sample["sourceHash"])
        return 0
    candidates = json.loads(args.candidates.read_text()) if args.candidates else {}
    candidates = candidates.get("candidates", candidates)
    if args.generate:
        from llm.call_registry import generate_detailed
        from scripts.translate_research_markdown import FEATURE, SYSTEM_PROMPT
        for sample in samples:
            lang = core.LANGUAGES.get(sample["lang"])
            system = lang.system_prompt if sample["kind"] == "archival" else SYSTEM_PROMPT
            prompt = sample.get("prompt", sample["source"])
            if args.context_chars and sample["kind"] == "archival":
                system += "\n참고 맥락 (번역·출력하지 말 것):\n" + json.dumps({
                    "before": sample["before"][-args.context_chars:],
                    "after": sample["after"][:args.context_chars]}, ensure_ascii=False)
            response = generate_detailed(lang.feature if sample["kind"] == "archival" else FEATURE,
                                         prompt, system=system)
            candidates[sample["id"]] = {"text": response.text or "", "usage": response.usage,
                "attempts": response.attempts, "latencyMs": response.latency_ms,
                "error": response.error, "truncated": response.truncated,
                "contextChars": args.context_chars}
            if response.error_kind in {"authentication", "quota", "policy", "configuration"}:
                break
    report = {"contextChars": args.context_chars, "candidates": candidates,
              "results": evaluate(samples, candidates),
              "rubric": "Human scores: 0=wrong/unusable, 1=major correction, 2=minor correction, 3=acceptable; null=unreviewed."}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from scripts.translate_research_markdown import atomic_write
        atomic_write(args.output, text + "\n")
    else:
        print(text)
    return int(any(r.get("missing") or r.get("problems") for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
