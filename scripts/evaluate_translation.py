#!/usr/bin/env python3
"""Frozen translation regression set. Default/list/scoring make no model calls."""
from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
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


def adapter_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "translation_runtime/__init__.py", "translation_runtime/structure.py",
        "scripts/translate_research_markdown.py", "runtime_tools/archival_translation/core.py")}


def generate_candidate(sample, *, mode="pipeline", context_chars=0, max_chars=None):
    """Compare raw model output separately from production translation adapters.

    Pipeline evaluation uses a fresh temporary cache, never the production cache
    or TM. Source fetching and publication are outside this frozen-sample test.
    """
    from llm.call_registry import generate_detailed, resolve
    from scripts.translate_research_markdown import FEATURE, SYSTEM_PROMPT, translate_markdown_with_retry
    lang = core.LANGUAGES.get(sample["lang"])
    archival = sample["kind"] == "archival"
    max_chars = max_chars if max_chars is not None else (core.Options().max_chars if archival else 8000)
    feature = lang.feature if archival else FEATURE
    system = lang.system_prompt if archival else SYSTEM_PROMPT
    prompt = sample.get("prompt", sample["source"])
    if context_chars and archival:
        prompt = "참고 맥락 (번역·출력하지 말 것):\n" + json.dumps({
            "before": sample["before"][-context_chars:],
            "after": sample["after"][:context_chars]}, ensure_ascii=False) + "\n\n" + prompt
    events = []
    def account(result):
        events.append({"usage": result.usage, "attempts": result.attempts,
                       "latencyMs": result.latency_ms, "errorKind": result.error_kind})
    candidate = {"mode": mode, "contextChars": context_chars, "maxChars": max_chars,
        "profile": dataclasses.asdict(resolve(feature)),
        "systemHash": hashlib.sha256(system.encode()).hexdigest(),
        "promptHash": hashlib.sha256(prompt.encode()).hexdigest(),
        "text": "", "error": None, "truncated": False}
    started = time.monotonic()
    try:
        if mode == "model":
            response = generate_detailed(feature, prompt, system=system)
            account(response)
            candidate.update(text=response.text or "", error=response.error,
                             errorKind=response.error_kind, truncated=response.truncated)
        else:
            with tempfile.TemporaryDirectory(prefix="translation-eval-") as directory:
                cache = Path(directory)
                if archival:
                    options = core.Options(max_chars=max_chars)
                    translated = {}
                    blocks = sample["blocks"]
                    groups = core.chunk_document({"offset": 0, "blocks": [b for _, b in blocks]}, max_chars)
                    source_body = core.render_chunk(blocks)
                    if not prompt.endswith(source_body):
                        raise ValueError("frozen archival prompt must end with the recorded source blocks")
                    prefix = prompt[:-len(source_body)]
                    for group in groups:
                        numbered = [(blocks[i][0], block) for i, block in group]
                        group_prompt = prefix + core.render_chunk(numbered)
                        translated.update(core._translate_chunk(numbered, [], core.Cache(cache / "chunks.jsonl"),
                            options, core.Stats(), lambda event: events.append(event) if event.get("event") == "usage" else None,
                            lang, prepared=(group_prompt, core._chunk_key(group_prompt, options, lang))))
                    candidate["text"] = "\n\n".join(
                        f"[[{idx}|{block['tag']}]]\n" + "\n".join(translated[idx])
                        for idx, block in sample["blocks"] if block["lines"])
                else:
                    candidate["text"] = translate_markdown_with_retry(sample["source"],
                        max_hangul_ratio=.03, cache_dir=cache, max_chars=max_chars, on_result=account)
    except Exception as exc:
        candidate["error"] = str(exc)
        candidate["errorKind"] = getattr(getattr(exc, "result", None), "error_kind", None)
    candidate["calls"] = events
    candidate["usage"] = {key: sum(event.get("usage", {}).get(key, 0) or 0 for event in events)
                          for key in {key for event in events for key in event.get("usage", {})}}
    candidate["attempts"] = sum(event.get("attempts", 0) for event in events)
    candidate["latencyMs"] = round((time.monotonic() - started) * 1000)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--candidates", type=Path, help="JSON: sample ID → text or {text, usage}")
    parser.add_argument("--generate", action="store_true", help="Explicitly call current registered models (billable)")
    parser.add_argument("--context-chars", type=int, default=0, choices=(0, 600, 1200))
    parser.add_argument("--id", action="append", help="Only evaluate selected frozen IDs")
    parser.add_argument("--mode", choices=("model", "pipeline"), default="pipeline")
    parser.add_argument("--max-chars", type=int, help="Pipeline chunk target; default is the adapter default.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_chars is not None and args.max_chars < 1:
        parser.error("--max-chars must be positive")
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
    supplied = json.loads(args.candidates.read_text()) if args.candidates else {}
    candidates = supplied.get("candidates", supplied)
    if args.generate:
        for sample in samples:
            candidate = generate_candidate(sample, mode=args.mode, context_chars=args.context_chars,
                                           max_chars=args.max_chars)
            candidates[sample["id"]] = candidate
            if candidate.get("errorKind") in {"authentication", "quota", "policy", "configuration"}:
                break
    report = {"adapterHashes": adapter_hashes() if args.generate else supplied.get("adapterHashes"),
              "mode": args.mode, "contextChars": args.context_chars, "candidates": candidates,
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
