"""Resumable Gemini inline Batch API transport for archival translation chunks."""

from __future__ import annotations

import fcntl
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import core as at

MAX_INLINE_BYTES = 19_000_000  # Gemini's request limit is 20 MB.
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _manifest_path(spec: dict, opts: at.Options) -> Path:
    return at._cache_path(spec, opts.cache_path).with_suffix(".batch.json")


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _save(path: Path, data: dict) -> None:
    from translation_runtime.storage import atomic_write

    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))


def _load(path: Path) -> dict:
    if not path.is_file():
        raise at.SpecError(f"배치 기록이 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _client():
    from google import genai
    from llm.call_registry import resolve_provider_connection

    connection = resolve_provider_connection("gemini")
    return genai.Client(
        api_key=connection.api_key,
        **({"http_options": {"base_url": connection.base_url}}
           if connection.base_url else {}),
    )


def _pending(spec: dict, opts: at.Options):
    from llm.call_registry import resolve

    if spec.get("frozen"):
        raise at.SpecError(f"{spec['id']}: frozen 스펙에는 배치를 제출하지 않는다")
    prepared = at.plan(spec, opts)
    lang = prepared["_lang"]
    profile = resolve(lang.feature)
    if profile.provider != "gemini":
        raise at.SpecError(f"{lang.feature}: Gemini Batch API는 gemini provider만 지원한다")
    cache = at.Cache(at._cache_path(spec, opts.cache_path))
    tm_filled = at._tm_prefill(prepared["_docs"], lang, lambda _: None, spec)
    pending = []
    for chunk in prepared["_chunks"]:
        if all(idx in tm_filled or not block["lines"] for idx, block in chunk):
            continue
        prompt, key = at._prepare_chunk(chunk, prepared["_glossary"], opts, lang)
        if at._cached_blocks(cache, key, chunk, lang, at._legacy_chunk_key(prompt, opts, lang))[0] is not None:
            continue
        if any(sum(map(len, block["lines"])) > opts.max_chars for _, block in chunk):
            raise at.SpecError("초대형 블록은 기존 동기 분할 경로로 번역해야 한다")
        pending.append((chunk, prompt, key))
    return prepared, profile, cache, pending


def _request_config(profile):
    from google.genai.types import GenerateContentConfig

    config = GenerateContentConfig(
        temperature=profile.temperature,
        max_output_tokens=profile.max_tokens,
        system_instruction=None,  # Set per request below, with language-specific rules.
        **({"thinking_config": {"thinking_level": profile.extra["thinking_level"]}}
           if profile.extra.get("thinking_level") else {}),
    )
    return config


def _batch_requests(pending, profile, lang):
    base = _request_config(profile)
    result = []
    for chunk, prompt, key in pending:
        config = base.model_copy(update={"system_instruction": lang.system_prompt})
        result.append({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "config": config,
            "metadata": {"key": key},
        })
    return result


def _state(job) -> str:
    value = getattr(job, "state", None)
    return getattr(value, "name", str(value))


def _audit_response(manifest: dict, entry: dict, response, profile) -> None:
    from llm.call_registry import _gemini_usage
    from llm.gateway import estimate_cost_usd, record_llm_call
    from llm.provider_registry import GEMINI_PRICING

    model_response = getattr(response, "response", None)
    usage = _gemini_usage(model_response) if model_response else {}
    cost = estimate_cost_usd(
        profile.model, tokens_in=usage.get("tokens_in", 0),
        tokens_out=usage.get("tokens_out", 0),
        cache_read=usage.get("cache_read", 0), token_semantics="gemini",
    )
    # Batch halves uncached input/output; cache-hit tokens retain the normal
    # cached-input rate in Google's pricing contract.
    pricing = GEMINI_PRICING.get(profile.model)
    cached_cost = (usage.get("cache_read", 0) * pricing.get("cached_input", pricing["input"])
                   if pricing else 0)
    batch_cost = (cost + cached_cost) * 0.5 if cost is not None else None
    record_llm_call(
        surface="archival_batch", caller=manifest["feature"], provider="gemini",
        model=profile.model, label=f"{manifest['spec_id']}:{manifest['job_name']}:{entry['key'][:12]}",
        status="ok" if model_response and not getattr(response, "error", None) else "error",
        error_excerpt=str(getattr(response, "error", None) or "")[:300] or None,
        tokens_in=usage.get("tokens_in", 0), tokens_out=usage.get("tokens_out", 0),
        cache_read=usage.get("cache_read", 0), cost_usd=batch_cost,
        token_semantics="gemini", estimate_cost=False,
    )


def _recover(client, manifest: dict, path: Path) -> None:
    if manifest.get("job_name"):
        return
    matches = [job.name for job in client.batches.list()
               if getattr(job, "display_name", None) == manifest["display_name"]]
    if len(matches) != 1:
        raise at.SpecError(
            f"제출 결과가 불확실하다 ({manifest['display_name']}, 일치 작업 {len(matches)}개). "
            "중복 과금을 피하려면 제공자 배치 목록을 확인할 것")
    manifest["job_name"] = matches[0]
    manifest["state"] = "submitted"
    _save(path, manifest)


def submit(spec: dict, opts: at.Options | None = None, *, client=None,
           new_batch: bool = False) -> dict:
    """Submit uncached chunks once; never automatically repeat an uncertain create."""
    opts = opts or at.Options()
    path = _manifest_path(spec, opts)
    with _locked(path):
        if path.exists():
            if not new_batch:
                raise at.SpecError(f"기존 배치 기록이 있다: {path}. 먼저 status/collect로 확인할 것")
            previous = _load(path)
            client = client or _client()
            _recover(client, previous, path)
            old_job = client.batches.get(name=previous["job_name"])
            old_state = _state(old_job)
            if old_state not in TERMINAL or (old_state == "JOB_STATE_SUCCEEDED" and
                    not all(e.get("audited") for e in previous["entries"])):
                raise at.SpecError("이전 배치가 진행 중이거나 아직 collect되지 않았다")
            archive = path.with_name(path.stem + "." + previous["display_name"] + ".json")
            if archive.exists():
                raise at.SpecError(f"이전 배치 보관 파일이 이미 있다: {archive}")
            path.replace(archive)
        prepared, profile, _, pending = _pending(spec, opts)
        if not pending:
            return {"state": "nothing_pending", "chunks": 0}
        at.preflight(opts, prepared["_lang"])
        requests = _batch_requests(pending, profile, prepared["_lang"])
        # SDK objects are serialized through Pydantic so the size check covers
        # system instructions and generation settings, not just user prompts.
        size = len(json.dumps(requests, default=lambda x: x.model_dump(exclude_none=True),
                              ensure_ascii=False).encode("utf-8"))
        if size > MAX_INLINE_BYTES:
            raise at.SpecError(f"inline 배치 {size:,}B가 안전 상한 {MAX_INLINE_BYTES:,}B를 초과한다")
        manifest = {
            "version": 1, "spec_id": spec["id"], "model": profile.model,
            "feature": prepared["_lang"].feature,
            "options": {"max_chars": opts.max_chars, "glossary_limit": opts.glossary_limit},
            "display_name": f"archival-{spec['id']}-{uuid.uuid4().hex[:12]}"[:128],
            "state": "submitting", "job_name": None,
            "entries": [{"key": key, "blocks": [idx for idx, _ in chunk],
                         "source_hashes": at._block_source_hashes(chunk),
                         "collected": False, "audited": False}
                        for chunk, _, key in pending],
        }
        _save(path, manifest)
        client = client or _client()
        job = client.batches.create(model=profile.model, src=requests,
                                    config={"display_name": manifest["display_name"]})
        manifest.update(state="submitted", job_name=job.name)
        _save(path, manifest)
        return {"state": "submitted", "job_name": job.name, "chunks": len(pending),
                "manifest": str(path)}


def status(spec: dict, opts: at.Options | None = None, *, client=None) -> dict:
    opts = opts or at.Options()
    path = _manifest_path(spec, opts)
    with _locked(path):
        manifest = _load(path)
        client = client or _client()
        _recover(client, manifest, path)
        job = client.batches.get(name=manifest["job_name"])
        return {"job_name": manifest["job_name"], "state": _state(job),
                "chunks": len(manifest["entries"]),
                "collected": sum(bool(e["collected"]) for e in manifest["entries"])}


def collect(spec: dict, opts: at.Options | None = None, *, client=None) -> dict:
    """Accept only source-matched, structurally valid responses into the normal cache."""
    opts = opts or at.Options()
    path = _manifest_path(spec, opts)
    with _locked(path):
        manifest = _load(path)
        prepared, profile, cache, pending = _pending(spec, opts)
        if manifest["spec_id"] != spec["id"] or manifest["model"] != profile.model:
            raise at.SpecError("배치와 현재 스펙/모델이 다르다")
        if manifest["options"] != {"max_chars": opts.max_chars, "glossary_limit": opts.glossary_limit}:
            raise at.SpecError("배치 제출 당시 청크 옵션과 다르다")
        current = {key: chunk for chunk, _, key in pending}
        for entry in manifest["entries"]:
            chunk = current.get(entry["key"])
            # A process can stop after the cache append but before recording
            # collected=true. _pending then omits that already-valid chunk.
            if not entry["collected"] and chunk is None and entry.get("audited"):
                cached = cache.get(entry["key"])
                if cached and cached.get("sourceHashes") == entry["source_hashes"]:
                    entry["collected"] = True
                    _save(path, manifest)
            if not entry["collected"] and (chunk is None or
                    entry["blocks"] != [idx for idx, _ in chunk] or
                    entry["source_hashes"] != at._block_source_hashes(chunk)):
                raise at.SpecError("배치 제출 후 원문·프롬프트·캐시가 변경됐다")
        client = client or _client()
        _recover(client, manifest, path)
        job = client.batches.get(name=manifest["job_name"])
        state = _state(job)
        if state != "JOB_STATE_SUCCEEDED":
            return {"state": state, "job_name": manifest["job_name"],
                    "error": str(getattr(job, "error", None) or "") if state in TERMINAL else None}
        responses = getattr(getattr(job, "dest", None), "inlined_responses", None)
        if responses is None or len(responses) != len(manifest["entries"]):
            raise at.SpecError("배치 응답 수가 제출 청크 수와 다르다")
        by_key = {r.metadata.get("key"): r for r in responses if getattr(r, "metadata", None)}
        if len(by_key) != len(responses) or set(by_key) != {e["key"] for e in manifest["entries"]}:
            raise at.SpecError("배치 응답 key가 누락·중복·변경됐다")
        failed = []
        for entry in manifest["entries"]:
            if entry["collected"]:
                continue
            response = by_key[entry["key"]]
            if not entry["audited"]:
                _audit_response(manifest, entry, response, profile)
                entry["audited"] = True
                _save(path, manifest)
            model_response = getattr(response, "response", None)
            if not model_response or getattr(response, "error", None):
                failed.append({"blocks": entry["blocks"], "error": str(getattr(response, "error", None))})
                continue
            candidates = getattr(model_response, "candidates", None) or []
            reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
            if reason and "STOP" not in reason:
                failed.append({"blocks": entry["blocks"], "error": f"incomplete output: {reason}"})
                continue
            try:
                got = at.parse_response(model_response.text or "")
                problems = at.validate(current[entry["key"]], got, prepared["_lang"])
            except (ValueError, TypeError, KeyError) as exc:
                problems = [str(exc)]
            if problems:
                failed.append({"blocks": entry["blocks"], "error": "; ".join(problems[:3])})
                continue
            cache.put(entry["key"], got, {"sourceHashes": entry["source_hashes"], "batchJob": manifest["job_name"]})
            entry["collected"] = True
            _save(path, manifest)
        return {"state": state, "job_name": manifest["job_name"],
                "collected": sum(bool(e["collected"]) for e in manifest["entries"]),
                "total": len(manifest["entries"]), "failed": failed,
                "next": "기존 동기 run 명령으로 실패 청크를 교정하고 조립"}
