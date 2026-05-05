"""Image generation and browser automation tools."""

from __future__ import annotations

import asyncio


def _build_generate_image_description() -> str:
    """Short generate_image description.

    Full per-model schemas live in replicate_image_service and are surfaced on
    parameter rejection to keep the default prompt small.
    """
    return (
        "Generate image via Replicate. Returns prediction_id, model, final "
        "prompt, image URL, local path. reference_image: FLUX editing only "
        "(local path / URL / data URI) — never with rd_fast / rd_plus."
    )


GENERATE_IMAGE_TOOL = {
    "name": "generate_image",
    "description": _build_generate_image_description(),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description prompt (English). The prompt is sent directly to the model with no automatic style prefix. Include all desired visual style, composition, and aesthetic details in the prompt itself.",
            },
            "style": {
                "type": "string",
                "enum": ["poster", "game", "pixel", "portrait", "detailed", "game_asset", "1_bit", "low_res", "mc_item", "default", "retro", "watercolor", "textured", "cartoon", "ui_element", "item_sheet", "character_turnaround", "environment", "isometric", "isometric_asset", "topdown_map", "topdown_asset", "classic", "topdown_item", "mc_texture", "skill_icon"],
                "description": "FLUX: poster | game | pixel. Retro Diffusion: default / retro / pixel / isometric_asset etc. (aliases accepted). Default: poster.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "match_input_image"],
                "description": "Output aspect ratio. Default: 1:1. Use match_input_image when editing from a reference photo.",
            },
            "model": {
                "type": "string",
                "enum": ["flux_schnell", "flux_dev", "flux_kontext_dev", "rd_fast", "rd_plus"],
                "description": "Model preset. FLUX: flux_schnell (fast), flux_dev (higher quality), flux_kontext_dev (reference-image editing). Retro Diffusion: rd_fast (fast pixel art), rd_plus (higher quality pixel art). Default: flux_schnell.",
            },
            "count": {
                "type": "integer",
                "description": "Number of images to generate in one batch (1-4). Single API call, no rate limit concern. Default: 1.",
            },
            "reference_image": {
                "type": "string",
                "description": "Optional reference image for FLUX editing only. Prefer a downloaded local file path under the project root. Remote URL and data URI are also accepted. When set, backend uses input_image-compatible Replicate model routing. Do not use with rd_fast or rd_plus.",
            },
        },
        "required": ["prompt"],
    },
}

BROWSE_WEB_TOOL = {
    "name": "browse_web",
    "description": (
        "AI-driven browser automation using browser-use. "
        "An AI agent will autonomously navigate websites, fill forms, click buttons, "
        "and extract information. Use for complex multi-step web interactions "
        "(e.g., login flows, form submissions, multi-page navigation, data extraction "
        "from dynamic sites). For simple page reads, prefer fetch_url (faster, cheaper)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language description of what to do in the browser. Be specific about the goal and expected output.",
            },
            "start_url": {
                "type": "string",
                "description": "Optional URL to navigate to before starting the task.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum browser interaction steps (default: 20, max: 50).",
            },
        },
        "required": ["task"],
    },
}

MEDIA_TOOLS = [GENERATE_IMAGE_TOOL, BROWSE_WEB_TOOL]

_last_image_gen_time: float = 0.0
_IMAGE_GEN_INTERVAL: float = 8.0


def _is_retryable_image_error(message: str) -> bool:
    lowered = str(message or "").lower()
    retry_markers = (
        "throttled",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "temporarily unstable",
        "temporarily",
        "network",
    )
    return any(marker in lowered for marker in retry_markers)


async def _exec_generate_image(
    prompt: str,
    style: str = "poster",
    aspect_ratio: str = "1:1",
    model: str | None = None,
    count: int = 1,
    reference_image: str | None = None,
) -> str:
    import time as _time

    global _last_image_gen_time
    from replicate_image_service import (
        generate_image,
        is_replicate_configured,
        is_retro_diffusion_model,
        normalize_retro_diffusion_style,
    )

    if not is_replicate_configured():
        return "ERROR: REPLICATE_API_TOKEN is not configured"

    requested_count = max(1, min(4, count))
    retro_model = is_retro_diffusion_model(model)
    normalized_style = normalize_retro_diffusion_style(style) if retro_model else style
    effective_count = 1 if retro_model else requested_count
    sequential_mode = retro_model and requested_count > 1

    async def _wait_turn(delay_floor: float = 0.0) -> None:
        global _last_image_gen_time
        elapsed = _time.monotonic() - _last_image_gen_time
        minimum_wait = max(_IMAGE_GEN_INTERVAL, delay_floor)
        if elapsed < minimum_wait and _last_image_gen_time > 0:
            await asyncio.sleep(minimum_wait - elapsed)

    try:
        if retro_model and reference_image:
            return "Image generation failed: reference_image is not supported with Retro Diffusion presets (rd_fast, rd_plus)"

        if not sequential_mode:
            await _wait_turn()
            result = await generate_image(
                prompt,
                model=model,
                style=normalized_style,
                aspect_ratio=aspect_ratio,
                num_outputs=effective_count,
                download=True,
                reference_image=reference_image,
            )
            _last_image_gen_time = _time.monotonic()
            urls = result.get("image_urls", [])
            local_paths = result.get("local_paths", [])
            style_line = normalized_style if retro_model else style
            lines = [
                f"Generated {len(urls)} image(s) in 1 API call.",
                f"  prediction_id: {result.get('prediction_id')}",
                f"  model: {result.get('model')}",
                f"  style_requested: {style}",
                f"  style_effective: {style_line}",
                f"  final_prompt: {result.get('prompt', '')[:300]}",
            ]
            if reference_image:
                lines.append(f"  reference_image: {result.get('reference_image')}")
                lines.append(f"  reference_image_source: {result.get('reference_image_source')}")
            for i, url in enumerate(urls):
                lp = local_paths[i] if i < len(local_paths) else "N/A"
                lines.append(f"  [{i+1}] url: {url}")
                lines.append(f"      local_path: {lp}")
            return "\n".join(lines)

        lines = [
            f"Requested {requested_count} Retro Diffusion image(s); executing sequentially to avoid credit/rate-limit failures.",
            f"  model: {model}",
            f"  style_requested: {style}",
            f"  style_effective: {normalized_style}",
        ]
        success_count = 0
        for idx in range(requested_count):
            backoff = 2.5
            last_error: Exception | None = None
            for attempt in range(1, 4):
                await _wait_turn(delay_floor=backoff)
                try:
                    result = await generate_image(
                        prompt,
                        model=model,
                        style=normalized_style,
                        aspect_ratio=aspect_ratio,
                        num_outputs=1,
                        download=True,
                        reference_image=reference_image,
                    )
                    _last_image_gen_time = _time.monotonic()
                    success_count += 1
                    url = (result.get("image_urls") or [None])[0]
                    local_path = (result.get("local_paths") or [None])[0]
                    lines.append(f"  [{idx+1}] prediction_id: {result.get('prediction_id')}")
                    lines.append(f"      url: {url}")
                    lines.append(f"      local_path: {local_path}")
                    if attempt > 1:
                        lines.append(f"      attempts: {attempt}")
                    last_error = None
                    break
                except Exception as exc:
                    _last_image_gen_time = _time.monotonic()
                    last_error = exc
                    if attempt < 3 and _is_retryable_image_error(str(exc)):
                        lines.append(f"  [{idx+1}] retrying after transient failure (attempt {attempt}/3): {exc}")
                        backoff = min(backoff * 2, 20.0)
                        continue
                    lines.append(f"  [{idx+1}] failed: {exc}")
                    break
            if last_error is not None:
                break
        lines.insert(1, f"  completed: {success_count}/{requested_count}")
        return "\n".join(lines)
    except Exception as exc:
        _last_image_gen_time = _time.monotonic()
        return f"Image generation failed: {exc}"


async def _exec_browse_web(task: str, start_url: str | None = None, max_steps: int = 20, **_kw) -> str:
    try:
        from browser.use_agent import browse
        from shared import diagnose_url_fetch_failure, extract_urls

        max_steps = max(1, min(int(max_steps), 50))
        result = await browse(task, max_steps=max_steps, start_url=start_url)

        parts = []
        if result["success"]:
            parts.append("[OK] Task completed")
        else:
            parts.append("[FAIL] Task did not complete successfully")

        parts.append(f"Steps: {result['steps']} | Duration: {result['duration_seconds']}s")

        if result["urls"]:
            parts.append(f"Visited: {', '.join(str(url) for url in result['urls'][:5])}")

        if result["result"]:
            text = result["result"]
            if len(text) > 15000:
                text = text[:15000] + f"\n... [truncated, total {len(result['result'])} chars]"
            parts.append(f"\n--- Result ---\n{text}")

        if result["extracted_content"]:
            content_str = "\n".join(str(content) for content in result["extracted_content"] if content)
            if content_str.strip():
                if len(content_str) > 10000:
                    content_str = content_str[:10000] + "\n... [truncated]"
                parts.append(f"\n--- Extracted ---\n{content_str}")

        if result["errors"]:
            errs = [str(error) for error in result["errors"] if error]
            if errs:
                parts.append(f"\nErrors: {'; '.join(errs[:3])}")
                task_urls = extract_urls(task)
                target_url = start_url or (task_urls[0] if task_urls else None)
                if target_url:
                    parts.append(diagnose_url_fetch_failure(target_url, errs))

        return "\n".join(parts)
    except Exception as exc:
        try:
            from shared import diagnose_url_fetch_failure, extract_urls

            task_urls = extract_urls(task)
            target_url = start_url or (task_urls[0] if task_urls else None)
            if target_url:
                return f"browse_web error: {exc}\n{diagnose_url_fetch_failure(target_url, [str(exc)])}"
        except Exception:
            pass
        return f"browse_web error: {exc}"


MEDIA_TOOL_HANDLERS = {
    "generate_image": _exec_generate_image,
    "browse_web": _exec_browse_web,
}
