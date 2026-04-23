"""replicate_image_service.py — Minimal Replicate image generation service for Leninbot.

MVP goals:
- Safe env-based configuration
- Simple model routing (default: FLUX Schnell on Replicate)
- Async-friendly polling flow
- Optional local file download for Telegram delivery / later agent workflows
- Small prompt helper for simple Soviet-style poster/game concept art
"""

import asyncio
import base64
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from secrets_loader import get_secret

PROJECT_ROOT = str(Path(__file__).resolve().parent)

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = (get_secret("REPLICATE_API_TOKEN", "") or "").strip()
REPLICATE_API_BASE = os.getenv("REPLICATE_API_BASE", "https://api.replicate.com/v1").rstrip("/")
REPLICATE_DEFAULT_MODEL = os.getenv(
    "REPLICATE_DEFAULT_MODEL",
    "black-forest-labs/flux-schnell",
).strip()
REPLICATE_WEBHOOK_URL = os.getenv("REPLICATE_WEBHOOK_URL", "").strip()
REPLICATE_POLL_INTERVAL = float(os.getenv("REPLICATE_POLL_INTERVAL", "1.5"))
REPLICATE_TIMEOUT_SEC = int(os.getenv("REPLICATE_TIMEOUT_SEC", "180"))
IMAGE_OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output", "generated_images")))

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "flux_schnell": {
        "model": "black-forest-labs/flux-schnell",
        "input": {
            "num_outputs": 1,
            "output_format": "png",
            "aspect_ratio": "1:1",
        },
    },
    "flux_dev": {
        "model": "black-forest-labs/flux-dev",
        "input": {
            "num_outputs": 1,
            "output_format": "png",
            "aspect_ratio": "1:1",
        },
    },
    "flux_kontext_dev": {
        "model": "black-forest-labs/flux-kontext-dev",
        "input": {
            "output_format": "png",
            "aspect_ratio": "1:1",
        },
    },
    "rd_fast": {
        "model": "retro-diffusion/rd-fast",
        "input": {
            "output_format": "png",
            "width": 256,
            "height": 256,
            "style": "portrait",
        },
    },
    "rd_plus": {
        "model": "retro-diffusion/rd-plus",
        "input": {
            "output_format": "png",
            "width": 256,
            "height": 256,
            "style": "portrait",
        },
    },
}

RETRO_DIFFUSION_MODELS = {"rd_fast", "rd_plus"}
RETRO_DIFFUSION_STYLE_ALIASES = {
    "portrait": "default",
    "detailed": "retro",
    "game_asset": "isometric_asset",
    "1_bit": "classic",
    "one_bit": "classic",
    "low_res": "low_res",
    "mc_item": "mc_item",
    "pixel": "default",
    "default": "default",
    "retro": "retro",
    "watercolor": "watercolor",
    "textured": "textured",
    "cartoon": "cartoon",
    "ui_element": "ui_element",
    "item_sheet": "item_sheet",
    "character_turnaround": "character_turnaround",
    "environment": "environment",
    "isometric": "isometric",
    "isometric_asset": "isometric_asset",
    "topdown_map": "topdown_map",
    "topdown_asset": "topdown_asset",
    "classic": "classic",
    "topdown_item": "topdown_item",
    "mc_texture": "mc_texture",
    "skill_icon": "skill_icon",
}


def _normalize_retro_style(style: str) -> str:
    key = str(style or "portrait").strip().lower().replace("-", "_").replace(" ", "_")
    return RETRO_DIFFUSION_STYLE_ALIASES.get(key, "default")


def is_retro_diffusion_model(model: str | None) -> bool:
    return (model or "") in RETRO_DIFFUSION_MODELS


def normalize_retro_diffusion_style(style: str | None) -> str:
    return _normalize_retro_style(str(style or "portrait"))


def _aspect_ratio_to_dimensions(aspect_ratio: str) -> tuple[int, int]:
    mapping = {
        "1:1": (256, 256),
        "16:9": (320, 180),
        "9:16": (180, 320),
        "4:3": (256, 192),
        "3:4": (192, 256),
    }
    return mapping.get(aspect_ratio, (256, 256))


@dataclass
class ReplicateErrorInfo:
    category: str
    user_message: str
    retryable: bool
    detail: str
    status_code: int | None = None


class ReplicateImageError(RuntimeError):
    def __init__(self, info: ReplicateErrorInfo):
        super().__init__(info.detail)
        self.info = info


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("detail", "error", "message", "title"):
            value = payload.get(key)
            if value:
                return str(value)
        if payload:
            return str(payload)
    if payload:
        return str(payload)
    return "Unknown Replicate API error"


def _classify_replicate_error(exc: Exception) -> ReplicateErrorInfo:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        status_code = response.status_code if response is not None else None
        payload = None
        if response is not None:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
        detail = _extract_error_message(payload)
        lowered = detail.lower()

        if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
            return ReplicateErrorInfo(
                category="rate_limit",
                user_message="⏳ 이미지 생성 한도에 걸렸다. 잠시 후 다시 시도해라.",
                retryable=True,
                detail=detail,
                status_code=status_code,
            )
        if status_code == 422:
            return ReplicateErrorInfo(
                category="invalid_request",
                user_message="⚠️ 이미지 요청이 거부됐다. 프롬프트나 옵션을 조금 단순하게 바꿔 다시 시도해라.",
                retryable=False,
                detail=detail,
                status_code=status_code,
            )
        if status_code in {408, 409, 425, 500, 502, 503, 504}:
            return ReplicateErrorInfo(
                category="temporary_api_error",
                user_message="🔄 이미지 서버가 일시적으로 불안정하다. 잠시 후 다시 시도해라.",
                retryable=True,
                detail=detail,
                status_code=status_code,
            )
        return ReplicateErrorInfo(
            category="api_error",
            user_message="❌ 이미지 생성 API가 요청을 처리하지 못했다.",
            retryable=False,
            detail=detail,
            status_code=status_code,
        )

    if isinstance(exc, requests.RequestException):
        return ReplicateErrorInfo(
            category="network_error",
            user_message="🌐 이미지 서버와 통신이 흔들렸다. 잠시 후 다시 시도해라.",
            retryable=True,
            detail=str(exc),
        )

    if isinstance(exc, TimeoutError):
        return ReplicateErrorInfo(
            category="timeout",
            user_message="⏱️ 이미지 생성 대기 시간이 초과됐다. 다시 시도해라.",
            retryable=True,
            detail=str(exc),
        )

    lowered = str(exc).lower()
    if "rate limit" in lowered or "too many requests" in lowered:
        return ReplicateErrorInfo(
            category="rate_limit",
            user_message="⏳ 이미지 생성 한도에 걸렸다. 잠시 후 다시 시도해라.",
            retryable=True,
            detail=str(exc),
        )

    return ReplicateErrorInfo(
        category="unknown",
        user_message="❌ 이미지 생성 중 예기치 않은 오류가 발생했다.",
        retryable=False,
        detail=str(exc),
    )


@lru_cache(maxsize=16)
def _get_model_metadata(model_name: str) -> dict[str, Any]:
    response = requests.get(
        f"{REPLICATE_API_BASE}/models/{model_name}",
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=16)
def _get_model_input_schema(model_name: str) -> dict[str, Any]:
    """Extract the Input schema properties from a model's OpenAPI spec.

    Returns a dict of {param_name: schema_dict} for valid input parameters.
    Empty dict if schema cannot be parsed (caller should skip filtering).
    """
    try:
        meta = _get_model_metadata(model_name)
        version = meta.get("latest_version") or {}
        openapi = version.get("openapi_schema") or {}
        schemas = openapi.get("components", {}).get("schemas", {})
        input_schema = schemas.get("Input", {})
        return input_schema.get("properties", {})
    except Exception as e:
        logger.debug("[replicate] Failed to parse input schema for %s: %s", model_name, e)
        return {}


def _filter_input_by_schema(model_name: str, payload_input: dict[str, Any]) -> dict[str, Any]:
    """Filter payload input to only include parameters the model accepts.

    Prevents 422 errors from sending unsupported parameters.
    Always passes through 'prompt' regardless of schema parsing.
    """
    schema_props = _get_model_input_schema(model_name)
    if not schema_props:
        return payload_input  # schema unavailable, pass through as-is
    valid_keys = set(schema_props.keys())
    valid_keys.add("prompt")  # always required
    filtered = {}
    for k, v in payload_input.items():
        if k in valid_keys:
            filtered[k] = v
        else:
            logger.info("[replicate] Dropping unsupported param '%s' for model %s", k, model_name)
    return filtered


def get_model_schemas_description() -> str:
    """Build a human-readable summary of each model's valid input parameters.

    Fetched live from Replicate API (cached). Used to populate the
    generate_image tool description so agents know valid params upfront.
    """
    lines = []
    for preset_key, preset in MODEL_PRESETS.items():
        model_name = preset["model"]
        try:
            props = _get_model_input_schema(model_name)
        except Exception:
            props = {}
        if not props:
            continue
        param_parts = []
        for pname, pschema in sorted(props.items()):
            if pname == "prompt":
                continue  # always present, not interesting
            info = pname
            ptype = pschema.get("type", "")
            default = pschema.get("default")
            enum = pschema.get("allOf", [{}])[0].get("enum") or pschema.get("enum")
            parts = []
            if ptype:
                parts.append(ptype)
            if enum:
                parts.append(f"enum={enum}")
            elif default is not None:
                parts.append(f"default={default}")
            if parts:
                info += f" ({', '.join(parts)})"
            param_parts.append(info)
        lines.append(f"  {preset_key} ({model_name}): {', '.join(param_parts)}")
    return "\n".join(lines)


@lru_cache(maxsize=16)
def _model_supports_official_endpoint(model_name: str) -> bool:
    data = _get_model_metadata(model_name)
    # Newer official models like black-forest-labs/flux-schnell reject POST /predictions
    # with {version, model}. They expose their own predictions URL instead.
    official_url = str((data.get("urls") or {}).get("predictions") or "").strip()
    return bool(official_url)


def is_replicate_configured() -> bool:
    return bool(REPLICATE_API_TOKEN)


def _headers() -> dict[str, str]:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is not configured")
    return {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }


def build_soviet_prompt(
    prompt: str,
    style: str = "poster",
    aspect_ratio: str = "1:1",
    *,
    model: str | None = None,
    apply_style_prefix: bool = True,
) -> str:
    base_prompt = prompt.strip().rstrip(".")

    if model in RETRO_DIFFUSION_MODELS:
        if apply_style_prefix:
            return (
                f"{base_prompt}. "
                "pixel art, grid-aligned, clean silhouette, limited palette, no text, no watermark, no logo."
            )
        return base_prompt

    if not apply_style_prefix:
        return base_prompt

    style_map = {
        "poster": "simple Soviet propaganda poster aesthetic, bold geometric composition, limited red black cream palette, clean silhouette, strong contrast",
        "game": "simple Soviet-style game concept art, readable shapes, stylized propaganda mood, limited palette, clear focal point",
        "pixel": "simple Soviet-inspired game key art, retro game readability, restricted palette, poster-like composition",
    }
    style_prefix = style_map.get(style, style_map["poster"])
    return (
        f"{style_prefix}. "
        f"{base_prompt}. "
        f"No text, no watermark, no logo. aspect ratio {aspect_ratio}."
    )


def _resolve_model_config(model: str | None = None) -> dict[str, Any]:
    if not model:
        preset = MODEL_PRESETS["flux_schnell"]
        resolved_model = REPLICATE_DEFAULT_MODEL or preset["model"]
        metadata = _get_model_metadata(resolved_model)
        return {
            "model": resolved_model,
            "version": ((metadata.get("latest_version") or {}).get("id") or "").strip(),
            "use_official_model_endpoint": _model_supports_official_endpoint(resolved_model),
            "official_prediction_url": str((metadata.get("urls") or {}).get("predictions") or "").strip(),
            "input": dict(preset["input"]),
        }
    if model in MODEL_PRESETS:
        preset = MODEL_PRESETS[model]
        metadata = _get_model_metadata(preset["model"])
        return {
            "model": preset["model"],
            "version": ((metadata.get("latest_version") or {}).get("id") or "").strip(),
            "use_official_model_endpoint": _model_supports_official_endpoint(preset["model"]),
            "official_prediction_url": str((metadata.get("urls") or {}).get("predictions") or "").strip(),
            "input": dict(preset["input"]),
        }
    metadata = _get_model_metadata(model)
    return {
        "model": model,
        "version": ((metadata.get("latest_version") or {}).get("id") or "").strip(),
        "use_official_model_endpoint": _model_supports_official_endpoint(model),
        "official_prediction_url": str((metadata.get("urls") or {}).get("predictions") or "").strip(),
        "input": dict(MODEL_PRESETS["flux_schnell"]["input"]),
    }


def create_prediction(
    prompt: str,
    *,
    model: str | None = None,
    aspect_ratio: str = "1:1",
    webhook_url: str | None = None,
    extra_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _resolve_model_config(model)
    payload_input: dict[str, Any] = {
        **cfg["input"],
        "prompt": prompt,
    }
    if model not in RETRO_DIFFUSION_MODELS:
        payload_input["aspect_ratio"] = aspect_ratio
    if extra_input:
        payload_input.update(extra_input)
    # Filter out parameters the model doesn't accept (prevents 422 errors)
    payload_input = _filter_input_by_schema(cfg["model"], payload_input)
    payload: dict[str, Any] = {"input": payload_input}
    official_prediction_url = str(cfg.get("official_prediction_url") or "").strip()
    if official_prediction_url:
        prediction_url = official_prediction_url
    elif cfg.get("use_official_model_endpoint"):
        prediction_url = f"{REPLICATE_API_BASE}/models/{cfg['model']}/predictions"
    else:
        prediction_url = f"{REPLICATE_API_BASE}/predictions"
        payload["version"] = cfg["version"]
    webhook = webhook_url or REPLICATE_WEBHOOK_URL
    if webhook:
        payload["webhook"] = webhook
        payload["webhook_events_filter"] = ["completed"]

    logger.info("[replicate] POST %s model=%s has_version=%s", prediction_url, cfg["model"], "version" in payload)
    try:
        response = requests.post(
            prediction_url,
            headers=_headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        info = _classify_replicate_error(exc)
        logger.warning("[replicate] create_prediction failed category=%s retryable=%s detail=%s", info.category, info.retryable, info.detail)
        raise ReplicateImageError(info) from exc


def get_prediction(prediction_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{REPLICATE_API_BASE}/predictions/{prediction_id}",
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


async def wait_for_prediction(
    prediction_id: str,
    *,
    poll_interval: float = REPLICATE_POLL_INTERVAL,
    timeout_sec: int = REPLICATE_TIMEOUT_SEC,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        try:
            prediction = await asyncio.to_thread(get_prediction, prediction_id)
        except Exception as exc:
            info = _classify_replicate_error(exc)
            logger.warning("[replicate] get_prediction failed category=%s retryable=%s detail=%s", info.category, info.retryable, info.detail)
            raise ReplicateImageError(info) from exc
        status = prediction.get("status")
        if status == "succeeded":
            return prediction
        if status in {"failed", "canceled"}:
            error = str(prediction.get("error") or f"Replicate prediction ended with status={status}")
            info = _classify_replicate_error(RuntimeError(error))
            raise ReplicateImageError(info)
        if time.monotonic() - started > timeout_sec:
            raise ReplicateImageError(_classify_replicate_error(TimeoutError(f"Replicate prediction timed out after {timeout_sec}s")))
        await asyncio.sleep(poll_interval)


def extract_output_urls(prediction: dict[str, Any]) -> list[str]:
    output = prediction.get("output")
    if isinstance(output, list):
        return [str(item) for item in output if item]
    if isinstance(output, str):
        return [output]
    return []


def _safe_slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "-", text).strip("-")
    return (text or "image")[:max_len]


def download_image(url: str, prompt: str, *, output_dir: Path | None = None) -> str:
    out_dir = output_dir or IMAGE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_slug(prompt)}.png"
    path = out_dir / filename
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return str(path)


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _encode_file_to_data_uri(path: str | Path) -> str:
    file_path = Path(path)
    mime_type = _guess_mime_type(file_path)
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def prepare_reference_image(reference_image: str) -> tuple[str, str]:
    """Normalize a local path / remote URL / data URI into a Replicate-safe input string.

    Returns (normalized_value, source_kind) where source_kind is one of
    local_path, remote_url, data_uri.
    """
    value = str(reference_image or "").strip()
    if not value:
        raise ValueError("reference_image is empty")
    if value.startswith("data:"):
        return value, "data_uri"
    if re.match(r"^https?://", value, re.IGNORECASE):
        return value, "remote_url"

    path = Path(value)
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {path}")
    return _encode_file_to_data_uri(path), "local_path"


async def generate_image(
    prompt: str,
    *,
    model: str | None = None,
    style: str = "poster",
    aspect_ratio: str = "1:1",
    num_outputs: int = 1,
    download: bool = True,
    extra_input: dict[str, Any] | None = None,
    reference_image: str | None = None,
) -> dict[str, Any]:
    if not is_replicate_configured():
        raise RuntimeError("REPLICATE_API_TOKEN is missing")

    num_outputs = max(1, min(4, num_outputs))
    merged_extra = dict(extra_input or {})
    resolved_model = model or REPLICATE_DEFAULT_MODEL
    reference_source: str | None = None

    if resolved_model in RETRO_DIFFUSION_MODELS:
        width, height = _aspect_ratio_to_dimensions(aspect_ratio)
        merged_extra.setdefault("width", width)
        merged_extra.setdefault("height", height)
        merged_extra["style"] = _normalize_retro_style(str(merged_extra.get("style") or style))
        if num_outputs > 1:
            logger.warning(
                "[replicate] retro diffusion models do not use flux num_outputs; forcing num_outputs=1 requested=%d model=%s",
                num_outputs,
                resolved_model,
            )
        num_outputs = 1
    if reference_image and resolved_model in RETRO_DIFFUSION_MODELS:
        raise ValueError("Retro Diffusion presets do not support reference_image in this wrapper")

    apply_style_prefix = False
    if reference_image:
        normalized_reference, reference_source = prepare_reference_image(reference_image)
        merged_extra["input_image"] = normalized_reference
        if model in {None, "flux_schnell", "flux_dev"}:
            resolved_model = "flux_kontext_dev"
        if num_outputs > 1:
            logger.warning(
                "[replicate] reference_image path does not support multi-output reliably; requested=%d forcing num_outputs=1 model=%s",
                num_outputs,
                resolved_model,
            )
            num_outputs = 1
        if aspect_ratio == "match_input_image":
            logger.info("[replicate] reference_image using caller-requested match_input_image aspect ratio")
        if resolved_model == "flux_kontext_dev":
            apply_style_prefix = False
    else:
        merged_extra["num_outputs"] = num_outputs

    final_prompt = build_soviet_prompt(
        prompt,
        style=style,
        aspect_ratio=aspect_ratio,
        model=resolved_model,
        apply_style_prefix=apply_style_prefix,
    )
    logger.info(
        "[replicate] create prediction model=%s style=%s num_outputs=%d reference=%s",
        resolved_model,
        style,
        num_outputs,
        bool(reference_image),
    )
    prediction = await asyncio.to_thread(
        create_prediction,
        final_prompt,
        model=resolved_model,
        aspect_ratio=aspect_ratio,
        extra_input=merged_extra,
    )
    prediction_id = prediction.get("id")
    if not prediction_id:
        raise ReplicateImageError(
            ReplicateErrorInfo(
                category="invalid_response",
                user_message="❌ 이미지 생성 응답이 비정상적이다. 다시 시도해라.",
                retryable=True,
                detail="Replicate response missing prediction id",
            )
        )

    completed = await wait_for_prediction(prediction_id)
    urls = extract_output_urls(completed)
    if not urls:
        raise ReplicateImageError(
            ReplicateErrorInfo(
                category="invalid_response",
                user_message="❌ 이미지 결과가 비어 있다. 다시 시도해라.",
                retryable=True,
                detail="Replicate returned no image output",
            )
        )

    # Download all output images
    local_paths: list[str] = []
    if download:
        for idx, url in enumerate(urls):
            try:
                suffix = f"_{idx+1}" if len(urls) > 1 else ""
                path = await asyncio.to_thread(download_image, url, f"{prompt}{suffix}")
                local_paths.append(path)
            except Exception as exc:
                logger.warning("[replicate] download_image #%d failed: %s", idx, exc)

    return {
        "prediction_id": prediction_id,
        "model": completed.get("model") or resolved_model,
        "prompt": final_prompt,
        "status": completed.get("status"),
        "image_urls": urls,
        "local_path": local_paths[0] if local_paths else None,
        "local_paths": local_paths,
        "reference_image": reference_image,
        "reference_image_source": reference_source,
        "raw": completed,
    }
