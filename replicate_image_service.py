"""replicate_image_service.py — Minimal Replicate image generation service for Leninbot.

MVP goals:
- Safe env-based configuration
- Simple model routing (default: FLUX Schnell on Replicate)
- Async-friendly polling flow
- Optional local file download for Telegram delivery / later agent workflows
- Small prompt helper for simple Soviet-style poster/game concept art
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.environ["PROJECT_ROOT"]

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
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
}


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
) -> str:
    style_map = {
        "poster": "simple Soviet propaganda poster aesthetic, bold geometric composition, limited red black cream palette, clean silhouette, strong contrast",
        "game": "simple Soviet-style game concept art, readable shapes, stylized propaganda mood, limited palette, clear focal point",
        "pixel": "simple Soviet-inspired game key art, retro game readability, restricted palette, poster-like composition",
    }
    style_prefix = style_map.get(style, style_map["poster"])
    return (
        f"{style_prefix}. "
        f"{prompt.strip()}. "
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
    payload: dict[str, Any] = {
        "input": {
            **cfg["input"],
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
        },
    }
    official_prediction_url = str(cfg.get("official_prediction_url") or "").strip()
    if official_prediction_url:
        prediction_url = official_prediction_url
    elif cfg.get("use_official_model_endpoint"):
        prediction_url = f"{REPLICATE_API_BASE}/models/{cfg['model']}/predictions"
    else:
        prediction_url = f"{REPLICATE_API_BASE}/predictions"
        payload["version"] = cfg["version"]
    if extra_input:
        payload["input"].update(extra_input)
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


async def generate_image(
    prompt: str,
    *,
    model: str | None = None,
    style: str = "poster",
    aspect_ratio: str = "1:1",
    download: bool = True,
    extra_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not is_replicate_configured():
        raise RuntimeError("REPLICATE_API_TOKEN is missing")

    final_prompt = build_soviet_prompt(prompt, style=style, aspect_ratio=aspect_ratio)
    logger.info("[replicate] create prediction model=%s style=%s", model or REPLICATE_DEFAULT_MODEL, style)
    prediction = await asyncio.to_thread(
        create_prediction,
        final_prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        extra_input=extra_input,
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

    local_path = None
    if download:
        try:
            local_path = await asyncio.to_thread(download_image, urls[0], prompt)
        except Exception as exc:
            info = _classify_replicate_error(exc)
            logger.warning("[replicate] download_image failed category=%s retryable=%s detail=%s", info.category, info.retryable, info.detail)
            raise ReplicateImageError(info) from exc

    return {
        "prediction_id": prediction_id,
        "model": completed.get("model") or model or REPLICATE_DEFAULT_MODEL,
        "prompt": final_prompt,
        "status": completed.get("status"),
        "image_urls": urls,
        "local_path": local_path,
        "raw": completed,
    }
