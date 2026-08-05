"""Shared boilerplate for the four FastAPI service entrypoints.

Each service module (api, a2a_api, email_api, novel_writer_api) keeps its
own FastAPI() construction, routers, health endpoints and uvicorn tail —
only the identical logging setup and CORS-origin parsing live here.
"""
import logging
import os

DEFAULT_CORS_ORIGINS = "https://cyber-lenin.com,http://localhost:3000"


def setup_service_logging(quiet_neo4j: bool = False) -> None:
    """LOG_LEVEL-driven basicConfig, identical across the API services."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(level)
    if quiet_neo4j:
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)


def parse_cors_origins(*service_envs: str) -> list[str]:
    """Comma-separated CORS origins: service-specific env(s) first, then the
    shared WEBCHAT_CORS_ORIGINS / CORS_ALLOW_ORIGINS chain, then defaults."""
    raw = ""
    for env_name in (*service_envs, "WEBCHAT_CORS_ORIGINS", "CORS_ALLOW_ORIGINS"):
        raw = os.getenv(env_name) or ""
        if raw:
            break
    raw = raw or DEFAULT_CORS_ORIGINS
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or [item.strip() for item in DEFAULT_CORS_ORIGINS.split(",")]
