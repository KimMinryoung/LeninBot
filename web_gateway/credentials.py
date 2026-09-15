"""Provider keys are read ONLY from this service's credential directory."""
import os
from pathlib import Path


def credential(name: str) -> str:
    if name not in {"TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY"}:
        return ""
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory:
        return ""
    try:
        return (Path(directory) / name.lower()).read_text().strip()
    except OSError:
        return ""
