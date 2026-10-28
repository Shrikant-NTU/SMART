from __future__ import annotations

import os
from pathlib import Path


def resolve_openai_api_key() -> str:
    load_local_env_file()
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY or OPENAI_API_KEY is not set. Export it in the same shell "
            "that runs main.py, or put API_KEY=... in sfm_refinement_agent/.env."
        )
    return api_key


def load_local_env_file() -> None:
    for env_path in _candidate_env_paths():
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            name = name.strip()
            if name not in {"API_KEY", "OPENAI_API_KEY"}:
                continue
            os.environ.setdefault(name, _clean_env_value(value))


def _candidate_env_paths() -> list[Path]:
    module_dir = Path(__file__).resolve().parent
    return [
        module_dir / ".env",
        Path.cwd() / ".env",
        Path.home() / ".sfm_refinement_agent.env",
    ]


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned
