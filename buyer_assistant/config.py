"""Настройки подключения к модели: переменные окружения > локальный .env (в Git не попадает)."""
from __future__ import annotations

import os
from pathlib import Path

KEYS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_USER_AGENT")
ROOT = Path(__file__).resolve().parent.parent


def load_settings(env_file=None) -> dict:
    settings = {"LLM_BASE_URL": "", "LLM_MODEL": "google/gemma-4-31B-it", "LLM_API_KEY": "",
                "LLM_USER_AGENT": "hackathon-team/1.0"}
    path = Path(env_file) if env_file else ROOT / ".env"
    try:
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip().strip('"').strip("'")
                    if key in KEYS and value:
                        settings[key] = value
    except OSError:
        pass
    for key in KEYS:                       # переменные окружения главнее (их может подменить проверяющий)
        if os.environ.get(key):
            settings[key] = os.environ[key]
    return settings
