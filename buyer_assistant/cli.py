"""Точка входа: ./run --catalog catalog.json < request.json > response.json"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .catalog import CatalogError, load_catalog
from .config import load_settings
from .llm import LLMError
from .pipeline import RequestError, process, validate_request

TIME_BUDGET_S = 100          # контракт: 120 с на запуск, держим запас


def _log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    started = time.monotonic()
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="ИИ-помощник покупателя")
    parser.add_argument("--catalog", required=True, help="путь к catalog.json")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))
        messages = validate_request(payload)
        catalog = load_catalog(args.catalog, _log)
    except (ValueError, RequestError, CatalogError) as exc:
        _log(f"ошибка входных данных: {exc}")
        return 2

    allow_fallback = os.environ.get("LLM_FALLBACK", "1") != "0"
    try:
        response = process(messages, catalog, load_settings(), started + TIME_BUDGET_S, _log, allow_fallback)
    except LLMError as exc:
        _log(f"ошибка модели: {exc}")
        return 3

    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0
