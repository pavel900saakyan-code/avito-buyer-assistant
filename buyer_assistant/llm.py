"""Клиент Chat Completions шлюза организаторов (только стандартная библиотека).

Учтено из LLM_API.md: явный User-Agent (иначе workers.dev отдаёт 403/1010), без редиректов,
json_schema strict с откатом на json_object при 400, уважение Retry-After при 429,
ограниченное число повторов, ключ не попадает в логи.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class LLMError(Exception):
    """Сбой обращения к модели (сеть, доступ, лимиты, невалидный ответ)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _validate_base_url(address: str):
    parts = urlsplit(address)
    loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if (not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
            or (parts.scheme != "https" and not (loopback and parts.scheme == "http"))):
        raise LLMError("LLM_BASE_URL должен быть HTTPS-адресом (или HTTP на loopback)")


def _post(url, headers, body: dict, timeout: float):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(1_000_000), {}
    except urllib.error.HTTPError as err:
        return err.code, err.read(1_000_000), dict(err.headers or {})


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def chat_json(settings: dict, messages: list, schema: dict, schema_name: str, max_tokens: int,
              deadline: float, log=lambda *_: None) -> dict:
    """Один логический вызов модели с ответом-JSON-объектом. deadline — time.monotonic()."""
    key, base = settings.get("LLM_API_KEY", ""), settings.get("LLM_BASE_URL", "")
    if not key or not base:
        raise LLMError("не заданы LLM_API_KEY / LLM_BASE_URL")
    _validate_base_url(base)
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
               "User-Agent": settings.get("LLM_USER_AGENT") or "hackathon-client/1.0"}

    use_schema, attempts = True, 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 5:
            raise LLMError("исчерпан лимит времени на вызов модели")
        attempts += 1
        response_format = ({"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
                           if use_schema else {"type": "json_object"})
        body = {"model": settings.get("LLM_MODEL") or "google/gemma-4-31B-it", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0, "response_format": response_format}
        try:
            status, raw, resp_headers = _post(url, headers, body, timeout=min(remaining - 2, 90))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            log(f"LLM: сетевая ошибка ({type(exc).__name__}), попытка {attempts}")
            if attempts < 3 and deadline - time.monotonic() > 20:
                time.sleep(1.5)
                continue
            raise LLMError("не удалось связаться с API модели") from exc

        text = raw.decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        if status == 200:
            return _parse_envelope(text)
        log(f"LLM: HTTP {status}: {text[:300]}")
        if status == 400 and use_schema and attempts < 3:
            use_schema = False                      # схема не принята шлюзом — просим json_object
            continue
        if status == 429 and attempts < 4:
            try:
                wait = float(resp_headers.get("Retry-After", 3))
            except ValueError:
                wait = 3.0
            wait = min(max(wait, 1.0), 20.0)
            if deadline - time.monotonic() > wait + 10:
                time.sleep(wait)
                continue
        if status in (502, 503) and attempts < 3 and deadline - time.monotonic() > 15:
            time.sleep(1.5)
            continue
        raise LLMError(f"API модели вернуло HTTP {status}")


def _parse_envelope(text: str) -> dict:
    try:
        envelope = json.loads(text)
        content = envelope["choices"][0]["message"]["content"]
        data = json.loads(_strip_fences(content))
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError("модель вернула ответ в неожиданном формате") from exc
    if not isinstance(data, dict):
        raise LLMError("модель вернула не JSON-объект")
    return data
