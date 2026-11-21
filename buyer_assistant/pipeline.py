"""Один запрос -> один ответ по контракту. Зависимости (извлечение условий) передаются параметром для тестов."""
from __future__ import annotations

from .conditions import Conditions, missing_questions
from .extract import extract_conditions
from .responder import solve


class RequestError(Exception):
    pass


def validate_request(payload) -> list:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise RequestError("ожидается объект с непустым массивом messages")
    messages = []
    for m in payload["messages"]:
        if (not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                or not isinstance(m.get("text"), str) or not m["text"].strip()):
            raise RequestError("каждое сообщение — объект с role (user|assistant) и непустым text")
        messages.append({"role": m["role"], "text": m["text"]})
    if messages[-1]["role"] != "user":
        raise RequestError("последняя реплика должна принадлежать user")
    return messages


def process(messages, catalog, settings, deadline, log=lambda *_: None, allow_fallback=True,
            extractor=extract_conditions) -> dict:
    cond: Conditions = extractor(messages, settings, deadline, log, allow_fallback)
    log(f"условия: city={cond.city_id} budget={cond.budget_rub} categories={cond.categories} "
        f"constraints={[(c.attribute, c.op, str(c.value)) for c in cond.constraints]}")
    questions = missing_questions(cond)
    if questions:
        return {"status": "clarify", "questions": questions}
    return solve(catalog, cond)
