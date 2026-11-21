"""Загрузка и проверка каталога. Описания объявлений НЕ сохраняются: они не нужны логике подбора
(источник фактов — структурированные поля), а значит не могут повлиять на ответ."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .conditions import CATEGORIES


class CatalogError(Exception):
    pass


@dataclass(frozen=True)
class Listing:
    id: str
    category: str
    title: str
    price_rub: int
    city_id: str
    attributes: dict


def _reject_constant(name):
    raise ValueError(f"недопустимая константа JSON: {name}")


def load_catalog(path, log=lambda *_: None) -> list:
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise CatalogError(f"не удалось прочитать каталог {path}: {exc}") from exc
    if not isinstance(data, list):
        raise CatalogError("каталог должен быть JSON-массивом объявлений")

    items, seen = [], set()
    for index, row in enumerate(data):
        problem = _check(row, seen)
        if problem:
            log(f"каталог: запись #{index} пропущена ({problem})")
            continue
        seen.add(row["id"])
        items.append(Listing(row["id"], row["category"], str(row.get("title", "")),
                             row["price_rub"], row["city_id"], dict(row["attributes"])))
    return items


def _check(row, seen):
    if not isinstance(row, dict):
        return "не объект"
    rid = row.get("id")
    if not isinstance(rid, str) or not rid:
        return "некорректный id"
    if rid in seen:
        return f"повтор id {rid}"
    if row.get("category") not in CATEGORIES:
        return f"неизвестная категория у {rid}"
    price = row.get("price_rub")
    if isinstance(price, bool) or not isinstance(price, int) or price <= 0:
        return f"некорректная цена у {rid}"
    if not isinstance(row.get("city_id"), str):
        return f"некорректный city_id у {rid}"
    if not isinstance(row.get("attributes"), dict):
        return f"attributes не объект у {rid}"
    return None
