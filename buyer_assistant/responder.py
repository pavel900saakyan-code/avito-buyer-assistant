"""Сборка ответов ok / no_match. Тексты строятся из структурированных полей каталога —
ни заголовки, ни описания объявлений в объяснения не попадают."""
from __future__ import annotations

from decimal import Decimal

from .conditions import (ATTR_CATEGORY, ATTR_LABEL, ATTR_UNIT, CATEGORIES, CITIES, Conditions)
from .matcher import _as_number, candidates_by_category, enumerate_sets, pick_options

OP_TEXT = {"lt": "менее", "lte": "не более", "gt": "более", "gte": "не менее", "eq": "ровно"}
ATTR_BY_CATEGORY = {v: k for k, v in ATTR_CATEGORY.items()}


def rub(n) -> str:
    return f"{int(n):,}".replace(",", " ") + " ₽"


def num(value) -> str:
    d = Decimal(value)
    return str(int(d)) if d == d.to_integral_value() else format(d.normalize(), "f")


def _constraint_text(c) -> str:
    if c.attribute == "adjustable_height":
        return "регулировка высоты " + ("есть" if c.value else "отсутствует")
    return f"{ATTR_LABEL[c.attribute]} {OP_TEXT[c.op]} {num(c.value)} {ATTR_UNIT[c.attribute]}"


def _describe(item, cond: Conditions) -> str:
    text = f"{CATEGORIES[item.category]} {item.id} — {rub(item.price_rub)}"
    attr = ATTR_BY_CATEGORY[item.category]
    value = item.attributes.get(attr)
    fact = None
    if attr == "adjustable_height":
        if isinstance(value, bool):
            fact = "регулировка высоты есть" if value else "регулировки высоты нет"
    elif _as_number(value) is not None:
        fact = f"{ATTR_LABEL[attr]} {num(_as_number(value))} {ATTR_UNIT[attr]}"
    if fact:
        wanted = [c for c in cond.constraints if c.attribute == attr]
        if wanted:
            fact += " (требование «" + "», «".join(_constraint_text(c) for c in wanted) + "» выполнено по данным каталога)"
        text += f": {fact}"
    return text


def build_ok(catalog, cond: Conditions, chosen) -> dict:
    by_id = {x.id: x for x in catalog}
    options = []
    for index, (total, ids) in enumerate(chosen):
        items = [by_id[i] for i in ids]
        lead = "Самый дешёвый из подходящих комплектов." if index == 0 else "Другой набор товаров, тоже в рамках бюджета."
        parts = "; ".join(_describe(x, cond) for x in items)
        explanation = (f"{lead} {parts}. Все товары в городе {CITIES[cond.city_id]}. "
                       f"Итого {rub(total)} при бюджете {rub(cond.budget_rub)} (остаётся {rub(cond.budget_rub - total)}).")
        options.append({"listing_ids": list(ids), "total_rub": total, "explanation": explanation})
    return {"status": "ok", "options": options}


def solve(catalog, cond: Conditions) -> dict:
    """Основные условия известны: вернуть ok с 1–3 комплектами или no_match с объяснением."""
    per_cat = candidates_by_category(catalog, cond)
    lists = [per_cat[c][1] for c in cond.categories]
    sets = enumerate_sets(lists, cond.budget_rub)
    if sets:
        return build_ok(catalog, cond, pick_options(sets))
    return {"status": "no_match", "message": _explain_no_match(catalog, cond, per_cat)}


def _explain_no_match(catalog, cond: Conditions, per_cat) -> str:
    city = CITIES[cond.city_id]
    problems = []
    for cat in cond.categories:
        in_city, ok = per_cat[cat]
        name = CATEGORIES[cat].lower()
        if not in_city:
            problems.append(f"в городе {city} в каталоге нет объявлений категории «{name}»")
        elif not ok:
            wanted = [c for c in cond.constraints if ATTR_CATEGORY[c.attribute] == cat]
            attr = ATTR_BY_CATEGORY[cat]
            unknown = sum(1 for x in in_city if attr not in x.attributes)
            text = f"среди {len(in_city)} объявлений категории «{name}» в городе {city} нет подходящих под условие ({'; '.join(_constraint_text(c) for c in wanted)})"
            if unknown:
                text += f"; у {unknown} из них значение не указано в характеристиках, поэтому соответствие подтвердить нельзя"
            problems.append(text)
    if problems:
        tail = "Можно изменить или смягчить условие, выбрать другой город или убрать часть категорий."
        return "Полный комплект не найден: " + "; ".join(problems) + ". " + tail

    cheapest = sum(per_cat[c][1][0].price_rub for c in cond.categories)
    message = (f"Подходящие товары есть, но самый дешёвый полный комплект стоит {rub(cheapest)}, "
               f"а бюджет — {rub(cond.budget_rub)}. Можно увеличить бюджет минимум до {rub(cheapest)}, "
               f"ослабить ограничения или убрать часть категорий.")
    alt = _cities_with_full_set(catalog, cond)
    if alt:
        message += " Комплект в рамках бюджета есть также в другом городе: " + ", ".join(CITIES[c] for c in alt) + "."
    return message


def _cities_with_full_set(catalog, cond: Conditions) -> list:
    found = []
    for city_id in CITIES:
        if city_id == cond.city_id:
            continue
        probe = Conditions(city_id=city_id, budget_rub=cond.budget_rub, categories=cond.categories,
                           constraints=cond.constraints)
        per_cat = candidates_by_category(catalog, probe)
        if all(per_cat[c][1] for c in cond.categories) and \
                sum(per_cat[c][1][0].price_rub for c in cond.categories) <= cond.budget_rub:
            found.append(city_id)
    return found
