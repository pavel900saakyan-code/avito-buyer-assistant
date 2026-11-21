"""Детерминированный подбор: фильтры по городу/характеристикам, перебор комплектов, выбор до 3 разных."""
from __future__ import annotations

import math
from decimal import Decimal

from .conditions import ATTR_CATEGORY, Conditions

_MISSING = object()


def _as_number(value):
    """Число из каталога как Decimal. bool (в Python это int!) и нечисла — не числа."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return Decimal(str(value))


def _compare(op, actual, wanted):
    if op == "lt":
        return actual < wanted
    if op == "lte":
        return actual <= wanted
    if op == "gt":
        return actual > wanted
    if op == "gte":
        return actual >= wanted
    return actual == wanted


def satisfies(listing, constraints) -> bool:
    """Подтверждено ли соответствие структурированными полями. Неизвестное значение — не подходит."""
    for c in constraints:
        if ATTR_CATEGORY[c.attribute] != listing.category:
            continue
        raw = listing.attributes.get(c.attribute, _MISSING)
        if raw is _MISSING:
            return False
        if c.attribute == "adjustable_height":
            if not isinstance(raw, bool) or raw != c.value:   # False/0 не путаем с «нет значения»
                return False
        else:
            number = _as_number(raw)
            if number is None or not _compare(c.op, number, c.value):
                return False
    return True


def candidates_by_category(catalog, cond: Conditions):
    """Для каждой запрошенной категории: (все в городе, подходящие по ограничениям, отсортированы по цене)."""
    result = {}
    for cat in cond.categories:
        in_city = [x for x in catalog if x.category == cat and x.city_id == cond.city_id]
        ok = sorted((x for x in in_city if satisfies(x, cond.constraints)), key=lambda x: (x.price_rub, x.id))
        result[cat] = (in_city, ok)
    return result


def enumerate_sets(lists, budget):
    """Все комплекты (по одному из каждого списка) с суммой <= бюджета. lists отсортированы по цене."""
    if not lists or any(not l for l in lists):
        return []
    n = len(lists)
    tail_min = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        tail_min[i] = tail_min[i + 1] + lists[i][0].price_rub
    out = []

    def rec(i, total, ids):
        if i == n:
            out.append((total, tuple(ids)))
            return
        for item in lists[i]:
            t = total + item.price_rub
            if t + tail_min[i + 1] > budget:
                break                       # дальше только дороже
            ids.append(item.id)
            rec(i + 1, t, ids)
            ids.pop()

    rec(0, 0, [])
    return out


def pick_options(sets, limit=3):
    """Самый дешёвый комплект + максимально непохожие на уже выбранные (при равенстве — дешевле)."""
    if not sets:
        return []
    chosen = [min(sets)]
    pool = [s for s in sets if s != chosen[0]]
    while pool and len(chosen) < limit:
        def key(s):
            dist = min(sum(a != b for a, b in zip(s[1], c[1])) for c in chosen)
            return (-dist, s[0], s[1])
        best = min(pool, key=key)
        chosen.append(best)
        pool.remove(best)
    return chosen
