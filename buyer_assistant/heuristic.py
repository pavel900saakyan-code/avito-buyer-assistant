"""Резервный разбор диалога правилами — только если модель недоступна или ответила мусором.
Покрывает явные формулировки из контракта; по качеству уступает модели."""
from __future__ import annotations

import re
from decimal import Decimal

from .conditions import CITIES, Conditions, Constraint

NUM = r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_B = r"(?<![а-яёa-z])"            # граница слова слева
_E = r"(?![а-яёa-z])"             # граница слова справа

_OP_WORDS = (r"(?:ширин\w*\s*)?(?:(?:не\s+)?(?:шире|уже|более|менее|больше|меньше|выше|ниже)"
             r"|не\s+(?:более|менее|больше|меньше)|максимум|минимум|от|до)?")
_RANGE = re.compile(_B + rf"от\s*(?P<a>{NUM})\s*(?:[а-яёa-z]+\s*)?до\s*(?P<b>{NUM})\s*(?P<unit>см|cm|мм|м|метр\w*|к|k|кельвин\w*){_E}")
_LENGTH = re.compile(_B + rf"(?P<op>{_OP_WORDS})\s*(?P<half>пол)?\s*(?P<num>{NUM})?\s*(?P<unit>см|cm|сантиметр\w*|мм|м|метр\w*){_E}")
_TEMP = re.compile(_B + rf"(?P<op>{_OP_WORDS})\s*(?P<num>{NUM})\s*(?P<unit>к|k|кельвин\w*){_E}")

_OP_TABLE = (("не шире", "lte"), ("не выше", "lte"), ("не более", "lte"), ("не больше", "lte"), ("максимум", "lte"),
             ("до", "lte"), ("не уже", "gte"), ("не менее", "gte"), ("не меньше", "gte"), ("минимум", "gte"),
             ("от", "gte"), ("уже", "lt"), ("меньше", "lt"), ("менее", "lt"), ("ниже", "lt"),
             ("шире", "gt"), ("больше", "gt"), ("более", "gt"), ("выше", "gt"))

CITY_STEMS = {"taganrog": r"(?:таганрог|таганрож)", "rostov-na-donu": r"(?:ростов|рнд)", "azov": r"азов"}
_CAT_STEMS = {"table": r"стол", "chair": r"кресл", "lamp": r"(?:ламп|светильник|торшер)"}


def _dec(text: str) -> Decimal:
    return Decimal(re.sub(r"[  ]", "", text).replace(",", "."))


def _op(word: str) -> str:
    w = re.sub(r"^ширин\w*\s*", "", " ".join((word or "").split()))
    for key, op in _OP_TABLE:
        if w.endswith(key):
            return op
    return "eq"


def _mask(text: str, span) -> str:
    return text[:span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _unit_factor_cm(unit: str) -> Decimal:
    if unit.startswith("мм"):
        return Decimal("0.1")
    if unit in ("см", "cm") or unit.startswith("сантиметр"):
        return Decimal(1)
    return Decimal(100)                       # м, метр...


def _extract_constraints(text: str):
    """Возвращает (ограничения, текст с «замаскированными» найденными фрагментами)."""
    found = []
    for m in list(_RANGE.finditer(text)):
        unit = m["unit"]
        if unit in ("к", "k") or unit.startswith("кельвин"):
            if _dec(m["a"]) >= 1000:
                found += [Constraint("color_temperature_k", "gte", _dec(m["a"])),
                          Constraint("color_temperature_k", "lte", _dec(m["b"]))]
                text = _mask(text, m.span())
        else:
            k = _unit_factor_cm(unit)
            found += [Constraint("width_cm", "gte", _dec(m["a"]) * k), Constraint("width_cm", "lte", _dec(m["b"]) * k)]
            text = _mask(text, m.span())
    for m in list(_TEMP.finditer(text)):
        value = _dec(m["num"])
        if value >= 1000:                      # «15к» — это тысячи рублей, а не кельвины
            found.append(Constraint("color_temperature_k", _op(m["op"]), value))
            text = _mask(text, m.span())
    for m in list(_LENGTH.finditer(text)):
        unit = m["unit"]
        if m["num"]:
            value = _dec(m["num"]) * _unit_factor_cm(unit)
        elif unit.startswith("метр"):
            value = Decimal(50) if m["half"] else Decimal(100)
        else:
            continue
        found.append(Constraint("width_cm", _op(m["op"]), value))
        text = _mask(text, m.span())
    if re.search(r"без\s+регулир|не\s+регулир|нерегулир", text):
        found.append(Constraint("adjustable_height", "eq", False))
    elif re.search(r"регулир", text):
        found.append(Constraint("adjustable_height", "eq", True))
    return found, text


def _extract_budget(text: str, bare_message: bool):
    value = None
    for m in re.finditer(NUM, text):
        number = _dec(m.group())
        after = text[m.end():m.end() + 12]
        before = text[max(0, m.start() - 20):m.start()]
        mult = None
        if re.match(rf"\s*(?:тыс|т\.?\s?р{_E}|[кk]{_E})", after):
            mult = 1000
        elif re.match(r"\s*(?:млн|миллион)", after):
            mult = 1_000_000
        elif re.match(r"\s*(?:₽|руб|р\.?" + _E + ")", after) or \
                re.search(r"(?:бюджет\w*|до|за\s+все?|не\s+более|не\s+дороже|максимум|в\s+пределах)\W*$", before):
            mult = 1
        elif bare_message:
            mult = 1
        if mult:
            value = int(number * mult)
    return value


def parse(messages: list) -> Conditions:
    cond = Conditions()
    cats_seen = {}
    for pos, message in enumerate(m for m in messages if m["role"] == "user"):
        text = " ".join(message["text"].lower().replace("ё", "е").split())
        constraints, rest = _extract_constraints(text)

        cities = [cid for cid, stem in CITY_STEMS.items() if re.search(_B + stem, rest)]
        if len(cities) == 1:
            cond.city_id, cond.city_problem = cities[0], False
        elif len(cities) > 1:
            cond.city_id, cond.city_problem = None, True

        bare = bool(re.fullmatch(rf"\s*(?:{NUM})\s*", rest))
        budget = _extract_budget(rest, bare)
        if budget is not None:
            cond.budget_rub = budget

        for cat, stem in _CAT_STEMS.items():
            m = re.search(_B + stem, rest)
            if m and cat not in cats_seen:
                cats_seen[cat] = (pos, m.start())

        for attr in {c.attribute for c in constraints}:           # новое условие заменяет прежнее
            cond.constraints = [c for c in cond.constraints if c.attribute != attr]
        cond.constraints += constraints
    cond.categories = [c for c, _ in sorted(cats_seen.items(), key=lambda kv: kv[1])]
    return cond
