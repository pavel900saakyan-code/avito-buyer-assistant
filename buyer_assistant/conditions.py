"""Справочники и модель условий покупателя (общие для LLM-разбора и fallback-парсера)."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

# Справочники из data/cities.json и data/categories.json (общие для учебных и проверочных данных).
CITIES = {"taganrog": "Таганрог", "rostov-na-donu": "Ростов-на-Дону", "azov": "Азов"}
CATEGORIES = {"table": "Стол", "chair": "Кресло", "lamp": "Лампа"}

# Характеристика -> категория, к которой она относится.
ATTR_CATEGORY = {"width_cm": "table", "adjustable_height": "chair", "color_temperature_k": "lamp"}
ATTR_LABEL = {"width_cm": "ширина", "adjustable_height": "регулировка высоты", "color_temperature_k": "цветовая температура"}
ATTR_UNIT = {"width_cm": "см", "color_temperature_k": "К"}

# lt/gt — строгие границы, lte/gte — включают границу, eq — точное значение.
OPS = ("lt", "lte", "gt", "gte", "eq")


@dataclass(frozen=True)
class Constraint:
    attribute: str
    op: str
    value: "Decimal | bool"  # bool — для adjustable_height, Decimal — для числовых


@dataclass
class Conditions:
    city_id: "str | None" = None
    city_problem: bool = False          # назван город вне справочника или несколько городов
    budget_rub: "int | None" = None
    categories: list = field(default_factory=list)
    category_problem: bool = False      # названа неподдерживаемая категория
    constraints: list = field(default_factory=list)


def missing_questions(cond: Conditions) -> list:
    """Вопросы по недостающим/неоднозначным основным условиям (порядок: бюджет, город, категории)."""
    cities = ", ".join(CITIES.values())
    cats = ", ".join(n.lower() for n in CATEGORIES.values())
    questions = []
    if cond.budget_rub is None:
        questions.append({"field": "budget_rub", "text": "Какой у вас бюджет на весь комплект (в рублях)?"})
    if cond.city_id is None:
        if cond.city_problem:
            text = f"Не удалось однозначно определить город: в каталоге есть только {cities}. В каком из них вы хотите купить?"
        else:
            text = f"В каком городе вы хотите купить товары? Доступны: {cities}."
        questions.append({"field": "city_id", "text": text})
    if not cond.categories or cond.category_problem:
        if cond.category_problem:
            text = f"Часть названных товаров в каталоге отсутствует. Доступны категории: {cats}. Что из этого вам нужно?"
        else:
            text = f"Какие товары вам нужны? Доступны категории: {cats}."
        questions.append({"field": "categories", "text": text})
    return questions
