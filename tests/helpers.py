import json
import os
import sys
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from buyer_assistant.catalog import Listing  # noqa: E402

CATALOG_PATH = os.path.join(ROOT, "data", "catalog.json")


def listing(id, category, price, city="taganrog", attrs=None, title="x"):
    return Listing(id, category, title, price, city, attrs or {})


def msgs(*texts):
    out = []
    for i, t in enumerate(texts):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "text": t})
    return out


def validate_schema(resp):
    """Схема ответа из репозитория (если установлен jsonschema)."""
    try:
        import jsonschema
    except ImportError:
        return
    with open(os.path.join(ROOT, "schemas", "response.schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator(schema).validate(resp)
