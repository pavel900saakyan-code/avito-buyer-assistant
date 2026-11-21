#!/usr/bin/env python3
"""Консольный чат для демонстрации: python chat.py [--catalog data/catalog.json]
Использует тот же конвейер, что и ./run; историю диалога хранит между репликами."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buyer_assistant.catalog import load_catalog  # noqa: E402
from buyer_assistant.config import load_settings  # noqa: E402
from buyer_assistant.llm import LLMError  # noqa: E402
from buyer_assistant.pipeline import process  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.json"))
    parser.add_argument("--verbose", action="store_true", help="показывать извлечённые моделью условия")
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    by_id = {x.id: x for x in catalog}
    settings = load_settings()
    log = (lambda *p: print("  [лог]", *p, file=sys.stderr)) if args.verbose else (lambda *p: None)
    history = []
    print("ИИ-помощник покупателя. Диалог запоминается; /new — начать заново; пустая строка или Ctrl+C — выход.\n")
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        if text == "/new":
            history.clear()
            print("— новый диалог —")
            continue
        history.append({"role": "user", "text": text})
        try:
            response = process(history, catalog, settings, time.monotonic() + 100, log)
        except LLMError as exc:
            print(f"Помощник: сервис модели недоступен ({exc})")
            history.pop()
            continue
        if response["status"] == "clarify":
            reply = " ".join(q["text"] for q in response["questions"])
            print("Помощник:", reply)
        elif response["status"] == "no_match":
            reply = response["message"]
            print("Помощник:", reply)
        else:
            print("Помощник: нашёл подходящие комплекты.")
            lines = []
            for i, opt in enumerate(response["options"], 1):
                print(f"\n  Вариант {i} — {opt['total_rub']:,} ₽".replace(",", " "))
                for lid in opt["listing_ids"]:
                    x = by_id[lid]
                    print(f"    • [{lid}] {x.title} — {x.price_rub:,} ₽".replace(",", " "))
                print("    " + opt["explanation"])
                lines.append(f"Вариант {i}: {', '.join(opt['listing_ids'])}, {opt['total_rub']} ₽")
            reply = " ".join(lines)
            print()
        history.append({"role": "assistant", "text": reply})


if __name__ == "__main__":
    main()
