"""Уведомление в Telegram после пересчёта: сводка, что изменилось, ссылка на дашборд.

Запуск: `python -m zakupki.notify --summary out/summary.md --url <ссылка на дашборд>`.
Токен бота и чат — из TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID; не заданы — уведомление пропускается.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from zakupki.config import Config

API = "https://api.telegram.org/bot{token}/sendMessage"


def _sections(text: str) -> tuple[str, list[str], dict[str, list[str]]]:
    """Заголовок-дата, первый абзац сводки и разделы «## …» построчно."""
    date, intro, sections, current = "", [], {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            if line.strip():
                sections[current].append(line.strip())
        elif line.startswith("Дата расчёта:"):
            date = line.split(":", 1)[1].strip()
        elif line.strip() and not line.startswith("#"):
            intro.append(line.strip())
    return date, intro, sections


def _changes(lines: list[str]) -> list[str]:
    """Только ненулевые пункты «Что изменилось»."""
    out = []
    for line in lines:
        if line.startswith("Пояснения:"):  # служебная строка в конце summary.md
            continue
        item = line.lstrip("- ").strip()
        if ":" in item:
            name, value = item.rsplit(":", 1)
            if value.strip() not in ("", "0"):
                out.append(f"• {name.strip()}: {value.strip()}")
        elif item:
            out.append(item)
    return out


def build_message(summary: str, url: str | None, cfg: Config) -> str:
    """Текст уведомления из summary.md."""
    date, intro, sections = _sections(summary)
    shown = ".".join(reversed(date.split("-"))) if date else ""
    parts = [f"План закупок — данные на {shown}" if shown else "План закупок пересчитан", ""]
    parts += intro
    if "Что изменилось" in sections:
        changes = _changes(sections["Что изменилось"])
        parts += ["", "С прошлого запуска:"] + (changes or ["изменений нет — план тот же"])
    if url:
        parts += ["", f"Дашборд: {url}"]
    text = "\n".join(parts).strip()
    if len(text) > cfg.telegram_max_chars:
        text = text[: cfg.telegram_max_chars - 1].rstrip() + "…"
    return text


def send(token: str, chat_id: str, text: str, cfg: Config) -> None:
    """Отправка сообщения через Bot API."""
    body = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode("utf-8")
    req = urllib.request.Request(API.format(token=token), data=body, method="POST")
    with urllib.request.urlopen(req, timeout=cfg.telegram_timeout_s) as resp:
        answer = json.loads(resp.read().decode("utf-8"))
    if not answer.get("ok"):
        raise RuntimeError(f"Telegram не принял сообщение: {answer.get('description', 'без описания')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zakupki.notify", description="Сводка плана в Telegram")
    parser.add_argument("--summary", default="out/summary.md", help="путь к summary.md")
    parser.add_argument("--url", default=None, help="ссылка на дашборд")
    args = parser.parse_args(argv)
    cfg = Config()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram: TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы — уведомление пропущено.")
        return 0
    path = Path(args.summary)
    if not path.exists():
        print(f"Telegram: нет файла {path}", file=sys.stderr)
        return 2
    text = build_message(path.read_text(encoding="utf-8"), args.url, cfg)
    try:
        send(token, chat_id, text, cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Telegram: не отправлено — {exc}", file=sys.stderr)
        return 1
    print("Telegram: сводка отправлена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
