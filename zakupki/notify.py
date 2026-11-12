"""Уведомление в Telegram после пересчёта: что заказать сегодня, у кого, риски, кнопка на дашборд.

Запуск: `python -m zakupki.notify --summary out/summary.md --plan out/plan.csv --risks out/risks.csv --url <дашборд>`.
Токен бота и чат — из TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID; не заданы — уведомление пропускается.
Нет plan.csv / risks.csv — сообщение собирается только из summary.md.
"""

import argparse
import html
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from zakupki import config as C
from zakupki.config import Config
from zakupki.explain import _n, _place_today

API = "https://api.telegram.org/bot{token}/sendMessage"
TOP_SUPPLIERS = 3


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
                out.append(f"• {html.escape(name.strip())}: {html.escape(value.strip())}")
        elif item:
            out.append(html.escape(item))
    return out


def _risk_counts(risks: pd.DataFrame | None, sections: dict[str, list[str]]) -> dict[str, int]:
    """Риски по типам; просрочка — по заказам (po_id), как на дашборде. Без risks.csv — из summary.md."""
    if risks is not None:
        counts = {str(k): int(v) for k, v in risks["risk_type"].value_counts().items()}
        late = risks[risks["risk_type"] == C.OVERDUE_PO]
        if len(late) and "po_id" in late:
            counts[C.OVERDUE_PO] = int(late["po_id"].nunique())
        return counts
    counts = {}
    for line in sections.get("Риски", []):
        name, _, value = line.lstrip("- ").rpartition(":")
        if value.strip().isdigit():
            counts[name.strip()] = int(value)
    return counts


def _today_block(plan: pd.DataFrame, date: str) -> list[str]:
    """«Разместить сегодня» и крупнейшие поставщики — та же выборка, что в сводке и на дашборде."""
    rows, _ = _place_today(plan, date or None)
    if not len(rows):
        return ["<b>Разместить сегодня</b>", "заказов нет — срок ни по одной позиции не наступил"]
    lines = [
        "<b>Разместить сегодня</b>",
        f"{len(rows)} поз. · {_n(rows['order_qty'].sum())} шт. · {rows['supplier_id'].nunique()} поставщ.",
        "",
        "<b>Крупнейшие заказы по поставщикам</b>",
    ]
    by_sup = (
        rows.groupby("supplier_id")["order_qty"]
        .agg(["sum", "count"])
        .sort_values(["sum", "count"], ascending=False, kind="mergesort")
        .head(TOP_SUPPLIERS)
    )
    for i, (sup, r) in enumerate(by_sup.iterrows(), 1):
        lines.append(
            f"{i}. <code>{html.escape(str(sup))}</code> — {_n(r['sum'])} шт. ({int(r['count'])} поз.)"
        )
    return lines


def _risk_block(counts: dict[str, int]) -> list[str]:
    items = [
        (C.SHORTAGE, "Риск дефицита", "поз."),
        (C.OVERDUE_PO, "Просрочено заказов поставщикам", ""),
        (C.EXCESS, "Избыток", "поз."),
        (C.DEAD, "Мёртвый ассортимент на складе", "поз."),
    ]
    lines = [
        f"• {label}: {counts[key]}{' ' + unit if unit else ''}"
        for key, label, unit in items
        if counts.get(key)
    ]
    return ["<b>Требует внимания</b>", *lines] if lines else []


def build_message(
    summary: str,
    cfg: Config,
    plan: pd.DataFrame | None = None,
    risks: pd.DataFrame | None = None,
) -> str:
    """HTML-текст уведомления (parse_mode=HTML)."""
    date, intro, sections = _sections(summary)
    shown = ".".join(reversed(date.split("-"))) if date else ""
    blocks = [[f"<b>📦 План закупок{' на ' + shown if shown else ''}</b>"]]
    if plan is not None:
        blocks.append(_today_block(plan, date))
    elif intro:
        blocks.append([html.escape(intro[0])])
    risk = _risk_block(_risk_counts(risks, sections))
    if risk:
        blocks.append(risk)
    if "Что изменилось" in sections:
        changes = _changes(sections["Что изменилось"])
        blocks.append(["<b>С прошлого запуска</b>", *(changes or ["изменений нет — план тот же"])])

    # Режем целыми блоками с конца, чтобы не порвать HTML-теги.
    while len(blocks) > 1 and len(_join(blocks)) > cfg.telegram_max_chars:
        blocks.pop()
    return _join(blocks)


def _join(blocks: list[list[str]]) -> str:
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def send(token: str, chat_id: str, text: str, url: str | None, cfg: Config) -> None:
    """Отправка сообщения через Bot API; ссылка — кнопкой под сообщением."""
    fields = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}
    if url:
        keyboard = {"inline_keyboard": [[{"text": "Открыть дашборд", "url": url}]]}
        fields["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    body = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(API.format(token=token), data=body, method="POST")
    with urllib.request.urlopen(req, timeout=cfg.telegram_timeout_s) as resp:
        answer = json.loads(resp.read().decode("utf-8"))
    if not answer.get("ok"):
        raise RuntimeError(f"Telegram не принял сообщение: {answer.get('description', 'без описания')}")


def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zakupki.notify", description="Сводка плана в Telegram")
    parser.add_argument("--summary", default="out/summary.md", help="путь к summary.md")
    parser.add_argument("--plan", default="out/plan.csv", help="путь к plan.csv")
    parser.add_argument("--risks", default="out/risks.csv", help="путь к risks.csv")
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
    plan = _read_csv(Path(args.plan), parse_dates=["order_date"])
    risks = _read_csv(Path(args.risks))
    text = build_message(path.read_text(encoding="utf-8"), cfg, plan, risks)
    try:
        send(token, chat_id, text, args.url or None, cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Telegram: не отправлено — {exc}", file=sys.stderr)
        return 1
    print("Telegram: сводка отправлена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
