"""Уведомление в Telegram: HTML-сообщение из summary.md, plan.csv и risks.csv, кнопка, пропуск без токена."""

import io
import json
import urllib.parse
import urllib.request

import pandas as pd

from zakupki import notify
from zakupki.config import Config

SUMMARY = """# Сводка плана закупки

Дата расчёта: 2026-08-13

На 13.08.2026 разместить заказ на 187 позиций, 14 087 шт.
Риск дефицита — 171 позиция.

## Итог

- SKU в расчёте: 791

## Риски

- РИСК ДЕФИЦИТА: 2
- ПРОСРОЧЕН ЗАКАЗ: 3

## Что изменилось

- Новых к заказу: 13
- Изменилось количество: 0
- Пришли в риск: 2

Пояснения: модель
"""


def _plan() -> pd.DataFrame:
    rows = [
        # срок наступил — в «сегодня»
        ("S1", "SUP-A", "ЗАКАЗАТЬ СЕЙЧАС", 1000, "2026-08-13"),
        ("S2", "SUP-A", "РИСК ДЕФИЦИТА", 500, "2026-08-01"),
        ("S3", "SUP-B", "ЗАКАЗАТЬ СЕЙЧАС", 1200, "2026-08-13"),
        ("S4", "SUP-C", "РИСК ДЕФИЦИТА", 100, "2026-08-10"),
        ("S5", "SUP-D", "ЗАКАЗАТЬ СЕЙЧАС", 50, "2026-08-13"),
        # не попадают: срок позже, другой статус, нулевое количество
        ("S6", "SUP-E", "ЗАКАЗАТЬ СЕЙЧАС", 9000, "2026-09-01"),
        ("S7", "SUP-E", "ЗАПЛАНИРОВАТЬ", 9000, "2026-08-13"),
        ("S8", "SUP-E", "РИСК ДЕФИЦИТА", 0, "2026-08-13"),
    ]
    df = pd.DataFrame(rows, columns=["sku_id", "supplier_id", "status", "order_qty", "order_date"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def _risks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_type": ["РИСК ДЕФИЦИТА"] * 4 + ["ПРОСРОЧЕН ЗАКАЗ"] * 3 + ["ИЗБЫТОК"],
            "po_id": [None] * 4 + ["PO-1", "PO-1", "PO-2"] + [None],
        }
    )


def test_full_message_today_suppliers_risks_changes():
    text = notify.build_message(SUMMARY, Config(), _plan(), _risks())
    assert text.startswith("<b>📦 План закупок на 13.08.2026</b>")
    assert "5 поз. · 2 850 шт. · 4 поставщ." in text
    a, b, c = text.index("SUP-A"), text.index("SUP-B"), text.index("SUP-C")
    assert a < b < c  # по убыванию штук: 1500, 1200, 100
    assert "1. <code>SUP-A</code> — 1 500 шт. (2 поз.)" in text
    assert "SUP-D" not in text  # только топ-3
    assert "SUP-E" not in text
    assert "• Риск дефицита: 4 поз." in text
    assert "• Просрочено заказов поставщикам: 2" in text  # заказов, не строк
    assert "• Избыток: 1 поз." in text
    assert "Мёртвый" not in text  # нулевые риски не шлём
    assert "• Новых к заказу: 13" in text
    assert "Изменилось количество" not in text
    assert "Пояснения" not in text and "SKU в расчёте" not in text


def test_without_csv_falls_back_to_summary():
    text = notify.build_message(SUMMARY, Config())
    assert "187 позиций, 14 087 шт." in text
    assert "• Риск дефицита: 2 поз." in text
    assert "• Просрочено заказов поставщикам: 3" in text


def test_nothing_today():
    plan = _plan()
    plan["order_date"] = pd.Timestamp("2026-12-01")
    assert "заказов нет" in notify.build_message(SUMMARY, Config(), plan, _risks())


def test_no_changes_and_first_run():
    same = SUMMARY.replace("- Новых к заказу: 13", "- Новых к заказу: 0").replace(
        "- Пришли в риск: 2", "- Пришли в риск: 0"
    )
    assert "изменений нет — план тот же" in notify.build_message(same, Config())
    first = SUMMARY.split("## Что изменилось")[0] + "## Что изменилось\n\nПрошлого запуска нет.\n"
    assert "Прошлого запуска нет." in notify.build_message(first, Config())


def test_html_is_escaped():
    text = notify.build_message(SUMMARY.replace("Новых к заказу", "Новых <к> & заказу"), Config())
    assert "Новых &lt;к&gt; &amp; заказу: 13" in text


def test_cut_drops_whole_blocks():
    text = notify.build_message(SUMMARY, Config(telegram_max_chars=300), _plan(), _risks())
    assert len(text) <= 300
    assert text.count("<b>") == text.count("</b>")
    assert "Разместить сегодня" in text and "С прошлого запуска" not in text


def test_no_token_skips_without_network(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.main(["--summary", str(tmp_path / "nope.md")]) == 0
    assert "пропущено" in capsys.readouterr().out


def test_send_posts_html_and_button(monkeypatch, tmp_path):
    (tmp_path / "summary.md").write_text(SUMMARY, encoding="utf-8")
    _plan().to_csv(tmp_path / "plan.csv", index=False)
    _risks().to_csv(tmp_path / "risks.csv", index=False)
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["body"] = req.full_url, urllib.parse.parse_qs(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    args = ["--summary", str(tmp_path / "summary.md"), "--plan", str(tmp_path / "plan.csv")]
    args += ["--risks", str(tmp_path / "risks.csv"), "--url", "https://example.org/"]
    assert notify.main(args) == 0
    body = seen["body"]
    assert seen["url"].endswith("/botT/sendMessage")
    assert body["chat_id"] == ["42"] and body["parse_mode"] == ["HTML"]
    assert "2 850 шт." in body["text"][0]
    button = json.loads(body["reply_markup"][0])["inline_keyboard"][0][0]
    assert button == {"text": "Открыть дашборд", "url": "https://example.org/"}


def test_send_reports_telegram_error(monkeypatch, tmp_path, capsys):
    (tmp_path / "summary.md").write_text(SUMMARY, encoding="utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout: io.BytesIO(json.dumps({"ok": False, "description": "bad"}).encode("utf-8")),
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert notify.main(["--summary", str(tmp_path / "summary.md"), "--plan", str(tmp_path / "x.csv")]) == 1
    assert "bad" in capsys.readouterr().err
