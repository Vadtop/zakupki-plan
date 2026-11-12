"""Уведомление в Telegram: текст из summary.md и пропуск без токена."""

import io
import json
import urllib.request

from zakupki import notify
from zakupki.config import Config

SUMMARY = """# Сводка плана закупки

Дата расчёта: 2026-08-13

На 13.08.2026 разместить заказ на 187 позиций, 14 087 шт.
Риск дефицита — 171 позиция.

## Итог

- SKU в расчёте: 791

## Что изменилось

- Новых к заказу: 13
- Изменилось количество: 0
- Пришли в риск: 2

Пояснения: модель
"""


def test_message_has_date_intro_changes_and_link():
    text = notify.build_message(SUMMARY, "https://example.org/plan/", Config())
    assert text.startswith("План закупок — данные на 13.08.2026")
    assert "187 позиций, 14 087 шт." in text
    assert "• Новых к заказу: 13" in text
    assert "• Пришли в риск: 2" in text
    assert "Изменилось количество" not in text  # нулевые пункты не шлём
    assert "Пояснения" not in text
    assert "SKU в расчёте" not in text  # остальные разделы — только на дашборде
    assert text.endswith("Дашборд: https://example.org/plan/")


def test_message_without_changes_and_first_run():
    same = SUMMARY.replace("- Новых к заказу: 13", "- Новых к заказу: 0").replace(
        "- Пришли в риск: 2", "- Пришли в риск: 0"
    )
    assert "изменений нет — план тот же" in notify.build_message(same, None, Config())
    first = SUMMARY.split("## Что изменилось")[0] + "## Что изменилось\n\nПрошлого запуска нет.\n"
    assert "Прошлого запуска нет." in notify.build_message(first, None, Config())


def test_message_is_cut_to_limit():
    long = SUMMARY.replace("Риск дефицита", "очень длинный текст " * 500)
    text = notify.build_message(long, None, Config(telegram_max_chars=200))
    assert len(text) <= 200 and text.endswith("…")


def test_no_token_skips_without_network(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.main(["--summary", str(tmp_path / "nope.md")]) == 0
    assert "пропущено" in capsys.readouterr().out


def test_send_posts_chat_and_text(monkeypatch, tmp_path):
    path = tmp_path / "summary.md"
    path.write_text(SUMMARY, encoding="utf-8")
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["body"] = req.full_url, req.data.decode("utf-8")
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert notify.main(["--summary", str(path), "--url", "https://example.org/"]) == 0
    assert seen["url"].endswith("/botT/sendMessage")
    assert "chat_id=42" in seen["body"] and "14+087" in seen["body"]


def test_telegram_error_returns_1(monkeypatch, tmp_path):
    path = tmp_path / "summary.md"
    path.write_text(SUMMARY, encoding="utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout: io.BytesIO(json.dumps({"ok": False, "description": "chat not found"}).encode()),
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert notify.main(["--summary", str(path)]) == 1
