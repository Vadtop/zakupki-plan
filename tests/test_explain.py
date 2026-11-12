"""Тесты пояснений: шаблон, подмена HTTP, кэш, белый список полей."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from zakupki import config as C
from zakupki.config import Config
from zakupki.explain import (
    build_row_payload,
    explain,
    fix_decimals,
    load_env,
    template_explanation,
    template_summary,
)
from zakupki.plan import PLAN_COLUMNS, RISK_COLUMNS, PlanResult

ENV = {"LLM_BASE_URL": "http://llm.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m"}


def _row(sku_id, status, **kw) -> dict:
    base = {
        "sku_id": sku_id,
        "category": "space_heaters",
        "supplier_id": "SUP-1",
        "status": status,
        "flags": "",
        "order_qty": 120,
        "order_date": pd.Timestamp("2026-09-01"),
        "stock_now": 40.0,
        "on_order": 300.0,
        "on_order_expected": 120.0,
        "lead_days": 123,
        "cover_days": 60.0,
        "cover_months": 2.0,
        "stockout_date": pd.Timestamp("2026-10-12"),
        "safety_stock": 15.0,
        "gap_months": 0.0,
        "reliability": 0.4,
        "avg_markup_pct_12m": 45.0,
        "level": 10.0,
        "trend": 1.1,
        "short_history": False,
        "need_qty": 120,
        "explanation": "",
    }
    base.update(kw)
    return base


def _result() -> PlanResult:
    rows = [
        _row("SKU-1", C.ORDER_NOW),
        _row("SKU-2", C.SHORTAGE, gap_months=3.5, stockout_date=pd.Timestamp("2026-09-20")),
        _row("SKU-3", C.OK, order_qty=0),
        _row("SKU-4", C.DEAD, order_qty=0, stock_now=7.0),
    ]
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    risks = pd.DataFrame(columns=RISK_COLUMNS)
    cal = pd.DataFrame({"month": pd.period_range("2026-08", periods=2, freq="M"), "positions": 1, "qty": 120})
    return PlanResult(plan=plan, risks=risks, sim=pd.DataFrame(), calendar=cal)


def _fake_post(calls: list):
    """Подмена HTTP: на пачку строк отвечает JSON по sku_id, на сводку — текстом."""

    def post(url, headers, body, timeout):
        calls.append(body)
        text = body["messages"][-1]["content"]
        rows = {}
        # Дефицит без «склад пуст N дн.» не принимается: фейк называет дыру из gap_days строки.
        for chunk in text.split('"sku_id": "')[1:]:
            sid = chunk.split('"', 1)[0]
            gap = re.search(r'"gap_days": (\d+)', chunk)
            rows[sid] = f"Пояснение модели для {sid}" + (f", склад пуст {gap.group(1)} дн." if gap else "")
        if rows:
            content = json.dumps(rows, ensure_ascii=False)
        else:
            content = "Строка сводки 1.\nВ расчёте 4 SKU."
        return {"choices": [{"message": {"content": content}}]}

    return post


def test_template_without_llm(tmp_path):
    """Без модели пояснения шаблонные, режим template, обращений нет."""
    result, summary, stats = explain(_result(), Config(), {}, tmp_path / "c.json", use_llm=False)
    row = result.plan.set_index("sku_id").loc["SKU-1"]
    assert "120" in row["explanation"]
    assert stats["mode"] == "template" and stats["calls"] == 0
    assert summary.strip()


def test_missing_key_falls_back_to_template(tmp_path):
    """Нет ключа в окружении: use_llm=True не ломает расчёт."""
    calls: list = []
    _, _, stats = explain(_result(), Config(), {}, tmp_path / "c.json", True, _post=_fake_post(calls))
    assert stats["mode"] == "template" and calls == []


@pytest.mark.parametrize("answer", ["error", "garbage", "badjson"])
def test_bad_model_keeps_plan_and_template(tmp_path, answer):
    """Ошибка или мусор от модели: шаблон, цифры плана те же."""

    def post(url, headers, body, timeout):
        if answer == "error":
            raise OSError("нет сети")
        if answer == "garbage":
            return {"unexpected": 1}
        return {"choices": [{"message": {"content": "это не json {"}}]}

    src = _result()
    before = src.plan.drop(columns="explanation").copy()
    result, summary, stats = explain(src, Config(), ENV, tmp_path / "c.json", True, _post=post)
    pd.testing.assert_frame_equal(result.plan.drop(columns="explanation"), before)
    assert result.plan.loc[0, "explanation"] == template_explanation(src.plan.iloc[0])
    assert stats["mode"] == ("mixed" if answer == "badjson" else "template")
    assert stats["errors"] >= 1
    assert summary.strip()


def test_model_text_used_and_cached(tmp_path):
    """Ответ модели попадает в строки ЗАКАЗАТЬ СЕЙЧАС и РИСК ДЕФИЦИТА; второй прогон — 0 обращений."""
    cache = tmp_path / "llm_cache.json"
    calls: list = []
    result, summary, stats = explain(_result(), Config(), ENV, cache, True, _post=_fake_post(calls))
    plan = result.plan.set_index("sku_id")
    assert plan.loc["SKU-1", "explanation"] == "Пояснение модели для SKU-1"
    assert plan.loc["SKU-2", "explanation"] == "Пояснение модели для SKU-2, склад пуст 106 дн."
    assert "модели" not in plan.loc["SKU-3", "explanation"]
    assert stats["mode"] == "llm" and stats["calls"] == 2 and summary.startswith("Строка сводки 1.")

    calls2: list = []
    result2, summary2, stats2 = explain(_result(), Config(), ENV, cache, True, _post=_fake_post(calls2))
    assert calls2 == [] and stats2["calls"] == 0 and stats2["cached"] >= 2
    assert result2.plan.loc[0, "explanation"] == "Пояснение модели для SKU-1"
    assert summary2 == summary


def _answer_post(rows: dict, summary: str):
    """Подмена HTTP с заданным ответом по строкам и сводке."""

    def post(url, headers, body, timeout):
        text = body["messages"][-1]["content"]
        content = json.dumps(rows, ensure_ascii=False) if '"sku_id"' in text else summary
        return {"choices": [{"message": {"content": content}}]}

    return post


def test_wrong_model_numbers_fall_back_to_template(tmp_path):
    """Чужие числа или отказ от заказа в ответе модели — шаблон, ответ в кэш не попадает."""
    post = _answer_post(
        {"SKU-1": "Заказать 999999 шт. до 01.01.2030.", "SKU-2": "Не заказывать: дефицита нет."},
        "Срочных закупок нет, все заказы можно отменить.",
    )
    src = _result()
    cache = tmp_path / "c.json"
    result, summary, stats = explain(src, Config(), ENV, cache, True, _post=post)
    plan = result.plan.set_index("sku_id")
    assert plan.loc["SKU-1", "explanation"] == template_explanation(src.plan.iloc[0])
    assert plan.loc["SKU-2", "explanation"] == template_explanation(src.plan.iloc[1])
    assert summary == template_summary(src)
    assert stats["mode"] == "template" and stats["errors"] == 3
    assert not cache.exists()


def test_correct_model_answer_accepted_and_bad_cache_rejected(tmp_path):
    """Числа и даты из данных (с пробелом в тысячах) — ответ принят; испорченный кэш — не принят."""
    post = _answer_post(
        {
            "SKU-1": "Заказать 120 шт. до 01.09.2026: плечо 123 дн., надёжность поставщика 40%.",
            "SKU-2": "Запас кончится 20.09.2026, склад пуст 106 дн.; в пути 300 шт.",
        },
        "В расчёте 4 SKU, к заказу 240 шт.",
    )
    cache = tmp_path / "c.json"
    result, summary, stats = explain(_result(), Config(), ENV, cache, True, _post=post)
    plan = result.plan.set_index("sku_id")
    assert plan.loc["SKU-1", "explanation"].startswith("Заказать 120 шт.")
    assert plan.loc["SKU-2", "explanation"].startswith("Запас кончится 20.09.2026")
    assert summary == "В расчёте 4 SKU, к заказу 240 шт."
    assert stats["mode"] == "llm" and stats["errors"] == 0

    data = json.loads(cache.read_text(encoding="utf-8"))
    data = {k: "Заказать 1 200 шт." if v.startswith("Заказать") else v for k, v in data.items()}
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    calls: list = []
    result2, _, stats2 = explain(_result(), Config(), ENV, cache, True, _post=_fake_post(calls))
    assert stats2["errors"] == 1 and len(calls) == 1
    assert result2.plan.loc[0, "explanation"] == "Пояснение модели для SKU-1"


def test_payload_whitelist():
    """В модель уходят только id и числа из белого списка плюс досчитанные кодом факты."""
    row = pd.Series(_row("SKU-1", C.ORDER_NOW, explanation="секрет", avg_markup_pct_12m=45.0))
    payload = build_row_payload(row)
    assert set(payload) <= {
        "sku_id",
        "category",
        "supplier_id",
        "status",
        "flags",
        "order_qty",
        "order_date",
        "stock_now",
        "on_order",
        "on_order_expected",
        "lead_days",
        "cover_months",
        "stockout_date",
        "gap_months",
        "reliability",
        "trend",
        "short_history",
        "on_order_reliability",
        "on_order_next_date",
        "on_order_next_qty",
        "new_order_arrival_date",
        "gap_days",
    }
    assert "explanation" not in payload and "avg_markup_pct_12m" not in payload
    assert payload["sku_id"] == "SKU-1" and payload["order_qty"] == 120
    # Досчитанное: исполнение поставщика транзита 120/300, приход нового заказа 01.09.2026 + 123 дн.
    assert payload["on_order_reliability"] == 0.4
    assert payload["new_order_arrival_date"] == "2027-01-02"
    # Без po даты ближайшего прихода нет, без дыры нет gap_days.
    assert "on_order_next_date" not in payload and "gap_days" not in payload

    shortage = pd.Series(_row("SKU-2", C.SHORTAGE, gap_months=1.546))
    assert build_row_payload(shortage, days_per_month=30.4)["gap_days"] == 47
    no_transit = pd.Series(_row("SKU-3", C.ORDER_NOW, on_order=0.0, on_order_expected=0.0))
    assert "on_order_reliability" not in build_row_payload(no_transit)


def test_template_summary_lines():
    """Сводка шаблоном: 5–7 строк."""
    text = template_summary(_result())
    assert 5 <= len([ln for ln in text.splitlines() if ln.strip()]) <= 7


def test_load_env(tmp_path, monkeypatch):
    """Разбор .env; переменные окружения главнее файла."""
    path = tmp_path / ".env"
    path.write_text(
        "# комментарий\nLLM_MODEL=file-model\nLLM_API_KEY='abc'\n\nLLM_BASE_URL=\n", encoding="utf-8"
    )
    monkeypatch.setenv("LLM_MODEL", "env-model")
    env = load_env(path)
    assert env["LLM_MODEL"] == "env-model"
    assert env["LLM_API_KEY"] == "abc"
    assert load_env(tmp_path / "нет.env") is not None


def test_past_order_date_reads_as_immediately(tmp_path):
    """Дата заказа в прошлом: в пояснении и в данных для модели «сразу», в плане дата прежняя."""
    past = pd.Timestamp("2026-03-08")
    rows = [
        _row("SKU-1", C.ORDER_NOW, order_date=past),
        _row("SKU-2", C.SHORTAGE, order_date=past, gap_months=2.0),
        _row("SKU-3", C.ORDER_NOW, order_date=pd.Timestamp("2026-09-01")),
    ]
    src = _result()
    src.plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    cfg = Config(today=pd.Timestamp("2026-08-13"))
    result, _, _ = explain(src, cfg, {}, tmp_path / "c.json", use_llm=False)
    text = result.plan.set_index("sku_id")["explanation"]
    assert "сразу" in text["SKU-1"] and "08.03.2026" not in text["SKU-1"]
    assert "сразу" in text["SKU-2"]
    assert "до 01.09.2026" in text["SKU-3"]
    assert (result.plan["order_date"] == src.plan["order_date"]).all()

    calls: list = []
    explain(src, cfg, ENV, tmp_path / "c2.json", True, _post=_fake_post(calls))
    sent = calls[0]["messages"][-1]["content"]
    assert "2026-03-08" not in sent


def test_prompts_found_inside_package():
    """Промпты лежат внутри пакета: находятся и после установки wheel, не только из корня репозитория."""
    import zakupki
    from zakupki.explain import PROMPTS_DIR, _read_prompt

    assert PROMPTS_DIR == Path(zakupki.__file__).resolve().parent / "prompts"
    assert "{rows_json}" in _read_prompt("explain.md")
    assert "{plan_json}" in _read_prompt("summary.md")


def test_template_summary_place_today():
    """Сводка называет «разместить сегодня» той же выборкой, что первый экран: срок наступил."""
    res = _result()
    plan = res.plan.copy()
    plan.loc[0, "order_date"] = pd.Timestamp("2026-08-10")
    plan.loc[1, "order_date"] = pd.Timestamp("2026-09-01")
    res.plan = plan
    text = template_summary(res, pd.Timestamp("2026-08-13"))
    q0, q1 = int(plan.loc[0, "order_qty"]), int(plan.loc[1, "order_qty"])
    assert f"Разместить сегодня: 1 SKU, {q0} шт." in text
    assert f"ещё 1 SKU, {q1} шт." in text


TODAY = pd.Timestamp("2026-08-13")


def _po(*orders) -> pd.DataFrame:
    """Заказы в пути в форме supply.open_orders: (sku_id, expected_date, qty_pending)."""
    return pd.DataFrame(
        {
            "sku_id": [o[0] for o in orders],
            "expected_date": pd.to_datetime([o[1] for o in orders]),
            "qty_pending": [o[2] for o in orders],
        }
    )


def _model_accepts(tmp_path, row: dict, text: str, po=None) -> bool:
    """Одна строка, модель отвечает text: True — ответ принят, False — ушёл шаблон."""
    src = _result()
    src.plan = pd.DataFrame([row], columns=PLAN_COLUMNS)
    post = _answer_post({row["sku_id"]: text}, "В расчёте 1 SKU.")
    cfg = Config(today=TODAY)
    result, _, _ = explain(src, cfg, ENV, tmp_path / "c.json", True, _post=post, po=po)
    got = result.plan.loc[0, "explanation"]
    if got == fix_decimals(text):
        return True
    assert got == template_explanation(src.plan.iloc[0], TODAY, cfg.days_per_month)
    return False


def test_transit_arrival_not_new_order_lead(tmp_path):
    """Заказ в пути: плечо нового заказа как срок прихода транзита — отказ; дата из po — принято."""
    row = _row("SKU-1", C.ORDER_NOW, on_order=500.0, on_order_expected=480.0, lead_days=97, reliability=0.96)
    po = _po(("SKU-1", "2026-08-14", 500))
    assert not _model_accepts(tmp_path / "a", row, "В пути 500 шт., придут через 97 дней.", po)
    good = "В пути 500 шт., придут 14.08.2026. Заказать 120 шт. до 01.09.2026."
    assert _model_accepts(tmp_path / "b", row, good, po)


def test_shortage_not_prevented_and_gap_in_days(tmp_path):
    """Дефицит: «избежать дефицита» — отказ; без дыры в днях — отказ; «склад пуст 47 дней» — принято."""
    row = _row("SKU-2", C.SHORTAGE, gap_months=1.546)
    deny = "Склад пуст 47 дней. Заказать 120 шт., чтобы избежать дефицита."
    assert not _model_accepts(tmp_path / "a", row, deny)
    no_gap = "Запас кончится 12.10.2026. Заказать 120 шт. до 01.09.2026."
    assert not _model_accepts(tmp_path / "b", row, no_gap)
    good = "Запас кончится 12.10.2026, склад пуст 47 дней, обычный заказ дыру не закроет. Заказать 120 шт. до 01.09.2026."
    assert _model_accepts(tmp_path / "c", row, good)
    closes = "Склад пуст 47 дней. Новый заказ закроет дыру. Заказать 120 шт. до 01.09.2026."
    assert not _model_accepts(tmp_path / "d", row, closes)
    not_closes = "Склад пуст 47 дней, новый заказ придёт и не закроет её. Заказать 120 шт. до 01.09.2026."
    assert _model_accepts(tmp_path / "e", row, not_closes)


def test_transit_reliability_of_other_supplier_rejected(tmp_path):
    """Надёжность нового поставщика (58%) приписана заказу в пути (96%) — отказ; свой процент — принято."""
    row = _row("SKU-1", C.ORDER_NOW, on_order=1500.0, on_order_expected=1440.0, reliability=0.58)
    assert not _model_accepts(tmp_path / "a", row, "В пути 1500 шт., поставщик ненадёжен (58%).")
    good = "В пути 1500 шт., их поставщик довозит 96%. Новый поставщик довозит 58%."
    assert _model_accepts(tmp_path / "b", row, good)


def test_delay_wording_rejected(tmp_path):
    """Про задержки в данных ничего нет: такой ответ — шаблон; без этих слов тот же ответ принят."""
    row = _row("SKU-1", C.ORDER_NOW)
    delay = "Есть риск, что заказ задержится. Заказать 120 шт. до 01.09.2026."
    assert not _model_accepts(tmp_path / "a", row, delay)
    assert _model_accepts(tmp_path / "b", row, "Заказать 120 шт. до 01.09.2026.")


def test_fix_decimals():
    """Десятичная точка → запятая, даты ДД.ММ.ГГГГ не трогаются."""
    assert fix_decimals("0.03 мес., 4.67") == "0,03 мес., 4,67"
    assert fix_decimals("до 13.08.2026") == "до 13.08.2026"
    assert fix_decimals("дыра 0.5 мес. с 13.08.2026.") == "дыра 0,5 мес. с 13.08.2026."


def test_model_decimal_dot_fixed_in_answer(tmp_path):
    """Ответ модели с точкой в дроби принимается уже с запятой."""
    row = _row("SKU-1", C.ORDER_NOW, cover_months=0.03)
    src = _result()
    src.plan = pd.DataFrame([row], columns=PLAN_COLUMNS)
    post = _answer_post({"SKU-1": "Запаса на 0.03 мес. Заказать 120 шт. до 01.09.2026."}, "В расчёте 1 SKU.")
    result, _, _ = explain(src, Config(today=TODAY), ENV, tmp_path / "c.json", True, _post=post)
    assert result.plan.loc[0, "explanation"] == "Запаса на 0,03 мес. Заказать 120 шт. до 01.09.2026."


def test_payload_next_arrival_from_po():
    """Ближайший приход — по минимальной expected_date своего SKU; прошедшая дата приводится к today."""
    row = pd.Series(_row("SKU-1", C.ORDER_NOW))
    po = _po(("SKU-1", "2026-10-01", 200), ("SKU-1", "2026-09-15", 100), ("SKU-9", "2026-08-20", 7))
    payload = build_row_payload(row, TODAY, po)
    assert payload["on_order_next_date"] == "2026-09-15" and payload["on_order_next_qty"] == 100

    past = _po(("SKU-1", "2026-07-01", 50), ("SKU-1", "2026-07-20", 30), ("SKU-1", "2026-09-15", 100))
    payload = build_row_payload(row, TODAY, past)
    assert payload["on_order_next_date"] == "2026-08-13" and payload["on_order_next_qty"] == 80


def test_template_transit_percent_from_on_order():
    """Шаблон: процент транзита — on_order_expected / on_order, а не надёжность нового поставщика."""
    row = pd.Series(_row("SKU-1", C.ORDER_NOW, on_order=300.0, on_order_expected=240.0, reliability=0.55))
    text = template_explanation(row)
    assert "исполнение их поставщика (80%)" in text
    assert "55%" not in text

    gap = pd.Series(_row("SKU-2", C.SHORTAGE, gap_months=1.546))
    text = template_explanation(gap, days_per_month=30.4)
    assert "склад пуст ≈47 дн." in text and "дыру не закроет" in text


# Кейсы REVIEW-8: payload — по цифрам реальных SKU, проверка — та же, что у ответа модели.
_P_A6E5E8 = {
    "sku_id": "SKU-A6E5E8",
    "status": C.SHORTAGE,
    "order_qty": 229,
    "order_date": "2026-08-13",
    "stock_now": 0,
    "on_order": 500,
    "on_order_expected": 500,
    "lead_days": 97,
    "reliability": 1.0,
    "on_order_reliability": 1.0,
    "on_order_next_date": "2026-08-14",
    "on_order_next_qty": 500,
    "new_order_arrival_date": "2026-11-18",
    "stockout_date": "2026-08-13",
    "gap_days": 1,
}
_P_C2CA37 = {
    "sku_id": "SKU-C2CA37",
    "status": C.SHORTAGE,
    "order_qty": 120,
    "order_date": "2026-08-13",
    "stock_now": 123,
    "on_order": 0,
    "on_order_expected": 0,
    "lead_days": 140,
    "reliability": 1.0,
    "new_order_arrival_date": "2026-12-31",
    "stockout_date": "2026-11-14",
    "gap_days": 47,
}
_P_BD8C22 = {
    "sku_id": "SKU-BD8C22",
    "status": C.SHORTAGE,
    "order_qty": 576,
    "order_date": "2026-08-13",
    "stock_now": 373,
    "on_order": 1500,
    "on_order_expected": 1433.9,
    "lead_days": 174,
    "reliability": 0.58,
    "on_order_reliability": 0.96,
    "on_order_next_date": "2026-08-16",
    "on_order_next_qty": 1500,
    "new_order_arrival_date": "2027-02-03",
    "stockout_date": "2026-11-30",
    "gap_days": 65,
}
_P_E342ED = {
    "sku_id": "SKU-E342ED",
    "status": C.SHORTAGE,
    "order_qty": 7,
    "order_date": "2026-08-13",
    "stock_now": 2,
    "on_order": 0,
    "on_order_expected": 0,
    "lead_days": 88,
    "reliability": 0.74,
    "new_order_arrival_date": "2026-11-09",
    "stockout_date": "2026-09-19",
    "gap_days": 51,
}


@pytest.mark.parametrize(
    "payload, text, ok",
    [
        (
            _P_A6E5E8,
            "Уже заказанные 500 шт. придут 18.11.2026. Склад пуст 1 день. Заказать 229 шт. до 13.08.2026.",
            False,
        ),
        (
            _P_BD8C22,
            "Уже заказано 1500 шт., их поставщик довозит 58%. Склад пуст 65 дней. Заказать 576 шт. до 13.08.2026.",
            False,
        ),
        (
            _P_A6E5E8,
            "В пути 500 шт. Они придут 18.11.2026. Склад пуст 1 день. Заказать 229 шт. до 13.08.2026.",
            False,
        ),
        (
            _P_C2CA37,
            "Склад пуст 47 дней. Заказ на 120 шт. устранит дефицит. Заказать 120 шт. до 13.08.2026.",
            False,
        ),
        (_P_C2CA37, "Дыра 47 дней. Новый заказ полностью закроет её. Заказать 120 шт. до 13.08.2026.", False),
        (
            _P_BD8C22,
            "В пути 1500 шт. Их поставщик довозит 58%. Склад пуст 65 дней. Заказать 576 шт. до 13.08.2026.",
            False,
        ),
        (_P_C2CA37, "Склад пуст 47 дней. Заказать 140 шт. до 13.08.2026.", False),
        (_P_C2CA37, "Склад пуст с 13.08.2026, дыра 47 дней. Заказать 120 шт. до 13.08.2026.", False),
        (
            _P_C2CA37,
            "Склад пуст 47 дней. Обычный заказ не предотвратит дефицит. Заказать 120 шт. до 13.08.2026.",
            True,
        ),
        (
            _P_A6E5E8,
            (
                "Склад пуст 1 день. В пути 500 шт. с приходом 14.08.2026, а новый заказ придёт 18.11.2026. "
                "Заказать 229 шт. до 13.08.2026."
            ),
            True,
        ),
        (_P_E342ED, "Склад пуст 51 день. Данных о задержках нет. Заказать 7 шт. до 13.08.2026.", True),
        (
            _P_BD8C22,
            (
                "Запас кончится 30.11.2026, заказ в пути придёт 16.08.2026, их поставщик довозит 96%. "
                "Склад пуст 65 дней, дыру надо закрывать отдельно; поставщик нового заказа довозит 58%. "
                "Заказать 576 шт. до 13.08.2026."
            ),
            True,
        ),
    ],
)
def test_review8_cases(payload, text, ok):
    """Роль числа — по соседним словам: чей приход, чей процент, сколько заказать; отрицание понимается."""
    from zakupki.explain import _text_ok

    assert _text_ok(text, payload, True) is ok
