"""Тесты дашборда: один файл, JSON-блок валиден, блоки на месте, один внешний скрипт."""

import json
import math
import re
import shutil
import subprocess

import pandas as pd
import pytest

from zakupki import config as C
from zakupki.config import Config
from zakupki.dashboard import TEMPLATE, build_data, render_dashboard
from zakupki.demand import build_demand
from zakupki.explain import explain
from zakupki.forecast import forecast, seasonal_index
from zakupki.history import diff_plans
from zakupki.plan import build_plan
from zakupki.supply import open_orders, supplier_reliability

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
BLOCKS = [
    "Сводка плана", "Что заказать", "Календарь заказов", "Риски", "Поставщики", "Низкая наценка",
    "Мёртвый ассортимент", "Сезонность категорий", "Что изменилось с прошлого запуска", 'id="btn-csv"',
    'id="f-status"', 'id="f-cat"', 'id="f-search"', "Ожидается", "Заказы поставщикам", "Как считается",
    "Почему столько", 'href="#sec-po"', 'href="#sec-how"',
]


def _render(small_dataset, tmp_path, prev=None):
    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    result, summary, _ = explain(result, cfg, {}, tmp_path / "c.json", use_llm=False)
    rel = supplier_reliability(ds)
    path = tmp_path / "dashboard.html"
    render_dashboard(
        result, index, demand, fc, rel, open_orders(ds, rel, cfg), diff_plans(prev, result.plan), summary,
        ds, path,
    )
    return path.read_text(encoding="utf-8"), result


def _data(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.DOTALL)
    assert match
    return json.loads(match.group(1))


def test_dashboard_structure(small_dataset, tmp_path):
    html, result = _render(small_dataset, tmp_path)
    for block in BLOCKS:
        assert block in html, block
    assert re.findall(r'<script[^>]+src="([^"]+)"', html) == [PLOTLY]
    data = _data(html)
    assert len(data["plan"]) == len(result.plan)
    assert data["diff"]["first_run"] is True
    assert set(data["charts"]) == set(result.plan["sku_id"])
    assert data["order_qty_total"] == int(result.plan["order_qty"].sum())
    assert len(data["calendar"]) == 12
    assert "__DATA__" not in html


def test_dashboard_json_is_escaped(small_dataset, tmp_path):
    """Текст с «</script>» в пояснении не ломает страницу."""
    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    result.plan.loc[0, "explanation"] = "</script><b>x</b>"
    rel = supplier_reliability(ds)
    path = tmp_path / "d.html"
    render_dashboard(result, index, demand, fc, rel, open_orders(ds, rel, cfg),
                     diff_plans(None, result.plan), "сводка", ds, path)
    html = path.read_text(encoding="utf-8")
    assert html.count("</script>") == 3
    assert _data(html)["plan"][0]["explanation"] == "</script><b>x</b>"


def test_dashboard_with_previous_snapshot(small_dataset, tmp_path):
    """Со снимком в данных есть списки изменений, а не first_run."""
    _, result = _render(small_dataset, tmp_path)
    expected = int((result.plan["order_qty"] > 0).sum())
    prev = result.plan.copy()
    prev["order_qty"] = 0
    html, _ = _render(small_dataset, tmp_path, prev=prev)
    assert _data(html)["diff"]["first_run"] is False
    assert len(_data(html)["diff"]["new_orders"]) == expected


def test_past_order_date_marked_asap(small_dataset, tmp_path):
    """«Заказать до» в прошлом у ЗАКАЗАТЬ СЕЙЧАС и РИСК ДЕФИЦИТА — признак «сразу», дата прежняя."""
    _, result = _render(small_dataset, tmp_path)
    plan = result.plan.copy()
    plan.loc[0, ["status", "order_date"]] = [C.ORDER_NOW, pd.Timestamp("2025-03-01")]
    plan.loc[1, ["status", "order_date"]] = [C.PLAN, pd.Timestamp("2025-03-01")]
    result.plan = plan
    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    _, fc = forecast(demand, index, cfg, ds.today)
    rel = supplier_reliability(ds)
    data = build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds)
    assert data["plan"][0]["order_asap"] is True
    assert data["plan"][0]["order_date"] == "2025-03-01"
    assert data["plan"][1]["order_asap"] is False


def _template_js() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    return text[text.index("const $ = "): text.index("// Шапка")]


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_computed_cell_text_shown_as_is(tmp_path):
    """Ячейка с готовым текстом (надёжность «96%») не идёт через формат числа."""
    script = tmp_path / "t.js"
    script.write_text(
        "const document = {};\n" + _template_js()
        + "\nconst td = {};\nfillCell(td, {f: (r) => fmtN(r.x * 100) + '%', n: 1}, {x: 0.96});\n"
        + "console.log(td.textContent);\n",
        encoding="utf-8",
    )
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "96%"


def test_chart_stock_not_clipped_at_zero():
    """Остаток на графике рисуется и ниже нуля — так виден дефицит."""
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "Math.max(0, p[1])" not in text


def rebuild_qty(step: dict) -> int:
    """Итог расшифровки «почему столько»: сумма шагов, вверх до целого и до партии."""
    need = step["forecast"] + step["safety"] - step["stock"] - step["transit"]
    need = max(need, step["bridge"])
    if need <= 0:
        return 0
    qty = math.ceil(round(need, 6))
    lot = step["lot"]
    return math.ceil(qty / lot) * lot if lot > 1 else qty


def check_steps(data: dict) -> int:
    """Шаги расшифровки дают order_qty у каждого SKU с заказом; возвращает число проверенных."""
    checked = 0
    for row in data["plan"]:
        step = data["steps"].get(row["sku_id"])
        if row["order_qty"] > 0:
            assert step is not None, row["sku_id"]
        if step is None:
            continue
        assert rebuild_qty(step) == row["need_qty"], row["sku_id"]
        if row["order_qty"] > 0:
            assert rebuild_qty(step) == row["order_qty"], row["sku_id"]
            assert step["ceil"] <= row["order_qty"] < step["ceil"] + step["lot"]
        checked += 1
    return checked


def test_steps_sum_to_order_qty(small_dataset, tmp_path):
    """Расшифровка в карточке SKU: прогноз + страховой − остаток − в пути → округление = order_qty."""
    html, result = _render(small_dataset, tmp_path)
    data = _data(html)
    assert check_steps(data) > 0
    assert (result.plan["order_qty"] > 0).any()


def test_steps_with_lot_and_transit(small_dataset, tmp_path):
    """Кратность партии и заказы в пути тоже сходятся с order_qty."""
    html, result = _render(small_dataset, tmp_path)
    data = _data(html)
    bd = result.breakdown.set_index("sku_id")
    for sku_id, step in data["steps"].items():
        assert step["lot"] == int(bd.loc[sku_id, "lot"])
        assert step["transit"] == pytest.approx(float(bd.loc[sku_id, "need_transit"]))
        assert step["transit"] <= step["pending"] + 1e-6


def test_supplier_orders_cover_all_orders(small_dataset, tmp_path):
    """Заказы поставщикам: «сейчас» + «позже» = все позиции с заказом, без повторов."""
    html, result = _render(small_dataset, tmp_path)
    data = _data(html)
    groups = data["orders"]["now"] + data["orders"]["later"]
    skus = [s for g in groups for s in g["skus"]]
    ordered = result.plan[result.plan["order_qty"] > 0]
    assert sorted(skus) == sorted(ordered["sku_id"])
    assert sum(g["qty"] for g in groups) == int(ordered["order_qty"].sum())
    by_sku = {r["sku_id"]: r for r in data["plan"]}
    for g in data["orders"]["now"]:
        for s in g["skus"]:
            r = by_sku[s]
            assert r["order_asap"] or r["order_date"] <= data["asof"] or r["status"] == C.SHORTAGE
        assert g["arrive_from"] <= g["arrive_to"]


def test_rules_use_config_values(small_dataset, tmp_path):
    """«Как считается» берёт числа из Config, а не из текста."""
    _, result = _render(small_dataset, tmp_path)
    ds = small_dataset
    cfg = Config(today=ds.today, min_markup_pct=31, z=2.05)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    _, fc = forecast(demand, index, cfg, ds.today)
    rel = supplier_reliability(ds)
    data = build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds, cfg)
    text = " ".join(data["rules"])
    assert "31%" in text and "2,05" in text
    assert 6 <= len(data["rules"]) <= 10
    assert "run_daily.ps1" in data["run_note"]


def test_supplier_orders_keep_rows_without_supplier(small_dataset, tmp_path):
    """Позиция без поставщика в справочнике не теряется: отдельная строка в конце."""
    _, result = _render(small_dataset, tmp_path)
    plan = result.plan.copy()
    idx = plan.index[plan["order_qty"] > 0][0]
    plan.loc[idx, "supplier_id"] = None
    result.plan = plan
    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    _, fc = forecast(demand, index, cfg, ds.today)
    rel = supplier_reliability(ds)
    data = build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds, cfg)
    groups = data["orders"]["now"] + data["orders"]["later"]
    assert sum(g["qty"] for g in groups) == int(plan["order_qty"].sum())
    assert any(g["supplier_id"] is None and plan.loc[idx, "sku_id"] in g["skus"] for g in groups)


def test_order_date_today_is_today(small_dataset, tmp_path):
    """Срок заказа ровно сегодня — в «Сегодня» и «сразу», а не в «скоро»; как в заказах поставщикам."""
    _, result = _render(small_dataset, tmp_path)
    ds = small_dataset
    plan = result.plan.copy()
    plan["order_qty"] = 0
    plan["status"] = C.OK
    plan.loc[0, ["status", "order_date", "order_qty"]] = [C.ORDER_NOW, ds.today, 500]
    result.plan = plan
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    _, fc = forecast(demand, index, cfg, ds.today)
    rel = supplier_reliability(ds)
    data = build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds)
    assert data["plan"][0]["order_asap"] is True
    assert data["today"]["n"] == 1 and data["today"]["qty"] == 500
    assert data["soon"]["n"] == 0
    assert sum(g["qty"] for g in data["orders"]["now"]) == 500


def test_first_screen_blocks_and_thresholds(small_dataset, tmp_path):
    """Первый экран: строка «Разместить сегодня», изменения, плитки, дефицит, поставщики, календарь; пороги в D.cfg."""
    html, _ = _render(small_dataset, tmp_path)
    for block_id in ("hero", "hero-diff", "kpis", "hero-act", "hero-suppliers", "hero-po", "hero-cal", "sec-summary", "summary"):
        assert f'id="{block_id}"' in html, block_id
    assert 'href="#top"' in html
    cfg = _data(html)["cfg"]
    assert cfg["long_gap_months"] == Config().long_gap_months == 3.0
    assert cfg["unreliable_below"] == Config().unreliable_below == 0.8


def test_soon_first_and_last(small_dataset, tmp_path):
    """«Скоро»: ближайший срок first и последний last; first не позже last."""
    _, result = _render(small_dataset, tmp_path)
    ds = small_dataset
    plan = result.plan.copy()
    plan["order_qty"] = 0
    plan["status"] = C.OK
    early, late = ds.today + pd.Timedelta(days=2), ds.today + pd.Timedelta(days=20)
    plan.loc[0, ["status", "order_date", "order_qty"]] = [C.ORDER_NOW, late, 10]
    plan.loc[1, ["status", "order_date", "order_qty"]] = [C.ORDER_NOW, early, 20]
    result.plan = plan
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    _, fc = forecast(demand, index, cfg, ds.today)
    rel = supplier_reliability(ds)
    soon = build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds)["soon"]
    assert soon["n"] == 2 and soon["qty"] == 30
    assert soon["first"] == early.strftime("%Y-%m-%d")
    assert soon["last"] == late.strftime("%Y-%m-%d")
    assert soon["first"] <= soon["last"]


def test_val_keeps_precision_for_single_rounding():
    """Дробные значения — с 4 знаками: шаблон округляет один раз, 6,3487 → 6,3, а не 6,35 → 6,4."""
    from zakupki.dashboard import _val

    assert _val(6.348684210526316) == 6.3487
    assert _val(6.348684210526316) != 6.35
    assert round(_val(23.947368421052634), 1) == 23.9
    assert _val(5.0) == 5 and _val(float("nan")) is None


def test_po_row_keys_ignore_nested_controls():
    """Enter/Space на кнопке CSV внутри строки поставщика не сворачивает строку."""
    html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r"row\.onkeydown = \(e\) => \{(.+?)\};\n", html)
    assert m, "обработчик клавиш строки поставщика"
    assert "e.target !== row" in m.group(1)


def test_generated_at_is_label_only(small_dataset):
    """Время запуска — только подпись в шапке: попадает в D как есть, без него — None."""
    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    rel = supplier_reliability(ds)
    po = open_orders(ds, rel, cfg)
    stamped = build_data(result, index, demand, fc, rel, po, {}, "", ds, cfg, "13.08.2026 09:15")
    plain = build_data(result, index, demand, fc, rel, po, {}, "", ds, cfg)
    assert stamped["generated_at"] == "13.08.2026 09:15"
    assert plain["generated_at"] is None
    stamped.pop("generated_at"), plain.pop("generated_at")
    assert stamped == plain
    assert "D.generated_at" in TEMPLATE.read_text(encoding="utf-8")


def _built(ds):
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    rel = supplier_reliability(ds)
    return build_data(result, index, demand, fc, rel, open_orders(ds, rel, cfg), {}, "", ds, cfg), demand, cfg


def test_overdue_stale_marked(make_dataset_factory):
    """Старая просрочка помечена stale в таблице просроченных; шаблон показывает «в расчёте не учтён»."""
    from tests.conftest import _default_orders

    old = dict(_default_orders()[0], po_id="PO-OLD", order_date="2025-12-01", promised_date="2026-03-01")
    data, _, _ = _built(make_dataset_factory(orders=_default_orders() + [old]))
    late = {r["po_id"]: r["stale"] for r in data["risks"]["overdue"]}
    assert late == {"PO-OLD": True}
    assert data["cfg"]["overdue_stale_days"] == 90
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "в расчёте не учтён" in html and "o.stale" in html


def test_unit_and_abc_in_plan_rows(small_dataset):
    """Комплект подписан «компл.», штучный — «шт.»; у каждой строки класс A/B/C."""
    data, _, _ = _built(small_dataset)
    uom = dict(zip(small_dataset.sku["sku_id"], small_dataset.sku["uom"], strict=True))
    for rec in data["plan"]:
        assert rec["unit"] == ("компл." if uom[rec["sku_id"]] == "set" else "шт.")
        assert rec["abc"] in ("A", "B", "C")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "r.unit" in html and 'k: "abc"' in html and "dead-note" in html


def test_abc_by_cumulative_share():
    """A — пока доля штук до SKU < 80%, B — < 95%, остальное и без продаж — C."""
    from zakupki.dashboard import _abc

    months = pd.period_range("2025-08", periods=12, freq="M")
    sold = {"S1": 70, "S2": 20, "S3": 6, "S4": 4, "S5": 0}
    demand = pd.DataFrame(
        [{"sku_id": k, "month": m, "sold_net": v / 12} for k, v in sold.items() for m in months]
    )
    got = _abc(demand, pd.Timestamp("2026-08-13"), Config())
    assert got == {"S1": "A", "S2": "A", "S3": "B", "S4": "C", "S5": "C"}


def test_accuracy_section_and_backtest_in_data(small_dataset):
    """Секция «Точность прогноза» перед «Как считается»; бэктест передаётся в D как есть."""
    from zakupki.backtest import backtest

    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    rel = supplier_reliability(ds)
    po = open_orders(ds, rel, cfg)
    bt = backtest(ds, cfg)
    data = build_data(result, index, demand, fc, rel, po, {}, "", ds, cfg, None, bt)
    assert data["backtest"] == bt and data["backtest"]["total"]["pairs"] > 0
    assert build_data(result, index, demand, fc, rel, po, {}, "", ds, cfg)["backtest"] is None
    json.dumps(data, ensure_ascii=False)
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="sec-acc"' in html and 'href="#sec-acc"' in html and "D.backtest" in html
    assert html.index('id="sec-acc"') < html.index('id="sec-how"')
    assert html.index('href="#sec-acc"') < html.index('href="#sec-how"')


def test_supplier_weight_known_part_only(small_dataset):
    """Вес поставщика — только по позициям с весом; без веса у всех — «не заведён» (None)."""
    from zakupki.dashboard import _orders_by_supplier

    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    plan = build_plan(ds, demand, index, params, fc, cfg).plan.copy()
    plan["status"], plan["order_date"] = C.ORDER_NOW, ds.today
    plan["order_qty"] = plan["sku_id"].map({"SKU-A": 10, "SKU-B": 0, "SKU-C": 5})
    weight = ds.sku.set_index("sku_id")["weight_kg"].astype("float64")
    asap = pd.Series(True, index=plan.index)
    groups = _orders_by_supplier(plan, asap, supplier_reliability(ds), ds.today, weight)
    sup1 = next(g for g in groups["now"] if g["supplier_id"] == "SUP-1")
    # SKU-A 10 × 4,5 кг; у SKU-C вес не заведён — в сумму не входит, но считается в n.
    assert (sup1["weight_kg"], sup1["weight_n"], sup1["n"]) == (45, 1, 2)
    no_w = weight.copy()
    no_w[:] = float("nan")
    sup1 = next(g for g in _orders_by_supplier(plan, asap, supplier_reliability(ds), ds.today, no_w)["now"]
                if g["supplier_id"] == "SUP-1")
    assert sup1["weight_kg"] is None and sup1["weight_n"] == 0


def _flat_fc(sku: str, per_day: float = 10.0) -> pd.DataFrame:
    """Прогноз ровно per_day штук в день на 24 месяца с 2026-08."""
    months = pd.period_range("2026-08", periods=24, freq="M")
    return pd.DataFrame({"sku_id": sku, "month": months, "qty": [per_day * m.days_in_month for m in months]})


EMPTY_PO = pd.DataFrame({"sku_id": pd.Series(dtype="object"), "expected_date": pd.Series(dtype="datetime64[ns]"),
                         "expected_qty": pd.Series(dtype="float64")})


def test_lost_units_hand_example():
    """Склад 100 шт. при 10 шт./день кончится на 10-й день; плечо 50 → без товара 40 дн. = 400 шт."""
    from zakupki.dashboard import _lost_units

    today = pd.Timestamp("2026-08-01")
    fc = _flat_fc("X")
    plan = pd.DataFrame([{"sku_id": "X", "status": C.SHORTAGE, "lead_days": 50, "stock_now": 100.0}])
    assert _lost_units(plan, fc, EMPTY_PO, today, Config())["X"] == pytest.approx(400.0)
    # Заказ в пути 1000 шт. на 30-й день закрывает дыру: без товара дни 10–30 = 200 шт.
    po = pd.DataFrame({"sku_id": ["X"], "expected_date": [today + pd.Timedelta(days=30)], "expected_qty": [1000.0]})
    assert _lost_units(plan, fc, po, today, Config())["X"] == pytest.approx(200.0)
    # Не дефицит — не считается.
    plan["status"] = C.ORDER_NOW
    assert _lost_units(plan, fc, EMPTY_PO, today, Config()) == {}


def _scenario_result(safety: float, order_qty: int):
    from zakupki.plan import BREAKDOWN_COLUMNS, PlanResult

    today = pd.Timestamp("2026-08-01")
    plan = pd.DataFrame([{"sku_id": "X", "status": C.SHORTAGE, "order_qty": order_qty, "lead_days": 50,
                          "stock_now": 0.0, "level": 300.0}])
    arrival = today + pd.Timedelta(days=50)
    step = {c: float("nan") for c in BREAKDOWN_COLUMNS}
    step.update({"sku_id": "X", "need_at": today, "need_arrival": arrival, "need_end": arrival + pd.Timedelta(days=30),
                 "need_safety": safety, "lot": 1, "lot_full": 1, "need_ceil": order_qty})
    bd = pd.DataFrame([step], columns=BREAKDOWN_COLUMNS)
    return PlanResult(plan=plan, risks=pd.DataFrame(), sim=pd.DataFrame(), calendar=pd.DataFrame(), breakdown=bd), today


def test_safety_scenario_caps_only_big_safety():
    """Страховой 1000 > спроса 60 дн. (600) → заказ 1300 → 900; страховой 200 < 600 → заказ тот же."""
    from zakupki.dashboard import _safety_scenario

    fc = _flat_fc("X")
    # Склад 0, в пути нет: заказ = прогноз 30 дн. (300) + страховой.
    result, today = _scenario_result(1000.0, 1300)
    alt = _safety_scenario(result, fc, EMPTY_PO, today, Config())
    assert alt["X"]["qty"] == 900 and alt["X"]["safety"] == pytest.approx(600.0)
    result, today = _scenario_result(200.0, 500)
    assert _safety_scenario(result, fc, EMPTY_PO, today, Config()) == {}
    assert _safety_scenario(*_scenario_result(1000.0, 1300)[:1], fc, EMPTY_PO, today, Config(safety_alt_days=0)) == {}


def test_requantify_with_plan_safety_gives_plan_qty(small_dataset):
    """Пересчёт с тем же страховым даёт то же количество, что в плане: правила партии и даты те же."""
    from zakupki.dashboard import _requantify

    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    po = open_orders(ds, supplier_reliability(ds, cfg), cfg)
    steps = result.breakdown.set_index("sku_id")
    rows = [r for r in result.plan.itertuples(index=False) if r.order_qty > 0]
    assert rows
    for r in rows:
        s = steps.loc[r.sku_id]
        assert _requantify(r, s, float(s.need_safety), fc, po, ds.today, cfg) == r.order_qty


def test_dashboard_has_weight_lost_and_scenario(small_dataset, tmp_path):
    """Новые поля в данных и в шаблоне: вес у поставщика, «потеряно», сценарий страхового."""
    data, _, cfg = _built(small_dataset)
    assert {"alt_qty", "alt_n", "alt_marked"} <= set(data["today"])
    assert data["today"]["alt_qty"] <= data["today"]["qty"]
    assert "lost_qty_total" in data and data["cfg"]["safety_alt_days"] == cfg.safety_alt_days
    for g in data["orders"]["now"] + data["orders"]["later"]:
        assert {"weight_kg", "weight_n"} <= set(g)
    assert any(f"{cfg.safety_alt_days} дн." in rule and "не применяет" in rule for rule in data["rules"])
    html = TEMPLATE.read_text(encoding="utf-8")
    for token in ('"Вес, кг"', "weightText", "вес не заведён", "у остальных вес не заведён", '"Ожидаемые потери, шт."',
                  "lost_qty_total", 'id="hero-alt"', "При сезонном потолке страхового запаса", "согласовать вручную",
                  "при сезонном потолке", "alt_qty"):
        assert token in html, token


def test_supplier_arrivals_per_sku(small_dataset):
    """Приход у поставщика — по каждой позиции (дата + плечо, штуки), а не одна ранняя дата на всё."""
    from zakupki.dashboard import _orders_by_supplier

    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    plan = build_plan(ds, demand, index, params, fc, cfg).plan.copy()
    plan["status"], plan["order_date"] = C.ORDER_NOW, ds.today
    plan["order_qty"] = plan["sku_id"].map({"SKU-A": 10, "SKU-B": 0, "SKU-C": 5})
    plan["lead_days"] = plan["sku_id"].map({"SKU-A": 30, "SKU-B": 30, "SKU-C": 90})
    asap = pd.Series(True, index=plan.index)
    sup1 = next(g for g in _orders_by_supplier(plan, asap, supplier_reliability(ds), ds.today)["now"]
                if g["supplier_id"] == "SUP-1")
    day = lambda n: (ds.today + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
    assert sorted(sup1["arrivals"]) == [[day(30), 10], [day(90), 5]]
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "g.arrivals" in html
