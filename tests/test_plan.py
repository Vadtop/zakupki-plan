"""Тесты плана закупки: количества, статусы, риски, календарь."""

import pandas as pd
import pytest

from tests.conftest import make_dataset
from zakupki import config as C
from zakupki.config import Config
from zakupki.plan import PlanResult, build_plan

TODAY = pd.Timestamp("2026-08-01")
PLAN_COLUMNS = [
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
    "cover_days",
    "cover_months",
    "stockout_date",
    "safety_stock",
    "gap_months",
    "reliability",
    "avg_markup_pct_12m",
    "level",
    "trend",
    "short_history",
    "need_qty",
    "explanation",
]


def sku_row(sku_id: str, markup: float | None = 40.0, supplier: str = "SUP-1", cover: int = 0) -> dict:
    return {
        "sku_id": sku_id,
        "category": "cat",
        "brand_id": "B",
        "supplier_id": supplier,
        "supply_type": "import",
        "uom": "pcs",
        "weight_kg": 1.0,
        "warranty_months": 0,
        "target_cover_days": cover,
        "price_decile": 5.0,
        "first_sale_month": "2023-01",
        "last_sale_month": "2026-07",
        "months_with_sales": 30,
        "qty_total": 1000,
        "avg_markup_pct_12m": markup,
        "on_order_qty": 0,
        "created": "2022-01",
    }


def order_row(
    po_id: str,
    sku_id: str,
    status: str,
    promised: str,
    pending: int,
    ordered: int = 0,
    received: int = 0,
    supplier: str = "SUP-1",
) -> dict:
    return {
        "po_id": po_id,
        "sku_id": sku_id,
        "supplier_id": supplier,
        "supply_type": "import",
        "order_date": "2026-01-01",
        "promised_date": promised,
        "planned_lead_days": 123,
        "status": status,
        "qty_ordered": ordered,
        "qty_confirmed": ordered,
        "qty_received": received,
        "qty_pending": pending,
    }


def run(skus, orders=None, receipts=None, cfg=None, short=(), sales=None):
    """skus: список (строка sku, остаток, месячный прогноз)."""
    rows = [s for s, _, _ in skus]
    stock = [{"sku_id": s["sku_id"], "month": "2026-08", "stock_qty": st} for s, st, _ in skus]
    ds = make_dataset(
        skus=rows,
        sales=sales or [],
        stock=stock,
        orders=orders or [],
        receipts=receipts or [],
        suppliers=[],
        today=TODAY,
    )
    months = pd.period_range("2026-08", periods=36, freq="M")
    fc = pd.concat(
        [pd.DataFrame({"sku_id": s["sku_id"], "month": months, "qty": float(q)}) for s, _, q in skus],
        ignore_index=True,
    )
    params = pd.DataFrame(
        {
            "sku_id": [s["sku_id"] for s, _, _ in skus],
            "level": [float(q) for _, _, q in skus],
            "trend": 1.0,
            "sigma": 0.0,
            "history_months": 24,
            "short_history": [s["sku_id"] in short for s, _, _ in skus],
        }
    )
    empty = pd.DataFrame()
    return build_plan(ds, empty, empty, params, fc, cfg or Config())


def row(result: PlanResult, sku_id: str) -> pd.Series:
    return result.plan[result.plan["sku_id"] == sku_id].iloc[0]


def test_need_qty_by_hand_and_shortage():
    # остаток 0, спрос 30 в месяц, плечо 123 дн.: приход 02.12; покрытие 30 дн. до 01.01 = 30 × 30/31 = 29.03 → 30;
    # продажи до прихода потеряны и не дозаказываются
    res = run([(sku_row("A"), 0, 30)])
    a = row(res, "A")
    assert a["status"] == C.SHORTAGE
    assert a["need_qty"] == 30 and a["order_qty"] == 30
    assert a["gap_months"] > 0
    risk = res.risks[res.risks["risk_type"] == C.SHORTAGE].iloc[0]
    assert risk["sku_id"] == "A" and risk["gap_months"] == pytest.approx(a["gap_months"])


def _lot_receipts(qty):
    return [{"sku_id": "A", "receipt_date": f"2025-0{i}-10", "qty": qty, "supplier_id": "SUP-1"} for i in (1, 2, 3)]


def test_lot_size_rounds_up():
    # потребность 60 × 30/31 = 58.1 → 59 → до кратности 50 → 100: +41 шт. — меньше месяца спроса
    res = run([(sku_row("A"), 0, 60)], receipts=_lot_receipts(50))
    assert row(res, "A")["order_qty"] == 100


def test_lot_not_applied_when_it_bloats_stock():
    # до партии 200 было бы +141 шт. — больше месяца спроса: не округляем, 59
    res = run([(sku_row("A"), 0, 60)], receipts=_lot_receipts(200))
    assert row(res, "A")["order_qty"] == 59
    bd = res.breakdown.set_index("sku_id").loc["A"]
    assert bd["lot"] == 1 and bd["lot_full"] == 200


@pytest.mark.parametrize("received, expected", [(None, 59), (50, 59)])
def test_open_order_not_reordered_and_reliability(received, expected):
    orders = [order_row("PO-1", "A", "open", "2026-09-15", 430, ordered=430)]
    if received is not None:
        orders.append(order_row("PO-0", "A", "closed", "2026-05-01", 0, ordered=100, received=received))
    res = run([(sku_row("A"), 0, 60)], orders=orders)
    a = row(res, "A")
    assert a["on_order"] == 430
    assert a["order_qty"] == expected
    if received is None:
        # дыру до 15.09 закроет заказ в пути — дефицит флагом; 430 шт. хватит до весны,
        # следующий заказ — по плану, а не «ничего не делать»
        assert a["status"] == C.PLAN and C.SHORTAGE in a["flags"]
        assert a["order_date"] > pd.Timestamp("2026-11-01")
    if received:
        # 430 × 0,5 = 215 придут 15.09; до прихода нового 02.12 продастся 241.9 > 215 → остаток к приходу 0;
        # заказ — только покрытие 58.1 → 59 (без надёжности 430 хватило бы и заказ 0)
        assert a["on_order_expected"] == pytest.approx(215)
        assert a["reliability"] == pytest.approx(0.5)


def test_low_markup_not_auto_ordered():
    res = run([(sku_row("A", markup=10.0), 0, 30)])
    a = row(res, "A")
    assert a["status"] == C.LOW_MARKUP
    assert a["order_qty"] == 0 and a["need_qty"] > 0


def test_dead_sku_sell_off_risk():
    res = run([(sku_row("A", markup=None), 40, 0), (sku_row("B", markup=None), 0, 0)])
    assert row(res, "A")["status"] == C.DEAD and row(res, "A")["order_qty"] == 0
    dead = res.risks[res.risks["risk_type"] == C.DEAD]
    assert list(dead["sku_id"]) == ["A"]
    assert "распродать" in dead.iloc[0]["detail"]


def test_excess_risk():
    res = run([(sku_row("A"), 5000, 30)])
    a = row(res, "A")
    assert a["status"] == C.EXCESS and a["order_qty"] == 0
    risk = res.risks[res.risks["risk_type"] == C.EXCESS].iloc[0]
    # покрытие 1096 дней до конца горизонта, плечо 123 + покрытие 30 → лишних (1096 − 153) / 30.4
    assert risk["excess_months"] == pytest.approx((1096 - 153) / 30.4, rel=1e-6)


def test_order_now_plan_ok_statuses():
    cfg = Config(excess_extra_months=36)
    res = run([(sku_row("N"), 141, 30), (sku_row("P"), 330, 30), (sku_row("K"), 600, 30)], cfg=cfg)
    n, p, k = row(res, "N"), row(res, "P"), row(res, "K")
    assert n["status"] == C.ORDER_NOW and n["order_qty"] > 0
    assert p["status"] == C.PLAN and p["order_qty"] > 0
    assert k["status"] == C.OK and k["order_qty"] == 0
    assert n["order_date"] <= TODAY + pd.Timedelta(days=30)


def test_flags_main_status_and_short_history():
    res = run([(sku_row("A", markup=10.0), 0, 30)], short=("A",))
    a = row(res, "A")
    assert a["status"] == C.LOW_MARKUP
    assert a["flags"] == f"{C.SHORTAGE}; {C.SHORT_HISTORY}"
    assert bool(a["short_history"]) is True


def test_overdue_order_in_risks():
    orders = [order_row("PO-9", "A", "open", "2026-07-01", 100, ordered=100)]
    res = run([(sku_row("A"), 0, 30)], orders=orders)
    r = res.risks[res.risks["risk_type"] == C.OVERDUE_PO].iloc[0]
    assert r["po_id"] == "PO-9" and r["sku_id"] == "A"
    assert r["promised_date"] == pd.Timestamp("2026-07-01") and r["qty"] == 100


def test_stale_overdue_not_in_transit_but_in_risks():
    """Просрочка 91 дн.: не «в пути» и не приход (нет избытка), но в рисках с пометкой; 89 дн. — в пути."""
    orders = [order_row("PO-OLD", "H", "open", "2026-05-02", 1000, ordered=1000)]
    res = run([(sku_row("H"), 0, 30)], orders=orders)
    h = row(res, "H")
    assert h["on_order"] == 0 and h["on_order_expected"] == 0
    assert h["status"] != C.EXCESS and h["order_qty"] > 0
    r = res.risks[res.risks["risk_type"] == C.OVERDUE_PO].iloc[0]
    assert r["po_id"] == "PO-OLD" and r["qty"] == 1000
    assert "просрочка 91 дн. — уточнить у поставщика, в расчёте не учтён" in r["detail"]

    fresh = run([(sku_row("H"), 0, 30)], orders=[order_row("PO-NEW", "H", "open", "2026-05-04", 1000, ordered=1000)])
    h = row(fresh, "H")
    assert h["on_order"] == 1000 and h["status"] == C.EXCESS
    r = fresh.risks[fresh.risks["risk_type"] == C.OVERDUE_PO].iloc[0]
    assert "не учтён" not in r["detail"]


def test_calendar_and_columns():
    res = run(
        [
            (sku_row("S"), 0, 30),
            (sku_row("N"), 141, 30),
            (sku_row("P"), 330, 30),
            (sku_row("D", markup=None), 5, 0),
        ]
    )
    assert list(res.plan.columns) == PLAN_COLUMNS
    assert list(res.plan["sku_id"].sort_values()) == ["D", "N", "P", "S"]
    order = [C.STATUS_ORDER.index(s) for s in res.plan["status"]]
    assert order == sorted(order)
    cal = res.calendar
    assert list(cal["month"]) == list(pd.period_range("2026-08", periods=12, freq="M"))
    aug = cal[cal["month"] == pd.Period("2026-08", freq="M")].iloc[0]
    # первые заказы SHORTAGE и ORDER_NOW — в августе
    assert aug["qty"] >= row(res, "S")["order_qty"] + row(res, "N")["order_qty"]
    p_month = row(res, "P")["order_date"].to_period("M")
    assert cal[cal["month"] == p_month].iloc[0]["positions"] >= 1
    # повторные заказы на ежемесячных пересмотрах: позиций больше, чем SKU с заказом
    assert cal["positions"].sum() > 3
    assert (cal["qty"] > 0).sum() >= 10
    sizes = res.sim.groupby("sku_id").size()
    assert set(sizes) == {12}


def test_deterministic():
    skus = [(sku_row("B"), 0, 30), (sku_row("A"), 141, 30)]
    assert run(skus).plan.to_csv() == run(skus).plan.to_csv()


def test_dead_without_stock_has_no_excess():
    res = run([(sku_row("A", markup=None), 0, 0)])
    a = row(res, "A")
    assert a["status"] == C.DEAD
    assert C.EXCESS not in a["flags"]
    assert res.risks[res.risks["risk_type"] == C.EXCESS].empty


def test_active_zero_forecast_zero_stock_not_excess():
    res = run([(sku_row("A"), 0, 0)])
    assert row(res, "A")["status"] != C.EXCESS
    assert res.risks[res.risks["risk_type"] == C.EXCESS].empty


def test_far_arrival_does_not_cancel_order():
    """Приход после конца периода (плечо + покрытие) не закрывает потребность сейчас."""
    orders = [order_row("PO-L", "L", "open", "2027-12-01", 1000, ordered=1000)]
    res = run([(sku_row("L"), 141, 30), (sku_row("N"), 141, 30)], orders=orders)
    assert row(res, "L")["need_qty"] > 0
    assert row(res, "L")["need_qty"] == row(res, "N")["need_qty"]


def test_zero_need_does_not_set_order_status():
    """Заказ в пути до обнуления закрывает потребность: «заказать» с нулём штук не ставится."""
    orders = [order_row("PO-L", "L", "open", "2026-12-15", 1000, ordered=1000)]
    res = run([(sku_row("L"), 141, 30), (sku_row("N"), 141, 30)], orders=orders)
    lone = row(res, "L")
    assert lone["need_qty"] == 0
    assert lone["status"] not in (C.ORDER_NOW, C.PLAN)
    assert row(res, "N")["status"] == C.ORDER_NOW
    ordered = res.plan[res.plan["status"].isin((C.ORDER_NOW, C.PLAN, C.SHORTAGE))]
    assert (ordered["order_qty"] > 0).all()


def test_shortage_risk_only_for_primary_shortage_and_not_for_dead():
    """Риск дефицита в risks — только у основного статуса; у мёртвого флага дефицита нет."""
    res = run(
        [
            (sku_row("S"), 0, 30),
            (sku_row("L", markup=10.0), 0, 30),
            (sku_row("D", markup=None), 0, 30),
        ]
    )
    short = res.risks[res.risks["risk_type"] == C.SHORTAGE]
    assert list(short["sku_id"]) == ["S"]
    assert (short["qty"] > 0).all()
    assert C.SHORTAGE not in row(res, "D")["flags"]
    assert C.SHORTAGE in row(res, "L")["flags"]


def test_excess_counts_orders_in_transit_with_zero_stock():
    """Остатка нет, но в пути запас на ~33 мес. при пороге 13 мес. — это избыток, а не норма."""
    orders = [order_row("PO-H", "H", "open", "2026-08-20", 1000, ordered=1000)]
    res = run([(sku_row("H"), 0, 30)], orders=orders)
    h = row(res, "H")
    assert h["stock_now"] == 0 and h["on_order"] == 1000
    assert h["status"] == C.EXCESS and h["order_qty"] == 0
    risk = res.risks[res.risks["risk_type"] == C.EXCESS]
    assert list(risk["sku_id"]) == ["H"]
    # 1000 шт. / 30 в месяц ≈ 33 мес. покрытия; сверх плеча 123 и покрытия 90 дней — больше 20 мес.
    assert risk.iloc[0]["excess_months"] > 20


def test_quantity_for_shown_order_date():
    """Количество считается на ту дату, что видит закупщик (REVIEW-2 № 3).

    Склад 85 при спросе ~1 шт./день, плечо 123: заказать до будущей даты; на эту дату
    к приходу склад пуст, заказ — полное покрытие, а не «то, что не хватит к приходу сегодняшнего».
    """
    res = run([(sku_row("A"), 160, 30)])
    a = row(res, "A")
    bd = res.breakdown.set_index("sku_id").loc["A"]
    assert bd["need_at"] == a["order_date"]


def test_bridge_before_transit_inside_cover():
    """Поставка в конце покрытия не отменяет заказ, если до неё склад опустеет (REVIEW-2 № 2)."""
    orders = [order_row("PO-L", "L", "open", "2026-12-25", 1000, ordered=1000)]
    res = run([(sku_row("L"), 0, 30)], orders=orders)
    lone = row(res, "L")
    assert lone["order_qty"] > 0
    bd = res.breakdown.set_index("sku_id").loc["L"]
    assert bd["need_bridge"] > 0


def test_excess_now_but_future_order_in_calendar():
    """Сейчас избыток, но в горизонте нужна закупка — она в календаре и с датой (REVIEW-3 № 3)."""
    res = run([(sku_row("A"), 400, 30)])
    a = row(res, "A")
    assert a["status"] == C.EXCESS and a["order_qty"] == 0
    assert a["need_qty"] > 0
    month = a["order_date"].to_period("M")
    cal = res.calendar.set_index("month")
    assert month in cal.index and cal.loc[month, "qty"] >= a["need_qty"]


def test_calendar_now_part_is_only_urgent():
    """Срочная часть текущего месяца — только «сразу», без заказов позже в месяце (REVIEW-3 № 4)."""
    res = run([(sku_row("S"), 0, 30), (sku_row("N"), 141, 30)])
    aug = res.calendar.iloc[0]
    urgent = res.plan[(res.plan["order_qty"] > 0) & (res.plan["order_date"] <= TODAY)]
    assert aug["now_qty"] == int(urgent["order_qty"].sum())
    assert aug["now_positions"] == len(urgent)
    assert aug["qty"] >= aug["now_qty"]


def test_review_not_rarer_than_sku_cover():
    """Покрытие SKU 10 дней — пересмотр не реже, без провалов между заказами (REVIEW-3 № 2)."""
    res = run([(sku_row("A", cover=10), 0, 30)])
    # 12 мес. по 10 дней — около 36 заказов; при шаге 30 дней было бы около 12
    assert res.calendar["positions"].sum() >= 30


def sale(sku_id: str, month: str, sold: int, returned: int = 0) -> dict:
    return {"sku_id": sku_id, "category": "cat", "month": month, "qty_sold": sold, "qty_returned": returned,
            "partial_month": 0}


def test_no_markup_with_sales_is_not_dead():
    """Наценки нет, а продажи за 12 полных месяцев есть: «нет наценки», не «мёртвый»; автозаказа нет."""
    sales = [sale("A", "2026-03", 20), sale("B", "2026-03", 2, returned=2), sale("C", "2025-06", 30)]
    skus = [(sku_row("A", markup=None), 0, 5), (sku_row("B", markup=None), 0, 5), (sku_row("C", markup=None), 0, 5)]
    res = run(skus, sales=sales)
    a = row(res, "A")
    assert a["status"] == C.NO_MARKUP
    assert a["order_qty"] == 0 and a["need_qty"] > 0
    from zakupki.explain import template_explanation

    assert "себестоимости" in template_explanation(a)
    # Продажи погашены возвратами или старше окна — мёртвый.
    assert row(res, "B")["status"] == C.DEAD
    assert row(res, "C")["status"] == C.DEAD


def test_overdue_counted_by_po_not_by_line():
    """Один просроченный заказ на две товарные строки — один заказ в сводке."""
    from zakupki.explain import template_summary

    orders = [order_row("PO-1", "A", "open", "2026-05-01", 50), order_row("PO-1", "B", "open", "2026-05-01", 30)]
    res = run([(sku_row("A"), 0, 30), (sku_row("B"), 0, 30)], orders=orders)
    assert (res.risks["risk_type"] == C.OVERDUE_PO).sum() == 2
    assert "Просроченных заказов поставщиков: 1 (2 строк)." in template_summary(res)
