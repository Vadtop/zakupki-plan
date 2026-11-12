"""Тесты симуляции запаса: покрытие, обнуление, страховой запас, дата заказа, помощники."""

import pandas as pd
import pytest

from zakupki.config import Config
from zakupki.simulation import (
    POSITION_COLUMNS,
    SIM_COLUMNS,
    forecast_between,
    run_simulation,
)

CFG = Config()
SKU = "S1"


def make_fc(start: str, values: list[float], sku: str = SKU) -> pd.DataFrame:
    months = pd.period_range(start, periods=len(values), freq="M")
    return pd.DataFrame({"sku_id": sku, "month": months, "qty": values})


def flat_fc(start: str, qty: float, n: int = 36, sku: str = SKU) -> pd.DataFrame:
    return make_fc(start, [qty] * n, sku)


def po(rows: list[tuple[str, float]], sku: str = SKU) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "po_id": [f"P{i}" for i in range(len(rows))],
            "sku_id": sku,
            "expected_date": [pd.Timestamp(d) for d, _ in rows],
            "expected_qty": [q for _, q in rows],
        }
    )


NO_PO = pd.DataFrame(columns=["po_id", "sku_id", "expected_date", "expected_qty"])


def run(fc, stock, today, open_po=NO_PO, lead=40, sigma=0.0, cfg=CFG, sku=SKU):
    params = pd.DataFrame({"sku_id": [sku], "level": [30.0], "trend": [1.0], "sigma": [sigma]})
    pos, sim = run_simulation(
        pd.Series({sku: float(stock)}),
        fc,
        open_po,
        pd.Series({sku: lead}),
        params,
        cfg,
        pd.Timestamp(today),
    )
    return pos.iloc[0], pos, sim


def test_columns_and_horizon():
    _, pos, sim = run(flat_fc("2026-01", 30), 90, "2026-01-01")
    assert list(pos.columns) == POSITION_COLUMNS
    assert list(sim.columns) == SIM_COLUMNS
    assert len(pos) == 1
    assert len(sim) == CFG.cover_horizon_months


def test_flat_demand_covers_three_months():
    row, _, _ = run(flat_fc("2026-01", 30), 90, "2026-01-01")
    assert row["stockout_date"] == pd.Timestamp("2026-04-01")
    assert row["cover_days"] == pytest.approx(91, abs=2)
    assert row["cover_months"] == pytest.approx(3, abs=0.1)


def test_current_month_subtracts_only_remaining_share():
    # today = 16 января: осталось 16 из 31 дня
    _, _, sim = run(flat_fc("2026-01", 31), 100, "2026-01-16")
    assert sim.iloc[0]["forecast"] == pytest.approx(16.0)
    assert sim.iloc[0]["stock_end"] == pytest.approx(84.0)


def test_arrival_pushes_stockout_and_overdue_goes_next_month():
    fc = flat_fc("2026-01", 30)
    base, _, _ = run(fc, 90, "2026-01-01")
    later, _, sim = run(fc, 90, "2026-01-01", open_po=po([("2026-02-10", 60)]))
    assert later["stockout_date"] > base["stockout_date"] + pd.Timedelta(days=50)
    assert sim.iloc[1]["arrivals"] == 60 and sim.iloc[0]["arrivals"] == 0
    # просроченный заказ: expected_date = today + 30 → следующий месяц
    _, _, sim2 = run(fc, 90, "2026-01-16", open_po=po([("2026-02-15", 40)]))
    assert list(sim2["arrivals"].iloc[:2]) == [0, 40]


def test_safety_stock_formula_and_zero_sigma():
    row, _, _ = run(flat_fc("2026-01", 30), 90, "2026-01-01", lead=76, sigma=10.0)
    assert row["safety_stock"] == pytest.approx(26.088, abs=0.01)
    zero, _, _ = run(flat_fc("2026-01", 30), 90, "2026-01-01", lead=76, sigma=0.0)
    assert zero["safety_stock"] == 0 and zero["safety_days"] == 0


def test_safety_stock_capped_by_months_of_level():
    """Большой σ: страховой запас не выше safety_cap_months × level (level в helper = 30)."""
    row, _, _ = run(flat_fc("2026-01", 30), 90, "2026-01-01", lead=76, sigma=100.0)
    assert row["safety_stock"] == pytest.approx(CFG.safety_cap_months * 30.0)
    off, _, _ = run(flat_fc("2026-01", 30), 90, "2026-01-01", lead=76, sigma=100.0,
                    cfg=Config(safety_cap_months=0))
    assert off["safety_stock"] > row["safety_stock"]


def test_order_date_is_stockout_minus_lead_minus_safety():
    fc = flat_fc("2026-01", 30)
    row, _, _ = run(fc, 90, "2026-01-01", lead=40, sigma=0.0)
    assert row["order_date"] == pd.Timestamp("2026-02-20")
    row, _, _ = run(fc, 90, "2026-01-01", lead=40, sigma=5.0)
    assert 9 <= row["safety_days"] <= 11
    expected = (
        pd.Timestamp("2026-04-01") - pd.Timedelta(days=40) - pd.Timedelta(days=round(row["safety_days"]))
    )
    assert row["order_date"] == expected


def test_seasonality_changes_order_date():
    heater = make_fc("2026-02", ([5.0] * 8 + [200.0, 60.0, 40.0, 20.0]) * 3)  # 360 в год, пик в октябре
    flat = flat_fc("2026-02", 30.0)  # те же 360 в год
    h, _, _ = run(heater, 150, "2026-02-01", lead=150)
    f, _, _ = run(flat, 150, "2026-02-01", lead=150)
    assert pd.Timestamp("2026-03-01") <= h["order_date"] < pd.Timestamp("2026-08-01")
    assert h["order_date"] != f["order_date"]
    assert not (pd.Timestamp("2026-03-01") <= f["order_date"] < pd.Timestamp("2026-08-01"))


def test_gap_months_when_stockout_before_lead():
    fc = flat_fc("2026-01", 30)
    row, _, _ = run(fc, 30, "2026-01-01", lead=123)
    assert row["gap_months"] > 0
    assert row["gap_months"] == pytest.approx(
        (pd.Timestamp("2026-01-01") + pd.Timedelta(days=123) - row["stockout_date"]).days / 30.4
    )
    ok, _, _ = run(fc, 900, "2026-01-01", lead=40)
    assert ok["gap_months"] == 0


def test_no_stockout_in_horizon_gives_nat():
    row, _, _ = run(flat_fc("2026-01", 1.0), 10000, "2026-01-01")
    assert pd.isna(row["stockout_date"]) and pd.isna(row["order_date"])
    assert row["cover_days"] == (pd.Timestamp("2029-01-01") - pd.Timestamp("2026-01-01")).days
    assert row["gap_months"] == 0


def test_forecast_between_partial_months():
    fc = make_fc("2026-01", [31.0, 28.0])
    assert forecast_between(fc, SKU, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), CFG) == 31
    # 16 января (16 дней) + 14 февраля (14 дней из 28 → половина)
    got = forecast_between(fc, SKU, pd.Timestamp("2026-01-16"), pd.Timestamp("2026-02-15"), CFG)
    assert got == pytest.approx(16.0 + 14.0)
    assert forecast_between(fc, SKU, pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-01"), CFG) == 0


def test_deterministic():
    fc = pd.concat([flat_fc("2026-01", 30), flat_fc("2026-01", 12, sku="S2")], ignore_index=True)
    params = pd.DataFrame(
        {"sku_id": ["S1", "S2"], "level": [30.0, 12.0], "trend": [1.0, 1.0], "sigma": [4.0, 2.0]}
    )
    args = (
        pd.Series({"S1": 90.0, "S2": 50.0}),
        fc,
        po([("2026-02-10", 20)]),
        pd.Series({"S1": 60, "S2": 90}),
        params,
        CFG,
        pd.Timestamp("2026-01-10"),
    )
    p1, s1 = run_simulation(*args)
    p2, s2 = run_simulation(*args)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(s1, s2)
    assert list(p1["sku_id"]) == ["S1", "S2"]


def test_safety_days_capped_by_lead_on_intermittent_demand():
    """Прерывистый спрос: большой σ при малом среднем не уводит дату заказа на годы назад."""
    lead = 120
    row, _, _ = run(flat_fc("2026-01", 1.0), 30, "2026-01-01", lead=lead, sigma=20.0)
    assert pd.notna(row["stockout_date"])
    assert row["safety_days"] <= lead
    assert row["order_date"] >= row["stockout_date"] - pd.Timedelta(days=2 * lead)


def test_stockout_before_month_end_arrival():
    """Приход в конце месяца не скрывает дефицит в его начале (REVIEW № 2)."""
    fc = make_fc("2026-01", [31.0] + [30.0] * 35)  # 1 шт./день
    row, _, sim = run(fc, 5, "2026-01-01", open_po=po([("2026-01-31", 100)]), lead=60)
    assert row["stockout_date"] == pd.Timestamp("2026-01-06")
    assert row["gap_months"] > 0
    # дыра — 25 дней без товара (06.01–31.01), а не всё время до плеча
    assert row["gap_months"] == pytest.approx(25 / 30.4)
    # продажи 06.01–30.01 потеряны: остаток на конец января = приход 100 − спрос 31.01 = 99
    assert sim.iloc[0]["arrivals"] == 100
    assert sim.iloc[0]["stock_end"] == pytest.approx(99.0)


def test_arrival_before_stockout_inside_month_keeps_cover():
    """Приход раньше обнуления в том же месяце — дефицита нет, обнуление позже."""
    fc = make_fc("2026-01", [31.0] + [30.0] * 35)
    row, _, _ = run(fc, 10, "2026-01-01", open_po=po([("2026-01-05", 100)]), lead=60)
    assert row["stockout_date"] > pd.Timestamp("2026-04-01")


def test_zero_exactly_at_arrival_is_not_shortage():
    """Склад обнуляется ровно в день прихода — дефицита нет, заказ — от настоящего обнуления (REVIEW-3 № 1)."""
    days = [pd.Period("2026-01", "M") + i for i in range(36)]
    fc = make_fc("2026-01", [float(m.days_in_month) for m in days])  # ровно 1 шт./день
    row, _, _ = run(fc, 100, "2026-01-01", open_po=po([("2026-04-11", 100)]), lead=60)
    assert row["gap_months"] == 0
    # 100 + 100 шт. хватит до 20.07; заказ за плечо 60 дн. до этого
    assert row["stockout_date"] == pd.Timestamp("2026-07-20")
    assert row["order_date"] == pd.Timestamp("2026-05-21")
