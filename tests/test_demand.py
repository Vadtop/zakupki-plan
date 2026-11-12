"""Спрос: продажи минус возвраты, упущенные месяцы, остаток на начало месяца."""

import pandas as pd

from zakupki.config import Config
from zakupki.demand import build_demand

TODAY = "2026-08-13"
CFG = Config()
X = "SKU-X"
CAT = "cat_x"


def _months(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def _sales(sku_id: str, category: str, months, qty, returned=0, partial=0) -> list[dict]:
    """Строки продаж: qty — число или функция от месяца."""
    rows = []
    for m in months:
        sold = qty(m) if callable(qty) else qty
        rows.append({
            "sku_id": sku_id, "category": category, "month": m, "qty_sold": sold,
            "qty_returned": returned, "partial_month": partial,
        })
    return rows


def _stock(sku_id: str, months, qty) -> list[dict]:
    return [{"sku_id": sku_id, "month": m, "stock_qty": qty} for m in months]


def test_columns_and_sold_net(make_dataset_factory):
    """Колонки ровно из interfaces.md; спрос = max(0, продажи − возвраты)."""
    months = _months("2025-01", "2025-06")
    ds = make_dataset_factory(
        sales=_sales(X, CAT, months, qty=10, returned=3),
        stock=_stock(X, months, 50),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    assert list(demand.columns) == [
        "sku_id", "category", "month", "demand", "sold_net", "lost", "stock_start",
    ]
    assert len(demand) == 6
    assert (demand["sold_net"] == 7).all()
    assert (demand["demand"] == 7.0).all()
    assert not demand["lost"].any()


def test_return_above_sales_gives_zero(make_dataset_factory):
    """Возврат больше продаж — спрос 0, месяц не упущенный (остаток был)."""
    months = _months("2025-01", "2025-03")
    ds = make_dataset_factory(
        sales=_sales(X, CAT, months, qty=4, returned=9),
        stock=_stock(X, months, 50),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    assert (demand["sold_net"] == 0).all()
    assert (demand["demand"] == 0.0).all()
    assert not demand["lost"].any()


def test_partial_and_current_month_excluded(make_dataset_factory):
    """partial_month=1 и месяц «сегодня» в спрос не попадают."""
    sales = [
        *_sales(X, CAT, _months("2026-05", "2026-06"), qty=10),
        *_sales(X, CAT, [pd.Period("2026-07", freq="M")], qty=10, partial=1),
        *_sales(X, CAT, [pd.Period("2026-08", freq="M")], qty=10),
    ]
    ds = make_dataset_factory(
        sales=sales,
        stock=_stock(X, _months("2026-05", "2026-08"), 50),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    assert list(demand["month"]) == _months("2026-05", "2026-06")


def test_lost_month_needs_zero_stock(make_dataset_factory):
    """Нулевые продажи при нулевом остатке — упущенный спрос: demand=NaN, lost=True."""
    months = _months("2024-01", "2025-12")
    stock_months = [m for m in months if not (pd.Period("2025-03", freq="M") <= m <= pd.Period("2025-05", freq="M"))]
    ds = make_dataset_factory(
        sales=_sales(X, CAT, months, qty=lambda m: 0 if m.month in (3, 4, 5) and m.year == 2025 else 10),
        stock=_stock(X, stock_months, 50),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    lost = demand[demand["lost"]]
    assert list(lost["month"]) == _months("2025-03", "2025-05")
    assert lost["demand"].isna().all()
    assert (lost["stock_start"] == 0.0).all()
    # Соседние месяцы с остатком не помечены.
    assert not demand[demand["month"] == pd.Period("2025-02", freq="M")]["lost"].iloc[0]


def test_zero_sales_with_stock_is_not_lost(make_dataset_factory):
    """Нулевые продажи при непустом остатке — честный ноль спроса."""
    months = _months("2024-01", "2024-06")
    ds = make_dataset_factory(
        sales=_sales(X, CAT, months, qty=0),
        stock=_stock(X, months, 40),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    assert (demand["demand"] == 0.0).all()
    assert not demand["lost"].any()


def test_months_before_first_snapshot_are_honest_zeros(make_dataset_factory):
    """До первого месяца снимков остаток неизвестен: нули — не упущенный спрос."""
    ds = make_dataset_factory(
        sales=_sales(X, CAT, _months("2023-01", "2024-06"), qty=0),
        stock=_stock(X, _months("2024-01", "2024-06"), 40),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    before = demand[demand["month"] < pd.Period("2024-01", freq="M")]
    assert len(before) == 12
    assert before["stock_start"].isna().all()
    assert not before["lost"].any()
    assert (before["demand"] == 0.0).all()


def test_missing_stock_row_after_snapshot_is_zero(make_dataset_factory):
    """Нет строки остатка за месяц ≥ первого снимка — остаток 0, а не NaN."""
    ds = make_dataset_factory(
        sales=_sales(X, CAT, _months("2024-01", "2024-04"), qty=0),
        stock=_stock(X, [pd.Period("2024-01", freq="M")], 40),
        today=TODAY,
    )
    demand = build_demand(ds, CFG)

    row = demand[demand["month"] == pd.Period("2024-02", freq="M")].iloc[0]
    assert row["stock_start"] == 0.0
    assert bool(row["lost"]) is True
    assert pd.isna(row["demand"])


def test_empty_sales_gives_empty_frame(make_dataset_factory):
    """Без продаж — пустой кадр с правильными колонками."""
    ds = make_dataset_factory(sales=[], stock=[], today=TODAY)
    demand = build_demand(ds, CFG)

    assert list(demand.columns) == [
        "sku_id", "category", "month", "demand", "sold_net", "lost", "stock_start",
    ]
    assert len(demand) == 0


def test_determinism(small_dataset):
    """Два вызова на одном входе дают равные кадры."""
    first = build_demand(small_dataset, CFG)
    second = build_demand(small_dataset, CFG)

    pd.testing.assert_frame_equal(first, second)


def test_default_dataset_marks_dead_sku_lost(small_dataset):
    """В наборе по умолчанию у мёртвого SKU-C месяцы после остатков упущены."""
    demand = build_demand(small_dataset, CFG)
    sku_c = demand[demand["sku_id"] == "SKU-C"]

    last_sold = sku_c[sku_c["month"] == pd.Period("2024-06", freq="M")].iloc[0]
    first_lost = sku_c[sku_c["month"] == pd.Period("2024-07", freq="M")].iloc[0]
    assert bool(last_sold["lost"]) is False
    assert bool(first_lost["lost"]) is True
    assert pd.isna(first_lost["demand"])
