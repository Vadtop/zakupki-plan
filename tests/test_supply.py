"""Поставки: плечо, надёжность поставщика, заказы в пути, кратность партии, остаток."""

from pathlib import Path

import pandas as pd
import pytest

from zakupki.config import Config
from zakupki.data import load_dataset
from zakupki.supply import current_stock, lead_days, lot_size, open_orders, supplier_reliability

RAW_DIR = Path("data/raw/share")
needs_raw = pytest.mark.skipif(not RAW_DIR.is_dir(), reason="нет data/raw/share")


# --- маленькие конструкторы строк набора ---

def _sku(sku_id: str, supplier_id: str = "SUP-1", **kw) -> dict:
    """Строка справочника товаров с разумными значениями по умолчанию."""
    row = {
        "sku_id": sku_id, "category": "sanitary_faucets", "brand_id": "BRD-1",
        "supplier_id": supplier_id, "supply_type": "import", "uom": "pcs", "weight_kg": 1.0,
        "warranty_months": 0, "target_cover_days": 0, "price_decile": 5.0,
        "first_sale_month": "2023-01", "last_sale_month": "2026-08", "months_with_sales": 44,
        "qty_total": 100, "avg_markup_pct_12m": 40.0, "on_order_qty": 0, "created": "2022-01",
    }
    row.update(kw)
    return row


def _order(
    po_id: str,
    sku_id: str,
    status: str,
    *,
    supplier_id: str = "SUP-1",
    lead: int = 123,
    ordered: int = 0,
    received: int = 0,
    pending: int = 0,
    order_date: str = "2026-05-01",
    promised_date: str = "2026-09-01",
) -> dict:
    """Строка заказа поставщику."""
    return {
        "po_id": po_id, "sku_id": sku_id, "supplier_id": supplier_id, "supply_type": "import",
        "order_date": order_date, "promised_date": promised_date, "planned_lead_days": lead,
        "status": status, "qty_ordered": ordered, "qty_confirmed": ordered,
        "qty_received": received, "qty_pending": pending,
    }


def _supplier(supplier_id: str, median_lead: float | None = None, **kw) -> dict:
    """Строка справочника поставщиков."""
    row = {
        "supplier_id": supplier_id, "supply_type": "import", "receipts_cnt": 3, "skus": 1,
        "median_days_between_receipts": 70.0, "median_planned_lead_days": median_lead,
    }
    row.update(kw)
    return row


def _receipt(sku_id: str, receipt_date: str, qty: int, supplier_id: str = "SUP-1") -> dict:
    """Строка прихода на склад."""
    return {"sku_id": sku_id, "receipt_date": receipt_date, "qty": qty, "supplier_id": supplier_id}


# --- плечо ---

def test_lead_days_median_of_positive_by_sku(make_dataset_factory):
    """Медиана положительных planned_lead_days по SKU; отрицательные не участвуют."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-2")],
        orders=[
            _order("PO-1", "SKU-A", "closed", lead=100, ordered=10, received=10),
            _order("PO-2", "SKU-A", "completed", lead=200, ordered=10, received=10),
            _order("PO-3", "SKU-A", "closed", lead=-13, ordered=10, received=10),
            _order("PO-4", "SKU-B", "closed", lead=90, ordered=10, received=10),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1", median_lead=150.0), _supplier("SUP-2", median_lead=101.0)],
    )

    result = lead_days(ds, Config())

    assert result.loc["SKU-A"] == 150  # медиана 100 и 200, а не 100 с минусом в наборе
    assert result.loc["SKU-B"] == 90
    assert result.dtype == "int64"


def test_lead_days_supplier_median_then_default(make_dataset_factory):
    """Нет заказов — медиана поставщика; нет и её — lead_default_days из Config."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-2"), _sku("SKU-C", "SUP-9")],
        orders=[],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1", median_lead=101.0), _supplier("SUP-2", median_lead=None)],
    )

    result = lead_days(ds, Config(lead_default_days=140))

    assert result.loc["SKU-A"] == 101
    assert result.loc["SKU-B"] == 140  # у поставщика медианы нет
    assert result.loc["SKU-C"] == 140  # поставщика нет в справочнике


def test_lead_days_clamped_to_bounds(make_dataset_factory):
    """Результат всегда в lead_min_days..lead_max_days."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-1")],
        orders=[
            _order("PO-1", "SKU-A", "closed", lead=500, ordered=10, received=10),
            _order("PO-2", "SKU-B", "closed", lead=5, ordered=10, received=10),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1", median_lead=123.0)],
    )

    result = lead_days(ds, Config())

    assert result.loc["SKU-A"] == Config().lead_max_days
    assert result.loc["SKU-B"] == Config().lead_min_days


# --- надёжность поставщика ---

def test_supplier_reliability_ratio_by_settled_orders(make_dataset_factory):
    """Надёжность = Σreceived/Σordered по closed и completed; open не считается."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-2")],
        orders=[
            _order("PO-1", "SKU-A", "closed", supplier_id="SUP-1", ordered=100, received=80),
            _order("PO-2", "SKU-A", "completed", supplier_id="SUP-1", ordered=100, received=100),
            _order("PO-3", "SKU-B", "open", supplier_id="SUP-2", ordered=900, received=0, pending=900),
            _order("PO-4", "SKU-B", "completed", supplier_id="SUP-2", ordered=50, received=25),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1"), _supplier("SUP-2")],
    )

    result = supplier_reliability(ds).set_index("supplier_id")

    assert result.loc["SUP-1", "reliability"] == pytest.approx(180 / 200)
    assert result.loc["SUP-1", "orders_n"] == 2
    assert result.loc["SUP-1", "received"] == 180
    assert result.loc["SUP-1", "ordered"] == 200
    # заказ в пути (900 шт) в надёжность не входит
    assert result.loc["SUP-2", "reliability"] == pytest.approx(0.5)
    assert result.loc["SUP-2", "orders_n"] == 1


def test_supplier_reliability_skips_zero_received_closed(make_dataset_factory):
    """closed с нулём получено — учёт/отмена: в надёжность не входит."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1")],
        orders=[
            _order("PO-1", "SKU-A", "closed", supplier_id="SUP-1", ordered=100, received=0),
            _order("PO-2", "SKU-A", "completed", supplier_id="SUP-1", ordered=100, received=90),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1")],
    )

    off = supplier_reliability(ds, Config(reliability_skip_zero_closed=False)).set_index("supplier_id")
    on = supplier_reliability(ds).set_index("supplier_id")

    assert off.loc["SUP-1", "reliability"] == pytest.approx(0.45)
    assert on.loc["SUP-1", "reliability"] == pytest.approx(0.9)


def test_supplier_reliability_falls_back_to_overall(make_dataset_factory):
    """Поставщик без закрытых заказов получает общую надёжность по всем поставщикам."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-2")],
        orders=[
            _order("PO-1", "SKU-A", "closed", supplier_id="SUP-1", ordered=100, received=50),
            _order("PO-2", "SKU-B", "open", supplier_id="SUP-2", ordered=100, received=0, pending=100),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1"), _supplier("SUP-2")],
    )

    result = supplier_reliability(ds).set_index("supplier_id")

    assert result.loc["SUP-2", "reliability"] == pytest.approx(0.5)  # общая = 50/100
    assert result.loc["SUP-2", "orders_n"] == 0
    assert result.loc["SUP-2", "ordered"] == 0


@needs_raw
def test_supplier_reliability_on_real_data():
    """Реальные данные: общая надёжность ≈ 0.39, как в брифе."""
    ds = load_dataset(RAW_DIR, pd.Timestamp("2026-08-13"))

    result = supplier_reliability(ds)

    settled = ds.orders[ds.orders["status"].isin(["closed", "completed"])]
    assert settled["qty_received"].sum() / settled["qty_ordered"].sum() == pytest.approx(0.39, abs=0.04)
    # каждый поставщик из справочника получил надёжность
    assert set(ds.suppliers["supplier_id"]) <= set(result["supplier_id"])
    assert result["reliability"].between(0.0, 1.0).all()


# --- заказы в пути ---

def test_open_orders_expected_qty_and_overdue(make_dataset_factory):
    """Только open с qty_pending > 0; ожидаемое = pending × надёжность; просрочка сдвинута."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A", "SUP-1"), _sku("SKU-B", "SUP-2")],
        orders=[
            # надёжность SUP-1 = 80/100 = 0.8
            _order("PO-1", "SKU-A", "closed", supplier_id="SUP-1", ordered=100, received=80),
            # в пути, придёт в будущем
            _order("PO-2", "SKU-A", "open", supplier_id="SUP-1", pending=50,
                   promised_date="2026-09-01"),
            # просрочен: обещано раньше today
            _order("PO-3", "SKU-A", "open", supplier_id="SUP-1", pending=25,
                   promised_date="2026-07-01"),
            # pending = 0 — не в пути
            _order("PO-4", "SKU-A", "open", supplier_id="SUP-1", pending=0,
                   promised_date="2026-09-01"),
            # закрытый заказ в пути не считается
            _order("PO-5", "SKU-B", "closed", supplier_id="SUP-2", ordered=10, received=0,
                   pending=10),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1"), _supplier("SUP-2")],
        today="2026-08-13",
    )
    reliability = supplier_reliability(ds)

    result = open_orders(ds, reliability, Config())
    result = result.set_index("po_id")

    assert sorted(result.index) == ["PO-2", "PO-3"]
    assert result.loc["PO-2", "expected_qty"] == pytest.approx(40.0)  # 50 × 0.8
    assert not bool(result.loc["PO-2", "overdue"])
    assert result.loc["PO-2", "promised_date"] == pd.Timestamp("2026-09-01")
    assert result.loc["PO-2", "expected_date"] == pd.Timestamp("2026-09-01")

    assert bool(result.loc["PO-3", "overdue"])
    assert result.loc["PO-3", "expected_qty"] == pytest.approx(20.0)  # 25 × 0.8
    assert result.loc["PO-3", "expected_date"] == pd.Timestamp("2026-09-12")  # today + 30 дней
    assert result.loc["PO-3", "qty_pending"] == 25


def test_open_orders_without_promised_date_is_overdue(make_dataset_factory):
    """Нет даты прихода — как просрочка: приход сегодня + сдвиг, overdue."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A")],
        orders=[_order("PO-1", "SKU-A", "open", pending=40, promised_date=None)],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1")],
        today="2026-08-13",
    )
    row = open_orders(ds, supplier_reliability(ds), Config()).set_index("po_id").loc["PO-1"]
    assert bool(row["overdue"])
    assert row["expected_date"] == pd.Timestamp("2026-09-12")  # today + 30 дней
    assert not bool(row["stale"])  # без даты срок просрочки не известен


def test_open_orders_stale_overdue_not_in_arrivals(make_dataset_factory):
    """Просрочка 91 дн. (> 90) — не приход: expected_qty 0, даты нет; 89 дн. — приход через 30 дн."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A")],
        orders=[
            _order("PO-OLD", "SKU-A", "open", pending=40, order_date="2025-12-01", promised_date="2026-05-14"),
            _order("PO-NEW", "SKU-A", "open", pending=30, order_date="2025-12-01", promised_date="2026-05-16"),
        ],
        sales=[], stock=[], receipts=[],
        suppliers=[_supplier("SUP-1")],
        today="2026-08-13",
    )
    result = open_orders(ds, supplier_reliability(ds), Config()).set_index("po_id")
    old, new = result.loc["PO-OLD"], result.loc["PO-NEW"]
    assert old["overdue_days"] == 91 and bool(old["overdue"]) and bool(old["stale"])
    assert old["expected_qty"] == 0 and pd.isna(old["expected_date"])
    assert old["qty_pending"] == 40
    assert new["overdue_days"] == 89 and bool(new["overdue"]) and not bool(new["stale"])
    assert new["expected_qty"] > 0 and new["expected_date"] == pd.Timestamp("2026-09-12")


@needs_raw
def test_open_orders_on_real_data():
    """Реальные данные: 31 заказ в пути на 20 039 шт."""
    ds = load_dataset(RAW_DIR, pd.Timestamp("2026-08-13"))

    result = open_orders(ds, supplier_reliability(ds), Config())

    assert len(result) == 31
    assert int(result["qty_pending"].sum()) == 20039
    assert (result["expected_qty"] <= result["qty_pending"]).all()
    assert (result["expected_qty"] >= 0).all()


# --- кратность партии ---

def test_lot_size_gcd_of_receipts(make_dataset_factory):
    """Приходы 200/400/600 → кратность 200."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A"), _sku("SKU-B"), _sku("SKU-C")],
        orders=[], sales=[], stock=[],
        receipts=[
            _receipt("SKU-A", "2024-05-13", 200),
            _receipt("SKU-A", "2025-05-13", 400),
            _receipt("SKU-A", "2026-05-13", 600),
            # два прихода — меньше lot_min_receipts
            _receipt("SKU-B", "2025-05-13", 300),
            _receipt("SKU-B", "2026-05-13", 300),
            # три прихода, но НОД = 1
            _receipt("SKU-C", "2024-05-13", 7),
            _receipt("SKU-C", "2025-05-13", 11),
            _receipt("SKU-C", "2026-05-13", 13),
        ],
        suppliers=[_supplier("SUP-1")],
    )

    result = lot_size(ds, Config())

    assert result.loc["SKU-A"] == 200
    assert result.loc["SKU-B"] == 1
    assert result.loc["SKU-C"] == 1
    assert result.dtype == "int64"


def test_lot_size_respects_min_receipts_config(make_dataset_factory):
    """Порог числа приходов берётся из Config, а не из кода."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A")],
        orders=[], sales=[], stock=[],
        receipts=[
            _receipt("SKU-A", "2024-05-13", 300),
            _receipt("SKU-A", "2025-05-13", 300),
        ],
        suppliers=[_supplier("SUP-1")],
    )

    assert lot_size(ds, Config()).loc["SKU-A"] == 1
    assert lot_size(ds, Config(lot_min_receipts=2)).loc["SKU-A"] == 300


# --- остаток на сегодня ---

def test_current_stock_partial_month(make_dataset_factory):
    """Остаток на 1-е число месяца today − продано + возвращено + приходы; не меньше нуля."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A"), _sku("SKU-B"), _sku("SKU-C")],
        sales=[
            {"sku_id": "SKU-A", "category": "sanitary_faucets", "month": "2026-08",
             "qty_sold": 40, "qty_returned": 10, "partial_month": 1},
            # возврат больше продаж → остаток растёт
            {"sku_id": "SKU-B", "category": "sanitary_faucets", "month": "2026-08",
             "qty_sold": 5, "qty_returned": 20, "partial_month": 1},
            # остатка 100, спрос 30, приходов нет → 70
            {"sku_id": "SKU-C", "category": "sanitary_faucets", "month": "2026-08",
             "qty_sold": 30, "qty_returned": 0, "partial_month": 1},
        ],
        stock=[
            {"sku_id": "SKU-A", "month": "2026-08", "stock_qty": 100},
            {"sku_id": "SKU-B", "month": "2026-08", "stock_qty": 5},
            {"sku_id": "SKU-C", "month": "2026-08", "stock_qty": 100},
        ],
        receipts=[
            {"sku_id": "SKU-A", "receipt_date": "2026-08-05", "qty": 25, "supplier_id": "SUP-1"},
            # приход прошлого месяца в остаток не входит
            {"sku_id": "SKU-A", "receipt_date": "2026-07-20", "qty": 500, "supplier_id": "SUP-1"},
            # приход позже today не входит
            {"sku_id": "SKU-A", "receipt_date": "2026-08-20", "qty": 900, "supplier_id": "SUP-1"},
        ],
        orders=[],
        suppliers=[_supplier("SUP-1")],
        today="2026-08-13",
    )

    result = current_stock(ds)

    # 100 − (40 − 10) + 25 = 95
    assert result.loc["SKU-A"] == pytest.approx(95.0)
    # 5 − 5 + 20 = 20
    assert result.loc["SKU-B"] == pytest.approx(20.0)
    # 100 − 30 = 70
    assert result.loc["SKU-C"] == pytest.approx(70.0)


def test_current_stock_returns_above_sales(make_dataset_factory):
    """Возвраты сверх продаж прибавляются к остатку: 100 − 0 + 20 = 120."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A")],
        sales=[{"sku_id": "SKU-A", "category": "sanitary_faucets", "month": "2026-08",
                "qty_sold": 0, "qty_returned": 20, "partial_month": 1}],
        stock=[{"sku_id": "SKU-A", "month": "2026-08", "stock_qty": 100}],
        receipts=[],
        orders=[],
        suppliers=[_supplier("SUP-1")],
        today="2026-08-13",
    )
    assert current_stock(ds).loc["SKU-A"] == pytest.approx(120.0)


def test_current_stock_missing_row_is_zero(make_dataset_factory):
    """Нет строки остатка за месяц today — остаток 0; отрицательный результат обрезан до 0."""
    ds = make_dataset_factory(
        skus=[_sku("SKU-A"), _sku("SKU-B")],
        sales=[
            {"sku_id": "SKU-A", "category": "sanitary_faucets", "month": "2026-08",
             "qty_sold": 10, "qty_returned": 0, "partial_month": 1},
            {"sku_id": "SKU-B", "category": "sanitary_faucets", "month": "2026-08",
             "qty_sold": 80, "qty_returned": 0, "partial_month": 1},
        ],
        stock=[
            # строки SKU-A за 2026-08 нет
            {"sku_id": "SKU-B", "month": "2026-08", "stock_qty": 50},
        ],
        receipts=[], orders=[],
        suppliers=[_supplier("SUP-1")],
        today="2026-08-13",
    )

    result = current_stock(ds)

    assert result.loc["SKU-A"] == pytest.approx(0.0)
    assert result.loc["SKU-B"] == pytest.approx(0.0)  # 50 − 80 → 0


@needs_raw
def test_current_stock_on_real_data():
    """Реальные данные: остаток неотрицателен и не больше снимка плюс приходы месяца."""
    ds = load_dataset(RAW_DIR, pd.Timestamp("2026-08-13"))

    result = current_stock(ds)

    assert len(result) == len(ds.sku)
    assert result.notna().all()
    assert (result >= 0).all()
    month = ds.today.to_period("M")
    snapshot = ds.stock[ds.stock["month"] == month].set_index("sku_id")["stock_qty"]
    receipts = ds.receipts[ds.receipts["receipt_date"] <= ds.today].groupby("sku_id")["qty"].sum()
    upper = snapshot.reindex(result.index).fillna(0) + receipts.reindex(result.index).fillna(0)
    assert (result <= upper).all()
