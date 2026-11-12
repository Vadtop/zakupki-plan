"""Синтетический набор данных для тестов: маленький, детерминированный, без файлов."""

import urllib.request
from collections.abc import Callable, Iterable

import pandas as pd
import pytest

from zakupki.data import Dataset, detect_data_date

TODAY = "2026-08-13"
LLM_VARS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


@pytest.fixture(autouse=True)
def _no_llm_no_network(monkeypatch):
    """Ни один тест не берёт ключи модели из окружения и не выходит в сеть."""
    for key in LLM_VARS:
        monkeypatch.delenv(key, raising=False)

    def blocked(*args, **kwargs):
        raise RuntimeError("тесты не ходят в сеть")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)


# Категории и SKU набора по умолчанию.
SKU_A = "SKU-A"  # сезонный: продажи почти только в октябре
SKU_B = "SKU-B"  # ровный спрос
SKU_C = "SKU-C"  # мёртвый: продаж в последние 12 месяцев нет


def _periods(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def _default_skus() -> list[dict]:
    return [
        {
            "sku_id": SKU_A,
            "category": "space_heaters",
            "brand_id": "BRD-1",
            "supplier_id": "SUP-1",
            "supply_type": "import",
            "uom": "pcs",
            "weight_kg": 4.5,
            "warranty_months": 24,
            "target_cover_days": 0,
            "price_decile": 7.0,
            "first_sale_month": "2023-01",
            "last_sale_month": "2026-08",
            "months_with_sales": 44,
            "qty_total": 4400,
            "avg_markup_pct_12m": 45.0,
            "on_order_qty": 300,
            "created": "2022-11",
        },
        {
            "sku_id": SKU_B,
            "category": "sanitary_faucets",
            "brand_id": "BRD-2",
            "supplier_id": "SUP-2",
            "supply_type": "import",
            "uom": "pcs",
            "weight_kg": 1.2,
            "warranty_months": 0,
            "target_cover_days": 60,
            "price_decile": 3.0,
            "first_sale_month": "2023-01",
            "last_sale_month": "2026-08",
            "months_with_sales": 44,
            "qty_total": 2200,
            "avg_markup_pct_12m": 30.0,
            "on_order_qty": 0,
            "created": "2022-12",
        },
        {
            "sku_id": SKU_C,
            "category": "string_trimmers",
            "brand_id": "BRD-3",
            "supplier_id": "SUP-1",
            "supply_type": "unknown",
            "uom": "set",
            "weight_kg": None,
            "warranty_months": 0,
            "target_cover_days": 0,
            "price_decile": None,
            "first_sale_month": "2023-01",
            "last_sale_month": "2024-06",
            "months_with_sales": 18,
            "qty_total": 900,
            "avg_markup_pct_12m": None,
            "on_order_qty": 0,
            "created": "2022-10",
        },
    ]


def _default_sales() -> list[dict]:
    rows: list[dict] = []
    for month in _periods("2023-01", "2026-08"):
        partial = 1 if month == pd.Period("2026-08", freq="M") else 0
        if month <= pd.Period("2024-06", freq="M"):
            sold_c = 50
        else:
            sold_c = 0
        rows.append(
            {
                "sku_id": SKU_A,
                "category": "space_heaters",
                "month": month,
                "qty_sold": 100 if month.month == 10 else 5,
                "qty_returned": 0,
                "partial_month": partial,
            }
        )
        rows.append(
            {
                "sku_id": SKU_B,
                "category": "sanitary_faucets",
                "month": month,
                "qty_sold": 50,
                "qty_returned": 2,
                "partial_month": partial,
            }
        )
        rows.append(
            {
                "sku_id": SKU_C,
                "category": "string_trimmers",
                "month": month,
                "qty_sold": sold_c,
                "qty_returned": 0,
                "partial_month": partial,
            }
        )
    return rows


def _default_stock() -> list[dict]:
    rows: list[dict] = []
    for month in _periods("2024-01", "2026-08"):
        rows.append({"sku_id": SKU_A, "month": month, "stock_qty": 200})
        rows.append({"sku_id": SKU_B, "month": month, "stock_qty": 120})
        if month <= pd.Period("2024-06", freq="M"):
            rows.append({"sku_id": SKU_C, "month": month, "stock_qty": 30})
    return rows


def _default_orders() -> list[dict]:
    return [
        {
            "po_id": "PO-1",
            "sku_id": SKU_A,
            "supplier_id": "SUP-1",
            "supply_type": "import",
            "order_date": "2026-05-01",
            "promised_date": "2026-09-01",
            "planned_lead_days": 123,
            "status": "open",
            "qty_ordered": 500,
            "qty_confirmed": 500,
            "qty_received": 0,
            "qty_pending": 300,
        },
        {
            "po_id": "PO-2",
            "sku_id": SKU_A,
            "supplier_id": "SUP-1",
            "supply_type": "import",
            "order_date": "2025-01-10",
            "promised_date": "2025-05-13",
            "planned_lead_days": 123,
            "status": "closed",
            "qty_ordered": 400,
            "qty_confirmed": 400,
            "qty_received": 380,
            "qty_pending": 20,
        },
        {
            "po_id": "PO-3",
            "sku_id": SKU_B,
            "supplier_id": "SUP-2",
            "supply_type": "import",
            "order_date": "2026-03-01",
            "promised_date": "2026-07-01",
            "planned_lead_days": 122,
            "status": "completed",
            "qty_ordered": 200,
            "qty_confirmed": 200,
            "qty_received": 200,
            "qty_pending": 0,
        },
        {
            "po_id": "PO-4",
            "sku_id": SKU_B,
            "supplier_id": "SUP-2",
            "supply_type": "import",
            "order_date": "2024-07-28",
            "promised_date": "2024-07-15",
            "planned_lead_days": -13,
            "status": "completed",
            "qty_ordered": 100,
            "qty_confirmed": 100,
            "qty_received": 100,
            "qty_pending": 0,
        },
    ]


def _default_receipts() -> list[dict]:
    return [
        {"sku_id": SKU_A, "receipt_date": "2024-05-13", "qty": 200, "supplier_id": "SUP-1"},
        {"sku_id": SKU_A, "receipt_date": "2025-05-13", "qty": 400, "supplier_id": "SUP-1"},
        {"sku_id": SKU_A, "receipt_date": "2026-08-01", "qty": 600, "supplier_id": "SUP-1"},
        {"sku_id": SKU_B, "receipt_date": "2025-06-01", "qty": 300, "supplier_id": "SUP-2"},
    ]


def _default_suppliers() -> list[dict]:
    return [
        {
            "supplier_id": "SUP-1",
            "supply_type": "import",
            "receipts_cnt": 3,
            "skus": 2,
            "median_days_between_receipts": 70.0,
            "median_planned_lead_days": 123.0,
        },
        {
            "supplier_id": "SUP-2",
            "supply_type": "import",
            "receipts_cnt": 1,
            "skus": 1,
            "median_days_between_receipts": None,
            "median_planned_lead_days": 101.0,
        },
    ]


def _frame(rows: list[dict], dtypes: dict[str, str]) -> pd.DataFrame:
    """Собирает DataFrame из списка словарей с заданными типами колонок."""
    columns = list(dtypes)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
    df = pd.DataFrame(rows, columns=columns)
    for column, dtype in dtypes.items():
        df[column] = df[column].astype(dtype)
    return df


def _to_period(series: pd.Series) -> pd.Series:
    """Месяц: строка 'YYYY-MM' или Period → Period[M]."""
    if isinstance(series.dtype, pd.PeriodDtype):
        return series
    return pd.to_datetime(series.astype("string"), format="%Y-%m", errors="coerce").dt.to_period("M")


def _to_timestamp(series: pd.Series) -> pd.Series:
    """Дата: строка или Timestamp → Timestamp без времени."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()
    return pd.to_datetime(series.astype("string"), errors="coerce").dt.normalize()


def _build_sales(rows: list[dict]) -> pd.DataFrame:
    df = _frame(
        rows,
        {
            "sku_id": "string",
            "category": "string",
            "month": "object",
            "qty_sold": "int64",
            "qty_returned": "int64",
            "partial_month": "int64",
        },
    )
    df["month"] = _to_period(df["month"])
    return df


def _build_stock(rows: list[dict]) -> pd.DataFrame:
    df = _frame(rows, {"sku_id": "string", "month": "object", "stock_qty": "int64"})
    df["month"] = _to_period(df["month"])
    return df


def _build_orders(rows: list[dict]) -> pd.DataFrame:
    df = _frame(
        rows,
        {
            "po_id": "string",
            "sku_id": "string",
            "supplier_id": "string",
            "supply_type": "string",
            "order_date": "object",
            "promised_date": "object",
            "planned_lead_days": "int64",
            "status": "string",
            "qty_ordered": "int64",
            "qty_confirmed": "int64",
            "qty_received": "int64",
            "qty_pending": "int64",
        },
    )
    df["order_date"] = _to_timestamp(df["order_date"])
    df["promised_date"] = _to_timestamp(df["promised_date"])
    return df


def _build_receipts(rows: list[dict]) -> pd.DataFrame:
    df = _frame(
        rows,
        {
            "sku_id": "string",
            "receipt_date": "object",
            "qty": "int64",
            "supplier_id": "string",
        },
    )
    df["receipt_date"] = _to_timestamp(df["receipt_date"])
    return df


def _build_sku(rows: list[dict]) -> pd.DataFrame:
    return _frame(
        rows,
        {
            "sku_id": "string",
            "category": "string",
            "brand_id": "string",
            "supplier_id": "string",
            "supply_type": "string",
            "uom": "string",
            "weight_kg": "float64",
            "warranty_months": "int64",
            "target_cover_days": "int64",
            "price_decile": "float64",
            "first_sale_month": "string",
            "last_sale_month": "string",
            "months_with_sales": "int64",
            "qty_total": "int64",
            "avg_markup_pct_12m": "float64",
            "on_order_qty": "int64",
            "created": "string",
        },
    )


def _build_suppliers(rows: list[dict]) -> pd.DataFrame:
    return _frame(
        rows,
        {
            "supplier_id": "string",
            "supply_type": "string",
            "receipts_cnt": "int64",
            "skus": "int64",
            "median_days_between_receipts": "float64",
            "median_planned_lead_days": "float64",
        },
    )


def make_dataset(
    *,
    skus: list[dict] | None = None,
    sales: list[dict] | None = None,
    stock: list[dict] | None = None,
    orders: list[dict] | None = None,
    receipts: list[dict] | None = None,
    suppliers: list[dict] | None = None,
    today: str | pd.Timestamp | None = TODAY,
    issues: Iterable[str] | None = None,
) -> Dataset:
    """Собирает Dataset из маленьких списков словарей; файлы не читает.

    Любой список можно заменить своим; today=None — дата берётся из данных.
    """
    ds_sales = _build_sales(sales if sales is not None else _default_sales())
    ds_stock = _build_stock(stock if stock is not None else _default_stock())
    ds_orders = _build_orders(orders if orders is not None else _default_orders())
    ds_receipts = _build_receipts(receipts if receipts is not None else _default_receipts())
    ds_today = (
        detect_data_date(ds_orders, ds_receipts, ds_sales)
        if today is None
        else pd.Timestamp(today).normalize()
    )
    # Как load_dataset: строки позже даты расчёта в данные не попадают.
    month = ds_today.to_period("M")
    ds_sales = ds_sales[ds_sales["month"] <= month].reset_index(drop=True)
    ds_stock = ds_stock[ds_stock["month"] <= month].reset_index(drop=True)
    ds_orders = ds_orders[ds_orders["order_date"] <= ds_today].reset_index(drop=True)
    ds_receipts = ds_receipts[ds_receipts["receipt_date"] <= ds_today].reset_index(drop=True)
    return Dataset(
        sku=_build_sku(skus if skus is not None else _default_skus()),
        sales=ds_sales,
        stock=ds_stock,
        orders=ds_orders,
        receipts=ds_receipts,
        suppliers=_build_suppliers(suppliers if suppliers is not None else _default_suppliers()),
        today=ds_today,
        issues=list(issues) if issues is not None else [],
    )


@pytest.fixture
def make_dataset_factory() -> Callable[..., Dataset]:
    """Фабрика набора: make_dataset(sales=[...], today=...) → Dataset."""
    return make_dataset


@pytest.fixture
def small_dataset() -> Dataset:
    """Набор по умолчанию: три SKU, 44 месяца продаж, остатки с 2024-01."""
    return make_dataset()
