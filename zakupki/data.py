"""Загрузка и проверка выгрузки ERP: единственное место, где читаются файлы."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Имена файлов выгрузки.
SKU_FILE = "sku_dim.csv"
SALES_FILE = "sales_monthly.csv"
STOCK_FILE = "stock_monthly.csv"
ORDERS_FILE = "purchase_orders.csv"
RECEIPTS_FILE = "receipts.csv"
SUPPLIERS_FILE = "supplier_dim.csv"

# Обязательные колонки каждого файла.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    SKU_FILE: (
        "sku_id", "category", "brand_id", "supplier_id", "supply_type", "uom", "weight_kg",
        "warranty_months", "target_cover_days", "price_decile", "first_sale_month", "last_sale_month",
        "months_with_sales", "qty_total", "avg_markup_pct_12m", "on_order_qty", "created",
    ),
    SALES_FILE: (
        "sku_id", "category", "month", "qty_sold", "qty_returned", "orders_cnt", "markup_pct",
        "partial_month",
    ),
    STOCK_FILE: ("sku_id", "month", "stock_qty"),
    ORDERS_FILE: (
        "po_id", "sku_id", "supplier_id", "supply_type", "order_date", "promised_date",
        "planned_lead_days", "status", "qty_ordered", "qty_confirmed", "qty_received", "qty_pending",
    ),
    RECEIPTS_FILE: ("sku_id", "receipt_date", "qty", "supplier_id", "supply_type"),
    SUPPLIERS_FILE: (
        "supplier_id", "supply_type", "receipts_cnt", "skus", "median_days_between_receipts",
        "median_planned_lead_days",
    ),
}

# Числовые колонки: непустое нечисловое значение — строка отбрасывается.
SKU_FLOATS = ("weight_kg", "price_decile", "avg_markup_pct_12m")
SKU_INTS = ("warranty_months", "target_cover_days", "months_with_sales", "qty_total", "on_order_qty")
SALES_INTS = ("qty_sold", "qty_returned", "partial_month")
ORDERS_INTS = ("planned_lead_days", "qty_ordered", "qty_confirmed", "qty_received", "qty_pending")
SUPPLIER_FLOATS = ("median_days_between_receipts", "median_planned_lead_days")
SUPPLIER_INTS = ("receipts_cnt", "skus")


class DataError(Exception):
    """Ошибка данных: текст показывается пользователю как есть."""


@dataclass
class Dataset:
    """Выгрузка ERP, приведённая к типам и обрезанная по дате расчёта."""

    sku: pd.DataFrame
    sales: pd.DataFrame
    stock: pd.DataFrame
    orders: pd.DataFrame
    receipts: pd.DataFrame
    suppliers: pd.DataFrame
    today: pd.Timestamp
    issues: list[str] = field(default_factory=list)


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    """Читает файл выгрузки, проверяя его наличие и обязательные колонки."""
    path = data_dir / name
    if not path.is_file():
        raise DataError(f"Нет файла {name} в {data_dir}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise DataError(f"В файле {name} нет колонки: {', '.join(missing)}")
    return df


def _note(issues: list[str], name: str, reason: str, n: int) -> None:
    """Записывает в issues, сколько строк отброшено и почему."""
    if n:
        issues.append(f"{name}: отброшено строк {n} — {reason}")


def _drop_non_numeric(
    df: pd.DataFrame, name: str, columns: tuple[str, ...], issues: list[str]
) -> pd.DataFrame:
    """Убирает строки с непустым нечисловым значением; пустая ячейка допустима."""
    keep = pd.Series(True, index=df.index)
    for c in columns:
        text = df[c].astype("string").str.strip()
        filled = (text.notna() & (text != "")).fillna(False).astype(bool)
        bad = filled & pd.to_numeric(df[c], errors="coerce").isna() & keep
        _note(issues, name, f"нечисловое значение в {c}", int(bad.sum()))
        keep &= ~bad
    return df[keep]


def _to_int(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Числовые колонки приводит к int64, пустое значение — 0."""
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")


def _to_float(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Числовые колонки приводит к float64, пустое значение — NaN."""
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")


def _drop_unknown_sku(
    df: pd.DataFrame, name: str, sku_ids: set[str], issues: list[str]
) -> pd.DataFrame:
    """Убирает строки с sku_id, которого нет в справочнике."""
    bad = ~df["sku_id"].isin(sku_ids)
    _note(issues, name, "неизвестный sku_id", int(bad.sum()))
    return df[~bad]


def _prepare_sku(raw: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    """Справочник товаров как есть; наценка — float, пусто = NaN."""
    sku = _drop_non_numeric(raw.copy(), SKU_FILE, SKU_FLOATS + SKU_INTS, issues)
    for c in ("sku_id", "category", "brand_id", "supplier_id", "supply_type", "uom",
              "first_sale_month", "last_sale_month", "created"):
        sku[c] = sku[c].astype("string")
    _to_float(sku, SKU_FLOATS)
    _to_int(sku, SKU_INTS)
    bad = sku["sku_id"].isna()
    _note(issues, SKU_FILE, "пустой sku_id", int(bad.sum()))
    return sku[~bad].reset_index(drop=True)


def _prepare_sales(raw: pd.DataFrame, sku_ids: set[str], issues: list[str]) -> pd.DataFrame:
    """Продажи помесячно: месяц — Period[M], количества — int64."""
    sales = _drop_non_numeric(raw.copy(), SALES_FILE, SALES_INTS, issues)
    sales["sku_id"] = sales["sku_id"].astype("string")
    sales["category"] = sales["category"].astype("string")
    sales["month"] = pd.to_datetime(sales["month"], format="%Y-%m", errors="coerce").dt.to_period("M")
    _to_int(sales, SALES_INTS)
    bad_month = sales["month"].isna()
    _note(issues, SALES_FILE, "неразбираемый месяц", int(bad_month.sum()))
    sales = sales[~bad_month]
    sales = _drop_unknown_sku(sales, SALES_FILE, sku_ids, issues)
    return sales[["sku_id", "category", "month", "qty_sold", "qty_returned", "partial_month"]].reset_index(
        drop=True
    )


def _prepare_stock(raw: pd.DataFrame, sku_ids: set[str], issues: list[str]) -> pd.DataFrame:
    """Остатки на 1-е число: месяц — Period[M]."""
    stock = _drop_non_numeric(raw.copy(), STOCK_FILE, ("stock_qty",), issues)
    stock["sku_id"] = stock["sku_id"].astype("string")
    stock["month"] = pd.to_datetime(stock["month"], format="%Y-%m", errors="coerce").dt.to_period("M")
    _to_int(stock, ("stock_qty",))
    bad_month = stock["month"].isna()
    _note(issues, STOCK_FILE, "неразбираемый месяц", int(bad_month.sum()))
    stock = stock[~bad_month]
    stock = _drop_unknown_sku(stock, STOCK_FILE, sku_ids, issues)
    stock = stock[["sku_id", "month", "stock_qty"]]
    exact = stock.duplicated()
    if exact.any():
        issues.append(f"{STOCK_FILE}: повторов снимка {int(exact.sum())} — оставлен один")
        stock = stock[~exact]
    clash = stock.duplicated(subset=["sku_id", "month"], keep=False)
    if clash.any():
        first = stock[clash].iloc[0]
        raise DataError(
            f"В файле {STOCK_FILE} разные остатки для {first['sku_id']} за {first['month']}"
        )
    return stock.reset_index(drop=True)


def _prepare_orders(raw: pd.DataFrame, sku_ids: set[str], issues: list[str]) -> pd.DataFrame:
    """Заказы поставщикам: даты — Timestamp, количества — int64."""
    orders = _drop_non_numeric(raw.copy(), ORDERS_FILE, ORDERS_INTS, issues)
    orders["po_id"] = orders["po_id"].astype("string")
    orders["sku_id"] = orders["sku_id"].astype("string")
    orders["supplier_id"] = orders["supplier_id"].astype("string")
    orders["status"] = orders["status"].astype("string")
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.normalize()
    orders["promised_date"] = pd.to_datetime(orders["promised_date"], errors="coerce").dt.normalize()
    _to_int(orders, ORDERS_INTS)
    bad_date = orders["order_date"].isna()
    _note(issues, ORDERS_FILE, "неразбираемая дата заказа", int(bad_date.sum()))
    orders = orders[~bad_date]
    return _drop_unknown_sku(orders, ORDERS_FILE, sku_ids, issues).reset_index(drop=True)


def _prepare_receipts(raw: pd.DataFrame, sku_ids: set[str], issues: list[str]) -> pd.DataFrame:
    """Приходы на склад: дата — Timestamp."""
    receipts = _drop_non_numeric(raw.copy(), RECEIPTS_FILE, ("qty",), issues)
    receipts["sku_id"] = receipts["sku_id"].astype("string")
    receipts["supplier_id"] = receipts["supplier_id"].astype("string")
    receipts["receipt_date"] = pd.to_datetime(receipts["receipt_date"], errors="coerce").dt.normalize()
    _to_int(receipts, ("qty",))
    bad_date = receipts["receipt_date"].isna()
    _note(issues, RECEIPTS_FILE, "неразбираемая дата прихода", int(bad_date.sum()))
    receipts = receipts[~bad_date]
    receipts = _drop_unknown_sku(receipts, RECEIPTS_FILE, sku_ids, issues)
    return receipts[["sku_id", "receipt_date", "qty", "supplier_id"]].reset_index(drop=True)


def _prepare_suppliers(raw: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    """Справочник поставщиков как есть."""
    suppliers = _drop_non_numeric(raw.copy(), SUPPLIERS_FILE, SUPPLIER_FLOATS + SUPPLIER_INTS, issues)
    suppliers["supplier_id"] = suppliers["supplier_id"].astype("string")
    suppliers["supply_type"] = suppliers["supply_type"].astype("string")
    _to_float(suppliers, SUPPLIER_FLOATS)
    _to_int(suppliers, SUPPLIER_INTS)
    return suppliers.reset_index(drop=True)


def detect_data_date(
    orders: pd.DataFrame, receipts: pd.DataFrame, sales: pd.DataFrame
) -> pd.Timestamp:
    """Дата данных — последняя из receipt_date и order_date; иначе конец последнего месяца продаж."""
    dates: list[pd.Timestamp] = []
    if receipts is not None and len(receipts):
        dates.append(pd.Timestamp(receipts["receipt_date"].max()))
    if orders is not None and len(orders):
        dates.append(pd.Timestamp(orders["order_date"].max()))
    dates = [d for d in dates if pd.notna(d)]
    if dates:
        return max(dates).normalize()
    if sales is not None and len(sales):
        last = sales["month"].max()
        if pd.notna(last):
            return pd.Period(last, freq="M").to_timestamp(how="end").normalize()
    raise DataError("Не удалось определить дату данных: нет приходов, заказов и продаж")


def load_dataset(data_dir: Path, today: pd.Timestamp | None) -> Dataset:
    """Читает выгрузку, проверяет её и обрезает данные позже даты расчёта."""
    data_dir = Path(data_dir)
    raw_sku = _read_csv(data_dir, SKU_FILE)
    raw_sales = _read_csv(data_dir, SALES_FILE)
    raw_stock = _read_csv(data_dir, STOCK_FILE)
    raw_orders = _read_csv(data_dir, ORDERS_FILE)
    raw_receipts = _read_csv(data_dir, RECEIPTS_FILE)
    raw_suppliers = _read_csv(data_dir, SUPPLIERS_FILE)

    issues: list[str] = []
    sku = _prepare_sku(raw_sku, issues)
    sku_ids = set(sku["sku_id"].tolist())
    sales = _prepare_sales(raw_sales, sku_ids, issues)
    stock = _prepare_stock(raw_stock, sku_ids, issues)
    orders = _prepare_orders(raw_orders, sku_ids, issues)
    receipts = _prepare_receipts(raw_receipts, sku_ids, issues)
    suppliers = _prepare_suppliers(raw_suppliers, issues)

    data_date = pd.Timestamp(today).normalize() if today is not None else detect_data_date(
        orders, receipts, sales
    )
    month = data_date.to_period("M")
    sales = sales[sales["month"] <= month].reset_index(drop=True)
    stock = stock[stock["month"] <= month].reset_index(drop=True)
    orders = orders[orders["order_date"] <= data_date].reset_index(drop=True)
    receipts = receipts[receipts["receipt_date"] <= data_date].reset_index(drop=True)
    no_date = (orders["status"] == "open") & (orders["qty_pending"] > 0) & orders["promised_date"].isna()
    if no_date.any():
        issues.append(
            f"{ORDERS_FILE}: у {int(no_date.sum())} открытых заказов нет даты прихода — "
            "считаются просроченными"
        )

    return Dataset(
        sku=sku,
        sales=sales,
        stock=stock,
        orders=orders,
        receipts=receipts,
        suppliers=suppliers,
        today=data_date,
        issues=issues,
    )
