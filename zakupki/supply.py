"""Поставки: плечо SKU, надёжность поставщика, открытые заказы, партия, остаток."""

import numpy as np
import pandas as pd

from zakupki.config import Config
from zakupki.data import Dataset

# Статусы заказов, закрытые для расчёта надёжности.
SETTLED_STATUSES = ("closed", "completed")


def _supplier_lead(ds: Dataset) -> pd.Series:
    """Медиана планового плеча поставщика из справочника: supplier_id → дней."""
    if ds.suppliers.empty:
        return pd.Series(dtype="float64")
    sup = ds.suppliers.drop_duplicates(subset="supplier_id").set_index("supplier_id")
    return sup["median_planned_lead_days"].astype("float64")


def lead_days(ds: Dataset, cfg: Config) -> pd.Series:
    """Плечо поставки по SKU: медиана положительных заказов, иначе поставщик, иначе дефолт.

    Ограничено cfg.lead_min_days..cfg.lead_max_days. Индекс — sku_id, значения int64.
    """
    sku_ids = ds.sku["sku_id"].astype("string")
    orders = ds.orders
    by_sku = pd.Series(np.nan, index=pd.Index(sku_ids, name="sku_id"), dtype="float64")

    if len(orders):
        positive = orders[orders["planned_lead_days"] > 0]
        if len(positive):
            medians = positive.groupby("sku_id")["planned_lead_days"].median()
            by_sku = by_sku.combine_first(medians.reindex(by_sku.index))

    supplier_of_sku = ds.sku.set_index("sku_id")["supplier_id"]
    sup_lead = _supplier_lead(ds)
    by_supplier = supplier_of_sku.map(sup_lead).reindex(by_sku.index)
    by_sku = by_sku.combine_first(by_supplier)

    by_sku = by_sku.fillna(float(cfg.lead_default_days))
    by_sku = by_sku.clip(lower=cfg.lead_min_days, upper=cfg.lead_max_days)
    return by_sku.round().astype("int64")


def supplier_reliability(ds: Dataset, cfg: Config | None = None) -> pd.DataFrame:
    """Надёжность поставщика: доля поставленного по закрытым и исполненным заказам.

    Колонки: supplier_id, reliability, orders_n, received, ordered. Поставщик без
    таких заказов получает общую надёжность по всем поставщикам.
    """
    suppliers = ds.suppliers["supplier_id"].astype("string") if len(ds.suppliers) else pd.Series(
        dtype="string"
    )
    if len(ds.orders):
        suppliers = pd.concat([suppliers, ds.orders["supplier_id"].astype("string")])
    ids = pd.Index(sorted(set(suppliers.dropna().tolist())), name="supplier_id")

    settled = ds.orders[ds.orders["status"].isin(SETTLED_STATUSES)] if len(ds.orders) else ds.orders
    cfg = cfg or Config()
    if cfg.reliability_skip_zero_closed and len(settled):
        # closed с нулём получено — учёт или отмена, а не недопоставка.
        zero_closed = (settled["status"] == "closed") & (settled["qty_received"] == 0)
        settled = settled[~zero_closed]
    totals = settled.groupby("supplier_id")[["qty_received", "qty_ordered"]].sum()
    counts = settled.groupby("supplier_id").size()

    received = totals["qty_received"].reindex(ids).fillna(0).astype("int64")
    ordered = totals["qty_ordered"].reindex(ids).fillna(0).astype("int64")
    orders_n = counts.reindex(ids).fillna(0).astype("int64")

    total_ordered = int(ordered.sum())
    # Общая надёжность; без закрытых заказов вовсе — считаем поставку полной.
    overall = float(received.sum()) / total_ordered if total_ordered else 1.0
    with_ratio = ordered > 0
    reliability = pd.Series(overall, index=ids, dtype="float64")
    reliability[with_ratio] = (received[with_ratio] / ordered[with_ratio]).astype("float64")

    return pd.DataFrame({
        "supplier_id": ids,
        "reliability": reliability.to_numpy(),
        "orders_n": orders_n.to_numpy(),
        "received": received.to_numpy(),
        "ordered": ordered.to_numpy(),
    })


def open_orders(ds: Dataset, reliability: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Открытые заказы в пути: когда и сколько ожидаемо придёт.

    Колонки: po_id, sku_id, supplier_id, qty_pending, promised_date, expected_date,
    expected_qty, overdue, stale, overdue_days. Просроченный (promised_date < сегодня или даты нет)
    переносится на сегодня + cfg.overdue_shift_days и помечается overdue. Просрочка больше
    cfg.overdue_stale_days — stale: в приходы не идёт (expected_qty 0, даты прихода нет).
    """
    columns = ["po_id", "sku_id", "supplier_id", "qty_pending", "promised_date",
               "expected_date", "expected_qty", "overdue", "stale", "overdue_days"]
    orders = ds.orders
    if not len(orders):
        return pd.DataFrame({
            "po_id": pd.Series(dtype="string"),
            "sku_id": pd.Series(dtype="string"),
            "supplier_id": pd.Series(dtype="string"),
            "qty_pending": pd.Series(dtype="int64"),
            "promised_date": pd.Series(dtype="datetime64[ns]"),
            "expected_date": pd.Series(dtype="datetime64[ns]"),
            "expected_qty": pd.Series(dtype="float64"),
            "overdue": pd.Series(dtype="bool"),
            "stale": pd.Series(dtype="bool"),
            "overdue_days": pd.Series(dtype="float64"),
        })

    pending = orders[(orders["status"] == "open") & (orders["qty_pending"] > 0)].copy()
    today = pd.Timestamp(ds.today).normalize()

    rates = reliability.set_index("supplier_id")["reliability"].astype("float64")
    # Запасной коэффициент для поставщика вне таблицы — общая надёжность.
    total_ordered = float(reliability["ordered"].sum())
    overall = float(reliability["received"].sum()) / total_ordered if total_ordered else 1.0
    rate = pending["supplier_id"].map(rates).fillna(overall).astype("float64")

    overdue = pending["promised_date"].isna() | (pending["promised_date"] < today)
    expected_date = pending["promised_date"].copy()
    expected_date.loc[overdue] = today + pd.Timedelta(days=int(cfg.overdue_shift_days))
    overdue_days = (today - pending["promised_date"]).dt.days.astype("float64").where(overdue)
    # Давняя просрочка, скорее всего, не придёт: в приходы не идёт, остаётся только в рисках.
    stale = (overdue_days > cfg.overdue_stale_days).astype("bool")
    expected_date.loc[stale] = pd.NaT

    expected_qty = pending["qty_pending"].astype("float64") * rate
    pending["expected_qty"] = expected_qty.where(~stale, 0.0).astype("float64")
    pending["expected_date"] = expected_date
    pending["overdue"] = overdue.astype("bool")
    pending["stale"] = stale
    pending["overdue_days"] = overdue_days

    return pending[columns].sort_values(["promised_date", "po_id"], kind="mergesort").reset_index(
        drop=True
    )


def lot_size(ds: Dataset, cfg: Config) -> pd.Series:
    """Кратность партии SKU: НОД количеств приходов при их числе ≥ cfg.lot_min_receipts.

    Иначе 1 — кратности нет. Индекс — sku_id, значения int64.
    """
    sku_ids = pd.Index(ds.sku["sku_id"].astype("string"), name="sku_id")
    sizes = pd.Series(1, index=sku_ids, dtype="int64")
    receipts = ds.receipts
    if not len(receipts):
        return sizes

    grouped = receipts.groupby("sku_id")["qty"]
    for sku_id, qty in grouped:
        values = [int(v) for v in qty.tolist() if v > 0]
        if len(values) < cfg.lot_min_receipts:
            continue
        step = values[0]
        for value in values[1:]:
            step = int(np.gcd(step, value))
        if step >= 2 and sku_id in sizes.index:
            sizes.loc[sku_id] = step
    return sizes


def current_stock(ds: Dataset) -> pd.Series:
    """Остаток на дату расчёта: снимок на 1-е число месяца today − продано + возвращено + приходы.

    Движения — с 1-го числа по today включительно. Нет строки остатка — 0.
    Не меньше 0 только итог. Индекс — sku_id.
    """
    sku_ids = pd.Index(ds.sku["sku_id"].astype("string"), name="sku_id")
    today = pd.Timestamp(ds.today).normalize()
    month = today.to_period("M")
    month_start = month.to_timestamp(how="start").normalize()

    stock = ds.stock
    snapshot = (
        stock.loc[stock["month"] == month].groupby("sku_id")["stock_qty"].sum()
        if len(stock) else pd.Series(dtype="float64")
    )

    sales = ds.sales
    if len(sales):
        in_month = sales[sales["month"] == month]
        net = in_month["qty_sold"] - in_month["qty_returned"]
        demand = net.groupby(in_month["sku_id"]).sum()
    else:
        demand = pd.Series(dtype="float64")

    receipts = ds.receipts
    if len(receipts):
        window = receipts[(receipts["receipt_date"] >= month_start) & (receipts["receipt_date"] <= today)]
        arrived = window.groupby("sku_id")["qty"].sum()
    else:
        arrived = pd.Series(dtype="float64")

    result = (
        snapshot.reindex(sku_ids).fillna(0)
        - demand.reindex(sku_ids).fillna(0)
        + arrived.reindex(sku_ids).fillna(0)
    )
    return result.clip(lower=0).astype("float64")
