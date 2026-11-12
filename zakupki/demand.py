"""Помесячный спрос: продажи минус возвраты, упущенные месяцы помечены."""

import numpy as np
import pandas as pd

from zakupki.config import Config
from zakupki.data import Dataset

# Колонки результата build_demand, порядок фиксирован.
DEMAND_COLUMNS = ["sku_id", "category", "month", "demand", "sold_net", "lost", "stock_start"]


def _empty_demand() -> pd.DataFrame:
    """Пустой кадр спроса с правильными типами колонок."""
    return pd.DataFrame({
        "sku_id": pd.Series(dtype="string"),
        "category": pd.Series(dtype="string"),
        "month": pd.Series(dtype="period[M]"),
        "demand": pd.Series(dtype="float64"),
        "sold_net": pd.Series(dtype="int64"),
        "lost": pd.Series(dtype="bool"),
        "stock_start": pd.Series(dtype="float64"),
    })


def _stock_start(sales: pd.DataFrame, stock: pd.DataFrame) -> np.ndarray:
    """Остаток на 1-е число месяца: нет строки после первого снимка — значит 0."""
    start = np.full(len(sales), np.nan, dtype="float64")
    if not len(stock):
        return start
    first_snapshot = stock["month"].min()
    lookup = stock.drop_duplicates(subset=["sku_id", "month"]).set_index(["sku_id", "month"])["stock_qty"]
    key = pd.MultiIndex.from_arrays([sales["sku_id"], sales["month"]])
    start = lookup.reindex(key).to_numpy(dtype="float64")
    # Снимков до первого месяца нет вообще: там остаток неизвестен, а не ноль.
    after_first = (sales["month"] >= first_snapshot).to_numpy()
    start = np.where(after_first & np.isnan(start), 0.0, start)
    return start


def build_demand(ds: Dataset, cfg: Config) -> pd.DataFrame:
    """Спрос по SKU и полным месяцам истории.

    Продажи минус возвраты (не меньше нуля). Месяц с нулевыми продажами при нулевом
    остатке на 1-е число — упущенный спрос: demand = NaN, lost = True.
    """
    today_month = ds.today.to_period("M")
    sales = ds.sales
    if not len(sales):
        return _empty_demand()

    sales = sales[(sales["partial_month"] == 0) & (sales["month"] < today_month)].copy()
    if not len(sales):
        return _empty_demand()

    sold_net = (sales["qty_sold"] - sales["qty_returned"]).clip(lower=0).astype("int64")
    stock_start = _stock_start(sales, ds.stock)
    lost = (sold_net == 0).to_numpy() & (stock_start == 0)
    demand = sold_net.astype("float64").where(~lost)

    result = pd.DataFrame({
        "sku_id": sales["sku_id"].to_numpy(),
        "category": sales["category"].to_numpy(),
        "month": sales["month"].to_numpy(),
        "demand": demand.to_numpy(),
        "sold_net": sold_net.to_numpy(),
        "lost": lost,
        "stock_start": stock_start,
    })
    return result.sort_values(["sku_id", "month"], kind="mergesort").reset_index(drop=True)[DEMAND_COLUMNS]
