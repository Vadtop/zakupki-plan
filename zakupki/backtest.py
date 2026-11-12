"""Точность прогноза: прогноз с прошлой даты против фактических продаж после неё.

Тот же расчёт спроса и прогноза, что в плане, но по данным до среза
(первое число месяца расчёта минус backtest_months). План не меняется.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from zakupki.config import Config
from zakupki.data import Dataset, load_dataset
from zakupki.demand import build_demand
from zakupki.forecast import MONTHS_PER_YEAR, forecast, seasonal_index

METHODS = ("model", "m12", "ly")


def _cut(ds: Dataset, date: pd.Timestamp) -> Dataset:
    """Набор, обрезанный на дату, как это делает load_dataset."""
    month = date.to_period("M")
    return Dataset(
        sku=ds.sku,
        sales=ds.sales[ds.sales["month"] <= month].reset_index(drop=True),
        stock=ds.stock[ds.stock["month"] <= month].reset_index(drop=True),
        orders=ds.orders[ds.orders["order_date"] <= date].reset_index(drop=True),
        receipts=ds.receipts[ds.receipts["receipt_date"] <= date].reset_index(drop=True),
        suppliers=ds.suppliers,
        today=date,
    )


def _datasets(src, cfg: Config) -> tuple[Dataset, Dataset, pd.Timestamp]:
    """Полная выгрузка, выгрузка на срез и дата среза."""
    full = src if isinstance(src, Dataset) else load_dataset(Path(src), cfg.today)
    start = full.today.to_period("M").to_timestamp() - pd.DateOffset(months=cfg.backtest_months)
    past = _cut(full, start) if isinstance(src, Dataset) else load_dataset(Path(src), start)
    return full, past, start


def wape(fc: pd.Series, fact: pd.Series) -> float | None:
    """Ошибка в долях: Σ|прогноз − факт| / Σфакт."""
    total = float(fact.sum())
    return None if total <= 0 else float((fc - fact).abs().sum() / total)


def bias(fc: pd.Series, fact: pd.Series) -> float | None:
    """Смещение в долях: Σ(прогноз − факт) / Σфакт; больше 0 — завышает."""
    total = float(fact.sum())
    return None if total <= 0 else float((fc.sum() - total) / total)


def _scores(rows: pd.DataFrame) -> dict:
    """Метрики на одном наборе пар — одинаковом для всех методов."""
    out = {
        "pairs": len(rows),
        "sku": int(rows["sku_id"].nunique()),
        "fact": int(rows["fact"].sum()),
        "model": round(float(rows["model"].sum()), 1),
    }
    for m in METHODS:
        out[f"wape_{m}"] = wape(rows[m], rows["fact"])
    out["bias_model"] = bias(rows["model"], rows["fact"])
    return out


def _in_stock(full: Dataset, fact: pd.DataFrame) -> pd.Series:
    """Маска по наличию: товар был на начало месяца, остался на начало следующего, продажи меньше доступного."""
    stock = full.stock.drop_duplicates(subset=["sku_id", "month"]).set_index(["sku_id", "month"])["stock_qty"]
    next_key = pd.MultiIndex.from_arrays([fact["sku_id"], fact["month"] + 1])
    next_start = stock.reindex(next_key).fillna(0).to_numpy(dtype="float64")
    rec = full.receipts.assign(month=full.receipts["receipt_date"].dt.to_period("M"))
    rec = rec.groupby(["sku_id", "month"])["qty"].sum()
    got = (
        rec.reindex(pd.MultiIndex.from_arrays([fact["sku_id"], fact["month"]]))
        .fillna(0)
        .to_numpy(dtype="float64")
    )
    start = fact["stock_start"].to_numpy(dtype="float64")
    sold = fact["fact"].to_numpy(dtype="float64")
    return pd.Series((start > 0) & (next_start > 0) & (sold < start + got), index=fact.index)


def compare(past_demand: pd.DataFrame, fc: pd.DataFrame, fact_demand: pd.DataFrame, months) -> pd.DataFrame:
    """Пары SKU × месяц после среза: факт, прогноз модели, M12, LY и признаки маски.

    SKU без истории на срез (нет прогноза модели) в пары не входят.
    """
    months = list(months)
    fact = fact_demand[fact_demand["month"].isin(months)]
    fact = fact[["sku_id", "category", "month", "sold_net", "stock_start"]].rename(
        columns={"sold_net": "fact"}
    )
    model = fc[fc["month"].isin(months)].rename(columns={"qty": "model"})
    rows = fact.merge(model, on=["sku_id", "month"], how="inner")
    # M12: среднее чистых продаж за год до среза, нули сохраняются.
    first = months[0]
    year = past_demand[past_demand["month"] >= first - MONTHS_PER_YEAR]
    rows["m12"] = rows["sku_id"].map(year.groupby("sku_id")["sold_net"].mean()).fillna(0.0).astype("float64")
    # LY: тот же месяц год назад, нет строки — 0.
    prev = past_demand[["sku_id", "month", "sold_net"]].copy()
    prev["month"] = prev["month"] + MONTHS_PER_YEAR
    rows = rows.merge(prev.rename(columns={"sold_net": "ly"}), on=["sku_id", "month"], how="left")
    rows["ly"] = rows["ly"].fillna(0).astype("float64")
    rows["fact"] = rows["fact"].astype("float64")
    rows["horizon"] = np.array([(m - first).n + 1 for m in rows["month"]], dtype="int64")
    return rows.sort_values(["sku_id", "month"], kind="mergesort").reset_index(drop=True)


def _empty(start: pd.Timestamp, new_sku: int = 0) -> dict:
    return {"start": f"{start:%Y-%m-%d}", "total": None, "horizons": [], "categories": [], "new_sku": new_sku}


def backtest(src, cfg: Config) -> dict:
    """Прогноз на срез против факта за backtest_months полных месяцев после него.

    src — папка выгрузки или уже загруженный Dataset. Пустой total — сравнивать не с чем.
    """
    full, past, start = _datasets(src, cfg)
    past_cfg = cfg.with_overrides(today=start)
    past_demand = build_demand(past, past_cfg)
    full_demand = build_demand(full, cfg.with_overrides(today=full.today))
    months = pd.period_range(start.to_period("M"), periods=cfg.backtest_months, freq="M")
    months = [m for m in months if m < full.today.to_period("M")]
    test_sku = set(full_demand.loc[full_demand["month"].isin(months), "sku_id"])
    if not len(past_demand) or not months:
        return _empty(start, len(test_sku))
    index = seasonal_index(past_demand, past_cfg)
    _, fc = forecast(past_demand, index, past_cfg, start)
    new_sku = len(test_sku - set(fc["sku_id"]))
    rows = compare(past_demand, fc, full_demand, months)
    rows = rows[_in_stock(full, rows)]
    if not len(rows) or float(rows["fact"].sum()) <= 0:
        return _empty(start, new_sku)
    horizons = [
        {"months": h, **_scores(rows[rows["horizon"] <= h])} for h in cfg.backtest_horizons if h < len(months)
    ]
    cats = [{"category": str(c), **_scores(part)} for c, part in rows.groupby("category", sort=True)]
    cats.sort(key=lambda c: -c["fact"])
    return {
        "start": f"{start:%Y-%m-%d}",
        "first_month": str(months[0]),
        "last_month": str(months[-1]),
        "months": len(months),
        "total": _scores(rows),
        "horizons": horizons,
        "categories": cats,
        "new_sku": new_sku,
    }
