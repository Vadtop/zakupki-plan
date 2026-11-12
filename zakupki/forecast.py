"""Сезонный индекс категорий и помесячный прогноз спроса по SKU."""

import numpy as np
import pandas as pd

from zakupki.config import Config

MONTHS_PER_YEAR = 12

PARAM_COLUMNS = ["sku_id", "level", "trend", "sigma", "history_months", "short_history"]
FC_COLUMNS = ["sku_id", "month", "qty"]


def _empty_index() -> pd.DataFrame:
    """Пустой сезонный индекс с правильными типами колонок."""
    return pd.DataFrame({
        "category": pd.Series(dtype="string"),
        "month_num": pd.Series(dtype="int64"),
        "index": pd.Series(dtype="float64"),
        "share": pd.Series(dtype="float64"),
    })


def _empty_params() -> pd.DataFrame:
    """Пустые параметры прогноза с правильными типами колонок."""
    return pd.DataFrame({
        "sku_id": pd.Series(dtype="string"),
        "level": pd.Series(dtype="float64"),
        "trend": pd.Series(dtype="float64"),
        "sigma": pd.Series(dtype="float64"),
        "history_months": pd.Series(dtype="int64"),
        "short_history": pd.Series(dtype="bool"),
    })


def _empty_fc() -> pd.DataFrame:
    """Пустой помесячный прогноз с правильными типами колонок."""
    return pd.DataFrame({
        "sku_id": pd.Series(dtype="string"),
        "month": pd.Series(dtype="period[M]"),
        "qty": pd.Series(dtype="float64"),
    })


def seasonal_index(demand: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Сезонный индекс категории по полным календарным годам.

    Полный год — год, в котором у категории есть все 12 месяцев. Индекс месяца —
    доля года, приведённая к среднему 1; share — та же доля года.
    """
    if not len(demand):
        return _empty_index()

    work = demand[["category", "month", "demand"]].copy()
    work["month_num"] = work["month"].dt.month.astype("int64")
    work["year"] = work["month"].dt.year.astype("int64")

    counts = (
        work.groupby(["category", "year"], sort=True)["month_num"]
        .nunique()
        .rename("months_n")
        .reset_index()
    )
    full_years = counts[counts["months_n"] == MONTHS_PER_YEAR][["category", "year"]]
    work = work.merge(full_years, on=["category", "year"], how="inner")
    if not len(work):
        return _empty_index()

    agg = work.groupby(["category", "month_num"], as_index=False)["demand"].sum(min_count=1)
    # Месяц без наблюдений (везде упущенный спрос) — неизвестен, а не ноль: берём средний месяц.
    agg["demand"] = agg["demand"].fillna(agg.groupby("category")["demand"].transform("mean")).fillna(0.0)
    total = agg.groupby("category")["demand"].transform("sum")

    agg["index"] = 1.0
    positive = total > 0
    agg.loc[positive, "index"] = agg.loc[positive, "demand"] / total[positive] * MONTHS_PER_YEAR
    agg["share"] = agg["index"] / MONTHS_PER_YEAR

    return (
        agg[["category", "month_num", "index", "share"]]
        .sort_values(["category", "month_num"], kind="mergesort")
        .reset_index(drop=True)
    )


def _index_lookup(index: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Индекс в виде словаря (категория, номер месяца) → значение."""
    if index is None or not len(index):
        return {}
    return {
        (str(c), int(m)): float(v)
        for c, m, v in zip(index["category"], index["month_num"], index["index"])
    }


def _trend(demand_values: np.ndarray, index_values: np.ndarray, cfg: Config) -> float:
    """Тренд: отношение очищенного среднего последних 6 мес к предыдущим 6.

    Сглаживается вдвое (`trend_damping`) и ограничивается `trend_min..trend_max`.
    """
    window = cfg.trend_window_months
    if len(demand_values) < 2 * window:
        return 1.0

    clean = np.full(len(demand_values), np.nan, dtype="float64")
    valid = (~np.isnan(demand_values)) & (index_values >= cfg.min_season_index)
    clean[valid] = demand_values[valid] / index_values[valid]

    recent = clean[-window:]
    previous = clean[-2 * window:-window]
    if np.isnan(recent).all() or np.isnan(previous).all():
        return 1.0

    mean_recent = float(np.nanmean(recent))
    mean_previous = float(np.nanmean(previous))
    if mean_previous <= 0:
        return cfg.trend_max if mean_recent > 0 else 1.0

    damped = 1.0 + cfg.trend_damping * (mean_recent / mean_previous - 1.0)
    return float(np.clip(damped, cfg.trend_min, cfg.trend_max))


def _cap_level(
    level: float, demand_values: np.ndarray, index_values: np.ndarray,
    short_history: bool, cfg: Config,
) -> float:
    """Потолок уровня: среднее очищенного спроса за последние месяцы.

    Действует при короткой истории или когда последние месяцы ниже
    `recent_drop_ratio` уровня. Месяцы без сезона и упущенные не считаются.
    """
    n = cfg.recent_level_months
    if n <= 0:
        return level
    tail_d, tail_i = demand_values[-n:], index_values[-n:]
    ok = (~np.isnan(tail_d)) & (tail_i >= cfg.min_season_index)
    if not ok.any():
        return level
    recent = float(np.mean(tail_d[ok] / tail_i[ok]))
    if short_history or recent < cfg.recent_drop_ratio * level:
        return min(level, recent)
    return level


def _params(demand: pd.DataFrame, index: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Уровень, тренд и σ по каждому SKU; короткая история помечена флагом."""
    if not len(demand):
        return _empty_params()

    lookup = _index_lookup(index)
    rows: list[dict] = []

    for sku_id, group in demand.groupby("sku_id", sort=True):
        group = group.sort_values("month", kind="mergesort")
        category = str(group["category"].iloc[0])
        history_months = len(group)
        window = group.tail(cfg.history_months)

        demand_values = window["demand"].to_numpy(dtype="float64")
        month_nums = window["month"].dt.month.to_numpy(dtype="int64")
        index_values = np.array(
            [lookup.get((category, int(m)), 1.0) for m in month_nums], dtype="float64"
        )

        usable = (~np.isnan(demand_values)) & (index_values >= cfg.min_season_index)
        lvl_d, lvl_i = demand_values[usable], index_values[usable]
        if not usable.any():
            # В окне всё упущено — уровень и σ по всей пригодной истории SKU.
            all_d = group["demand"].to_numpy(dtype="float64")
            all_i = np.array(
                [lookup.get((category, int(m)), 1.0) for m in group["month"].dt.month], dtype="float64"
            )
            ok = (~np.isnan(all_d)) & (all_i >= cfg.min_season_index)
            lvl_d, lvl_i = all_d[ok], all_i[ok]
        level = float(np.mean(lvl_d / lvl_i)) if len(lvl_d) else 0.0

        short_history = history_months < cfg.short_history_months
        trend = 1.0 if short_history else _trend(demand_values, index_values, cfg)

        sigma = 0.0
        if len(lvl_d) >= 2:
            residual = lvl_d - level * lvl_i
            sigma = float(np.std(residual, ddof=1))
            if not np.isfinite(sigma):
                sigma = 0.0

        level = _cap_level(level, demand_values, index_values, short_history, cfg)

        rows.append({
            "sku_id": sku_id,
            "level": level,
            "trend": trend,
            "sigma": sigma,
            "history_months": history_months,
            "short_history": short_history,
        })

    return pd.DataFrame(rows, columns=PARAM_COLUMNS)


def _forecast_rows(
    params: pd.DataFrame, demand: pd.DataFrame, index: pd.DataFrame, cfg: Config,
    today_month: pd.Period,
) -> pd.DataFrame:
    """Помесячный прогноз: уровень × индекс месяца × тренд."""
    if not len(params):
        return _empty_fc()

    lookup = _index_lookup(index)
    categories = demand.drop_duplicates(subset=["sku_id"]).set_index("sku_id")["category"]
    months = pd.period_range(today_month, periods=cfg.cover_horizon_months, freq="M")
    month_nums = np.array([m.month for m in months], dtype="int64")

    rows: list[dict] = []
    for _, row in params.sort_values("sku_id", kind="mergesort").iterrows():
        sku_id = row["sku_id"]
        category = str(categories.get(sku_id, ""))
        index_values = np.array(
            [lookup.get((category, int(m)), 1.0) for m in month_nums], dtype="float64"
        )
        qty = float(row["level"]) * index_values * float(row["trend"])
        for month, value in zip(months, qty):
            rows.append({"sku_id": sku_id, "month": month, "qty": float(value)})

    return (
        pd.DataFrame(rows, columns=FC_COLUMNS)
        .sort_values(["sku_id", "month"], kind="mergesort")
        .reset_index(drop=True)
    )


def forecast(
    demand: pd.DataFrame, index: pd.DataFrame, cfg: Config, today: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Параметры по SKU и прогноз на `cover_horizon_months` месяцев от месяца today."""
    today_month = pd.Period(pd.Timestamp(today), freq="M")
    params = _params(demand, index, cfg)
    fc = _forecast_rows(params, demand, index, cfg, today_month)
    return params, fc
