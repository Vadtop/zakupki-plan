"""Сезонный индекс, уровень, тренд, σ и помесячный прогноз."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zakupki.config import Config
from zakupki.data import load_dataset
from zakupki.demand import build_demand
from zakupki.forecast import forecast, seasonal_index

TODAY = pd.Timestamp("2026-08-13")
CFG = Config()
X = "SKU-X"
CAT = "cat_x"
REAL_DATA = Path("data/raw/share")


def _months(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def _sales(sku_id: str, category: str, months, qty, returned=0, partial=0) -> list[dict]:
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


def _flat_demand(sku_id: str, category: str, months, qty, stock_qty=1000) -> pd.DataFrame:
    """Готовый кадр спроса без упущенных месяцев — для тестов прогноза."""
    values = [qty(m) if callable(qty) else qty for m in months]
    return pd.DataFrame({
        "sku_id": pd.Series([sku_id] * len(months), dtype="string"),
        "category": pd.Series([category] * len(months), dtype="string"),
        "month": pd.Series(months, dtype="period[M]"),
        "demand": pd.Series(values, dtype="float64"),
        "sold_net": pd.Series(values, dtype="int64"),
        "lost": pd.Series([False] * len(months), dtype="bool"),
        "stock_start": pd.Series([float(stock_qty)] * len(months), dtype="float64"),
    })


def _seasonal_demand(sku_id: str, category: str, years, peak_month: int, peak=100, base=10) -> pd.DataFrame:
    """Спрос с одним пиком в году — сезонный профиль категории."""
    rows = []
    for year in years:
        for month in range(1, 13):
            rows.append({
                "sku_id": sku_id, "category": category,
                "month": pd.Period(f"{year}-{month:02d}", freq="M"),
                "demand": float(peak if month == peak_month else base),
                "sold_net": int(peak if month == peak_month else base),
                "lost": False, "stock_start": 1000.0,
            })
    return pd.DataFrame(rows)


# --- seasonal_index ---

def test_seasonal_index_uses_full_years_only():
    """Неполный год в индекс не идёт: 2024 без декабря не влияет."""
    demand = pd.concat([
        _seasonal_demand(X, CAT, [2023, 2024], peak_month=10),
        _flat_demand(X, CAT, _months("2024-01", "2024-11"), 0),
    ], ignore_index=True)
    # Убираем декабрь 2024 — год становится неполным.
    demand = demand[demand["month"] != pd.Period("2024-12", freq="M")]

    index = seasonal_index(demand, CFG)
    assert set(index["month_num"]) == set(range(1, 13))
    # Считается по 2023: 11 месяцев по 10 и октябрь 100.
    total = 11 * 10 + 100
    expected_oct = 100 / total * 12
    actual_oct = index[(index["category"] == CAT) & (index["month_num"] == 10)]["index"].iloc[0]
    assert actual_oct == pytest.approx(expected_oct)


def test_seasonal_index_mean_one_and_share_sums_to_one():
    """Средний индекс по категории — 1, share в сумме — 1; 12 месяцев."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=5)
    index = seasonal_index(demand, CFG)

    assert len(index) == 12
    assert index["index"].mean() == pytest.approx(1.0)
    assert index["share"].sum() == pytest.approx(1.0)
    assert (index["share"] == index["index"] / 12).all()


def test_seasonal_index_peak_in_peak_month():
    """Максимум индекса — в месяц пика."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10)
    index = seasonal_index(demand, CFG)

    assert index.loc[index["index"].idxmax(), "month_num"] == 10


def test_seasonal_index_empty():
    """Пустой спрос — пустой индекс с нужными колонками."""
    index = seasonal_index(pd.DataFrame({
        "sku_id": pd.Series(dtype="string"), "category": pd.Series(dtype="string"),
        "month": pd.Series(dtype="period[M]"), "demand": pd.Series(dtype="float64"),
        "sold_net": pd.Series(dtype="int64"), "lost": pd.Series(dtype="bool"),
        "stock_start": pd.Series(dtype="float64"),
    }), CFG)

    assert list(index.columns) == ["category", "month_num", "index", "share"]
    assert len(index) == 0


# --- level ---

def test_lost_months_do_not_lower_level():
    """Три упущенных месяца не занижают уровень против того же ряда без них."""
    months = _months("2024-01", "2025-12")
    with_lost = _flat_demand(X, CAT, months, 20)
    lost_months = [pd.Period("2025-03", freq="M"), pd.Period("2025-07", freq="M"), pd.Period("2025-11", freq="M")]
    with_lost.loc[with_lost["month"].isin(lost_months), "demand"] = np.nan
    with_lost.loc[with_lost["month"].isin(lost_months), "lost"] = True

    without = _flat_demand(X, CAT, months, 20)

    index = pd.DataFrame({
        "category": pd.Series([CAT] * 12, dtype="string"),
        "month_num": pd.Series(range(1, 13), dtype="int64"),
        "index": pd.Series([1.0] * 12, dtype="float64"),
        "share": pd.Series([1 / 12] * 12, dtype="float64"),
    })
    params_lost, _ = forecast(with_lost, index, CFG, TODAY)
    params_clean, _ = forecast(without, index, CFG, TODAY)

    level_lost = params_lost["level"].iloc[0]
    level_clean = params_clean["level"].iloc[0]
    assert level_lost == pytest.approx(level_clean, rel=0.05)
    assert level_lost == pytest.approx(20.0, rel=0.05)


def test_seasonal_sku_gets_nonzero_level():
    """SKU с продажами только в сезон получает ненулевой уровень."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10, peak=100, base=0)
    index = seasonal_index(demand, CFG)
    params, _ = forecast(demand, index, CFG, TODAY)

    assert params["level"].iloc[0] > 0


def test_months_below_min_season_index_excluded_from_level():
    """Месяцы с индексом ниже min_season_index в уровень не идут."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10, peak=100, base=0)
    index = seasonal_index(demand, CFG)
    # Ровно так, как считает код: только месяцы с индексом ≥ порога.
    lookup = dict(zip(index["month_num"], index["index"]))
    usable = demand[demand["month"].dt.month.map(lookup) >= CFG.min_season_index]
    expected = float((usable["demand"] / usable["month"].dt.month.map(lookup)).mean())

    params, _ = forecast(demand, index, CFG, TODAY)
    assert params["level"].iloc[0] == pytest.approx(expected)


# --- trend ---

def _trend_params(demand: pd.DataFrame, cfg: Config = CFG):
    index = pd.DataFrame({
        "category": pd.Series([CAT] * 12, dtype="string"),
        "month_num": pd.Series(range(1, 13), dtype="int64"),
        "index": pd.Series([1.0] * 12, dtype="float64"),
        "share": pd.Series([1 / 12] * 12, dtype="float64"),
    })
    params, _ = forecast(demand, index, cfg, TODAY)
    return params["trend"].iloc[0]


# Тренд считается по последним 12 полным месяцам: 6 последних против 6 предыдущих.
TREND_MONTHS = _months("2025-01", "2025-12")
TREND_HALF = pd.Period("2025-07", freq="M")


def test_trend_capped_on_doubling():
    """Удвоение продаж: тренд не выше trend_max."""
    demand = _flat_demand(X, CAT, TREND_MONTHS, lambda m: 20 if m < TREND_HALF else 40)

    assert _trend_params(demand) == pytest.approx(CFG.trend_max)


def test_trend_floored_on_fall():
    """Падение продаж вдвое: тренд не ниже trend_min."""
    demand = _flat_demand(X, CAT, TREND_MONTHS, lambda m: 40 if m < TREND_HALF else 20)

    assert _trend_params(demand) == pytest.approx(CFG.trend_min)


def test_damping_applied_between_limits():
    """Небольшой рост сглаживается вдвое и остаётся внутри границ."""
    demand = _flat_demand(X, CAT, TREND_MONTHS, lambda m: 100 if m < TREND_HALF else 110)

    # Отношение 1.1, сглаживание 0.5 → 1 + 0.5 * 0.1.
    assert _trend_params(demand) == pytest.approx(1.05)


def test_short_history_flag_and_unit_trend():
    """История меньше 12 полных месяцев — short_history=True и тренд 1."""
    demand = _flat_demand(X, CAT, _months("2026-01", "2026-06"), 30)
    index = pd.DataFrame({
        "category": pd.Series([CAT] * 12, dtype="string"),
        "month_num": pd.Series(range(1, 13), dtype="int64"),
        "index": pd.Series([1.0] * 12, dtype="float64"),
        "share": pd.Series([1 / 12] * 12, dtype="float64"),
    })
    params, _ = forecast(demand, index, CFG, TODAY)

    assert bool(params["short_history"].iloc[0]) is True
    assert params["trend"].iloc[0] == pytest.approx(1.0)
    assert params["history_months"].iloc[0] == 6
    assert params["level"].iloc[0] == pytest.approx(30.0)


def test_level_falls_back_to_earlier_history_when_window_all_lost():
    """Все 12 мес. окна упущены — уровень по более ранней пригодной истории (REVIEW № 4)."""
    demand = _flat_demand(X, CAT, _months("2024-08", "2026-07"), 30)
    demand["demand"] = demand["demand"].astype("float64")
    demand.loc[demand["month"] >= pd.Period("2025-08", freq="M"), ["demand", "lost"]] = [np.nan, True]
    demand.loc[demand["month"] == pd.Period("2024-08", freq="M"), "demand"] = 42.0
    params, fc = forecast(demand, _flat_index(), CFG, TODAY)
    row = params.iloc[0]
    assert row["level"] == pytest.approx((42 + 11 * 30) / 12)
    assert fc["qty"].iloc[0] > 0
    # σ по тем же месяцам, что и уровень
    vals = np.array([42.0] + [30.0] * 11)
    assert row["sigma"] == pytest.approx(float(np.std(vals - row["level"], ddof=1)))


def test_level_zero_when_no_usable_history():
    """Пригодных месяцев нет вовсе — уровень 0, как раньше."""
    demand = _flat_demand(X, CAT, _months("2025-08", "2026-07"), 30)
    demand["demand"] = np.nan
    params, _ = forecast(demand, _flat_index(), CFG, TODAY)
    assert params["level"].iloc[0] == 0.0 and params["sigma"].iloc[0] == 0.0


def _flat_index() -> pd.DataFrame:
    return pd.DataFrame({
        "category": pd.Series([CAT] * 12, dtype="string"),
        "month_num": pd.Series(range(1, 13), dtype="int64"),
        "index": pd.Series([1.0] * 12, dtype="float64"),
        "share": pd.Series([1 / 12] * 12, dtype="float64"),
    })


def _level(demand, cfg=CFG) -> float:
    params, _ = forecast(demand, _flat_index(), cfg, TODAY)
    return float(params["level"].iloc[0])


def test_level_capped_by_recent_months_on_short_history():
    """Короткая история: уровень не выше среднего последних 3 месяцев."""
    demand = _flat_demand(X, CAT, _months("2026-02", "2026-07"), lambda m: 100 if m < _months("2026-05", "2026-05")[0] else 20)

    assert _level(demand) == pytest.approx(20.0)
    assert _level(demand, Config(recent_level_months=0)) == pytest.approx(60.0)


def test_level_capped_when_recent_fall_below_half():
    """Длинная история: последние 3 месяца ниже 50% уровня режут уровень."""
    demand = _flat_demand(X, CAT, _months("2025-08", "2026-07"), lambda m: 100 if m < _months("2026-05", "2026-05")[0] else 20)

    assert _level(demand) == pytest.approx(20.0)


def test_level_not_capped_when_recent_stable():
    """Спад меньше порога — уровень прежний (среднее за 12 месяцев)."""
    demand = _flat_demand(X, CAT, _months("2025-08", "2026-07"), lambda m: 100 if m < _months("2026-05", "2026-05")[0] else 70)

    assert _level(demand) == pytest.approx((9 * 100 + 3 * 70) / 12)


# --- sigma ---

def test_sigma_is_spread_around_seasonal_forecast():
    """σ — разброс спроса вокруг уровень × индекс, а не сырой разброс продаж.

    Спрос точно повторяет сезонный профиль категории: остаток от прогноза нулевой,
    хотя сырой разброс продаж большой.
    """
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10, peak=100, base=10)
    index = seasonal_index(demand, CFG)
    params, _ = forecast(demand, index, CFG, TODAY)

    assert params["sigma"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    raw = demand["demand"].to_numpy(dtype="float64")
    assert float(np.std(raw, ddof=1)) > 10.0


def test_sigma_grows_with_deviation_from_profile():
    """Отклонение одного месяца от профиля даёт ненулевую σ."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10, peak=100, base=10)
    index = seasonal_index(demand, CFG)
    clean, _ = forecast(demand, index, CFG, TODAY)

    deviated = demand.copy()
    extra = 300.0
    mask = deviated["month"] == pd.Period("2025-06", freq="M")
    deviated.loc[mask, "demand"] = deviated.loc[mask, "demand"] + extra
    dirty, _ = forecast(deviated, index, CFG, TODAY)

    assert dirty["sigma"].iloc[0] > clean["sigma"].iloc[0]


# --- forecast ---

def test_forecast_rows_and_formula():
    """Строк — cover_horizon_months на SKU, с месяца today; qty = уровень × индекс × тренд."""
    months = _months("2024-01", "2025-12")
    demand = _flat_demand(X, CAT, months, 30)
    index = pd.DataFrame({
        "category": pd.Series([CAT] * 12, dtype="string"),
        "month_num": pd.Series(range(1, 13), dtype="int64"),
        "index": pd.Series([1.0] * 12, dtype="float64"),
        "share": pd.Series([1 / 12] * 12, dtype="float64"),
    })
    params, fc = forecast(demand, index, CFG, TODAY)

    assert list(fc.columns) == ["sku_id", "month", "qty"]
    assert len(fc) == CFG.cover_horizon_months
    assert fc["month"].iloc[0] == pd.Period("2026-08", freq="M")
    assert fc["month"].iloc[-1] == pd.Period("2026-08", freq="M") + CFG.cover_horizon_months - 1
    expected = params["level"].iloc[0] * 1.0 * params["trend"].iloc[0]
    assert fc["qty"].iloc[0] == pytest.approx(expected)


def test_forecast_follows_seasonal_profile():
    """Прогноз повторяет сезонный профиль: пик — в октябре."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10)
    index = seasonal_index(demand, CFG)
    _, fc = forecast(demand, index, CFG, pd.Timestamp("2025-12-01"))

    october = fc[fc["month"].dt.month == 10]["qty"].iloc[0]
    july = fc[fc["month"].dt.month == 7]["qty"].iloc[0]
    assert october > july


def test_forecast_short_history_uses_category_index():
    """Короткая история — индекс категории и среднее за то, что есть (очищенное от сезонности)."""
    demand = _seasonal_demand(X, CAT, [2023, 2024, 2025], peak_month=10)
    index = seasonal_index(demand, CFG)
    short_months = _months("2026-01", "2026-06")
    short = _flat_demand(X, CAT, short_months, 30)
    params, fc = forecast(short, index, CFG, TODAY)

    lookup = dict(zip(index["month_num"], index["index"]))
    expected_level = float(np.mean([30.0 / lookup[m.month] for m in short_months]))
    assert params["level"].iloc[0] == pytest.approx(expected_level)
    assert bool(params["short_history"].iloc[0]) is True
    assert params["trend"].iloc[0] == pytest.approx(1.0)

    october_index = lookup[10]
    october = fc[fc["month"].dt.month == 10]["qty"].iloc[0]
    assert october == pytest.approx(expected_level * october_index * 1.0)


def test_forecast_empty_params():
    """Нет спроса — пустые параметры и пустой прогноз."""
    demand = pd.DataFrame({
        "sku_id": pd.Series(dtype="string"), "category": pd.Series(dtype="string"),
        "month": pd.Series(dtype="period[M]"), "demand": pd.Series(dtype="float64"),
        "sold_net": pd.Series(dtype="int64"), "lost": pd.Series(dtype="bool"),
        "stock_start": pd.Series(dtype="float64"),
    })
    params, fc = forecast(demand, seasonal_index(demand, CFG), CFG, TODAY)

    assert list(params.columns) == ["sku_id", "level", "trend", "sigma", "history_months", "short_history"]
    assert len(params) == 0
    assert list(fc.columns) == ["sku_id", "month", "qty"]
    assert len(fc) == 0


# --- детерминизм ---

def test_determinism(small_dataset):
    """Два вызова на одном входе дают равные кадры."""
    demand = build_demand(small_dataset, CFG)
    index_first = seasonal_index(demand, CFG)
    index_second = seasonal_index(demand, CFG)
    pd.testing.assert_frame_equal(index_first, index_second)

    params_first, fc_first = forecast(demand, index_first, CFG, small_dataset.today)
    params_second, fc_second = forecast(demand, index_second, CFG, small_dataset.today)
    pd.testing.assert_frame_equal(params_first, params_second)
    pd.testing.assert_frame_equal(fc_first, fc_second)


def test_default_dataset_params(small_dataset):
    """На наборе по умолчанию: сезонный SKU-A получает ненулевой уровень и сезонный пик."""
    demand = build_demand(small_dataset, CFG)
    index = seasonal_index(demand, CFG)
    params, fc = forecast(demand, index, CFG, small_dataset.today)

    assert set(params["sku_id"]) == {"SKU-A", "SKU-B", "SKU-C"}
    level_a = params[params["sku_id"] == "SKU-A"]["level"].iloc[0]
    assert level_a > 0
    fc_a = fc[fc["sku_id"] == "SKU-A"]
    october = fc_a[fc_a["month"].dt.month == 10]["qty"].iloc[0]
    july = fc_a[fc_a["month"].dt.month == 7]["qty"].iloc[0]
    assert october > july


# --- реальные данные ---

@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="нет папки data/raw/share")
def test_real_seasonality():
    """Реальные данные: обогреватели — пик в октябре и провал летом; смесители ровные."""
    ds = load_dataset(REAL_DATA, TODAY)
    demand = build_demand(ds, Config(today=ds.today))
    index = seasonal_index(demand, Config(today=ds.today))

    heaters = index[index["category"] == "space_heaters"].set_index("month_num")["index"]
    assert heaters.idxmax() == 10
    assert (heaters.loc[[5, 6, 7, 8]] < 0.15).all()

    faucets = index[index["category"] == "sanitary_faucets"].set_index("month_num")["index"]
    assert len(faucets) == 12
    assert faucets.between(0.6, 1.4).all()


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="нет папки data/raw/share")
def test_real_forecast_shape():
    """Реальные данные: прогноз по каждому SKU на cover_horizon_months месяцев."""
    ds = load_dataset(REAL_DATA, TODAY)
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)

    assert len(params) == len(demand["sku_id"].unique())
    assert len(fc) == len(params) * cfg.cover_horizon_months
    assert fc["month"].min() == pd.Period("2026-08", freq="M")
    assert (fc["qty"] >= 0).all()


def test_month_without_observations_is_not_zero_season():
    """Месяц категории, где спрос везде упущен, — неизвестен, а не ноль (REVIEW-3 № 5)."""
    months = _months("2024-01", "2025-12")
    demand = _flat_demand("A", "cat", months, 30.0)
    summer = demand["month"].dt.month.isin([6, 7, 8])
    demand.loc[summer, "demand"] = np.nan
    demand.loc[summer, "lost"] = True
    index = seasonal_index(demand, Config()).set_index("month_num")["index"]
    assert index.loc[[6, 7, 8]].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert index.loc[1] == pytest.approx(1.0)
