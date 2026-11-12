"""Точность прогноза: метрики, маска по наличию, срез без подглядывания в будущее."""

import pandas as pd
import pytest

from tests.conftest import SKU_B, SKU_C, _default_sales, _default_stock, make_dataset
from zakupki.backtest import _cut, _in_stock, backtest, bias, compare, wape
from zakupki.config import Config
from zakupki.demand import build_demand
from zakupki.forecast import forecast, seasonal_index

CUT = pd.Period("2025-08", freq="M")


def _run(ds):
    return backtest(ds, Config(today=ds.today))


def test_wape_and_bias_by_hand():
    fc, fact = pd.Series([12.0, 6.0, 0.0]), pd.Series([10.0, 5.0, 3.0])
    # |2| + |1| + |3| = 6 из 18; (18 − 18) / 18 = 0
    assert wape(fc, fact) == pytest.approx(6 / 18)
    assert bias(fc, fact) == pytest.approx(0.0)
    assert bias(pd.Series([12.0, 6.0]), pd.Series([10.0, 5.0])) == pytest.approx(0.2)
    assert wape(fc, pd.Series([0.0, 0.0, 0.0])) is None


def test_default_dataset_cut_and_pairs(small_dataset):
    r = _run(small_dataset)
    assert r["start"] == "2025-08-01"
    assert (r["first_month"], r["last_month"], r["months"]) == ("2025-08", "2026-07", 12)
    # SKU-C без остатка после 2024-06 — все его месяцы вне маски
    assert (r["total"]["pairs"], r["total"]["sku"]) == (24, 2)
    assert r["total"]["fact"] == (100 + 11 * 5) + 12 * 48  # SKU-A: пик в октябре; SKU-B: 50 − 2 в месяц
    assert [h["months"] for h in r["horizons"]] == [4, 6, 7]
    assert [h["pairs"] for h in r["horizons"]] == [8, 12, 14]
    assert {c["category"] for c in r["categories"]} == {"space_heaters", "sanitary_faucets"}
    assert r["new_sku"] == 0


def test_stockout_month_excluded_for_all_methods():
    stock = [s for s in _default_stock() if not (s["sku_id"] == SKU_B and s["month"] == pd.Period("2025-10", "M"))]
    r = _run(make_dataset(stock=stock))
    # нет остатка на 01.10 → выпадают октябрь (начало) и сентябрь (начало следующего)
    assert r["total"]["pairs"] == 22
    for k in ("wape_model", "wape_m12", "wape_ly"):
        assert r["total"][k] is not None


def test_mask_does_not_depend_on_forecast(small_dataset):
    ds = small_dataset
    cfg = Config(today=ds.today)
    start = CUT.to_timestamp()
    past = _cut(ds, start)
    past_cfg = cfg.with_overrides(today=start)
    past_demand = build_demand(past, past_cfg)
    _, fc = forecast(past_demand, seasonal_index(past_demand, past_cfg), past_cfg, start)
    months = list(pd.period_range(CUT, periods=12, freq="M"))
    full_demand = build_demand(ds, cfg)
    rows = compare(past_demand, fc, full_demand, months)
    other = compare(past_demand, fc.assign(qty=fc["qty"] * 100), full_demand, months)
    assert _in_stock(ds, rows).tolist() == _in_stock(ds, other).tolist()
    assert set(rows.loc[~_in_stock(ds, rows), "sku_id"]) == {SKU_C}


def test_sales_after_cut_do_not_change_model():
    base = _run(make_dataset())
    sales = _default_sales()
    for s in sales:
        if s["sku_id"] == SKU_B and s["month"] >= CUT:
            s["qty_sold"] = 100  # меньше остатка 120 — маска та же
    changed = _run(make_dataset(sales=sales))
    assert changed["total"]["pairs"] == base["total"]["pairs"]
    assert changed["total"]["model"] == base["total"]["model"]
    assert [h["model"] for h in changed["horizons"]] == [h["model"] for h in base["horizons"]]
    assert changed["total"]["fact"] > base["total"]["fact"]


def test_new_sku_without_history_counted():
    sales = _default_sales() + [
        {"sku_id": "SKU-N", "category": "space_heaters", "month": m, "qty_sold": 7, "qty_returned": 0,
         "partial_month": 0}
        for m in pd.period_range("2025-09", "2026-07", freq="M")
    ]
    r = _run(make_dataset(sales=sales))
    assert r["new_sku"] == 1
    assert r["total"]["sku"] == 2  # новый SKU в оценку не попал


def test_no_data_before_cut_gives_empty_result():
    sales = [s for s in _default_sales() if s["month"] >= pd.Period("2025-09", "M")]
    r = _run(make_dataset(sales=sales))
    assert r["total"] is None
    assert r["horizons"] == [] and r["categories"] == []
    assert r["start"] == "2025-08-01"
    assert r["new_sku"] == 3
