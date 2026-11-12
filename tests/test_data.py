"""Загрузка и проверка выгрузки: типы, дата расчёта, отбрасывание битых строк."""

from pathlib import Path

import pandas as pd
import pytest

from zakupki.data import (
    ORDERS_FILE,
    RECEIPTS_FILE,
    SALES_FILE,
    SKU_FILE,
    STOCK_FILE,
    SUPPLIERS_FILE,
    DataError,
    detect_data_date,
    load_dataset,
)

RAW_DIR = Path("data/raw/share")
needs_raw = pytest.mark.skipif(not RAW_DIR.is_dir(), reason="нет data/raw/share")

# Минимальные выгрузки для синтетических тестов загрузки.
SKU_CSV = """sku_id,category,brand_id,supplier_id,supply_type,uom,weight_kg,warranty_months,target_cover_days,price_decile,first_sale_month,last_sale_month,months_with_sales,qty_total,avg_markup_pct_12m,on_order_qty,created
SKU-A,space_heaters,BRD-1,SUP-1,import,pcs,4.5,24,0,7.0,2023-01,2026-08,44,4400,45.0,300,2022-11
SKU-B,sanitary_faucets,BRD-2,SUP-2,import,pcs,,0,60,3.0,2023-01,2026-08,44,2200,,0,2022-12
"""

SALES_CSV = """sku_id,category,month,qty_sold,qty_returned,orders_cnt,markup_pct,partial_month
SKU-A,space_heaters,2026-06,10,0,2,45,0
SKU-A,space_heaters,2026-07,12,0,2,45,0
SKU-A,space_heaters,2026-08,5,0,1,45,1
SKU-A,space_heaters,2026-09,99,0,1,45,0
SKU-B,sanitary_faucets,2026-07,50,2,5,30,0
SKU-GHOST,sanitary_faucets,2026-07,7,0,1,30,0
SKU-B,sanitary_faucets,не-месяц,3,0,1,30,0
"""

STOCK_CSV = """sku_id,month,stock_qty
SKU-A,2026-06,200
SKU-A,2026-07,180
SKU-A,2026-08,150
SKU-A,2026-09,140
SKU-B,2026-07,120
"""

ORDERS_CSV = """po_id,sku_id,supplier_id,supply_type,order_date,promised_date,planned_lead_days,status,qty_ordered,qty_confirmed,qty_received,qty_pending
PO-1,SKU-A,SUP-1,import,2026-05-01,2026-09-01,123,open,500,500,0,300
PO-2,SKU-A,SUP-1,import,2026-09-05,2027-01-05,122,open,100,100,0,100
PO-3,SKU-B,SUP-2,import,2024-07-28,2024-07-15,-13,completed,100,100,100,0
PO-4,SKU-B,SUP-2,import,не-дата,2026-07-15,10,closed,50,50,0,50
PO-5,SKU-B,SUP-2,import,2026-08-05,2026-12-05,122,open,80,80,0,80
"""

RECEIPTS_CSV = """sku_id,receipt_date,qty,supplier_id,supply_type
SKU-A,2026-08-01,600,SUP-1,import
SKU-A,2026-09-10,100,SUP-1,import
SKU-B,не-дата,50,SUP-2,import
"""

SUPPLIERS_CSV = """supplier_id,supply_type,receipts_cnt,skus,median_days_between_receipts,median_planned_lead_days
SUP-1,import,3,2,70.0,123.0
SUP-2,import,1,1,,101.0
"""

FILES = {
    SKU_FILE: SKU_CSV,
    SALES_FILE: SALES_CSV,
    STOCK_FILE: STOCK_CSV,
    ORDERS_FILE: ORDERS_CSV,
    RECEIPTS_FILE: RECEIPTS_CSV,
    SUPPLIERS_FILE: SUPPLIERS_CSV,
}


def write_share(tmp_path: Path, **overrides: str) -> Path:
    """Раскладывает шесть CSV выгрузки в папку; overrides заменяют содержимое файла."""
    content = FILES | overrides
    for name, text in content.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


# --- реальные данные -------------------------------------------------------


@needs_raw
def test_load_real_dataset_types_and_counts():
    ds = load_dataset(RAW_DIR, pd.Timestamp("2026-08-13"))

    assert len(ds.sku) == 791
    assert int(ds.sku["avg_markup_pct_12m"].isna().sum()) == 378
    assert ds.sku["avg_markup_pct_12m"].dtype == "float64"
    assert ds.sales["month"].dtype == pd.PeriodDtype("M")
    assert ds.stock["month"].dtype == pd.PeriodDtype("M")
    assert pd.api.types.is_datetime64_any_dtype(ds.orders["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(ds.orders["promised_date"])
    assert pd.api.types.is_datetime64_any_dtype(ds.receipts["receipt_date"])
    assert ds.today == pd.Timestamp("2026-08-13")
    assert ds.issues == []


@needs_raw
def test_detect_data_date_on_real_data():
    ds = load_dataset(RAW_DIR, None)
    assert detect_data_date(ds.orders, ds.receipts, ds.sales) == pd.Timestamp("2026-08-01")
    assert ds.today == pd.Timestamp("2026-08-01")


@needs_raw
def test_explicit_today_wins_over_data_date():
    assert load_dataset(RAW_DIR, pd.Timestamp("2026-08-13")).today == pd.Timestamp("2026-08-13")
    assert load_dataset(RAW_DIR, None).today == pd.Timestamp("2026-08-01")


# --- проверки файлов -------------------------------------------------------


def test_missing_file_message(tmp_path):
    write_share(tmp_path)
    (tmp_path / SKU_FILE).unlink()
    with pytest.raises(DataError) as err:
        load_dataset(tmp_path, pd.Timestamp("2026-08-13"))
    assert str(err.value) == f"Нет файла {SKU_FILE} в {tmp_path}"


def test_missing_column_message(tmp_path):
    broken = SKU_CSV.replace("category,", "", 1)
    write_share(tmp_path, **{SKU_FILE: broken})
    with pytest.raises(DataError) as err:
        load_dataset(tmp_path, pd.Timestamp("2026-08-13"))
    message = str(err.value)
    assert "category" in message and SKU_FILE in message


# --- битые строки и обрезка ------------------------------------------------


def dropped(issues: list[str], name: str, reason: str) -> int:
    """Сколько строк файла отброшено по указанной причине — из строк issues."""
    total = 0
    for line in issues:
        if line.startswith(name) and reason in line:
            total += int(line.split("отброшено строк ")[1].split(" ")[0])
    return total


def test_bad_rows_dropped_and_reported(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))

    # неизвестный sku_id в продажах и неразбираемый месяц
    assert "SKU-GHOST" not in set(ds.sales["sku_id"])
    assert ds.sales["month"].notna().all()
    assert dropped(ds.issues, SALES_FILE, "неизвестный sku_id") == 1
    assert dropped(ds.issues, SALES_FILE, "неразбираемый месяц") == 1
    # неразбираемые даты в заказах и приходах
    assert ds.orders["order_date"].notna().all()
    assert (ds.orders["po_id"] != "PO-4").all()
    assert ds.receipts["receipt_date"].notna().all()
    assert dropped(ds.issues, ORDERS_FILE, "неразбираемая дата заказа") == 1
    assert dropped(ds.issues, RECEIPTS_FILE, "неразбираемая дата прихода") == 1


def test_issues_are_russian_and_counted(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))
    assert ds.issues
    assert all("отброшено строк" in line for line in ds.issues)
    assert all(line.startswith((SALES_FILE, ORDERS_FILE, RECEIPTS_FILE, STOCK_FILE)) for line in ds.issues)


def test_future_rows_trimmed(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))

    assert ds.sales["month"].max() == pd.Period("2026-08", freq="M")
    assert ds.stock["month"].max() == pd.Period("2026-08", freq="M")
    assert ds.orders["order_date"].max() <= ds.today
    assert ds.receipts["receipt_date"].max() <= ds.today
    assert "PO-2" not in set(ds.orders["po_id"])


def test_earlier_today_trims_more(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-07-01"))

    assert ds.sales["month"].max() == pd.Period("2026-07", freq="M")
    assert ds.stock["month"].max() == pd.Period("2026-07", freq="M")
    assert ds.receipts.empty
    assert ds.today == pd.Timestamp("2026-07-01")


def test_negative_lead_days_kept(tmp_path):
    """Отрицательное плечо в данных не чиним — только загружаем."""
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))
    row = ds.orders.loc[ds.orders["po_id"] == "PO-3", "planned_lead_days"]
    assert row.tolist() == [-13]


def test_detect_data_date_uses_orders_and_receipts(tmp_path):
    """Последняя из дат заказов (2026-08-05) и приходов (2026-08-01) — дата заказа."""
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-20"))
    assert detect_data_date(ds.orders, ds.receipts, ds.sales) == pd.Timestamp("2026-08-05")


def test_detect_data_date_falls_back_to_sales(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))
    empty_orders = ds.orders.iloc[0:0]
    empty_receipts = ds.receipts.iloc[0:0]
    assert detect_data_date(empty_orders, empty_receipts, ds.sales) == pd.Timestamp("2026-08-31")


def test_detect_data_date_without_any_dates(tmp_path):
    ds = load_dataset(write_share(tmp_path), pd.Timestamp("2026-08-13"))
    empty = ds.sales.iloc[0:0]
    with pytest.raises(DataError):
        detect_data_date(ds.orders.iloc[0:0], ds.receipts.iloc[0:0], empty)


# --- фабрика синтетического набора -----------------------------------------


def test_factory_dataset_types(make_dataset_factory):
    """Набор из conftest пригоден для тестов: типы и связи как у реальной выгрузки."""
    ds = make_dataset_factory()

    assert set(ds.sku["sku_id"]) == {"SKU-A", "SKU-B", "SKU-C"}
    assert len(ds.sales) == 3 * 44
    assert ds.sales["month"].dtype == pd.PeriodDtype("M")
    assert ds.sales["month"].min() == pd.Period("2023-01", freq="M")
    assert ds.sales["month"].max() == pd.Period("2026-08", freq="M")
    assert ds.sales.loc[ds.sales["month"] == pd.Period("2026-08", freq="M"), "partial_month"].eq(1).all()
    assert ds.stock["month"].min() == pd.Period("2024-01", freq="M")
    assert pd.api.types.is_datetime64_any_dtype(ds.orders["order_date"])
    assert ds.receipts["receipt_date"].max() == pd.Timestamp("2026-08-01")
    assert ds.today == pd.Timestamp("2026-08-13")
    assert ds.issues == []


def test_factory_overrides_and_data_date(make_dataset_factory):
    """Свои списки заменяют набор целиком; today=None — дата из данных."""
    ds = make_dataset_factory(
        receipts=[{"sku_id": "SKU-A", "receipt_date": "2026-03-04", "qty": 10, "supplier_id": "SUP-1"}],
        orders=[],
        today=None,
    )
    assert len(ds.receipts) == 1
    assert ds.orders.empty
    assert ds.today == pd.Timestamp("2026-03-04")


def test_non_numeric_qty_dropped_empty_is_zero(tmp_path):
    """Непустое нечисловое количество — строка отброшена и записана в issues; пустая ячейка — 0."""
    sales = SALES_CSV.replace("SKU-A,space_heaters,2026-06,10,", "SKU-A,space_heaters,2026-06,bad,")
    sales = sales.replace("SKU-A,space_heaters,2026-07,12,", "SKU-A,space_heaters,2026-07,,")
    stock = STOCK_CSV.replace("SKU-B,2026-07,120", "SKU-B,2026-07,много")
    ds = load_dataset(write_share(tmp_path, **{SALES_FILE: sales, STOCK_FILE: stock}), pd.Timestamp("2026-08-13"))

    a = ds.sales[ds.sales["sku_id"] == "SKU-A"].set_index("month")["qty_sold"]
    assert pd.Period("2026-06", "M") not in a.index
    assert a[pd.Period("2026-07", "M")] == 0
    assert dropped(ds.issues, SALES_FILE, "нечисловое значение в qty_sold") == 1
    assert dropped(ds.issues, STOCK_FILE, "нечисловое значение в stock_qty") == 1
    assert (ds.stock["sku_id"] != "SKU-B").all()


def test_stock_exact_duplicate_kept_once(tmp_path):
    """Точный повтор строки снимка: остаётся одна, в issues — сколько повторов."""
    stock = STOCK_CSV + "SKU-A,2026-08,150\n"
    ds = load_dataset(write_share(tmp_path, **{STOCK_FILE: stock}), pd.Timestamp("2026-08-13"))
    a = ds.stock[(ds.stock["sku_id"] == "SKU-A") & (ds.stock["month"] == pd.Period("2026-08", "M"))]
    assert len(a) == 1 and a["stock_qty"].iloc[0] == 150
    assert f"{STOCK_FILE}: повторов снимка 1 — оставлен один" in ds.issues


def test_stock_conflicting_duplicate_is_error(tmp_path):
    """Разные остатки одного SKU за один месяц — DataError с SKU и месяцем."""
    stock = STOCK_CSV + "SKU-A,2026-08,999\n"
    with pytest.raises(DataError, match="SKU-A за 2026-08"):
        load_dataset(write_share(tmp_path, **{STOCK_FILE: stock}), pd.Timestamp("2026-08-13"))


def test_open_order_without_promised_date_reported(tmp_path):
    """Открытый заказ с неразборчивой или пустой датой прихода остаётся и попадает в issues."""
    orders = ORDERS_CSV.replace("2026-05-01,2026-09-01", "2026-05-01,bad").replace(
        "2026-08-05,2026-12-05", "2026-08-05,"
    )
    ds = load_dataset(write_share(tmp_path, **{ORDERS_FILE: orders}), pd.Timestamp("2026-08-13"))
    assert {"PO-1", "PO-5"} <= set(ds.orders["po_id"])
    assert f"{ORDERS_FILE}: у 2 открытых заказов нет даты прихода — считаются просроченными" in ds.issues
