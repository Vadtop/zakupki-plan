"""Тесты команды запуска: аргументы, файлы результата, ошибки данных, детерминизм."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from zakupki import config as C
from zakupki.cli import PLAN_FILE, RISKS_FILE, SUMMARY_FILE, main

TODAY = "2026-08-13"
DATA_DATE = "2026-08-01"

SKU_COLUMNS = [
    "sku_id", "category", "brand_id", "supplier_id", "supply_type", "uom", "weight_kg",
    "warranty_months", "target_cover_days", "price_decile", "first_sale_month", "last_sale_month",
    "months_with_sales", "qty_total", "avg_markup_pct_12m", "on_order_qty", "created",
]
SALES_COLUMNS = [
    "sku_id", "category", "month", "qty_sold", "qty_returned", "orders_cnt", "markup_pct",
    "partial_month",
]
STOCK_COLUMNS = ["sku_id", "month", "stock_qty"]
ORDER_COLUMNS = [
    "po_id", "sku_id", "supplier_id", "supply_type", "order_date", "promised_date",
    "planned_lead_days", "status", "qty_ordered", "qty_confirmed", "qty_received", "qty_pending",
]
RECEIPT_COLUMNS = ["sku_id", "receipt_date", "qty", "supplier_id", "supply_type"]
SUPPLIER_COLUMNS = [
    "supplier_id", "supply_type", "receipts_cnt", "skus", "median_days_between_receipts",
    "median_planned_lead_days",
]

# Три SKU: сезонный, ровный и мёртвый — чтобы в плане были разные статусы.
SKUS = [
    {
        "sku_id": "SKU-A", "category": "space_heaters", "brand_id": "BRD-1", "supplier_id": "SUP-1",
        "supply_type": "import", "uom": "pcs", "weight_kg": 4.5, "warranty_months": 24,
        "target_cover_days": 0, "price_decile": 7.0, "first_sale_month": "2023-01",
        "last_sale_month": "2026-08", "months_with_sales": 44, "qty_total": 4400,
        "avg_markup_pct_12m": 45.0, "on_order_qty": 300, "created": "2022-11",
    },
    {
        "sku_id": "SKU-B", "category": "sanitary_faucets", "brand_id": "BRD-2", "supplier_id": "SUP-2",
        "supply_type": "import", "uom": "pcs", "weight_kg": 1.2, "warranty_months": 0,
        "target_cover_days": 0, "price_decile": 3.0, "first_sale_month": "2023-01",
        "last_sale_month": "2026-08", "months_with_sales": 44, "qty_total": 2200,
        "avg_markup_pct_12m": 30.0, "on_order_qty": 0, "created": "2022-12",
    },
    {
        "sku_id": "SKU-C", "category": "string_trimmers", "brand_id": "BRD-3", "supplier_id": "SUP-1",
        "supply_type": "unknown", "uom": "set", "weight_kg": "", "warranty_months": 0,
        "target_cover_days": 0, "price_decile": "", "first_sale_month": "2023-01",
        "last_sale_month": "2024-06", "months_with_sales": 18, "qty_total": 900,
        "avg_markup_pct_12m": "", "on_order_qty": 0, "created": "2022-10",
    },
]


def _write(name: str, columns: list[str], rows: list[dict], path: Path) -> None:
    """Пишет CSV с заданными колонками."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_data_dir(root: Path) -> Path:
    """Собирает маленькую выгрузку ERP в папке root и возвращает её."""
    months = [str(m) for m in pd.period_range("2023-01", "2026-08", freq="M")]
    sales: list[dict] = []
    for i, month in enumerate(months):
        partial = 1 if month == "2026-08" else 0
        sold_c = 50 if month <= "2024-06" else 0
        sales.append({
            "sku_id": "SKU-A", "category": "space_heaters", "month": month,
            "qty_sold": 100 if month.endswith("-10") else 5, "qty_returned": 0,
            "orders_cnt": 1, "markup_pct": 45.0, "partial_month": partial,
        })
        sales.append({
            "sku_id": "SKU-B", "category": "sanitary_faucets", "month": month,
            "qty_sold": 40 + (i * 7) % 21, "qty_returned": 2, "orders_cnt": 1, "markup_pct": 30.0,
            "partial_month": partial,
        })
        sales.append({
            "sku_id": "SKU-C", "category": "string_trimmers", "month": month,
            "qty_sold": sold_c, "qty_returned": 0, "orders_cnt": 1, "markup_pct": 20.0,
            "partial_month": partial,
        })
    stock: list[dict] = []
    for month in [str(m) for m in pd.period_range("2024-01", "2026-08", freq="M")]:
        stock.append({"sku_id": "SKU-A", "month": month, "stock_qty": 200})
        stock.append({"sku_id": "SKU-B", "month": month, "stock_qty": 120})
        if month <= "2024-06":
            stock.append({"sku_id": "SKU-C", "month": month, "stock_qty": 30})
    orders = [
        {
            "po_id": "PO-1", "sku_id": "SKU-A", "supplier_id": "SUP-1", "supply_type": "import",
            "order_date": "2026-05-01", "promised_date": "2026-09-01", "planned_lead_days": 123,
            "status": "open", "qty_ordered": 500, "qty_confirmed": 500, "qty_received": 0,
            "qty_pending": 300,
        },
        {
            "po_id": "PO-2", "sku_id": "SKU-A", "supplier_id": "SUP-1", "supply_type": "import",
            "order_date": "2025-01-10", "promised_date": "2025-05-13", "planned_lead_days": 123,
            "status": "closed", "qty_ordered": 400, "qty_confirmed": 400, "qty_received": 380,
            "qty_pending": 20,
        },
        {
            "po_id": "PO-3", "sku_id": "SKU-B", "supplier_id": "SUP-2", "supply_type": "import",
            "order_date": "2026-03-01", "promised_date": "2026-07-01", "planned_lead_days": 122,
            "status": "completed", "qty_ordered": 200, "qty_confirmed": 200, "qty_received": 200,
            "qty_pending": 0,
        },
    ]
    receipts = [
        {"sku_id": "SKU-A", "receipt_date": "2024-05-13", "qty": 200, "supplier_id": "SUP-1",
         "supply_type": "import"},
        {"sku_id": "SKU-A", "receipt_date": "2025-05-13", "qty": 400, "supplier_id": "SUP-1",
         "supply_type": "import"},
        {"sku_id": "SKU-A", "receipt_date": DATA_DATE, "qty": 600, "supplier_id": "SUP-1",
         "supply_type": "import"},
        {"sku_id": "SKU-B", "receipt_date": "2025-06-01", "qty": 300, "supplier_id": "SUP-2",
         "supply_type": "import"},
    ]
    suppliers = [
        {"supplier_id": "SUP-1", "supply_type": "import", "receipts_cnt": 3, "skus": 2,
         "median_days_between_receipts": 70.0, "median_planned_lead_days": 123.0},
        {"supplier_id": "SUP-2", "supply_type": "import", "receipts_cnt": 1, "skus": 1,
         "median_days_between_receipts": "", "median_planned_lead_days": 101.0},
    ]
    root.mkdir(parents=True, exist_ok=True)
    _write("sku_dim.csv", SKU_COLUMNS, SKUS, root / "sku_dim.csv")
    _write("sales_monthly.csv", SALES_COLUMNS, sales, root / "sales_monthly.csv")
    _write("stock_monthly.csv", STOCK_COLUMNS, stock, root / "stock_monthly.csv")
    _write("purchase_orders.csv", ORDER_COLUMNS, orders, root / "purchase_orders.csv")
    _write("receipts.csv", RECEIPT_COLUMNS, receipts, root / "receipts.csv")
    _write("supplier_dim.csv", SUPPLIER_COLUMNS, suppliers, root / "supplier_dim.csv")
    return root


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Запуск из пустой папки: .env проекта не читается."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Маленькая выгрузка на диске."""
    return write_data_dir(tmp_path / "data")


def read_plan(out_dir: Path) -> pd.DataFrame:
    """Читает plan.csv как DataFrame."""
    return pd.read_csv(out_dir / PLAN_FILE)


def test_missing_dir_prints_russian_error(tmp_path, capsys):
    """Нет папки выгрузки — сообщение по-русски, код 2, без трассировки."""
    code = main(["run", "--data", str(tmp_path / "nope"), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert "Нет файла sku_dim.csv" in captured.err
    assert "Traceback" not in captured.err


def test_missing_column_names_file_and_column(tmp_path, capsys):
    """Нет колонки — в сообщении её имя и файл."""
    root = write_data_dir(tmp_path / "data")
    path = root / "sku_dim.csv"
    text = path.read_text(encoding="utf-8").replace("avg_markup_pct_12m,", "")
    path.write_text(text, encoding="utf-8")
    code = main(["run", "--data", str(root), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert "sku_dim.csv" in captured.err
    assert "avg_markup_pct_12m" in captured.err


def test_writes_three_files_and_prints_five_to_eight_lines(data_dir, tmp_path, capsys):
    """Прогон пишет plan.csv, risks.csv, summary.md и печатает 5–8 строк итога."""
    out_dir = tmp_path / "out"
    code = main(["run", "--data", str(data_dir), "--today", TODAY, "--no-llm", "--out", str(out_dir)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert code == 0
    assert (out_dir / PLAN_FILE).is_file()
    assert (out_dir / RISKS_FILE).is_file()
    assert (out_dir / SUMMARY_FILE).is_file()
    assert 5 <= len(lines) <= 8
    assert any("Дата расчёта" in ln for ln in lines)


def test_plan_columns_match_interfaces(data_dir, tmp_path):
    """plan.csv — колонки ровно из interfaces.md, по строке на SKU."""
    out_dir = tmp_path / "out"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--out", str(out_dir)])
    plan = read_plan(out_dir)
    assert list(plan.columns) == [
        "sku_id", "category", "supplier_id", "status", "flags", "order_qty", "order_date",
        "stock_now", "on_order", "on_order_expected", "lead_days", "cover_days", "cover_months",
        "stockout_date", "safety_stock", "gap_months", "reliability", "avg_markup_pct_12m",
        "level", "trend", "short_history", "need_qty", "explanation",
    ]
    assert sorted(plan["sku_id"]) == ["SKU-A", "SKU-B", "SKU-C"]


def test_without_today_uses_data_date(data_dir, tmp_path, capsys):
    """Без --today берётся дата данных и печатается в итоге."""
    code = main(["run", "--data", str(data_dir), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0
    assert DATA_DATE in out
    assert "дата данных" in out


def test_earlier_today_gives_other_plan(data_dir, tmp_path):
    """Другой --today — другой plan.csv."""
    first, second = tmp_path / "first", tmp_path / "second"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--out", str(first)])
    main(["run", "--data", str(data_dir), "--today", "2026-06-01", "--out", str(second)])
    assert (first / PLAN_FILE).read_bytes() != (second / PLAN_FILE).read_bytes()


def test_two_runs_are_byte_identical(data_dir, tmp_path):
    """Один вход → побайтно одинаковые plan.csv и risks.csv."""
    first, second = tmp_path / "first", tmp_path / "second"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--out", str(first)])
    main(["run", "--data", str(data_dir), "--today", TODAY, "--out", str(second)])
    assert (first / PLAN_FILE).read_bytes() == (second / PLAN_FILE).read_bytes()
    assert (first / RISKS_FILE).read_bytes() == (second / RISKS_FILE).read_bytes()


def test_min_markup_flag_moves_sku_to_low_markup(data_dir, tmp_path):
    """--min-markup меняет порог: при 50 SKU с наценкой 30 попадает в НИЗКАЯ НАЦЕНКА."""
    low, high = tmp_path / "low", tmp_path / "high"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--min-markup", "25", "--out", str(low)])
    main(["run", "--data", str(data_dir), "--today", TODAY, "--min-markup", "50", "--out", str(high)])
    low_plan, high_plan = read_plan(low), read_plan(high)
    assert (high_plan["status"] == C.LOW_MARKUP).sum() > (low_plan["status"] == C.LOW_MARKUP).sum()
    assert "SKU-B" in set(high_plan.loc[high_plan["status"] == C.LOW_MARKUP, "sku_id"])


def test_z_flag_reaches_safety_stock(data_dir, tmp_path):
    """--z идёт в Config: при z=0 страховой запас нулевой, при z=3 — больше нуля."""
    zero, big = tmp_path / "zero", tmp_path / "big"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--z", "0", "--out", str(zero)])
    main(["run", "--data", str(data_dir), "--today", TODAY, "--z", "3", "--out", str(big)])
    assert read_plan(zero)["safety_stock"].sum() == 0
    assert read_plan(big)["safety_stock"].sum() > 0


def test_cover_days_flag_changes_order_qty(data_dir, tmp_path):
    """--cover-days идёт в Config: больше покрытие — больше заказ."""
    short, long = tmp_path / "short", tmp_path / "long"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--cover-days", "30", "--out", str(short)])
    main(["run", "--data", str(data_dir), "--today", TODAY, "--cover-days", "180", "--out", str(long)])
    assert read_plan(long)["order_qty"].sum() > read_plan(short)["order_qty"].sum()


def test_summary_has_required_sections(data_dir, tmp_path):
    """summary.md: дата, статусы, штук к заказу, просроченные, надёжность, качество данных, факты."""
    out_dir = tmp_path / "out"
    main(["run", "--data", str(data_dir), "--today", TODAY, "--out", str(out_dir)])
    text = (out_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    assert f"Дата расчёта: {TODAY}" in text
    assert "К заказу всего:" in text
    assert "Просроченных заказов поставщиков:" in text
    assert "Общая надёжность поставщиков:" in text
    assert "нулевым приходом" in text
    assert "## Качество данных" in text
    assert "## Проверочные факты" in text
    assert "space_heaters" in text
    assert "Пояснения: шаблон" in text
    for status in C.STATUS_ORDER:
        assert f"- {status}:" in text


def test_summary_lists_dropped_rows(tmp_path, capsys):
    """Битые строки не роняют расчёт и попадают в «Качество данных»."""
    root = write_data_dir(tmp_path / "data")
    path = root / "sales_monthly.csv"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("SKU-НЕТ,space_heaters,2026-01,5,0,1,45.0,0\n")
    out_dir = tmp_path / "out"
    code = main(["run", "--data", str(root), "--today", TODAY, "--out", str(out_dir)])
    text = (out_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    assert code == 0
    assert "неизвестный sku_id" in text
    assert "sales_monthly.csv" in text


def test_bad_today_prints_russian_error(data_dir, tmp_path, capsys):
    """Кривой --today — сообщение по-русски и код 2."""
    code = main(["run", "--data", str(data_dir), "--today", "вчера", "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert "--today" in captured.err
    assert "Traceback" not in captured.err


def test_tests_isolated_from_env_and_network(tmp_path):
    """Тесты не видят .env текущей папки, LLM_* окружения и не могут выйти в сеть."""
    import os
    import urllib.request

    assert Path.cwd() == tmp_path
    assert not [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if k in os.environ]
    with pytest.raises(RuntimeError):
        urllib.request.urlopen("http://llm.test/v1/chat/completions", timeout=1)


def test_summary_season_month_in_prepositional_case(small_dataset):
    """«пик индекса в октябре», а не «в октябрь»; дефицит в статусах и рисках — одно число."""
    from zakupki.config import Config
    from zakupki.demand import build_demand
    from zakupki.forecast import forecast, seasonal_index
    from zakupki.plan import build_plan
    from zakupki.report import build_summary

    ds = small_dataset
    cfg = Config(today=ds.today)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)
    text = build_summary(result, ds, "", {"mode": "template"}, None, index)
    assert "пик индекса в октябре" in text
    shortage = int((result.plan["status"] == C.SHORTAGE).sum())
    assert text.count(f"- {C.SHORTAGE}: {shortage}") == 2


def test_without_today_warns_about_data_date(data_dir, tmp_path, capsys):
    """Без --today — заметное предупреждение: дата из последнего заказа/прихода, передайте --today."""
    code = main(["run", "--data", str(data_dir), "--no-llm", "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 0
    assert "ВНИМАНИЕ" in err
    assert "01.08.2026" in err
    assert "последнего заказа/прихода" in err
    assert "--today" in err


def test_with_today_no_data_date_warning(data_dir, tmp_path, capsys):
    """С явным --today предупреждения нет."""
    main(["run", "--data", str(data_dir), "--today", TODAY, "--no-llm", "--out", str(tmp_path / "out")])
    assert "ВНИМАНИЕ" not in capsys.readouterr().err
