"""Прогон по реальной выгрузке; пропускается, если папки с данными нет."""

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from zakupki import config as C
from zakupki.cli import main
from zakupki.plan import PLAN_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "share"

pytestmark = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="нет папки data/raw/share")


def test_real_run_matches_known_numbers(tmp_path):
    """Известные цифры: 791 SKU, 334 мёртвых, 44 с продажами без наценки (ревью 4), 68 с низкой наценкой."""
    out_dir = tmp_path / "out"
    code = main(["run", "--data", str(DATA_DIR), "--today", "2026-08-13", "--no-llm", "--out", str(out_dir)])
    assert code == 0
    plan = pd.read_csv(out_dir / "plan.csv")
    assert len(plan) == 791
    assert list(plan.columns) == PLAN_COLUMNS
    counts = plan["status"].value_counts()
    assert counts[C.DEAD] == 334
    assert counts[C.NO_MARKUP] == 44
    assert counts[C.LOW_MARKUP] == 68
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "Дата расчёта: 2026-08-13" in summary
    assert "Пояснения: шаблон" in summary


def test_real_run_plan_and_risks_agree(tmp_path):
    """Строки «заказать» с количеством, дефицит в рисках — ровно по основному статусу."""
    out_dir = tmp_path / "out"
    main(["run", "--data", str(DATA_DIR), "--today", "2026-08-13", "--no-llm", "--out", str(out_dir)])
    plan = pd.read_csv(out_dir / "plan.csv", parse_dates=["order_date"])
    risks = pd.read_csv(out_dir / "risks.csv")
    ordered = plan[plan["status"].isin([C.ORDER_NOW, C.PLAN, C.SHORTAGE])]
    assert (ordered["order_qty"] > 0).all()
    short = set(risks.loc[risks["risk_type"] == C.SHORTAGE, "sku_id"])
    assert short == set(plan.loc[plan["status"] == C.SHORTAGE, "sku_id"])
    dead = plan[plan["status"] == C.DEAD]
    assert not dead["flags"].fillna("").str.contains(C.SHORTAGE, regex=False).any()
    now = plan[plan["status"] == C.ORDER_NOW]
    assert (now["order_date"] >= pd.Timestamp("2025-01-01")).all()


def test_real_run_zero_stock_with_big_order_in_transit_is_excess(tmp_path):
    """Обогреватели без остатка, ~1000 шт. в пути и покрытием 35,7 мес. — избыток, а не норма."""
    out_dir = tmp_path / "out"
    main(["run", "--data", str(DATA_DIR), "--today", "2026-08-13", "--no-llm", "--out", str(out_dir)])
    plan = pd.read_csv(out_dir / "plan.csv").set_index("sku_id")
    risks = pd.read_csv(out_dir / "risks.csv")
    heaters = ["SKU-032D7B", "SKU-CE59B5", "SKU-ED0D18"]
    assert (plan.loc[heaters, "stock_now"] == 0).all()
    assert (plan.loc[heaters, "on_order"] > 800).all()
    assert (plan.loc[heaters, "status"] == C.EXCESS).all()
    assert set(heaters) <= set(risks.loc[risks["risk_type"] == C.EXCESS, "sku_id"])


def _dashboard_data(out_dir: Path) -> dict:
    html = (out_dir / "dashboard.html").read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.DOTALL)
    return json.loads(match.group(1))


def test_real_run_steps_match_order_qty(tmp_path):
    """Все 791 SKU: шаги «почему столько» в дашборде дают ровно order_qty из plan.csv."""
    out_dir = tmp_path / "out"
    main(["run", "--data", str(DATA_DIR), "--today", "2026-08-13", "--no-llm", "--out", str(out_dir)])
    plan = pd.read_csv(out_dir / "plan.csv").set_index("sku_id")
    data = _dashboard_data(out_dir)
    ordered = plan[plan["order_qty"] > 0]
    assert set(ordered.index) <= set(data["steps"])
    for sku_id, step in data["steps"].items():
        need = step["forecast"] + step["safety"] - step["stock"] - step["transit"]
        qty = 0 if need <= 0 else math.ceil(round(need, 6))
        if qty and step["lot"] > 1:
            qty = math.ceil(qty / step["lot"]) * step["lot"]
        assert qty == plan.loc[sku_id, "need_qty"], sku_id
        if plan.loc[sku_id, "order_qty"] > 0:
            assert qty == plan.loc[sku_id, "order_qty"], sku_id
    orders = data["orders"]["now"] + data["orders"]["later"]
    assert sum(g["qty"] for g in orders) == int(plan["order_qty"].sum())


def test_real_run_dashboard_deterministic(tmp_path):
    """Два прогона в разные папки — одинаковый dashboard.html, кроме подписи времени пересчёта."""
    stamp = re.compile(rb'"generated_at":"[^"]*"')
    pages = []
    for name in ("a", "b"):
        out_dir = tmp_path / name
        main(["run", "--data", str(DATA_DIR), "--today", "2026-08-13", "--no-llm", "--out", str(out_dir)])
        pages.append(stamp.sub(b"", (out_dir / "dashboard.html").read_bytes()))
    assert pages[0] == pages[1]
