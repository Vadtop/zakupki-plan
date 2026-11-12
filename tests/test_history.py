"""Тесты снимков и сравнения планов."""

import pandas as pd

from zakupki import config as C
from zakupki.history import diff_plans, previous_snapshot, save_snapshot


def _plan(rows) -> pd.DataFrame:
    """rows: (sku_id, status, flags, order_qty, order_date)."""
    df = pd.DataFrame(rows, columns=["sku_id", "status", "flags", "order_qty", "order_date"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


PREV = _plan([
    ("A", C.ORDER_NOW, "", 100, "2026-09-01"),
    ("B", C.PLAN, "", 50, "2026-10-01"),
    ("C", C.SHORTAGE, "", 70, "2026-07-01"),
    ("D", C.OK, "", 0, None),
])
CUR = _plan([
    ("A", C.ORDER_NOW, "", 100, "2026-09-01"),
    ("B", C.PLAN, "", 50, "2026-10-11"),
    ("C", C.ORDER_NOW, "", 70, "2026-08-20"),
    ("D", C.ORDER_NOW, "", 30, "2026-09-05"),
    ("E", C.SHORTAGE, "", 0, None),
    ("F", C.LOW_MARKUP, C.SHORTAGE, 0, None),
])


def test_first_run():
    diff = diff_plans(None, CUR)
    assert diff["first_run"] is True
    assert diff["new_orders"] == [] and diff["date_shifts"] == []


def test_diff_between_snapshots():
    diff = diff_plans(PREV, CUR)
    assert diff["first_run"] is False
    assert [r["sku_id"] for r in diff["new_orders"]] == ["D"]
    shifts = {r["sku_id"]: r for r in diff["date_shifts"]}
    assert shifts["B"]["days"] == 10 and shifts["B"]["from"] == "2026-10-01" and shifts["B"]["to"] == "2026-10-11"
    assert shifts["C"]["days"] == 50
    assert "A" not in shifts
    assert [r["sku_id"] for r in diff["left_risk"]] == ["C"]
    assert [r["sku_id"] for r in diff["new_risk"]] == ["E"]
    assert diff["qty_changes"] == [] and diff["cancelled"] == []


def test_snapshot_roundtrip_and_same_date(tmp_path):
    """Снимок читается обратно; на ту же дату находится прежний снимок, ранее — нет."""
    plan_file = tmp_path / "plan.csv"
    PREV.to_csv(plan_file, index=False)
    out = tmp_path / "out"
    assert previous_snapshot(out, pd.Timestamp("2026-08-13")) is None
    path = save_snapshot(out, pd.Timestamp("2026-08-13"), [plan_file])
    assert path == out / "history" / "2026-08-13" and (path / "plan.csv").exists()
    same = previous_snapshot(out, pd.Timestamp("2026-08-13"))
    assert same is not None and list(same["sku_id"]) == ["A", "B", "C", "D"]
    assert same["order_date"].iloc[0] == pd.Timestamp("2026-09-01")
    assert previous_snapshot(out, pd.Timestamp("2026-08-12")) is None
    assert previous_snapshot(out, pd.Timestamp("2026-09-01")) is not None


def test_numeric_sku_id_stays_string(tmp_path):
    """sku_id «1» после снимка остаётся строкой: та же позиция не новая и не исчезнувшая."""
    plan = _plan([("1", C.ORDER_NOW, "", 120, "2026-09-01")])
    plan_file = tmp_path / "plan.csv"
    plan.to_csv(plan_file, index=False)
    out = tmp_path / "out"
    save_snapshot(out, pd.Timestamp("2026-08-13"), [plan_file])
    prev = previous_snapshot(out, pd.Timestamp("2026-08-13"))
    assert prev["sku_id"].iloc[0] == "1"
    diff = diff_plans(prev, plan)
    assert diff["new_orders"] == [] and diff["gone"] == []


def test_qty_change_and_cancel():
    """Количество у существующего заказа и отмена видны с прежним и новым значением."""
    prev = _plan([
        ("A", C.ORDER_NOW, "", 100, "2026-09-01"),
        ("B", C.PLAN, "", 50, "2026-10-01"),
        ("C", C.PLAN, "", 40, "2026-10-01"),
    ])
    cur = _plan([
        ("A", C.ORDER_NOW, "", 500, "2026-09-01"),
        ("B", C.OK, "", 0, None),
        ("C", C.PLAN, "", 40, "2026-10-01"),
    ])
    diff = diff_plans(prev, cur)
    assert diff["qty_changes"] == [{"sku_id": "A", "from": 100, "to": 500}]
    assert diff["cancelled"] == [{"sku_id": "B", "status": C.OK, "from": 50}]
    assert diff["new_orders"] == [] and diff["date_shifts"] == []
    same = diff_plans(cur, cur)
    assert same["qty_changes"] == [] and same["cancelled"] == []


def test_risk_only_by_main_status():
    """Флаг дефицита при другом основном статусе в историю риска не попадает — как в risks.csv."""
    prev = _plan([("A", C.LOW_MARKUP, "", 0, None)])
    cur = _plan([("A", C.LOW_MARKUP, C.SHORTAGE, 0, None)])
    diff = diff_plans(prev, cur)
    assert diff["new_risk"] == [] and diff["left_risk"] == []


def test_summary_lists_qty_changes_and_cancelled():
    from zakupki.report import _diff_lines

    diff = diff_plans(
        _plan([("A", C.ORDER_NOW, "", 100, "2026-09-01"), ("B", C.PLAN, "", 50, "2026-10-01")]),
        _plan([("A", C.ORDER_NOW, "", 300, "2026-09-01"), ("B", C.OK, "", 0, None)]),
    )
    lines = _diff_lines(diff)
    assert "- Изменилось количество: 1" in lines and "- Отменены: 1" in lines


def test_gone_sku_listed():
    """SKU с заказом в прошлом плане, которого нет в новом, попадает в «Пропали из выгрузки»."""
    from zakupki.report import _diff_lines

    prev = _plan([
        ("A", C.ORDER_NOW, "", 100, "2026-09-01"),
        ("B", C.PLAN, "", 50, "2026-10-01"),
        ("Z", C.OK, "", 0, None),
    ])
    cur = _plan([("B", C.PLAN, "", 50, "2026-10-01")])
    diff = diff_plans(prev, cur)
    assert diff["gone"] == [{"sku_id": "A", "status": C.ORDER_NOW, "from": 100}]
    assert diff["cancelled"] == [] and diff["new_orders"] == []
    assert "- Пропали из выгрузки: 1" in _diff_lines(diff)
    assert diff_plans(None, cur)["gone"] == [] and diff_plans(cur, cur)["gone"] == []
