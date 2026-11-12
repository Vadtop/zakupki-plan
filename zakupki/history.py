"""Снимки запусков в out/history/<дата>/ и сравнение плана с прошлым снимком."""

import shutil
from pathlib import Path

import pandas as pd

from zakupki import config as C

HISTORY_DIR = "history"
SNAPSHOT_PLAN = "plan.csv"
DATE_FORMAT = "%Y-%m-%d"


def previous_snapshot(out_dir: Path, today: pd.Timestamp) -> pd.DataFrame | None:
    """Последний снимок плана с датой не позже today; нет снимков — None."""
    root = Path(out_dir) / HISTORY_DIR
    if not root.is_dir():
        return None
    limit = pd.Timestamp(today).normalize()
    found: list[tuple[pd.Timestamp, Path]] = []
    for folder in root.iterdir():
        stamp = pd.to_datetime(folder.name, format=DATE_FORMAT, errors="coerce")
        if folder.is_dir() and not pd.isna(stamp) and stamp <= limit and (folder / SNAPSHOT_PLAN).exists():
            found.append((stamp, folder))
    if not found:
        return None
    _, latest = max(found)
    plan = pd.read_csv(latest / SNAPSHOT_PLAN, dtype={"sku_id": str})
    for column in ("order_date", "stockout_date"):
        if column in plan:
            plan[column] = pd.to_datetime(plan[column])
    if "flags" in plan:
        plan["flags"] = plan["flags"].astype("object").fillna("")
    return plan


def save_snapshot(out_dir: Path, today: pd.Timestamp, files: list[Path]) -> Path:
    """Копирует файлы результата в out/history/<дата>/, перезаписывая снимок этой даты."""
    target = Path(out_dir) / HISTORY_DIR / pd.Timestamp(today).strftime(DATE_FORMAT)
    target.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target / Path(file).name)
    return target


def _in_risk(plan: pd.DataFrame) -> set[str]:
    """SKU с риском дефицита — только основной статус, как в risks.csv."""
    return set(plan.loc[plan["status"] == C.SHORTAGE, "sku_id"])


def _iso(value) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime(DATE_FORMAT)


def diff_plans(prev: pd.DataFrame | None, cur: pd.DataFrame) -> dict:
    """Что изменилось: новые к заказу, количества, отмены, пропавшие, сдвиги дат, ушедшие и пришедшие в риск."""
    empty = {
        "new_orders": [], "qty_changes": [], "cancelled": [], "gone": [], "date_shifts": [],
        "left_risk": [], "new_risk": [],
    }
    if prev is None:
        return {"first_run": True, **empty}
    old = prev.set_index("sku_id")
    new = cur.set_index("sku_id")

    new_orders, qty_changes, cancelled, date_shifts = [], [], [], []
    for sku_id, row in new.sort_index(kind="mergesort").iterrows():
        was = old.loc[sku_id] if sku_id in old.index else None
        had = was is not None and was["order_qty"] > 0
        if row["order_qty"] <= 0:
            if had:
                cancelled.append({"sku_id": sku_id, "status": row["status"], "from": int(was["order_qty"])})
            continue
        if not had:
            new_orders.append({
                "sku_id": sku_id, "status": row["status"], "order_qty": int(row["order_qty"]),
                "order_date": _iso(row["order_date"]),
            })
            continue
        if int(row["order_qty"]) != int(was["order_qty"]):
            qty_changes.append({"sku_id": sku_id, "from": int(was["order_qty"]), "to": int(row["order_qty"])})
        if pd.notna(row["order_date"]) and pd.notna(was["order_date"]):
            days = int((row["order_date"] - was["order_date"]).days)
            if days != 0:
                date_shifts.append({
                    "sku_id": sku_id, "from": _iso(was["order_date"]), "to": _iso(row["order_date"]),
                    "days": days,
                })

    # Был заказ, а в новой выгрузке SKU нет вообще: прежняя рекомендация не должна пропасть молча.
    gone = [
        {"sku_id": sku_id, "status": row["status"], "from": int(row["order_qty"])}
        for sku_id, row in old.sort_index(kind="mergesort").iterrows()
        if sku_id not in new.index and row["order_qty"] > 0
    ]

    before, after = _in_risk(prev), _in_risk(cur)
    return {
        "first_run": False,
        "new_orders": new_orders,
        "qty_changes": qty_changes,
        "cancelled": cancelled,
        "gone": gone,
        "date_shifts": date_shifts,
        "left_risk": [{"sku_id": s} for s in sorted(before - after)],
        "new_risk": [{"sku_id": s} for s in sorted(after - before)],
    }
