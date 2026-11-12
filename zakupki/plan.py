"""План закупки: сколько заказать, основной статус и флаги, риски, календарь заказов."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from zakupki import config as C
from zakupki.config import Config
from zakupki.data import Dataset
from zakupki.simulation import run_simulation
from zakupki.supply import current_stock, lead_days, lot_size, open_orders, supplier_reliability

PLAN_COLUMNS = [
    "sku_id",
    "category",
    "supplier_id",
    "status",
    "flags",
    "order_qty",
    "order_date",
    "stock_now",
    "on_order",
    "on_order_expected",
    "lead_days",
    "cover_days",
    "cover_months",
    "stockout_date",
    "safety_stock",
    "gap_months",
    "reliability",
    "avg_markup_pct_12m",
    "level",
    "trend",
    "short_history",
    "need_qty",
    "explanation",
]
RISK_COLUMNS = [
    "risk_type",
    "sku_id",
    "category",
    "supplier_id",
    "detail",
    "gap_months",
    "excess_months",
    "po_id",
    "promised_date",
    "qty",
]
# Статусы, по которым SKU идёт в автозаказ и в календарь.
ORDERED_STATUSES = (C.SHORTAGE, C.ORDER_NOW, C.PLAN)
RISK_ORDER = [C.SHORTAGE, C.OVERDUE_PO, C.EXCESS, C.DEAD]
# Шаги расчёта количества для показа в дашборде (в plan.csv не пишутся).
BREAKDOWN_COLUMNS = [
    "sku_id",
    "need_at",
    "need_arrival",
    "need_end",
    "need_forecast",
    "need_safety",
    "need_stock",
    "need_transit",
    "need_bridge",
    "need_raw",
    "need_ceil",
    "lot",
    "lot_full",
    "sigma",
    "safety_raw",
    "safety_cap",
]


@dataclass
class PlanResult:
    plan: pd.DataFrame
    risks: pd.DataFrame
    sim: pd.DataFrame
    calendar: pd.DataFrame
    breakdown: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BREAKDOWN_COLUMNS))


def _num(value, default=float("nan")) -> float:
    return default if value is None or pd.isna(value) else float(value)


def _daily_demand(fc: pd.DataFrame, sku_id: str, today: pd.Timestamp, days: int) -> np.ndarray:
    """Прогноз по дням от today: месячный прогноз поровну на дни месяца."""
    rows = fc[fc["sku_id"] == sku_id]
    by_month = dict(zip(rows["month"], rows["qty"]))
    dates = pd.date_range(today, periods=days, freq="D")
    qty = np.array([float(by_month.get(m, 0.0)) for m in dates.to_period("M")], dtype="float64")
    return qty / dates.days_in_month.to_numpy(dtype="float64")


def _arrival_days(po: pd.DataFrame, sku_id: str, today: pd.Timestamp) -> dict[int, float]:
    """Приходы заказов в пути: номер дня от today → штук (с поправкой на надёжность)."""
    out: dict[int, float] = {}
    if po is None or po.empty:
        return out
    rows = po[po["sku_id"] == sku_id]
    for date, qty in zip(rows["expected_date"], rows["expected_qty"]):
        if pd.isna(date) or pd.isna(qty):
            continue
        day = max((pd.Timestamp(date).normalize() - today).days, 0)
        out[day] = out.get(day, 0.0) + float(qty)
    return out


def _need_parts(at, lead, cover_days, safety, lot, dem, arr, today, stock_now, max_over=math.inf) -> dict:
    """Шаги потребности на заказ, размещённый в at: он придёт через плечо и покроет cover_days.

    Остаток к приходу считается по дням: приход — в начале своего дня, продажи при пустом
    складе потеряны и не дозаказываются. Заказ должен (1) довести остаток к концу покрытия
    до страхового и (2) не дать складу опустеть до прихода заказов в пути внутри покрытия.
    """
    start = max((pd.Timestamp(at) - today).days, 0) + int(lead)
    stop = start + int(cover_days)
    stock = stock_now
    for t in range(start):
        stock = max(stock + arr.get(t, 0.0) - dem[t], 0.0)
    demand = transit = bridge = 0.0
    for t in range(start, stop):
        transit += arr.get(t, 0.0)
        demand += dem[t]
        bridge = max(bridge, demand - stock - transit)
    arrival = today + pd.Timedelta(days=start)
    parts = {
        "need_at": pd.Timestamp(at),
        "need_arrival": arrival,
        "need_end": arrival + pd.Timedelta(days=int(cover_days)),
        "need_forecast": demand,
        "need_safety": safety,
        "need_stock": stock,
        "need_transit": transit,
        "lot": lot,
    }
    total = demand + safety - stock - transit
    need = max(total, bridge)
    parts["need_bridge"] = bridge if bridge > total else 0.0
    parts["need_raw"] = need
    if need <= 0:
        parts["need_ceil"] = 0
        parts["need_qty"] = 0
        parts["lot_full"] = lot
        return parts
    qty = math.ceil(round(need, 6))
    parts["need_ceil"] = int(qty)
    rounded = int(math.ceil(qty / lot) * lot) if lot > 1 else int(qty)
    # До партии — только если это не раздувает склад больше чем на max_over штук.
    parts["lot_full"] = lot
    if rounded - qty > max_over:
        rounded, parts["lot"] = int(qty), 1
    parts["need_qty"] = rounded
    return parts


def _statuses(markup, pos, cover_target, today, cfg, in_transit: float = 0.0, sold: float = 0.0) -> dict:
    """Какие статусы сработали у SKU (по независимым условиям)."""
    order_date = pos.get("order_date", pd.NaT)
    gap = _num(pos.get("gap_months"), 0.0)
    cover_days = _num(pos.get("cover_all_days"), _num(pos.get("cover_days")))
    lead = _num(pos.get("lead_days"), 0.0)
    horizon_end = today + pd.DateOffset(months=cfg.horizon_months)
    # Пустая наценка — не значит «нет продаж»: у части SKU продажи есть, а себестоимости нет.
    no_markup = pd.isna(markup)
    dead = no_markup and sold <= 0
    gap_hit = gap > 0
    # Мёртвый не закупается, поэтому и дефицита у него не бывает.
    shortage = gap_hit and not dead
    order_now = (
        not gap_hit
        and pd.notna(order_date)
        and order_date <= today + pd.Timedelta(days=cfg.order_now_window_days)
    )
    # Избыток — по покрытию с учётом заказов в пути: деньги заморожены и в остатке, и в пути.
    has_goods = _num(pos.get("stock_now"), 0.0) + in_transit > 0
    return {
        C.DEAD: dead,
        C.LOW_MARKUP: (not no_markup) and markup < cfg.min_markup_pct,
        C.NO_MARKUP: no_markup and not dead,
        C.SHORTAGE: shortage,
        C.ORDER_NOW: order_now,
        C.EXCESS: (not dead)
        and has_goods
        and pd.notna(cover_days)
        and cover_days > lead + cover_target + cfg.excess_extra_months * cfg.days_per_month,
        C.PLAN: (not gap_hit) and (not order_now) and pd.notna(order_date) and order_date < horizon_end,
    }


def _sold_recent(ds: Dataset, cfg: Config) -> pd.Series:
    """Чистые продажи SKU (продано − возвраты) за последние полные месяцы до даты расчёта."""
    s = ds.sales
    if not len(s):
        return pd.Series(dtype="float64")
    month = ds.today.to_period("M")
    s = s[(s["partial_month"] == 0) & (s["month"] < month) & (s["month"] >= month - cfg.dead_window_months)]
    return (s["qty_sold"] - s["qty_returned"]).groupby(s["sku_id"]).sum().astype("float64")


def _reliability_by_sku(ds: Dataset, rel: pd.DataFrame) -> pd.Series:
    """Надёжность SKU по его поставщику; нет поставщика в таблице — общая."""
    total = float(rel["ordered"].sum())
    overall = float(rel["received"].sum()) / total if total else 1.0
    rates = rel.set_index("supplier_id")["reliability"].astype("float64")
    suppliers = ds.sku.set_index("sku_id")["supplier_id"]
    return suppliers.map(rates).fillna(overall).astype("float64")


def _build_rows(ds, params, fc, cfg, today, parts) -> tuple[list, list, list, list]:
    """Строки плана и риски по каждому SKU."""
    pos_by_sku = parts["positions"].set_index("sku_id")
    prm = params.set_index("sku_id") if len(params) else pd.DataFrame()
    po, lot = parts["po"], parts["lot"]
    # Старая просрочка (stale) — не «в пути»: она только в рисках.
    live = po[~po["stale"]] if len(po) else po
    on_order = live.groupby("sku_id")["qty_pending"].sum() if len(live) else pd.Series(dtype="float64")
    on_expected = live.groupby("sku_id")["expected_qty"].sum() if len(live) else pd.Series(dtype="float64")
    plan_rows, risks, steps, calendar_src = [], [], [], []
    sold = _sold_recent(ds, cfg)
    for sku in ds.sku.sort_values("sku_id", kind="mergesort").itertuples(index=False):
        sid = sku.sku_id
        pos = pos_by_sku.loc[sid].to_dict() if sid in pos_by_sku.index else {}
        target = int(sku.target_cover_days)
        cover_t = target if target > 0 else cfg.default_cover_days
        lead = int(pos.get("lead_days", parts["lead"].get(sid, cfg.lead_default_days)))
        stock_now = _num(pos.get("stock_now"), float(parts["stock"].get(sid, 0.0)))
        safety = _num(pos.get("safety_stock"), 0.0)
        order_date = pos.get("order_date", pd.NaT)
        trig = _statuses(
            sku.avg_markup_pct_12m, pos, cover_t, today, cfg, _num(on_order.get(sid), 0.0), _num(sold.get(sid), 0.0)
        )

        need, step = 0, None
        closed = False  # дефицит до прихода закроет уже заказанное, новый заказ — позже
        if sid in pos_by_sku.index and (pd.notna(order_date) or trig[C.SHORTAGE]):
            lot_n = int(lot.get(sid, 1))
            level = _num(prm["level"].get(sid), 0.0) if len(prm) else 0.0
            over = cfg.lot_max_overage_months * level if cfg.lot_max_overage_months > 0 else math.inf
            far = (order_date - today).days if pd.notna(order_date) else 0
            days = max(int(cfg.horizon_months * cfg.days_per_month), far) + lead + cover_t + 1
            dem, arr = _daily_demand(fc, sid, today, days), _arrival_days(po, sid, today)
            # Количество — на ту дату, которую увидит закупщик: срок наступил или дефицит — сегодня.
            now = trig[C.SHORTAGE] or pd.isna(order_date) or order_date <= today
            at = today if now else order_date
            step = _need_parts(at, lead, cover_t, safety, lot_n, dem, arr, today, stock_now, over)
            if trig[C.SHORTAGE] and step["need_qty"] == 0 and pd.notna(order_date) and order_date > today:
                closed = True
                step = _need_parts(order_date, lead, cover_t, safety, lot_n, dem, arr, today, stock_now, over)
            if step["need_qty"] == 0 and (closed or not now) and cover_t > 0:
                # На найденную дату заказывать нечего — ищем следующий пересмотр, где потребность есть.
                horizon_end = today + pd.DateOffset(months=cfg.horizon_months)
                probe = step["need_at"] + pd.Timedelta(days=cover_t)
                while probe < horizon_end:
                    nxt = _need_parts(probe, lead, cover_t, safety, lot_n, dem, arr, today, stock_now, over)
                    if nxt["need_qty"] > 0:
                        step, order_date = nxt, probe
                        break
                    probe += pd.Timedelta(days=cover_t)
            if closed or (not now and step["need_at"] != pos.get("order_date")):
                order_date = step["need_at"]
            calendar_src.append((sid, step["need_at"], lead, cover_t, safety, lot_n, dem, arr, stock_now, over))
        if step is not None and (closed or not now) and step["need_qty"] > 0:
            # Дата заказа могла сдвинуться: статусы заказа — по ней.
            horizon_end = today + pd.DateOffset(months=cfg.horizon_months)
            soon = order_date <= today + pd.Timedelta(days=cfg.order_now_window_days)
            trig[C.ORDER_NOW] = soon
            trig[C.PLAN] = (not soon) and order_date < horizon_end
        if step is not None:
            need = step.pop("need_qty")
            extra = {k: _num(pos.get(k)) for k in ("sigma", "safety_raw", "safety_cap")}
            steps.append({"sku_id": sid, **step, **extra})
        if need == 0:
            # Заказывать нечего: статусы заказа не ставятся.
            trig[C.ORDER_NOW] = trig[C.PLAN] = False

        # Дыру закроет уже заказанное, новый заказ не нужен: дефицит — флагом, не основным.
        demoted = need == 0 or closed
        pick = [s for s in C.STATUS_ORDER if trig.get(s) and not (s == C.SHORTAGE and demoted)]
        primary = pick[0] if pick else C.OK
        flags = [s for s in C.STATUS_ORDER if trig.get(s) and s != primary]
        short = bool(prm["short_history"].get(sid, False)) if len(prm) else False
        if short:
            flags.append(C.SHORT_HISTORY)
        order_qty = need if primary in ORDERED_STATUSES else 0

        gap = _num(pos.get("gap_months"), 0.0)
        cover_days = _num(pos.get("cover_all_days"), _num(pos.get("cover_days")))
        base = {"sku_id": sid, "category": sku.category, "supplier_id": sku.supplier_id}
        if primary == C.SHORTAGE:
            risks.append(
                {
                    **base,
                    "risk_type": C.SHORTAGE,
                    "gap_months": gap,
                    "qty": order_qty,
                    "detail": f"Обнулится раньше прихода заказа: дыра {gap:.1f} мес.",
                }
            )
        if trig[C.EXCESS]:
            extra = (cover_days - lead - cover_t) / cfg.days_per_month
            risks.append(
                {
                    **base,
                    "risk_type": C.EXCESS,
                    "excess_months": extra,
                    "detail": f"Покрытие {cover_days / cfg.days_per_month:.1f} мес., лишних {extra:.1f}",
                }
            )
        if trig[C.DEAD] and stock_now > 0:
            risks.append(
                {
                    **base,
                    "risk_type": C.DEAD,
                    "qty": stock_now,
                    "detail": f"Нет продаж за 12 мес., остаток {stock_now:.0f} шт.: распродать",
                }
            )
        plan_rows.append(
            {
                **base,
                "status": primary,
                "flags": "; ".join(flags),
                "order_qty": int(order_qty),
                "order_date": step["need_at"] if step is not None else order_date,
                "stock_now": stock_now,
                "on_order": _num(on_order.get(sid), 0.0),
                "on_order_expected": _num(on_expected.get(sid), 0.0),
                "lead_days": lead,
                "cover_days": _num(pos.get("cover_days")),
                "cover_months": _num(pos.get("cover_months")),
                "stockout_date": pos.get("stockout_date", pd.NaT),
                "safety_stock": safety,
                "gap_months": gap,
                "reliability": float(parts["reliability"].get(sid, 1.0)),
                "avg_markup_pct_12m": sku.avg_markup_pct_12m,
                "level": _num(prm["level"].get(sid)) if len(prm) else float("nan"),
                "trend": _num(prm["trend"].get(sid)) if len(prm) else float("nan"),
                "short_history": short,
                "need_qty": int(need),
                "explanation": "",
            }
        )
    return plan_rows, risks, steps, calendar_src


def _overdue_risks(po: pd.DataFrame, ds: Dataset) -> list[dict]:
    """Просроченные заказы в пути: по строке на товарную строку заказа (в одном po_id бывает несколько SKU)."""
    if po.empty:
        return []
    cats = ds.sku.set_index("sku_id")["category"]
    rows = []
    for r in po[po["overdue"]].itertuples(index=False):
        detail = (
            f"Заказ {r.po_id} просрочен, обещан {r.promised_date:%Y-%m-%d}"
            if pd.notna(r.promised_date)
            else f"Заказ {r.po_id}: нет даты прихода"
        )
        if r.stale:
            detail += f"; просрочка {r.overdue_days:.0f} дн. — уточнить у поставщика, в расчёте не учтён"
        rows.append(
            {
                "risk_type": C.OVERDUE_PO,
                "sku_id": r.sku_id,
                "category": cats.get(r.sku_id),
                "supplier_id": r.supplier_id,
                "po_id": r.po_id,
                "promised_date": r.promised_date,
                "qty": float(r.qty_pending),
                "detail": detail,
            }
        )
    return rows


def _calendar(plan: pd.DataFrame, src: list, today: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """Заказы по месяцам: первый заказ из плана и повторные на пересмотрах.

    Пересмотр — раз в default_cover_days (не реже покрытия SKU) после первого заказа; на каждом
    та же потребность с учётом уже запланированных заказов. positions — число заказов, now_* —
    срочная часть текущего месяца (срок сегодня или прошёл, риск дефицита).
    """
    months = pd.period_range(today.to_period("M"), periods=cfg.horizon_months, freq="M")
    horizon_end = today + pd.DateOffset(months=cfg.horizon_months)
    by_sku = plan.set_index("sku_id")
    ordered = plan["status"].isin(ORDERED_STATUSES) & (plan["order_qty"] > 0)
    # Сейчас избыток, но в горизонте потребность появится — будущая закупка тоже в календаре.
    later = plan["status"].isin((C.EXCESS, C.OK)) & (plan["need_qty"] > 0) & (plan["order_date"] < horizon_end)
    chosen = set(plan.loc[ordered | later, "sku_id"])
    orders: list[tuple[pd.Period, int, bool]] = []
    for sid, at, lead, cover_t, safety, lot_n, dem, arr, stock_now, over in src:
        if sid not in chosen:
            continue
        arr = dict(arr)
        row = by_sku.loc[sid]
        review = max(pd.Timestamp(row["order_date"]), today)
        qty = int(row["order_qty"] if row["order_qty"] > 0 else row["need_qty"])
        urgent = bool(row["status"] in ORDERED_STATUSES and (row["status"] == C.SHORTAGE or review <= today))
        # Пересмотр не реже покрытия SKU, иначе между заказами провалы.
        step_days = min(int(cfg.default_cover_days), int(cover_t)) if cover_t > 0 else int(cfg.default_cover_days)
        while True:
            orders.append((review.to_period("M"), qty, urgent))
            urgent = False
            day = (review - today).days + int(lead)
            arr[day] = arr.get(day, 0.0) + qty
            qty = 0
            while qty == 0 and step_days > 0:
                review += pd.Timedelta(days=step_days)
                if review >= horizon_end:
                    break
                qty = _need_parts(review, lead, cover_t, safety, lot_n, dem, arr, today, stock_now, over)["need_qty"]
            if qty == 0:
                break
    cal = pd.DataFrame({"month": months, "positions": 0, "qty": 0, "now_positions": 0, "now_qty": 0})
    if orders:
        frame = pd.DataFrame(orders, columns=["month", "qty", "urgent"])
        grouped = frame.groupby("month")["qty"].agg(["size", "sum"])
        cal["positions"] = grouped["size"].reindex(months).fillna(0).astype("int64").to_numpy()
        cal["qty"] = grouped["sum"].reindex(months).fillna(0).astype("int64").to_numpy()
        # Срочная часть: срок сегодня или прошёл, риск дефицита — только первый заказ SKU.
        urgent = frame[frame["urgent"]].groupby("month")["qty"].agg(["size", "sum"])
        cal["now_positions"] = urgent["size"].reindex(months).fillna(0).astype("int64").to_numpy()
        cal["now_qty"] = urgent["sum"].reindex(months).fillna(0).astype("int64").to_numpy()
    return cal


def build_plan(ds: Dataset, demand, index, params, fc, cfg: Config) -> PlanResult:
    """Собирает план закупки: количества, статусы, риски, симуляция и календарь."""
    today = pd.Timestamp(ds.today).normalize()
    rel = supplier_reliability(ds, cfg)
    po = open_orders(ds, rel, cfg)
    lead = lead_days(ds, cfg)
    stock = current_stock(ds)
    positions, sim = run_simulation(stock, fc, po, lead, params, cfg, today)
    parts = {
        "positions": positions,
        "sim": sim,
        "po": po,
        "lead": lead,
        "stock": stock,
        "lot": lot_size(ds, cfg),
        "reliability": _reliability_by_sku(ds, rel),
    }
    plan_rows, risk_rows, steps, calendar_src = _build_rows(ds, params, fc, cfg, today, parts)
    risk_rows += _overdue_risks(po, ds)

    plan = pd.DataFrame(plan_rows, columns=PLAN_COLUMNS)
    plan["order_date"] = pd.to_datetime(plan["order_date"])
    plan["stockout_date"] = pd.to_datetime(plan["stockout_date"])
    plan["_rank"] = plan["status"].map({s: i for i, s in enumerate(C.STATUS_ORDER)})
    plan = plan.sort_values(["_rank", "order_date", "sku_id"], kind="mergesort")
    plan = plan.drop(columns="_rank").reset_index(drop=True)

    risks = pd.DataFrame(risk_rows, columns=RISK_COLUMNS)
    risks["promised_date"] = pd.to_datetime(risks["promised_date"])
    risks["_rank"] = risks["risk_type"].map({s: i for i, s in enumerate(RISK_ORDER)})
    risks["_po"] = risks["po_id"].astype("object").fillna("")
    risks = risks.sort_values(["_rank", "sku_id", "_po"], kind="mergesort")
    risks = risks.drop(columns=["_rank", "_po"]).reset_index(drop=True)

    sim_out = sim.groupby("sku_id", sort=True).head(cfg.horizon_months).reset_index(drop=True)
    breakdown = pd.DataFrame(steps, columns=BREAKDOWN_COLUMNS)
    return PlanResult(
        plan=plan, risks=risks, sim=sim_out, calendar=_calendar(plan, calendar_src, today, cfg), breakdown=breakdown
    )
