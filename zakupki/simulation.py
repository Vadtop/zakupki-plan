"""Симуляция запаса по месяцам: покрытие, дата обнуления, страховой запас, дата заказа."""

import math

import numpy as np
import pandas as pd

from zakupki.config import Config

POSITION_COLUMNS = [
    "sku_id",
    "stock_now",
    "lead_days",
    "safety_stock",
    "safety_days",
    "cover_days",
    "cover_all_days",
    "cover_months",
    "stockout_date",
    "order_date",
    "gap_months",
    "sigma",
    "safety_raw",
    "safety_cap",
]
SIM_COLUMNS = ["sku_id", "month", "forecast", "arrivals", "stock_end"]


def _month_start(month: pd.Period) -> pd.Timestamp:
    return month.to_timestamp(how="start").normalize()


def _month_days(month: pd.Period) -> int:
    return int(month.days_in_month)


def forecast_between(
    fc: pd.DataFrame, sku_id: str, start: pd.Timestamp, end: pd.Timestamp, cfg: Config
) -> float:
    """Прогноз спроса за [start, end); неполные месяцы берутся долей дней."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return 0.0
    rows = fc[fc["sku_id"] == sku_id]
    total = 0.0
    for month, qty in zip(rows["month"], rows["qty"]):
        m_start = _month_start(month)
        m_end = m_start + pd.Timedelta(days=_month_days(month))
        overlap = (min(end, m_end) - max(start, m_start)).days
        if overlap > 0:
            total += float(qty) * overlap / _month_days(month)
    return total


def _arrival_events(open_po: pd.DataFrame, sku_id: str) -> list[tuple[pd.Timestamp, float]]:
    """Приходы SKU списком (дата, qty) по возрастанию даты."""
    if open_po is None or open_po.empty:
        return []
    rows = open_po[open_po["sku_id"] == sku_id]
    out = [
        (pd.Timestamp(date).normalize(), float(qty))
        for date, qty in zip(rows["expected_date"], rows["expected_qty"])
        if not (pd.isna(date) or pd.isna(qty))
    ]
    return sorted(out, key=lambda e: e[0])


def _simulate_sku(
    sku_id: str,
    months: list[pd.Period],
    qty: np.ndarray,
    events: list[tuple[pd.Timestamp, float]],
    stock_now: float,
    today: pd.Timestamp,
):
    """Помесячный прогон: строки sim, дата обнуления и отрезки без товара (смещения в днях от today).

    Внутри месяца спрос идёт равномерно по дням, приход — в свою дату; всё, что
    раньше первого месяца (просрочка), приходит в его начале.
    """
    fc_eff = qty.astype("float64").copy()
    if months[0] == today.to_period("M"):
        days = _month_days(months[0])
        fc_eff[0] *= (days - (today.day - 1)) / days
    stock = stock_now
    ends = np.zeros(len(months))
    arrivals = np.zeros(len(months))
    offset = None
    gaps, out_since = [], None
    k = 0
    for i, month in enumerate(months):
        if i == 0:
            start = today
            span = _month_days(month) - (today.day - 1)
        else:
            start = _month_start(month)
            span = _month_days(month)
        m_end = _month_start(month) + pd.Timedelta(days=_month_days(month))
        # Приходы месяца: (день от start, qty); конец месяца — точка проверки без прихода.
        points = []
        while k < len(events) and events[k][0] < m_end:
            points.append((min(max((events[k][0] - start).days, 0), span), events[k][1]))
            k += 1
        points.append((span, 0.0))
        daily = fc_eff[i] / span if span > 0 else 0.0
        pos = 0
        base = (start - today).days
        for day, q in points:
            # Дефицит — когда спроса больше, чем товара; ноль ровно к приходу — не дефицит.
            if out_since is None and daily > 0 and day > pos and stock - daily * (day - pos) < -1e-9:
                out_since = base + pos + max(stock, 0.0) / daily
                offset = out_since if offset is None else offset
            # Продажи при пустом складе потеряны: остаток не уходит в минус.
            stock = max(stock - daily * (day - pos), 0.0)
            pos = day
            stock += q
            arrivals[i] += q
            if out_since is not None and stock > 0:
                if base + day - out_since > 1e-9:
                    gaps.append((out_since, base + day))
                out_since = None
        ends[i] = stock
    if out_since is not None:
        gaps.append((out_since, math.inf))
    sim = pd.DataFrame(
        {
            "sku_id": sku_id,
            "month": months,
            "forecast": fc_eff,
            "arrivals": arrivals,
            "stock_end": ends,
        }
    )
    return sim, offset, gaps


def run_simulation(
    stock_now: pd.Series,
    fc: pd.DataFrame,
    open_po: pd.DataFrame,
    lead: pd.Series,
    params: pd.DataFrame,
    cfg: Config,
    today: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Симуляция запаса по SKU из fc: positions (одна строка на SKU) и sim (помесячно)."""
    today = pd.Timestamp(today).normalize()
    sigma_of = params.set_index("sku_id")["sigma"] if len(params) else pd.Series(dtype="float64")
    level_of = params.set_index("sku_id")["level"] if len(params) else pd.Series(dtype="float64")
    positions, sims = [], []
    for sku_id, grp in fc.groupby("sku_id", sort=True):
        grp = grp.sort_values("month", kind="mergesort")
        months = list(grp["month"])
        qty = grp["qty"].to_numpy(dtype="float64")
        s_now = float(stock_now.get(sku_id, 0.0))
        lead_d = int(lead.get(sku_id, cfg.lead_default_days))
        sigma = float(sigma_of.get(sku_id, 0.0)) if sku_id in sigma_of.index else 0.0
        if pd.isna(sigma):
            sigma = 0.0

        sim, offset, gaps = _simulate_sku(
            sku_id, months, qty, _arrival_events(open_po, sku_id), s_now, today
        )
        sims.append(sim)

        safety = cfg.z * sigma * math.sqrt(lead_d / cfg.days_per_month)
        safety_raw, safety_cap = safety, float("nan")
        # Потолок: не больше safety_cap_months месяцев спроса (0 — без потолка).
        if cfg.safety_cap_months > 0 and sku_id in level_of.index:
            safety_cap = cfg.safety_cap_months * float(level_of[sku_id])
            safety = min(safety, safety_cap)
        lead_end = today + pd.Timedelta(days=lead_d)
        lead_fc = forecast_between(fc, sku_id, today, lead_end, cfg)
        daily = lead_fc / lead_d if lead_d > 0 else 0.0
        safety_days = safety / daily if daily > 0 else 0.0
        # Не больше плеча: при прерывистом спросе σ велик, а средний прогноз мал.
        safety_days = min(safety_days, float(lead_d))

        # Дыра — дни без товара до прихода нового заказа (через плечо).
        out_days = sum(max(min(lead_d, b) - round(a), 0) for a, b in gaps)
        gap = out_days / cfg.days_per_month
        horizon_end = _month_start(months[-1]) + pd.Timedelta(days=_month_days(months[-1]))
        horizon_days = float((horizon_end - today).days)
        # Покрытие — до первого обнуления: столько хватит запаса с учётом заказов в пути.
        cover_days = horizon_days if offset is None else float(round(offset))
        stockout = pd.NaT if offset is None else today + pd.Timedelta(days=cover_days)
        # Следующий заказ и избыток — от первого обнуления, которое новый заказ ещё успеет закрыть:
        # дыру, которую закроет заказ в пути раньше, чем придёт новый, пропускаем.
        final = next((a for a, b in gaps if b > lead_d), None)
        cover_all_days = horizon_days if final is None else float(round(final))
        order = (
            pd.NaT
            if final is None
            else today
            + pd.Timedelta(days=cover_all_days)
            - pd.Timedelta(days=lead_d)
            - pd.Timedelta(days=round(safety_days))
        )
        positions.append(
            {
                "sku_id": sku_id,
                "stock_now": s_now,
                "lead_days": lead_d,
                "safety_stock": safety,
                "safety_days": safety_days,
                "cover_days": cover_days,
                "cover_all_days": cover_all_days,
                "cover_months": cover_days / cfg.days_per_month,
                "stockout_date": stockout,
                "order_date": order,
                "gap_months": gap,
                "sigma": sigma,
                "safety_raw": safety_raw,
                "safety_cap": safety_cap,
            }
        )
    pos_df = pd.DataFrame(positions, columns=POSITION_COLUMNS)
    pos_df["stockout_date"] = pd.to_datetime(pos_df["stockout_date"])
    pos_df["order_date"] = pd.to_datetime(pos_df["order_date"])
    sim_df = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame(columns=SIM_COLUMNS)
    return pos_df, sim_df[SIM_COLUMNS]
