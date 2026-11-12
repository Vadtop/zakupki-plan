"""Дашборд закупщика: один HTML-файл, данные внутри JSON-блоком, графики — Plotly с CDN."""

import json
import math
from pathlib import Path

import pandas as pd

from zakupki import config as C
from zakupki.config import Config
from zakupki.data import Dataset
from zakupki.plan import PlanResult, _arrival_days, _daily_demand, _need_parts
from zakupki.simulation import _arrival_events, _simulate_sku, forecast_between

TEMPLATE = Path(__file__).resolve().parent / "templates" / "dashboard.html"
PLACEHOLDER = "__DATA__"
CHART_MONTHS = 12  # сколько месяцев прогноза и симуляции показывать на графике SKU
PLAN_FIELDS = [
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
    "cover_months",
    "stockout_date",
    "gap_months",
    "reliability",
    "avg_markup_pct_12m",
    "need_qty",
    "explanation",
]
DATE_FIELDS = {"order_date", "stockout_date", "promised_date"}
# Статусы, у которых наступившая (сегодня или раньше) дата заказа показывается как «сразу».
ASAP_STATUSES = (C.ORDER_NOW, C.SHORTAGE)
# Названия категорий для показа; неизвестный код показывается как есть.
CATEGORY_NAMES = C.CATEGORY_NAMES
# Единицы из sku_dim.uom для подписей карточки; неизвестная — «шт.»
UNIT_NAMES = {"pcs": "шт.", "set": "компл."}


def _val(value, digits: int = 4):
    """Значение для JSON: NaN/NaT → None, даты → ISO, числа округлены.

    4 знака по умолчанию: шаблон сам округляет до десятых, второе округление здесь
    давало бы 6,3487 → 6,35 → 6,4.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (bool, str)):
        return value
    number = float(value)
    return int(number) if number == int(number) else round(number, digits)


def _records(df: pd.DataFrame, fields: list[str]) -> list[dict]:
    """Строки таблицы как список словарей только по нужным полям."""
    cols = [f for f in fields if f in df.columns]
    return [{c: _val(v) for c, v in zip(cols, row, strict=True)} for row in df[cols].itertuples(index=False)]


def _charts(plan: pd.DataFrame, demand: pd.DataFrame, fc: pd.DataFrame, sim: pd.DataFrame) -> dict:
    """Ряды для графика по клику на SKU: история спроса, прогноз, остаток и приходы."""
    hist = {k: g for k, g in demand.sort_values(["sku_id", "month"], kind="mergesort").groupby("sku_id")}
    fcast = {k: g for k, g in fc.sort_values(["sku_id", "month"], kind="mergesort").groupby("sku_id")}
    sims = {k: g for k, g in sim.sort_values(["sku_id", "month"], kind="mergesort").groupby("sku_id")}
    out: dict = {}
    for sku_id in plan["sku_id"]:
        h, f, s = hist.get(sku_id), fcast.get(sku_id), sims.get(sku_id)
        out[sku_id] = {
            "h": []
            if h is None
            else [
                [str(m), _val(d, 1), int(lost)]
                for m, d, lost in zip(h["month"], h["demand"], h["lost"], strict=True)
            ],
            "f": []
            if f is None
            else [[str(m), _val(q, 1)] for m, q in zip(f["month"], f["qty"], strict=True)][:CHART_MONTHS],
            "s": []
            if s is None
            else [
                [str(m), _val(e, 1), _val(a, 1)]
                for m, e, a in zip(s["month"], s["stock_end"], s["arrivals"], strict=True)
            ],
        }
    return out


def _suppliers(reliability: pd.DataFrame, open_po: pd.DataFrame, ds: Dataset) -> list[dict]:
    """Поставщики: надёжность, заказов в расчёте, медиана плеча, открытых заказов."""
    sup = reliability[["supplier_id", "reliability", "orders_n"]].copy()
    lead = ds.suppliers.drop_duplicates("supplier_id").set_index("supplier_id")["median_planned_lead_days"]
    sup["median_lead"] = sup["supplier_id"].map(lead)
    opened = open_po.groupby("supplier_id").size() if len(open_po) else pd.Series(dtype="int64")
    sup["open_n"] = sup["supplier_id"].map(opened).fillna(0).astype("int64")
    sup = sup.sort_values(["open_n", "supplier_id"], ascending=[False, True], kind="mergesort")
    return _records(sup, ["supplier_id", "reliability", "orders_n", "median_lead", "open_n"])


def _steps(result: PlanResult, open_po: pd.DataFrame) -> dict:
    """Шаги расчёта количества по SKU: прогноз, страховой, остаток, в пути, округление."""
    out: dict = {}
    bd = result.breakdown
    if bd is None or not len(bd):
        return out
    for r in bd.itertuples(index=False):
        pending = 0.0
        if len(open_po):
            po = open_po[open_po["sku_id"] == r.sku_id]
            rows = po[(po["expected_date"] >= r.need_arrival) & (po["expected_date"] < r.need_end)]
            pending = float(rows["qty_pending"].sum())
        # Точность 6 знаков: из этих чисел дашборд и тест собирают итог заново.
        out[r.sku_id] = {
            "at": _val(r.need_at),
            "arrival": _val(r.need_arrival),
            "end": _val(r.need_end),
            "forecast": _val(r.need_forecast, 6),
            "safety": _val(r.need_safety, 6),
            "safety_raw": _val(r.safety_raw, 6),
            "safety_cap": _val(r.safety_cap, 6),
            "sigma": _val(r.sigma, 6),
            "stock": _val(r.need_stock, 6),
            "transit": _val(r.need_transit, 6),
            "pending": _val(pending, 6),
            "bridge": _val(r.need_bridge, 6),
            "raw": _val(r.need_raw, 6),
            "ceil": int(r.need_ceil),
            "lot": int(r.lot),
            "lot_full": int(r.lot_full),
        }
    return out


def _orders_by_supplier(
    plan: pd.DataFrame, asap: pd.Series, reliability: pd.DataFrame, today, weight: pd.Series | None = None
) -> dict:
    """Заказы поставщикам: что разместить сейчас и что позже, по каждому поставщику.

    weight — вес единицы по SKU (sku_dim.weight_kg); вес группы — только по позициям, где он заведён.
    """
    rel = reliability.set_index("supplier_id")["reliability"]
    chosen = plan[plan["order_qty"] > 0].copy()
    chosen["asap"] = asap[chosen.index]
    unit_kg = (
        chosen["sku_id"].map(weight) if weight is not None else pd.Series(float("nan"), index=chosen.index)
    )
    chosen["_kg"] = (chosen["order_qty"] * unit_kg.where(unit_kg > 0)).astype("float64")
    now_mask = chosen["asap"] | (chosen["order_date"] <= today) | (chosen["status"] == C.SHORTAGE)
    groups: dict = {"now": [], "later": []}
    for key, part in (("now", chosen[now_mask]), ("later", chosen[~now_mask])):
        # Поставщик в справочнике может быть пустым — такие позиции идут отдельной строкой.
        for sup, g in part.groupby("supplier_id", sort=True, dropna=False):
            g = g.sort_values(["order_date", "sku_id"], kind="mergesort")
            start = pd.Series(today, index=g.index) if key == "now" else g["order_date"]
            # Приход = день размещения + плечо; «сейчас» размещаем в дату данных.
            arrive = start + pd.to_timedelta(g["lead_days"], unit="D")
            groups[key].append(
                {
                    "supplier_id": None if pd.isna(sup) else str(sup),
                    "n": len(g),
                    "qty": int(g["order_qty"].sum()),
                    "place_by": _val(g["order_date"].min()),
                    "asap": bool(g["asap"].any()),
                    "arrive_from": _val(arrive.min()),
                    "arrive_to": _val(arrive.max()),
                    "short_n": int((g["status"] == C.SHORTAGE).sum()),
                    # Известная часть веса: weight_n позиций из n, у остальных вес не заведён.
                    "weight_kg": _val(g["_kg"].sum(), 2) if g["_kg"].notna().any() else None,
                    "weight_n": int(g["_kg"].notna().sum()),
                    "reliability": None if pd.isna(sup) else _val(rel.get(sup), 4),
                    "skus": [str(x) for x in g["sku_id"]],
                    # Приход по каждой позиции: медиану на первом экране считают по SKU, а не по поставщику.
                    "arrivals": [[_val(a), int(q)] for a, q in zip(arrive, g["order_qty"], strict=True)],
                }
            )
        groups[key].sort(key=lambda x: (x["supplier_id"] is None, -x["qty"], x["supplier_id"] or ""))
    return groups


def _sku_gaps(sid: str, fc: pd.DataFrame, open_po: pd.DataFrame, stock_now: float, today) -> list:
    """Отрезки без товара (дни от today) — тот же прогон, что в simulation.run_simulation."""
    grp = fc[fc["sku_id"] == sid].sort_values("month", kind="mergesort")
    if not len(grp):
        return []
    _, _, gaps = _simulate_sku(
        sid,
        list(grp["month"]),
        grp["qty"].to_numpy(dtype="float64"),
        _arrival_events(open_po, sid),
        stock_now,
        today,
    )
    return gaps


def _lost_units(
    plan: pd.DataFrame, fc: pd.DataFrame, open_po: pd.DataFrame, today, cfg: Config
) -> dict[str, float]:
    """«Потеряно, шт.» у РИСК ДЕФИЦИТА: прогноз спроса за дни без товара до прихода заказа, сделанного сегодня.

    Те же дни, что в «дыре» (gap_months): от обнуления до min(конец отрезка, сегодня + плечо).
    """
    today = pd.Timestamp(today).normalize()
    out: dict[str, float] = {}
    for r in plan[plan["status"] == C.SHORTAGE].itertuples(index=False):
        lead = int(r.lead_days)
        lost = 0.0
        for a, b in _sku_gaps(r.sku_id, fc, open_po, float(r.stock_now), today):
            start, stop = round(a), min(lead, b)
            if stop > start:
                lost += forecast_between(
                    fc, r.sku_id, today + pd.Timedelta(days=start), today + pd.Timedelta(days=int(stop)), cfg
                )
        out[r.sku_id] = lost
    return out


def _requantify(row, step, safety: float, fc: pd.DataFrame, open_po: pd.DataFrame, today, cfg: Config) -> int:
    """Количество на ту же дату заказа с другим страховым; остальное — как в plan._build_rows."""
    lead = int(row.lead_days)
    cover = int((step.need_end - step.need_arrival).days)
    far = max((pd.Timestamp(step.need_at) - today).days, 0)
    days = max(int(cfg.horizon_months * cfg.days_per_month), far) + lead + cover + 1
    level = 0.0 if pd.isna(row.level) else float(row.level)
    over = cfg.lot_max_overage_months * level if cfg.lot_max_overage_months > 0 else math.inf
    parts = _need_parts(
        step.need_at,
        lead,
        cover,
        safety,
        int(step.lot_full),
        _daily_demand(fc, row.sku_id, today, days),
        _arrival_days(open_po, row.sku_id, today),
        today,
        float(row.stock_now),
        over,
    )
    return int(parts["need_qty"])


def _safety_scenario(result: PlanResult, fc: pd.DataFrame, open_po: pd.DataFrame, today, cfg: Config) -> dict:
    """Сценарий: страховой ≤ прогноза спроса за safety_alt_days дн. после прихода; даты, склад, в пути, партия те же.

    Только для показа, план его не применяет. Возвращает SKU, у которых заказ уменьшается: {sku: {qty, safety}}.
    """
    today = pd.Timestamp(today).normalize()
    bd = result.breakdown
    if bd is None or not len(bd) or cfg.safety_alt_days <= 0:
        return {}
    steps = bd.set_index("sku_id")
    plan = result.plan
    out: dict = {}
    for r in plan[plan["order_qty"] > 0].itertuples(index=False):
        if r.sku_id not in steps.index:
            continue
        s = steps.loc[r.sku_id]
        arrival = pd.Timestamp(s.need_arrival)
        cap = forecast_between(fc, r.sku_id, arrival, arrival + pd.Timedelta(days=cfg.safety_alt_days), cfg)
        if cap >= float(s.need_safety):
            continue
        qty = _requantify(r, s, cap, fc, open_po, today, cfg)
        if qty < int(r.order_qty):
            out[r.sku_id] = {"qty": qty, "safety": cap}
    return out


def _num_text(v: float) -> str:
    """Число для текста правил: 1.65 → «1,65»."""
    return f"{v:g}".replace(".", ",")


def _service_level(z: float) -> str:
    """Вероятность не остаться без товара за плечо при нормальном спросе: z=1,65 → «95%»."""
    return f"{50 * (1 + math.erf(z / math.sqrt(2))):.0f}%"


def _rules(cfg: Config) -> list[str]:
    """Правила расчёта простым языком; числа — из Config."""
    n = _num_text
    return [
        (
            "Спрос — продажи минус возвраты по полным месяцам. Месяц без продаж при пустом складе на начало месяца — "
            "упущенный спрос: его не считаем нулём, а восполняем по сезону, иначе прогноз занижается."
        ),
        (
            f"Сезонность — по категории: доля каждого месяца в годовых продажах за все полные годы. Уровень SKU — "
            f"средний спрос за последние {cfg.history_months} мес. без сезонности (месяцы с индексом ниже "
            f"{n(cfg.min_season_index)} не берём). Тренд — последние {cfg.trend_window_months} мес. к предыдущим, "
            f"смягчён вдвое и ограничен {n(cfg.trend_min)}…{n(cfg.trend_max)}."
        ),
        (
            f"Плечо — медиана планового срока по заказам этого SKU, иначе по поставщику, иначе {cfg.lead_default_days} дн.; "
            f"в пределах {cfg.lead_min_days}…{cfg.lead_max_days} дн."
        ),
        (
            f"Страховой запас = {n(cfg.z)} × разброс месячного спроса × √(плечо в месяцах): "
            f"целевой уровень сервиса {_service_level(cfg.z)} при нормальном спросе, на истории не проверен. "
            f"Не больше {n(cfg.safety_cap_months)} мес. спроса, чтобы редкие всплески не раздували заказ; "
            "если страховой — большая часть заказа вне сезона, количество стоит согласовать вручную. "
            f"Сценарий «сезонный потолок»: страховой не больше прогноза спроса за {cfg.safety_alt_days} дн. после "
            "прихода, остальное как в плане — даты, склад, заказы в пути, партия. У части сегодняшних заказов склад "
            "к приходу уже больше спроса на несколько месяцев, и заказ возникает только из-за страхового запаса. "
            "План сценарий не применяет: такая политика на истории не проверена, а меньший страховой — выше риск "
            "дефицита. Позиции, где сценарий уменьшает заказ, помечены «согласовать вручную», обе цифры — в карточке SKU."
        ),
        (
            "Заказы в пути учитываются не целиком, а с надёжностью поставщика — долей реально полученных штук "
            "по закрытым заказам. Обещанная дата прошла — считаем, что придёт через "
            f"{cfg.overdue_shift_days} дн., и показываем в рисках. Просрочка больше {cfg.overdue_stale_days} дн. "
            "в расчёт не идёт — только в рисках: уточнить у поставщика. Уже заказанное повторно не заказываем."
        ),
        (
            f"Сколько: заказ придёт через плечо и должен покрыть {cfg.default_cover_days} дн. до следующего пересмотра "
            "(закупщики планируют раз в месяц): прогноз на это окно + страховой − остаток к приходу − заказы в пути "
            "с приходом внутри окна. Остаток к приходу считается по дням: продажи при пустом складе потеряны и не "
            "дозаказываются. Если склад опустеет до прихода уже заказанного, заказ закрывает и эту нехватку."
        ),
        (
            "Вверх до целого и до кратности партии — общего делителя прошлых приходов, если их не меньше "
            f"{cfg.lot_min_receipts}; до партии не округляем, если это добавит больше "
            f"{n(cfg.lot_max_overage_months)} мес. спроса — склад не раздуваем."
        ),
        (
            f"Когда: дата обнуления склада − плечо − страховой запас в днях. Срок в ближайшие "
            f"{cfg.order_now_window_days} дн. — «заказать сейчас». Склад опустеет раньше, чем придёт заказ, сделанный "
            "сегодня, — «риск дефицита»: заказ сразу, «дыра» показана в месяцах, «ожидаемые потери» — прогноз спроса за дни без товара до прихода, шт."
        ),
        (
            f"Не закупаем: наценка за 12 мес. ниже {n(cfg.min_markup_pct)}% — «низкая наценка», решение за закупщиком; "
            "продаж за 12 мес. нет — «мёртвый», остаток распродать. Запас дольше плеча + покрытия + "
            f"{cfg.excess_extra_months} мес. — «избыток», деньги заморожены."
        ),
        (
            "Все цифры считает код: один и тот же вход даёт один и тот же план. Модель ИИ только пишет пояснения "
            "к готовым числам; без неё пояснения шаблонные, а расчёт тот же."
        ),
    ]


RUN_NOTE = (
    "Регулярный запуск — одна команда: python -m zakupki run --data <папка выгрузки> --today <дата данных>; "
    "на Windows — scripts/run_daily.ps1 в Планировщике заданий. Каждый запуск сохраняет снимок плана "
    "и показывает, что изменилось с прошлого раза: новые к заказу, количества, отмены, сдвиги дат, риск дефицита."
)


def _abc(demand: pd.DataFrame, today, cfg: Config) -> dict[str, str]:
    """Класс ABC по чистым продажам в штуках за cfg.abc_months полных месяцев: A — 80%, B — до 95%, C."""
    if not len(demand) or today is None:
        return {}
    month = pd.Timestamp(today).to_period("M")
    m = demand["month"]
    if not isinstance(m.dtype, pd.PeriodDtype):
        m = pd.to_datetime(m).dt.to_period("M")
    win = demand[(m < month) & (m >= month - cfg.abc_months)]
    sold = win.groupby("sku_id")["sold_net"].sum().sort_values(ascending=False, kind="mergesort")
    total = float(sold.sum())
    if total <= 0:
        return {}
    before = (sold.cumsum() - sold) / total  # доля штук до этого SKU
    return {
        str(k): ("A" if b < cfg.abc_a_share else "B" if b < cfg.abc_b_share else "C") if v > 0 else "C"
        for k, b, v in zip(sold.index, before, sold, strict=True)
    }


def build_data(
    result: PlanResult,
    index,
    demand,
    fc,
    reliability,
    open_po,
    diff: dict,
    summary: str,
    ds: Dataset,
    cfg: Config | None = None,
    generated_at: str | None = None,
    backtest: dict | None = None,
) -> dict:
    """Собирает все данные дашборда в один словарь для JSON.

    generated_at — время запуска из cli, только подпись в шапке; в расчёт не идёт.
    backtest — точность прогноза (zakupki.backtest), только показывается.
    """
    cfg = cfg or Config()
    plan, risks = result.plan, result.risks
    asap = plan["status"].isin(ASAP_STATUSES) & (plan["order_date"] <= ds.today)
    plan_rows = _records(plan, PLAN_FIELDS)
    uom = dict(zip(ds.sku["sku_id"], ds.sku["uom"], strict=True)) if "uom" in ds.sku else {}
    abc = _abc(demand, ds.today, cfg)
    weight = ds.sku.set_index("sku_id")["weight_kg"].astype("float64") if "weight_kg" in ds.sku else None
    lost = _lost_units(plan, fc, open_po, ds.today, cfg)
    alt = _safety_scenario(result, fc, open_po, ds.today, cfg)
    for rec, flag in zip(plan_rows, asap, strict=True):
        sid = rec["sku_id"]
        rec["order_asap"] = bool(flag)
        unit_kg = weight.get(sid) if weight is not None else None
        rec["weight_kg"] = _val(unit_kg) if unit_kg is not None and unit_kg > 0 else None
        if sid in lost:
            rec["lost_qty"] = _val(lost[sid], 1)
        if sid in alt:
            rec["alt_qty"] = alt[sid]["qty"]
            rec["alt_safety"] = _val(alt[sid]["safety"], 1)
        rec["abc"] = abc.get(rec["sku_id"], "C")
        rec["unit"] = UNIT_NAMES.get(str(uom.get(rec["sku_id"], "")).strip().lower(), "шт.")
    counts = plan["status"].value_counts()
    qty_by_status = plan.groupby("status")["order_qty"].sum()
    # «Сегодня»: срок заказа сегодня или прошёл; «скоро»: заказать сейчас, но срок ещё не наступил.
    to_order = plan["order_qty"] > 0
    today_rows = plan[asap & to_order]
    soon_rows = plan[(plan["status"] == C.ORDER_NOW) & ~asap & to_order]
    stock = plan.set_index("sku_id")["stock_now"]
    short = risks[risks["risk_type"] == C.SHORTAGE].copy()
    short["lost_qty"] = short["sku_id"].map(lost).astype("float64")
    alt_today = [
        alt[s]["qty"] if s in alt else int(q) for s, q in zip(today_rows["sku_id"], today_rows["order_qty"])
    ]
    excess = risks[risks["risk_type"] == C.EXCESS].copy()
    excess["stock_now"] = excess["sku_id"].map(stock)
    overdue = risks[risks["risk_type"] == C.OVERDUE_PO].copy()
    stale = open_po[open_po["stale"]] if len(open_po) and "stale" in open_po else open_po.iloc[0:0]
    stale_keys = set(zip(stale.get("po_id", []), stale.get("sku_id", [])))
    overdue["stale"] = [(p, s) in stale_keys for p, s in zip(overdue["po_id"], overdue["sku_id"])]
    calendar = result.calendar.copy()
    calendar["month"] = calendar["month"].astype(str)
    season = {
        str(cat): [_val(v, 4) for v in grp.sort_values("month_num")["share"]]
        for cat, grp in index.groupby("category")
    }
    return {
        "asof": ds.today.strftime("%Y-%m-%d"),
        "summary": summary,
        "statuses": C.STATUS_ORDER,
        "counts": {s: int(counts.get(s, 0)) for s in C.STATUS_ORDER},
        "total_sku": len(plan),
        "order_qty_total": int(plan["order_qty"].sum()),
        "qty_by_status": {s: int(qty_by_status.get(s, 0)) for s in C.STATUS_ORDER},
        "today": {
            "n": len(today_rows),
            "qty": int(today_rows["order_qty"].sum()),
            "by_status": {s: int((today_rows["status"] == s).sum()) for s in ASAP_STATUSES},
            # Сценарий сезонного потолка страхового — только показ, план не меняется.
            "alt_qty": int(sum(alt_today)),
            "alt_n": int(sum(1 for q in alt_today if q > 0)),
            "alt_marked": int(sum(1 for s in today_rows["sku_id"] if s in alt)),
        },
        "lost_qty_total": _val(float(short["lost_qty"].sum()), 1),
        "soon": {
            "n": len(soon_rows),
            "qty": int(soon_rows["order_qty"].sum()),
            "first": _val(soon_rows["order_date"].min()) if len(soon_rows) else None,
            "last": _val(soon_rows["order_date"].max()) if len(soon_rows) else None,
        },
        "categories": {
            c: CATEGORY_NAMES.get(c, c)
            for c in sorted({str(v) for v in plan["category"].dropna()} | set(season))
        },
        "plan": plan_rows,
        "calendar": _records(calendar, ["month", "positions", "qty", "now_positions", "now_qty"]),
        "risks": {
            "short": _records(short, ["sku_id", "category", "supplier_id", "gap_months", "lost_qty", "qty"]),
            "overdue": _records(overdue, ["po_id", "sku_id", "supplier_id", "promised_date", "qty", "stale"]),
            "excess": _records(excess, ["sku_id", "category", "stock_now", "excess_months"]),
        },
        "suppliers": _suppliers(reliability, open_po, ds),
        "season": season,
        "charts": _charts(plan, demand, fc, result.sim),
        "diff": diff,
        "steps": _steps(result, open_po),
        "orders": _orders_by_supplier(plan, asap, reliability, ds.today, weight),
        "cfg": {
            "min_markup_pct": cfg.min_markup_pct,
            "overdue_shift_days": cfg.overdue_shift_days,
            "overdue_stale_days": cfg.overdue_stale_days,
            "excess_extra_months": cfg.excess_extra_months,
            "safety_cap_months": cfg.safety_cap_months,
            "safety_alt_days": cfg.safety_alt_days,
            "z": cfg.z,
            "default_cover_days": cfg.default_cover_days,
            "horizon_months": cfg.horizon_months,
            "days_per_month": cfg.days_per_month,
            "lot_max_overage_months": cfg.lot_max_overage_months,
            "long_gap_months": cfg.long_gap_months,
            "unreliable_below": cfg.unreliable_below,
        },
        "rules": _rules(cfg),
        "run_note": RUN_NOTE,
        "generated_at": generated_at,
        "backtest": backtest,
    }


def render_dashboard(
    result: PlanResult,
    index,
    demand,
    fc,
    reliability,
    open_po,
    diff: dict,
    summary: str,
    ds: Dataset,
    out_path: Path,
    cfg: Config | None = None,
    generated_at: str | None = None,
    backtest: dict | None = None,
) -> None:
    """Пишет out/dashboard.html: шаблон с данными внутри."""
    data = build_data(
        result, index, demand, fc, reliability, open_po, diff, summary, ds, cfg, generated_at, backtest
    )
    # «<» экранируется, чтобы текст внутри данных не закрыл тег script.
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
