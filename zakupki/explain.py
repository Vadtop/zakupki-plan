"""Пояснения к плану: модель пишет текст по готовым цифрам, без ключа — шаблон из цифр."""

import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from zakupki import config as C
from zakupki.config import Config
from zakupki.plan import PlanResult

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# Статусы, для которых просим пояснение у модели.
LLM_STATUSES = (C.ORDER_NOW, C.SHORTAGE)
# Белый список полей строки, уходящих в модель: id и цифры.
PAYLOAD_FIELDS = [
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
    "trend",
    "short_history",
]
MAX_TEXT_LEN = 600
# Про сроки в данных ничего нет: надёжность — доля довезённых штук, задержку из неё не вывести.
DELAY_PHRASES = ("задерж", "опозда", "просроч", "сорв")
# При дефиците обычный заказ дыру не закрывает: обещать обратное нельзя.
SHORTAGE_DENY = ("не допустить дефицит", "не будет дефицит", "не было дефицит", "не возникнет дефицит")
# «избежит / предотвратит / устранит / закроет (её) дефицит / дыру» — без «не» прямо перед глаголом.
PROMISE_RE = re.compile(
    r"(?<!не )\b(?:избеж\w*|предотвра\w*|устран\w*|ликвидир\w*|закро\w*|закрыва\w*|закрыт\w*|покро\w*|перекро\w*)"
    r"\s+(?:\w+\s+){0,2}?(?:дефицит\w*|нехватк\w*|дыр\w*|разрыв\w*|её|его)\b"
)
# Граница предложения: знак и пробел перед заглавной (точки внутри дат и «шт.» перед строчной — не граница).
SENTENCE_RE = re.compile(r"[.;!?](?=\s+[А-ЯЁA-Z]|\s*$)")
# Чей факт: последнее перед числом слово-метка в том же предложении.
NEW_MARK_RE = re.compile(r"нов\w*")
TRANSIT_MARK_RE = re.compile(r"в пути|уже заказ\w*|заказанн\w*|ед[уе]т\b|транзит\w*|\bони\b|\bих\b")
ORDER_QTY_RE = re.compile(
    r"(?:заказать|закажите|дозаказать|разместить|заказ на)\s+(?:ещё\s+)?(\d{1,3}(?:[  ]\d{3})+|\d+)\s*шт"
)
EMPTY_NOW_RE = re.compile(r"склад\s+(?:уже\s+)?(?:пуст\s+(?:с|сейчас|уже)\b|пустой)")
DECIMAL_DOT_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
CLAUSE_RE = re.compile(r"(?<!шт)[.;!?](?:\s|$)")
DAYS_RE = r"\s*(?:дн|день|дня|дней)"
LLM_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError)
# Слова, отменяющие рекомендацию заказать: с ними ответ модели не принимается.
DENY_PHRASES = ("не заказывать", "не нужно", "отменить")
DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
NUM_RE = re.compile(r"\d{1,3}(?:[  ]\d{3})+(?!\d)|\d+(?:[.,]\d+)?")


def load_env(path: Path) -> dict[str, str]:
    """Читает строки KEY=VALUE из .env; переменные окружения главнее файла."""
    env: dict[str, str] = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'\"")
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _clean(value):
    """Значение для JSON: NaN → None, даты → ISO, числа округлены."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.strftime("%Y-%m-%d")
    if isinstance(value, bool) or hasattr(value, "dtype") and str(value.dtype) == "bool":
        return bool(value)
    if isinstance(value, (int, float)) or hasattr(value, "dtype"):
        if pd.isna(value):
            return None
        number = float(value)
        return int(number) if number == int(number) else round(number, 2)
    return None if pd.isna(value) else str(value)


def _order_asap(row: pd.Series, today) -> bool:
    """Заказ нужен сразу: «заказать сейчас» или дефицит с датой заказа раньше today."""
    if today is None or row["status"] not in LLM_STATUSES or pd.isna(row["order_date"]):
        return False
    return pd.Timestamp(row["order_date"]) < pd.Timestamp(today)


def _next_arrival(po: pd.DataFrame | None, sku_id: str, today) -> tuple:
    """Ближайший приход заказов в пути: дата и штук по заказу (без поправки), иначе (None, None)."""
    if po is None or po.empty:
        return None, None
    rows = po[(po["sku_id"] == sku_id) & po["expected_date"].notna()]
    if rows.empty:
        return None, None
    dates = pd.to_datetime(rows["expected_date"]).dt.normalize()
    if today is not None:
        dates = dates.clip(lower=pd.Timestamp(today).normalize())
    first = dates.min()
    return first, float(rows.loc[dates == first, "qty_pending"].sum())


def _derived(row: pd.Series, payload: dict, today, po, days_per_month: float) -> dict:
    """Готовые факты для модели: чей приход когда, исполнение поставщика именно транзита, дыра в днях."""
    out: dict = {}
    on_order = float(row.get("on_order", 0) or 0)
    if on_order > 0:
        out["on_order_reliability"] = _clean(float(row["on_order_expected"]) / on_order)
        date, qty = _next_arrival(po, row["sku_id"], today)
        if date is not None:
            out["on_order_next_date"] = _clean(date)
            out["on_order_next_qty"] = _clean(qty)
    if payload.get("order_date") and pd.notna(row.get("lead_days")):
        arrival = pd.Timestamp(payload["order_date"]) + pd.Timedelta(days=int(row["lead_days"]))
        out["new_order_arrival_date"] = _clean(arrival)
    gap = row.get("gap_months")
    if gap is not None and pd.notna(gap) and float(gap) > 0:
        out["gap_days"] = _gap_days(gap, days_per_month)
    return out


def _gap_days(gap_months, days_per_month: float) -> int:
    return max(round(float(gap_months) * days_per_month), 1)


def build_row_payload(row: pd.Series, today=None, po=None, days_per_month: float = 30.4) -> dict:
    """Поля строки для модели — белый список и досчитанные кодом факты; прошедшая дата заказа → today."""
    payload = {name: _clean(row[name]) for name in PAYLOAD_FIELDS if name in row.index}
    if "order_date" in payload and _order_asap(row, today):
        payload["order_date"] = _clean(pd.Timestamp(today))
    payload.update(_derived(row, payload, today, po, days_per_month))
    return payload


def fix_decimals(text: str) -> str:
    """Десятичная точка → запятая; даты ДД.ММ.ГГГГ не трогаются."""
    return DECIMAL_DOT_RE.sub(r"\1,\2", text)


def _n(value) -> str:
    """Число с пробелом между тысячами."""
    return f"{float(value):,.0f}".replace(",", " ")


def _d(value) -> str:
    """Дата ДД.ММ.ГГГГ; пусто — прочерк."""
    return "—" if pd.isna(value) else pd.Timestamp(value).strftime("%d.%m.%Y")


def template_explanation(row: pd.Series, today=None, days_per_month: float = 30.4) -> str:
    """Пояснение из цифр без модели; дата заказа раньше today пишется как «сразу»."""
    status = row["status"]
    qty, lead = _n(row["order_qty"]), _n(row["lead_days"])
    transit = ""
    if row["on_order"] > 0:
        # Процент — у поставщика заказов в пути: он может быть не тем, у кого размещаем новый заказ.
        po_rel = f"{float(row['on_order_expected']) / float(row['on_order']) * 100:.0f}%"
        transit = (
            f" В пути {_n(row['on_order'])} шт., с поправкой на исполнение их поставщика ({po_rel}) "
            f"ожидается {_n(row['on_order_expected'])} шт."
        )
    if status == C.ORDER_NOW:
        when = "сразу" if _order_asap(row, today) else f"до {_d(row['order_date'])}"
        text = (
            f"Заказать {qty} шт. {when}: запаса на {float(row['cover_months']):.1f} мес., "
            f"плечо поставки {lead} дн., учтены сезон и тренд.{transit}"
        )
    elif status == C.SHORTAGE:
        text = (
            f"Запас кончится {_d(row['stockout_date'])}, следующий приход позже: склад пуст "
            f"≈{_gap_days(row['gap_months'], days_per_month)} дн. Обычный заказ эту дыру не закроет — "
            f"решить отдельно. Заказать {qty} шт. сразу, плечо нового заказа {lead} дн.{transit}"
        )
    elif status == C.PLAN:
        text = f"Запланировать заказ {qty} шт. до {_d(row['order_date'])}: плечо {lead} дн.{transit}"
    elif status == C.EXCESS:
        if float(row["on_order_expected"]) > 0:
            text = (
                f"Склад {_n(row['stock_now'])} шт., в пути ожидается {_n(row['on_order_expected'])} шт.: "
                "с ними запаса больше нужного — не заказывать."
            )
        else:
            text = f"Запаса на {float(row['cover_months']):.1f} мес., больше нужного: не заказывать."
    elif status == C.LOW_MARKUP:
        text = "Наценка ниже порога: автозаказ отключён, решение за закупщиком."
    elif status == C.NO_MARKUP:
        text = (
            "Продажи есть, но наценка не посчитана — в ERP нет себестоимости: автозаказ отключён, "
            f"проверить цену. Потребность для справки — {_n(row['need_qty'])} шт."
        )
    elif status == C.DEAD:
        text = f"Продаж за последние 12 месяцев нет, остаток {_n(row['stock_now'])} шт.: не закупать."
    else:
        text = f"Запас в норме на {float(row['cover_months']):.1f} мес.: заказ не нужен."
    if bool(row["short_history"]):
        text += " Истории продаж мало, оценка грубая."
    return fix_decimals(text)


def _place_today(plan: pd.DataFrame, today) -> tuple[pd.DataFrame, pd.DataFrame]:
    """«Разместить сегодня» как на первом экране дашборда и «скоро» — срок заказа ещё не наступил."""
    act = plan["status"].isin(LLM_STATUSES) & (plan["order_qty"] > 0)
    if today is None:
        return plan[act], plan.iloc[0:0]
    due = plan["order_date"] <= pd.Timestamp(today)
    return plan[act & due], plan[act & ~due]


def _summary_payload(result: PlanResult, cfg: Config) -> dict:
    """Агрегаты плана для сводки: числа без id позиций."""
    plan = result.plan
    counts = plan["status"].value_counts()
    top = plan.groupby("category")["order_qty"].sum().sort_values(ascending=False, kind="mergesort")
    today_rows, _ = _place_today(plan, cfg.today)
    risks = result.risks["risk_type"].value_counts() if len(result.risks) else pd.Series(dtype="int64")
    overdue = result.risks[result.risks["risk_type"] == C.OVERDUE_PO] if len(result.risks) else None
    if overdue is not None and len(overdue) and "po_id" in overdue:
        risks[C.OVERDUE_PO] = overdue["po_id"].nunique()  # заказов, а не строк — как на дашборде
    return {
        "date": None if cfg.today is None else pd.Timestamp(cfg.today).strftime("%Y-%m-%d"),
        "sku_total": len(plan),
        "statuses": {s: int(counts.get(s, 0)) for s in C.STATUS_ORDER},
        "order_qty_total": int(plan["order_qty"].sum()),
        "place_today_sku": len(today_rows),
        "place_today_qty": int(today_rows["order_qty"].sum()),
        "top_categories": {
            C.CATEGORY_NAMES.get(str(k), str(k)): int(v) for k, v in top.head(3).items() if v > 0
        },
        "risks": {str(k): int(v) for k, v in risks.items()},
    }


def template_summary(result: PlanResult, today=None) -> str:
    """Сводка плана из цифр, 5–7 строк."""
    plan = result.plan
    today_rows, soon_rows = _place_today(plan, today)
    counts = plan["status"].value_counts()

    def cnt(status) -> int:
        return int(counts.get(status, 0))

    lines = [
        f"В расчёте {len(plan)} SKU, к заказу всего {_n(plan['order_qty'].sum())} шт.",
        (
            f"Разместить сегодня: {len(today_rows)} SKU, {_n(today_rows['order_qty'].sum())} шт. "
            f"(срок заказа наступил); ещё {len(soon_rows)} SKU, {_n(soon_rows['order_qty'].sum())} шт. — "
            "срок позже."
        ),
        f"Риск дефицита: {cnt(C.SHORTAGE)} SKU — запас кончится раньше, чем придёт новый заказ.",
        f"Запланировать на ближайшие месяцы: {cnt(C.PLAN)} SKU.",
        f"Избыток: {cnt(C.EXCESS)} SKU; запас в норме: {cnt(C.OK)} SKU.",
        (
            f"Не закупаем: низкая наценка — {cnt(C.LOW_MARKUP)} SKU, нет наценки (нет себестоимости) — "
            f"{cnt(C.NO_MARKUP)} SKU, мёртвый ассортимент — {cnt(C.DEAD)} SKU."
        ),
    ]
    risks = result.risks
    late = risks[risks["risk_type"] == C.OVERDUE_PO] if len(risks) else risks
    # Счёт — по заказам (po_id), а не по товарным строкам заказа.
    lines.append(
        f"Просроченных заказов поставщиков: {late['po_id'].nunique() if len(late) else 0} ({len(late)} строк)."
    )
    return "\n".join(lines)


def _default_post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    """POST JSON на OpenAI-совместимый адрес, ответ — разобранный JSON."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_cache(path: Path) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def _facts(payload, nums: set, dates: set, words: list) -> None:
    """Собирает из данных для модели допустимые числа, даты и строки-идентификаторы."""
    if isinstance(payload, dict):
        for value in payload.values():
            _facts(value, nums, dates, words)
    elif isinstance(payload, bool) or payload is None:
        return
    elif isinstance(payload, (int, float)):
        value = float(payload)
        nums.update({round(value, 2), round(value, 1), float(round(value)), float(round(value * 100))})
    elif isinstance(payload, str):
        iso = ISO_RE.match(payload)
        if iso:
            y, m, d = (int(x) for x in iso.groups())
            dates.add((d, m, y))
            nums.update({float(d), float(m), float(y)})
        elif payload:
            words.append(payload)


def _meaning_ok(text: str, payload: dict) -> bool:
    """Смысл не спорит с расчётом: нет задержек из надёжности, дефицит не «предотвращается»,
    у заказа в пути нет плеча и даты нового заказа и надёжности чужого поставщика."""
    low = text.lower()
    starts = [0] + [m.end() for m in SENTENCE_RE.finditer(text)]

    def sentence(pos: int) -> str:
        """Предложение, в котором стоит pos (целиком)."""
        a = max(s for s in starts if s <= pos)
        b = min([s for s in starts if s > pos] + [len(low)])
        return low[a:b]

    def owner(pos: int) -> str:
        """Чей факт в позиции pos: «new» / «transit» / «» — по последней метке до него в предложении."""
        head = low[max(s for s in starts if s <= pos) : pos]
        marks = [(m.start(), "new") for m in NEW_MARK_RE.finditer(head)]
        marks += [(m.start(), "transit") for m in TRANSIT_MARK_RE.finditer(head)]
        return max(marks)[1] if marks else ""

    for m in re.finditer("|".join(DELAY_PHRASES), low):
        if "данн" not in sentence(m.start()):  # «данных о задержках нет» — можно
            return False
    for m in ORDER_QTY_RE.finditer(low):
        if int(re.sub(r"\D", "", m.group(1))) != int(payload.get("order_qty") or 0):
            return False
    if payload.get("stock_now") and EMPTY_NOW_RE.search(low):
        return False
    if payload.get("status") == C.SHORTAGE:
        if any(p in low for p in SHORTAGE_DENY):
            return False
        for m in PROMISE_RE.finditer(low):
            if "отдельн" not in sentence(m.start()):
                return False
        gap = payload.get("gap_days")
        if gap and not re.search(rf"(?<!\d){gap}{DAYS_RE}", low):
            return False
    if not payload.get("on_order"):
        return True

    # Есть заказ в пути: факты нового заказа — только рядом с «новый», факты транзита — не рядом с ним.
    def day_str(key):
        v = payload.get(key)
        return pd.Timestamp(v).strftime("%d.%m.%Y") if v else None

    new_arr, po_arr = day_str("new_order_arrival_date"), day_str("on_order_next_date")
    if new_arr and new_arr != po_arr:
        for m in re.finditer(re.escape(new_arr), low):
            if owner(m.start()) != "new":
                return False
    if po_arr and po_arr != new_arr:
        for m in re.finditer(re.escape(po_arr), low):
            if owner(m.start()) == "new":
                return False
    lead, gap = payload.get("lead_days"), payload.get("gap_days")
    if lead is not None:
        for m in re.finditer(rf"(?<!\d){int(lead)}{DAYS_RE}", low):
            who, sent = owner(m.start()), sentence(m.start())
            if who == "transit" or (
                who == ""
                and not (lead == gap and re.search(r"пуст|дыр|без товара", sent))
                and "плеч" not in sent
            ):
                return False
    rel, po_rel = payload.get("reliability"), payload.get("on_order_reliability")
    if rel is not None and po_rel is not None:
        new_p, po_p = round(rel * 100), round(po_rel * 100)
        for m in re.finditer(r"(?<!\d)(\d+)\s*%", low):
            p = int(m.group(1))
            if p not in (new_p, po_p):
                return False
            if new_p == po_p:
                continue
            who = owner(m.start())
            if (p == new_p and who != "new") or (p == po_p and who == "new"):
                return False
    return True


def _text_ok(text: str, payload: dict, with_order: bool) -> bool:
    """Ответ модели годен: все числа и даты — из данных, нет отказа от заказа, смысл не спорит с расчётом."""
    if with_order and any(p in text.lower() for p in DENY_PHRASES):
        return False
    if "sku_id" in payload and not _meaning_ok(text, payload):
        return False
    nums: set = set()
    dates: set = set()
    words: list = []
    _facts(payload, nums, dates, words)
    for word in sorted(words, key=len, reverse=True):
        text = text.replace(word, " ")
    for d, m, y in DATE_RE.findall(text):
        if (int(d), int(m), int(y)) not in dates:
            return False
    text = DATE_RE.sub(" ", text)
    for token in NUM_RE.findall(text):
        value = float(token.replace(" ", "").replace(" ", "").replace(",", "."))
        if round(value, 2) not in nums:
            return False
    return True


def _key(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _ask(post: Callable, env: dict[str, str], prompt: str, timeout: int) -> str:
    """Один вызов модели с температурой 0; возвращает текст ответа или бросает исключение."""
    url = env["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {env['LLM_API_KEY']}"}
    body = {"model": env["LLM_MODEL"], "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    answer = post(url, headers, body, timeout)
    content = answer["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("пустой ответ модели")
    return content.strip()


def _parse_rows(content: str) -> dict[str, str]:
    """Ответ по строкам: JSON-объект sku_id → текст (возможна обёртка ```json)."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        raise TypeError("ответ не объект")
    return {str(k): v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}


def explain(
    result: PlanResult,
    cfg: Config,
    env: dict[str, str],
    cache_path: Path,
    use_llm: bool,
    _post: Callable | None = None,
    po: pd.DataFrame | None = None,
) -> tuple[PlanResult, str, dict]:
    """Заполняет explanation и пишет сводку. Расчёт плана не меняется ни при каких ответах модели.

    po — заказы в пути (supply.open_orders): из них модель получает дату ближайшего прихода.
    """
    plan = result.plan.copy()
    dpm = cfg.days_per_month
    plan["explanation"] = [template_explanation(row, cfg.today, dpm) for _, row in plan.iterrows()]
    summary = template_summary(result, cfg.today)
    stats = {"mode": "template", "calls": 0, "cached": 0, "errors": 0}

    ready = all(env.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    if not (use_llm and ready):
        return replace(result, plan=plan), summary, stats

    post = _post or _default_post
    cache = _load_cache(cache_path)
    model = env["LLM_MODEL"]
    row_prompt = _read_prompt("explain.md")
    targets = plan[plan["status"].isin(LLM_STATUSES)]
    payloads = {row["sku_id"]: build_row_payload(row, cfg.today, po, dpm) for _, row in targets.iterrows()}
    keys = {
        sid: _key(model, row_prompt, json.dumps(p, ensure_ascii=False, sort_keys=True))
        for sid, p in payloads.items()
    }
    texts: dict[str, str] = {}
    for sid, key in keys.items():
        if key in cache:
            if _text_ok(cache[key], payloads[sid], True):
                texts[sid] = cache[key]
                stats["cached"] += 1
            else:  # кэш противоречит цифрам — спросим заново
                stats["errors"] += 1

    pending = [sid for sid in payloads if sid not in texts]
    for start in range(0, len(pending), cfg.llm_batch):
        batch = pending[start : start + cfg.llm_batch]
        rows_json = json.dumps([payloads[s] for s in batch], ensure_ascii=False, indent=1)
        stats["calls"] += 1
        try:
            got = _parse_rows(
                _ask(post, env, row_prompt.replace("{rows_json}", rows_json), cfg.llm_timeout_s)
            )
        except LLM_ERRORS:  # сбой сети или формата → шаблон, расчёт не страдает
            stats["errors"] += 1
            continue
        for sid in batch:
            text = fix_decimals(got.get(sid) or "")
            if text and len(text) <= MAX_TEXT_LEN and _text_ok(text, payloads[sid], True):
                texts[sid] = text
                cache[keys[sid]] = text
            else:
                stats["errors"] += 1

    summary_ok = False
    sum_payload = _summary_payload(result, cfg)
    sum_order = sum_payload["order_qty_total"] > 0
    sum_prompt = _read_prompt("summary.md").replace(
        "{plan_json}", json.dumps(sum_payload, ensure_ascii=False, indent=1)
    )
    sum_key = _key(model, sum_prompt)
    if sum_key in cache and _text_ok(cache[sum_key], sum_payload, sum_order):
        summary, summary_ok = cache[sum_key], True
        stats["cached"] += 1
    else:
        if sum_key in cache:
            stats["errors"] += 1
        stats["calls"] += 1
        try:
            text = fix_decimals(_ask(post, env, sum_prompt, cfg.llm_timeout_s))
            if _text_ok(text, sum_payload, sum_order):
                summary, summary_ok = text, True
                cache[sum_key] = text
            else:
                stats["errors"] += 1
        except LLM_ERRORS:
            stats["errors"] += 1

    if texts:
        mask = plan["sku_id"].isin(texts)
        plan.loc[mask, "explanation"] = plan.loc[mask, "sku_id"].map(texts)
    if cache:
        _save_cache(cache_path, cache)

    good = len(texts) + int(summary_ok)
    total = len(payloads) + 1
    stats["mode"] = "llm" if good == total else "template" if good == 0 else "mixed"
    return replace(result, plan=plan), summary, stats
