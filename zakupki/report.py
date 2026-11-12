"""Сводка цифр в summary.md: статусы, заказы, риски, качество данных, проверочные факты."""

from pathlib import Path

import pandas as pd

from zakupki import config as C
from zakupki.data import Dataset
from zakupki.plan import RISK_ORDER, PlanResult
from zakupki.supply import SETTLED_STATUSES, supplier_reliability

# Русские названия месяцев по номеру 1..12, в предложном падеже («в октябре»).
MONTH_NAMES = [
    "январе", "феврале", "марте", "апреле", "мае", "июне",
    "июле", "августе", "сентябре", "октябре", "ноябре", "декабре",
]
# Категория, по которой в сводке показывается месяц пика сезона.
SEASON_CATEGORY = "space_heaters"
MODE_NAMES = {"llm": "модель", "template": "шаблон", "mixed": "смешанно"}


def mode_name(mode: str) -> str:
    """Русское название режима пояснений."""
    return MODE_NAMES.get(mode, MODE_NAMES["template"])


def _status_lines(plan: pd.DataFrame) -> list[str]:
    """Число SKU по каждому статусу — все статусы, даже с нулём."""
    counts = plan["status"].value_counts()
    return [f"- {status}: {int(counts.get(status, 0))}" for status in C.STATUS_ORDER]


def _risk_lines(risks: pd.DataFrame) -> list[str]:
    """Число рисков по типам."""
    counts = risks["risk_type"].value_counts() if len(risks) else pd.Series(dtype="int64")
    return [f"- {risk}: {int(counts.get(risk, 0))}" for risk in RISK_ORDER]


def _reliability_lines(ds: Dataset) -> list[str]:
    """Надёжность поставщиков и оговорка про закрытые заказы с нулевым приходом."""
    rel = supplier_reliability(ds)
    ordered = int(rel["ordered"].sum())
    received = int(rel["received"].sum())
    overall = received / ordered if ordered else 1.0
    lines = [
        (
            f"- Общая надёжность поставщиков: {overall:.2f} "
            f"(получено {received} из {ordered} шт. по закрытым и исполненным заказам)"
        )
    ]
    if len(ds.orders):
        settled = ds.orders[ds.orders["status"].isin(SETTLED_STATUSES)]
        zero = settled[(settled["qty_received"] == 0) & (settled["qty_ordered"] > 0)]
        lines.append(
            f"- Заказов `closed`/`completed` с нулевым приходом: {len(zero)} из {len(settled)} — "
            "похоже на отменённые, в надёжность не входят"
        )
    return lines


def _fact_lines(plan: pd.DataFrame, index: pd.DataFrame) -> list[str]:
    """Проверочные факты: сезон обогревателей, мёртвые, низкая наценка."""
    lines: list[str] = []
    if len(index):
        cat = index[index["category"] == SEASON_CATEGORY]
        if len(cat):
            peak = cat.loc[cat["index"].idxmax()]
            lines.append(
                f"- Сезон {SEASON_CATEGORY}: пик индекса в {MONTH_NAMES[int(peak['month_num']) - 1]} "
                f"({float(peak['index']):.2f})"
            )
    dead = int((plan["status"] == C.DEAD).sum())
    low = int((plan["status"] == C.LOW_MARKUP).sum())
    lines.append(f"- Мёртвый ассортимент (нет продаж 12 мес.): {dead} SKU")
    lines.append(f"- Низкая наценка: {low} SKU")
    lines.append(f"- Нет наценки, продажи есть (нет себестоимости): {int((plan['status'] == C.NO_MARKUP).sum())} SKU")
    return lines


def _diff_lines(diff: dict | None) -> list[str]:
    """Раздел «Что изменилось»; без прошлого снимка — одна строка."""
    if not diff:
        return []
    if diff.get("first_run"):
        return ["## Что изменилось", "", "Прошлого запуска нет."]
    lines = ["## Что изменилось", ""]
    lines.append(f"- Новых к заказу: {len(diff.get('new_orders') or [])}")
    lines.append(f"- Изменилось количество: {len(diff.get('qty_changes') or [])}")
    lines.append(f"- Отменены: {len(diff.get('cancelled') or [])}")
    lines.append(f"- Пропали из выгрузки: {len(diff.get('gone') or [])}")
    lines.append(f"- Сдвинулись даты заказа: {len(diff.get('date_shifts') or [])}")
    lines.append(f"- Ушли из риска: {len(diff.get('left_risk') or [])}")
    lines.append(f"- Пришли в риск: {len(diff.get('new_risk') or [])}")
    return lines


def build_summary(
    result: PlanResult, ds: Dataset, summary_text: str, stats: dict, diff: dict | None, index: pd.DataFrame
) -> str:
    """Собирает текст summary.md из готовых цифр."""
    plan, risks = result.plan, result.risks
    order_qty = int(plan["order_qty"].sum())
    late = risks[risks["risk_type"] == C.OVERDUE_PO] if len(risks) else risks
    overdue = f"{late['po_id'].nunique()} ({len(late)} строк)" if len(late) else "0"

    parts: list[str] = ["# Сводка плана закупки", "", f"Дата расчёта: {ds.today:%Y-%m-%d}", ""]
    if summary_text.strip():
        parts += [summary_text.strip(), ""]
    parts += ["## Итог", "", f"- SKU в расчёте: {len(plan)}"]
    parts += [f"- К заказу всего: {order_qty} шт."]
    parts += [f"- Просроченных заказов поставщиков: {overdue}"]
    parts += _reliability_lines(ds)
    parts += ["", "## SKU по статусам", ""] + _status_lines(plan)
    parts += ["", "## Риски", ""] + _risk_lines(risks)
    parts += ["", "## Проверочные факты", ""] + _fact_lines(plan, index)
    parts += ["", "## Качество данных", ""]
    parts += [f"- {issue}" for issue in ds.issues] or ["- Отброшенных строк нет"]
    diff_lines = _diff_lines(diff)
    if diff_lines:
        parts += [""] + diff_lines
    parts += ["", f"Пояснения: {mode_name(stats.get('mode', 'template'))}"]
    return "\n".join(parts) + "\n"


def write_summary(
    result: PlanResult,
    ds: Dataset,
    summary_text: str,
    stats: dict,
    diff: dict | None,
    index: pd.DataFrame,
    path: Path,
) -> None:
    """Пишет summary.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_summary(result, ds, summary_text, stats, diff, index), encoding="utf-8")
