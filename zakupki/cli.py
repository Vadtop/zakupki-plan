"""Команда запуска: читает выгрузку, считает план и пишет файлы в out/."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from zakupki import config as C
from zakupki.backtest import backtest
from zakupki.config import Config
from zakupki.dashboard import render_dashboard
from zakupki.data import DataError, load_dataset
from zakupki.demand import build_demand
from zakupki.explain import explain, load_env
from zakupki.forecast import forecast, seasonal_index
from zakupki.history import diff_plans, previous_snapshot, save_snapshot
from zakupki.plan import RISK_ORDER, build_plan
from zakupki.report import mode_name, write_summary
from zakupki.supply import open_orders, supplier_reliability

PLAN_FILE = "plan.csv"
RISKS_FILE = "risks.csv"
SUMMARY_FILE = "summary.md"
DASHBOARD_FILE = "dashboard.html"
CACHE_FILE = "llm_cache.json"
ENV_FILE = ".env"


def build_parser() -> argparse.ArgumentParser:
    """Собирает разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(prog="zakupki", description="Планирование закупок импортёра")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="посчитать план по выгрузке и записать файлы в out/")
    run_cmd.add_argument("--data", required=True, type=Path, help="папка с выгрузкой ERP")
    run_cmd.add_argument(
        "--today", default=None, help="дата расчёта, YYYY-MM-DD; без неё берётся дата данных"
    )
    run_cmd.add_argument("--out", default=Path("out"), type=Path, help="папка результата")
    run_cmd.add_argument("--no-llm", action="store_true", help="не вызывать модель: пояснения шаблонные")
    run_cmd.add_argument("--min-markup", type=float, default=None, help="порог наценки, процентов")
    run_cmd.add_argument("--cover-days", type=int, default=None, help="покрытие по умолчанию, дней")
    run_cmd.add_argument("--z", type=float, default=None, help="коэффициент страхового запаса")
    return parser


def _parse_today(value: str | None) -> pd.Timestamp | None:
    """Разбирает --today; None остаётся None — тогда дата берётся из данных."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DataError(f"Не разобрать дату --today: {value}. Ожидается YYYY-MM-DD") from exc
    if pd.isna(stamp):
        raise DataError(f"Не разобрать дату --today: {value}. Ожидается YYYY-MM-DD")
    return stamp.normalize()


def _counts_line(counts: pd.Series, order: list[str]) -> str:
    """Строка «статус число, ...» в заданном порядке."""
    return ", ".join(f"{name} {int(counts.get(name, 0))}" for name in order)


def _data_date_warning(ds) -> str:
    """Предупреждение, когда дата расчёта не задана и взята из самих данных."""
    has_dates = len(ds.orders) > 0 or len(ds.receipts) > 0
    source = "последнего заказа/прихода" if has_dates else "последнего месяца продаж"
    return (
        f"ВНИМАНИЕ: дата расчёта не задана и взята из {source} ({ds.today:%d.%m.%Y}); "
        "для точного расчёта передайте --today ГГГГ-ММ-ДД"
    )


def _print_result(
    data_dir: Path, ds, result, out_dir: Path, explicit_today: bool, stats: dict, diff: dict
) -> None:
    """Печатает 5–8 строк итога по-русски."""
    plan = result.plan
    source = "--today" if explicit_today else "дата данных"
    statuses = _counts_line(plan["status"].value_counts(), C.STATUS_ORDER)
    risk_counts = result.risks["risk_type"].value_counts() if len(result.risks) else pd.Series(dtype="int64")
    risks = _counts_line(risk_counts, RISK_ORDER)
    print(f"Выгрузка: {data_dir}")
    print(f"Дата расчёта: {ds.today:%Y-%m-%d} ({source})")
    print(f"SKU: {len(plan)}; к заказу: {int(plan['order_qty'].sum())} шт.")
    print(f"Статусы: {statuses}")
    print(f"Риски: {risks}")
    print(
        f"Файлы: {out_dir / DASHBOARD_FILE}, {out_dir / PLAN_FILE}, {out_dir / RISKS_FILE}, {out_dir / SUMMARY_FILE}"
    )
    if diff["first_run"]:
        print("Что изменилось: прошлого запуска нет")
    else:
        print(
            f"Что изменилось: новых к заказу {len(diff['new_orders'])}, количество {len(diff['qty_changes'])}, "
            f"отменено {len(diff['cancelled'])}, пропало из выгрузки {len(diff.get('gone') or [])}, "
            f"сдвигов дат {len(diff['date_shifts'])}, "
            f"ушло из риска {len(diff['left_risk'])}, пришло в риск {len(diff['new_risk'])}"
        )
    print(f"Пояснения: {mode_name(stats.get('mode', 'template'))}")


def run(args: argparse.Namespace) -> int:
    """Считает план по выгрузке, пишет файлы в out/ и снимок в out/history/<дата>/."""
    cfg = Config().with_overrides(
        today=_parse_today(args.today),
        min_markup_pct=args.min_markup,
        default_cover_days=args.cover_days,
        z=args.z,
    )
    ds = load_dataset(args.data, cfg.today)
    if args.today is None:
        print(_data_date_warning(ds), file=sys.stderr)
    demand = build_demand(ds, cfg)
    index = seasonal_index(demand, cfg)
    params, fc = forecast(demand, index, cfg, ds.today)
    result = build_plan(ds, demand, index, params, fc, cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(Path(ENV_FILE))
    reliability = supplier_reliability(ds, cfg)
    po = open_orders(ds, reliability, cfg)
    result, summary_text, stats = explain(
        result, cfg.with_overrides(today=ds.today), env, out_dir / CACHE_FILE, use_llm=not args.no_llm, po=po
    )
    diff = diff_plans(previous_snapshot(out_dir, ds.today), result.plan)

    result.plan.to_csv(out_dir / PLAN_FILE, index=False)
    result.risks.to_csv(out_dir / RISKS_FILE, index=False)
    write_summary(result, ds, summary_text, stats, diff, index, out_dir / SUMMARY_FILE)
    render_dashboard(
        result,
        index,
        demand,
        fc,
        reliability,
        po,
        diff,
        summary_text,
        ds,
        out_dir / DASHBOARD_FILE,
        cfg.with_overrides(today=ds.today),
        generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"),  # только подпись в шапке
        backtest=backtest(args.data, cfg.with_overrides(today=ds.today)),  # только дашборд, в план не идёт
    )
    files = [out_dir / name for name in (PLAN_FILE, RISKS_FILE, SUMMARY_FILE, DASHBOARD_FILE)]
    save_snapshot(out_dir, ds.today, files)

    _print_result(
        args.data, ds, result, out_dir, explicit_today=args.today is not None, stats=stats, diff=diff
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа: разбирает аргументы, печатает ошибку данных и возвращает код выхода."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = {"run": run}[args.command]
    try:
        return handler(args)
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 2
