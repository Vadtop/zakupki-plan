# Границы и правила проекта

Читать первым. Имена — ровно такие; новое общее имя — только через спецификацию.

## Правила проекта

- **Цифры считает только код.** Модель получает готовые числа и пишет текст; на план она не влияет.
- **Все пороги и константы — в `zakupki/config.py`** (`Config`). В остальных модулях числовых порогов нет, только `cfg.<поле>`.
- **Детерминизм:** никакой случайности, сортировки с `kind="mergesort"` или по полному ключу, даты — из `cfg.today`, никогда `date.today()` / `datetime.now()` в расчёте.
- **`data/raw/` только читать.**
- Python ≥ 3.11; зависимости — pandas, numpy; тесты — pytest. HTTP — `urllib.request`. Ничего больше.
- Комментарии и docstring — по-русски, коротко, по делу. Ни слова про ИИ-генерацию, Claude, Kimi, промпты для исполнителя.
- Тесты — на синтетических данных из `tests/conftest.py` (фабрика маленького набора), не на `data/raw/`. Один тест-прогон реальных данных — `tests/test_real_data.py`, пропускается, если папки нет.
- Ошибки пользователю — по-русски, через `DataError(message)`; CLI ловит её, печатает сообщение, выход 2.

## Стек и команды

- Всё запускается **только в песочнице** `autopilot-sandbox:1` (там pandas 3.1, numpy 2.4, pytest 9.2, ruff 0.17, `python3`); код совместим с pandas ≥ 2.2 и Python ≥ 3.11; `pyproject.toml` (таск 01) объявляет `requires-python = ">=3.11"`, зависимости `pandas>=2.2`, `numpy>=1.26`, dev — `pytest`, `ruff`; настройки ruff — `line-length = 110`.
- Запуск: `python -m zakupki run --data data/raw/share --today 2026-08-13`.
- **Одна проверка:** `python3 -m ruff check zakupki tests --extend-ignore EXE002 && python3 -m pytest -q` (на ПК — через `docker run --rm --network none -v "$(pwd -W):/work" -w /work autopilot-sandbox:1 …`).
- Один файл тестов: `python -m pytest -q tests/test_plan.py`.
- Не трогать: `data/raw/`, `.autopilot/`, `BRIEF.md`, `TODO.md`. В `CLAUDE.md` менять только текст между `<!-- autopilot:start -->` и строкой `## Как здесь работает Autopilot` (таск 02).

## Границы, решённые в спецификации

Все DataFrame — pandas. Месяц — `pd.Period(freq="M")` в колонке `month`. Даты — `pd.Timestamp` (без времени).

### `zakupki/config.py`
```python
@dataclass(frozen=True)
class Config:
    today: pd.Timestamp | None = None    # None → дата данных
    min_markup_pct: float = 25.0
    default_cover_days: int = 90
    z: float = 1.65
    lead_default_days: int = 123
    lead_min_days: int = 30
    lead_max_days: int = 210
    order_now_window_days: int = 30
    overdue_shift_days: int = 30
    excess_extra_months: int = 6
    horizon_months: int = 12
    cover_horizon_months: int = 36
    history_months: int = 12          # окно уровня и σ
    short_history_months: int = 12    # меньше → МАЛО ИСТОРИИ
    trend_window_months: int = 6
    trend_damping: float = 0.5
    trend_min: float = 0.8
    trend_max: float = 1.25
    min_season_index: float = 0.15    # месяцы с меньшим индексом не идут в уровень
    lot_min_receipts: int = 3
    days_per_month: float = 30.4
    llm_batch: int = 25
    llm_timeout_s: int = 60
    def with_overrides(self, **kw) -> "Config"   # dataclasses.replace, None игнорируется
```
Статусы и флаги — константы там же:
`DEAD="МЁРТВЫЙ"`, `LOW_MARKUP="НИЗКАЯ НАЦЕНКА"`, `SHORTAGE="РИСК ДЕФИЦИТА"`, `ORDER_NOW="ЗАКАЗАТЬ СЕЙЧАС"`, `EXCESS="ИЗБЫТОК"`, `PLAN="ЗАПЛАНИРОВАТЬ"`, `OK="ЗАПАС В НОРМЕ"`, `SHORT_HISTORY="МАЛО ИСТОРИИ"`, `STATUS_ORDER=[DEAD, LOW_MARKUP, SHORTAGE, ORDER_NOW, EXCESS, PLAN, OK]`; тип риска `OVERDUE_PO="ПРОСРОЧЕН ЗАКАЗ"`.

### `zakupki/data.py` — загрузка и проверка
```python
class DataError(Exception): ...
@dataclass
class Dataset:
    sku: pd.DataFrame        # sku_dim как есть + avg_markup_pct_12m float (NaN = пусто)
    sales: pd.DataFrame      # sku_id, category, month(Period), qty_sold, qty_returned, partial_month
    stock: pd.DataFrame      # sku_id, month(Period), stock_qty
    orders: pd.DataFrame     # purchase_orders; order_date, promised_date — Timestamp
    receipts: pd.DataFrame   # sku_id, receipt_date(Timestamp), qty, supplier_id
    suppliers: pd.DataFrame  # supplier_dim
    today: pd.Timestamp      # итоговая дата расчёта
    issues: list[str]        # «Качество данных»: что отброшено и почему, по-русски
def load_dataset(data_dir: Path, today: pd.Timestamp | None) -> Dataset
def detect_data_date(orders, receipts, sales) -> pd.Timestamp   # последняя дата receipt_date / order_date; на реальных данных 2026-08-01
```
Владеет: чтением CSV, проверкой файлов/колонок (`DataError`), отбрасыванием битых строк (в `issues`), обрезкой по `today` (продажи с месяцем > месяца `today`, приходы и заказы с датой > `today`). Остальные модули файлов не читают.

### `zakupki/demand.py` — спрос
```python
def build_demand(ds: Dataset, cfg: Config) -> pd.DataFrame
```
Возвращает по каждому SKU и полному месяцу (без `partial_month=1` и без месяца `today`):
`sku_id, category, month, demand (float, NaN если lost), sold_net (int), lost (bool), stock_start (float|NaN)`.
`lost` = `sold_net == 0` и остаток на 1-е число == 0, только для месяцев ≥ первого месяца снимков; отсутствующая строка остатка в таком месяце = 0.

### `zakupki/forecast.py` — прогноз
```python
def seasonal_index(demand: pd.DataFrame, cfg: Config) -> pd.DataFrame
    # category, month_num(1..12), index (среднее 1 по категории), share (доля года)
def forecast(demand: pd.DataFrame, index: pd.DataFrame, cfg: Config, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]
    # 1) params: sku_id, level, trend, sigma, history_months, short_history(bool)
    # 2) fc: sku_id, month (cover_horizon_months месяцев от месяца today, включая его), qty (float)
```
Месяц `today` в прогнозе — целиком; симуляция вычитает уже прошедшую долю (см. plan).

### `zakupki/supply.py` — поставки
```python
def lead_days(ds: Dataset, cfg: Config) -> pd.Series              # index sku_id → int дней
def supplier_reliability(ds: Dataset) -> pd.DataFrame             # supplier_id, reliability, orders_n, received, ordered
def open_orders(ds: Dataset, reliability: pd.DataFrame, cfg: Config) -> pd.DataFrame
    # po_id, sku_id, supplier_id, qty_pending, promised_date, expected_date, expected_qty, overdue(bool)
def lot_size(ds: Dataset, cfg: Config) -> pd.Series               # sku_id → int (1 = кратности нет)
def current_stock(ds: Dataset) -> pd.Series                       # sku_id → float, на дату today
```

### `zakupki/simulation.py` — симуляция запаса и даты (таск 04a)
```python
def run_simulation(stock_now: pd.Series, fc: pd.DataFrame, open_po: pd.DataFrame, lead: pd.Series,
                   params: pd.DataFrame, cfg: Config, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]
    # positions — одна строка на sku_id из fc:
    #   sku_id, stock_now, lead_days, safety_stock, safety_days, cover_days, cover_months,
    #   stockout_date (NaT — не обнуляется за cover_horizon_months), order_date (NaT — нет), gap_months
    # sim — sku_id, month, forecast, arrivals, stock_end; cover_horizon_months месяцев от месяца today
def projected_stock(sim: pd.DataFrame, sku_id: str, at: pd.Timestamp, cfg: Config) -> float
    # остаток по симуляции на дату (внутри месяца — пропорционально), без ухода ниже 0 не обрезать
def forecast_between(fc: pd.DataFrame, sku_id: str, start: pd.Timestamp, end: pd.Timestamp, cfg: Config) -> float
    # прогноз спроса за период, неполные месяцы — долей дней
def arrivals_after(open_po: pd.DataFrame, sku_id: str, at: pd.Timestamp) -> float
    # сумма expected_qty с expected_date > at
```
Правила: в месяце today прогноз вычитается только за оставшиеся дни; приходы — `expected_qty` в месяц `expected_date`;
`stockout_date` интерполируется внутри месяца; `cover_days = stockout_date − today` (нет обнуления — дни до конца горизонта),
`cover_months = cover_days / days_per_month`; `safety_stock = z × sigma × √(lead_days / days_per_month)`;
`safety_days = safety_stock / средний дневной прогноз на плече` (прогноз 0 → 0);
`order_date = stockout_date − lead_days − safety_days`; `gap_months` — месяцев от `stockout_date` до `today + lead_days`, если обнуление раньше, иначе 0.

### `zakupki/plan.py` — количества, статусы, риски, календарь (таск 04b)
```python
@dataclass
class PlanResult:
    plan: pd.DataFrame       # по одной строке на каждый SKU из sku_dim, колонки ниже
    risks: pd.DataFrame      # risk_type, sku_id, category, supplier_id, detail, gap_months, excess_months, po_id, promised_date, qty
    sim: pd.DataFrame        # sku_id, month, stock_end, arrivals, forecast — первые horizon_months
    calendar: pd.DataFrame   # month, positions, qty
def build_plan(ds, demand, index, params, fc, cfg) -> PlanResult
```
Колонки `plan` (порядок фиксирован, это и есть `plan.csv`):
`sku_id, category, supplier_id, status, flags, order_qty, order_date, stock_now, on_order (сырые штуки в пути), on_order_expected (× надёжность), lead_days, cover_days, cover_months, stockout_date, safety_stock, gap_months, reliability, avg_markup_pct_12m, level, trend, short_history, need_qty, explanation`.
- `order_qty` — к автозаказу (0 у МЁРТВЫЙ, НИЗКАЯ НАЦЕНКА, ЗАПАС В НОРМЕ, ИЗБЫТОК); `need_qty` — расчётная потребность до отсечений.
- `flags` — сработавшие статусы ниже основного и `МАЛО ИСТОРИИ`, через «; ».
- `explanation` — пусто в `build_plan`, заполняет `explain`.
- Сортировка: по `STATUS_ORDER`, потом `order_date`, потом `sku_id`.

### `zakupki/explain.py` — пояснения
```python
def explain(result: PlanResult, cfg: Config, env: dict[str, str], cache_path: Path, use_llm: bool, _post=None) -> tuple[PlanResult, str, dict]
    # возвращает план с заполненным explanation, сводку (5–7 строк) и stats {"mode": "llm"|"template"|"mixed", "calls": n, "cached": n, "errors": n}
def template_explanation(row: pd.Series) -> str
def template_summary(result: PlanResult) -> str
def build_row_payload(row: pd.Series) -> dict    # белый список полей: id и цифры
def load_env(path: Path) -> dict[str, str]       # .env + os.environ (окружение главнее)
```
Промпты читаются из `prompts/explain.md` и `prompts/summary.md`; `{rows_json}` / `{plan_json}` — места подстановки.

### `zakupki/history.py` — снимки и «что изменилось»
```python
def previous_snapshot(out_dir: Path, today: pd.Timestamp) -> pd.DataFrame | None   # последний снимок с датой ≤ today
def diff_plans(prev: pd.DataFrame | None, cur: pd.DataFrame) -> dict
    # {"first_run": bool, "new_orders": [...], "date_shifts": [{"sku_id","from","to","days"}], "left_risk": [...], "new_risk": [...]}
def save_snapshot(out_dir: Path, today: pd.Timestamp, files: list[Path]) -> Path   # out/history/YYYY-MM-DD/
```

### `zakupki/dashboard.py` — HTML
```python
def render_dashboard(result: PlanResult, index, demand, fc, reliability, open_po, diff: dict, summary: str, ds: Dataset, out_path: Path) -> None
```
Шаблон `zakupki/templates/dashboard.html`, данные — один `<script type="application/json" id="data">`. Plotly `https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js`.

### `zakupki/cli.py` + `zakupki/__main__.py`
`python -m zakupki run --data DIR [--today YYYY-MM-DD] [--out out] [--no-llm] [--min-markup N] [--cover-days N] [--z N]`.
Порядок: load → demand → index/forecast → plan → explain → previous_snapshot/diff → plan.csv, risks.csv, summary.md, dashboard.html → save_snapshot. Печатает 5–8 строк итога. `DataError` → сообщение, код 2.
`zakupki/report.py`: `def write_summary(result, ds, summary_text, stats, diff, index, path)` — `summary.md`.

## Швы для тестов

- `build_demand`, `seasonal_index`, `forecast`, `build_plan` — чистые функции над DataFrame: тесты собирают `Dataset` из `tests/conftest.py`.
- `explain(..., use_llm=False)` и подмена HTTP через параметр `_post` (вызываемый объект) — без сети.
- `diff_plans` — на двух DataFrame.

## Что построили законченные таски

_(пополняется по ходу сборки)_

### T01 — данные и конфиг
- `Config` — все поля как выше; `with_overrides(**kw)`: `None` игнорируется, неизвестное поле → `TypeError`.
- `load_dataset(data_dir, today)`, `detect_data_date(orders, receipts, sales)` (нет приходов и заказов → конец последнего месяца продаж), `DataError`, `Dataset` — как выше.
- Обрезка: продажи и остатки — `month <= месяц today`, заказы и приходы — `дата <= today`. Месяц today остаётся в `sales` (нужен для остатка на сегодня), но `build_demand` исключает его по дате.
- Типы: `sales.month`, `stock.month` — `Period[M]`; `orders.order_date/promised_date`, `receipts.receipt_date` — `Timestamp`; `sku.avg_markup_pct_12m` — float64, NaN = пусто.
- `tests/conftest.py`: `make_dataset(*, skus, sales, stock, orders, receipts, suppliers, today="2026-08-13", issues) -> Dataset`; фикстуры `make_dataset_factory`, `small_dataset`; константы `TODAY`, `SKU_A`, `SKU_B`, `SKU_C`. Свои фикстуры следующих тасков — в их тест-файлах.

### T03 — поставки
- `lead_days(ds, cfg)` → Series int64 по `sku_id`; `lot_size(ds, cfg)` → Series int64 (1 = нет кратности); `current_stock(ds)` → Series float64.
- `supplier_reliability(ds)` → `supplier_id, reliability, orders_n, received, ordered`.
- `open_orders(ds, reliability, cfg)` → `po_id, sku_id, supplier_id, qty_pending, promised_date, expected_date, expected_qty, overdue`.
- `current_stock` берёт спрос с 1-го по today из строки продаж месяца today (другой гранулярности нет); нетто-спрос не меньше 0.

### T02 — спрос и прогноз
- `build_demand(ds, cfg)` → `sku_id, category, month(Period[M]), demand(float, NaN = lost), sold_net(int64), lost(bool), stock_start(float|NaN)`.
- `seasonal_index(demand, cfg)` → `category, month_num(1..12), index(среднее 1), share(сумма 1)`; индекс — по сумме спроса полных календарных лет.
- `forecast(demand, index, cfg, today)` → `(params, fc)`: params — `sku_id, level, trend, sigma, history_months(int64), short_history(bool)`; fc — `sku_id, month(Period[M]), qty(float)`, `cover_horizon_months` строк от месяца today включительно.
- σ при < 2 пригодных месяцах = 0; уровень без пригодных месяцев (все lost или индекс < min_season_index) = 0.

### T04a — симуляция
- `run_simulation(stock_now, fc, open_po, lead, params, cfg, today)` → `(positions, sim)`; колонки — `POSITION_COLUMNS`, `SIM_COLUMNS` в `simulation.py`; одна строка positions на SKU из `fc`.
- `sim.forecast` в месяце today уже уменьшен на долю оставшихся дней; приходы ложатся в месяц `expected_date` (раньше месяца today → в первый месяц).
- `forecast_between(fc, sku_id, start, end, cfg)` и `projected_stock(sim, sku_id, at, cfg)` — интервал `[start, end)`, внутри месяца линейно; `projected_stock` ниже нуля не обрезает. `arrivals_after(open_po, sku_id, at)` — `expected_date > at`.
- `cover_days`, `stockout_date` округлены до дня; не обнуляется за горизонт → NaT, `cover_days` до конца горизонта, `gap_months` 0. `safety_days = safety_stock / (прогноз за плечо / lead_days)`.

### T04b — план
- `build_plan(ds, demand, index, params, fc, cfg)` → `PlanResult(plan, risks, sim, calendar)`; колонки — `PLAN_COLUMNS`, `RISK_COLUMNS` в `plan.py`. `demand`, `index` в расчёте не используются.
- `risks.risk_type`: РИСК ДЕФИЦИТА (`gap_months`, `qty`), ПРОСРОЧЕН ЗАКАЗ (`po_id`, `promised_date`, `qty`), ИЗБЫТОК (`excess_months` = сверх плеча+покрытия; остаток + в пути > 0 и не мёртвый — F2), МЁРТВЫЙ («распродать», только с остатком).
- `calendar`: `month, positions, qty` — 12 месяцев от месяца today; прошедшие даты заказа — в текущем месяце.
- `order_date` в plan — дата из симуляции, у дефицитных бывает в прошлом. ЗАКАЗАТЬ СЕЙЧАС не ставится при РИСК ДЕФИЦИТА.
- Реальные данные, today 2026-08-13 (после F2): МЁРТВЫЙ 378, ДЕФИЦИТ 179, СЕЙЧАС 69, НАЦЕНКА 68, ИЗБЫТОК 49, ПЛАН 37, НОРМА 11; к заказу 158 245 шт.; рисков: дефицит 179, избыток 57, мёртвый 41, просрочен 9.

### T05 — CLI и сводка
- `cli.main(argv=None) -> int`, `build_parser()`, `run(args) -> int`; константы `PLAN_FILE`, `RISKS_FILE`, `SUMMARY_FILE`. Флаги: `--data --today --out --no-llm --min-markup --cover-days --z`.
- Места для explain / diff / dashboard / снимка оставлены в `cli.run` комментариями «Шаг 06»; `--no-llm` пока ничего не выключает.
- `report.write_summary(result, ds, summary_text, stats, diff, index, path)`, `report.build_summary(...) -> str`, `report.mode_name(mode) -> str`.
- `tests/test_real_data.py` — прогон по `data/raw/share`, пропуск без папки. Цифры совпали с T04b.
- EXE002 в ruff — бит исполняемости от монтирования с Windows, игнорируется флагом.

### T06 — дашборд, пояснения, история
- `explain(...)`: `_post(url, headers, body, timeout) -> dict` в формате OpenAI; модель — только для основных статусов ЗАКАЗАТЬ СЕЙЧАС и РИСК ДЕФИЦИТА, остальным шаблон; кэш `out/llm_cache.json`; пустые LLM_* → шаблон.
- `diff_plans`: new_orders {sku_id, status, order_qty, order_date}, date_shifts {sku_id, from, to, days}, left_risk/new_risk {sku_id}.
- `cli`: `DASHBOARD_FILE`, `CACHE_FILE`, `ENV_FILE` (.env в текущей папке). Дашборд на реальных данных 1,44 МБ.

### F1 — ремонт ветки
- `safety_days = min(…, lead_days)` (D01); ЗАКАЗАТЬ СЕЙЧАС / ЗАПЛАНИРОВАТЬ только при need > 0 (D03); риск ДЕФИЦИТА в risks — только у основного статуса, у МЁРТВЫЙ флага нет (D02).
- `template_explanation(row, today=None)`, `build_row_payload(row, today=None)`: прошедшая дата заказа → «сразу» / today. `dashboard.ASAP_STATUSES`, поле `order_asap` в JSON плана.
- `tests/conftest._no_llm_no_network` (autouse: delenv LLM_*, urlopen → RuntimeError), `tests/test_cli._isolated_cwd`.

### F2 — приёмка
- `plan._statuses(..., in_transit=0.0)`: избыток при остатке + в пути > 0. `cli._data_date_warning(ds)` → stderr, только без `--today`.
- `run_daily.ps1 -Today` (по умолчанию `$env:DATA_DATE`); cron и workflow передают `DATA_DATE` в `--today`.
