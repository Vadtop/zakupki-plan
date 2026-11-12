"""Пороги и константы проекта: всё настраиваемое живёт здесь."""

from dataclasses import dataclass, fields, replace

import pandas as pd

# Статусы плана. Основной статус у SKU один, по старшинству STATUS_ORDER.
DEAD = "МЁРТВЫЙ"
LOW_MARKUP = "НИЗКАЯ НАЦЕНКА"
NO_MARKUP = "НЕТ НАЦЕНКИ"  # продажи есть, наценка не посчитана (нет себестоимости)
SHORTAGE = "РИСК ДЕФИЦИТА"
ORDER_NOW = "ЗАКАЗАТЬ СЕЙЧАС"
EXCESS = "ИЗБЫТОК"
PLAN = "ЗАПЛАНИРОВАТЬ"
OK = "ЗАПАС В НОРМЕ"
SHORT_HISTORY = "МАЛО ИСТОРИИ"
STATUS_ORDER = [DEAD, LOW_MARKUP, NO_MARKUP, SHORTAGE, ORDER_NOW, EXCESS, PLAN, OK]

# Русские названия категорий из выгрузки — для дашборда и сводки модели
CATEGORY_NAMES = {
    "concrete_mixers": "Бетономешалки",
    "grain_mills": "Зернодробилки",
    "sanitary_faucets": "Смесители",
    "space_heaters": "Обогреватели",
    "string_trimmers": "Триммеры",
    "trimmer_blades_accessories": "Ножи и леска для триммеров",
}

# Типы рисков.
OVERDUE_PO = "ПРОСРОЧЕН ЗАКАЗ"


@dataclass(frozen=True)
class Config:
    """Все пороги расчёта. Меняется только через with_overrides."""

    today: pd.Timestamp | None = None  # None → дата данных
    min_markup_pct: float = 25.0
    default_cover_days: int = 30  # до следующего пересмотра: закупщики планируют раз в месяц (ТЗ)
    z: float = 1.65
    lead_default_days: int = 123
    lead_min_days: int = 30
    lead_max_days: int = 210
    order_now_window_days: int = 30
    overdue_shift_days: int = 30
    overdue_stale_days: int = 90  # просрочка больше N дн. — не приход, только риск
    excess_extra_months: int = 6
    horizon_months: int = 12
    cover_horizon_months: int = 36
    history_months: int = 12  # окно уровня и σ
    dead_window_months: int = 12  # нет чистых продаж за N полных мес. → МЁРТВЫЙ
    short_history_months: int = 12  # меньше → МАЛО ИСТОРИИ
    trend_window_months: int = 6
    trend_damping: float = 0.5
    trend_min: float = 0.8
    trend_max: float = 1.25
    min_season_index: float = 0.15  # месяцы с меньшим индексом не идут в уровень
    recent_level_months: int = 3  # потолок уровня: среднее последних N мес. (0 = выкл.)
    recent_drop_ratio: float = 0.5  # потолок и при падении: последние N мес. < доли уровня
    safety_cap_months: float = 2.0  # страховой запас ≤ N мес. спроса (0 = без потолка)
    safety_alt_days: int = 60  # дашборд: сценарий «страховой ≤ прогноза N дн. после прихода», план не меняет
    reliability_skip_zero_closed: bool = True  # closed с qty_received=0 не считать недопоставкой
    lot_min_receipts: int = 3
    lot_max_overage_months: float = 1.0  # до партии не округляем, если это добавит больше N мес. спроса
    days_per_month: float = 30.4
    abc_months: int = 12  # ABC по чистым продажам, шт., за N полных месяцев
    abc_a_share: float = 0.80  # A — первые 80% штук
    abc_b_share: float = 0.95  # B — до 95%, дальше C
    llm_batch: int = 25
    llm_timeout_s: int = 60
    telegram_timeout_s: int = 20  # уведомление в Telegram после пересчёта
    telegram_max_chars: int = 4000  # предел сообщения Telegram — 4096 символов
    long_gap_months: float = 3.0  # дашборд: «дыра» от N мес. — нужен местный поставщик
    unreliable_below: float = 0.8  # дашборд: надёжность поставщика ниже — метка «недовоз»
    backtest_months: int = 12  # точность прогноза: срез на N мес. раньше, факт — N мес. после
    backtest_horizons: tuple[int, ...] = (4, 6, 7)  # горизонты 1–N мес.; плечо импорта 4–6 мес.

    def with_overrides(self, **kw) -> "Config":
        """Копия с изменёнными полями; None игнорируется, неизвестное поле — ошибка."""
        known = {f.name for f in fields(self)}
        unknown = set(kw) - known
        if unknown:
            raise TypeError(f"Неизвестное поле Config: {', '.join(sorted(unknown))}")
        return replace(self, **{k: v for k, v in kw.items() if v is not None})
