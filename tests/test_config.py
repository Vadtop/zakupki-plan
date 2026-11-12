"""Пороги и константы: значения и поведение берём из interfaces.md."""

import dataclasses

import pandas as pd
import pytest

from zakupki.config import (
    DEAD,
    EXCESS,
    LOW_MARKUP,
    NO_MARKUP,
    OK,
    ORDER_NOW,
    OVERDUE_PO,
    PLAN,
    SHORT_HISTORY,
    SHORTAGE,
    STATUS_ORDER,
    Config,
)

# Ожидаемые значения — из раздела «zakupki/config.py» в interfaces.md.
DEFAULTS = {
    "today": None,
    "min_markup_pct": 25.0,
    "default_cover_days": 30,
    "lot_max_overage_months": 1.0,
    "z": 1.65,
    "lead_default_days": 123,
    "lead_min_days": 30,
    "lead_max_days": 210,
    "order_now_window_days": 30,
    "overdue_shift_days": 30,
    "overdue_stale_days": 90,
    "excess_extra_months": 6,
    "horizon_months": 12,
    "cover_horizon_months": 36,
    "history_months": 12,
    "dead_window_months": 12,
    "short_history_months": 12,
    "trend_window_months": 6,
    "trend_damping": 0.5,
    "trend_min": 0.8,
    "trend_max": 1.25,
    "min_season_index": 0.15,
    "recent_level_months": 3,
    "recent_drop_ratio": 0.5,
    "safety_cap_months": 2.0,
    "safety_alt_days": 60,
    "reliability_skip_zero_closed": True,
    "lot_min_receipts": 3,
    "days_per_month": 30.4,
    "abc_months": 12,
    "abc_a_share": 0.80,
    "abc_b_share": 0.95,
    "llm_batch": 25,
    "llm_timeout_s": 60,
    "telegram_timeout_s": 20,
    "telegram_max_chars": 4000,
    "long_gap_months": 3.0,
    "unreliable_below": 0.8,
    "backtest_months": 12,
    "backtest_horizons": (4, 6, 7),
}


def test_defaults_match_spec():
    assert dataclasses.asdict(Config()) == DEFAULTS


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.z = 2.0


def test_with_overrides_changes_only_given_fields():
    base = Config()
    changed = base.with_overrides(min_markup_pct=30, default_cover_days=60, z=1.28)
    assert (changed.min_markup_pct, changed.default_cover_days, changed.z) == (30, 60, 1.28)
    # остальные поля не тронуты
    assert dataclasses.asdict(changed) | {"min_markup_pct": 25.0, "default_cover_days": 30, "z": 1.65} == DEFAULTS
    assert dataclasses.asdict(base) == DEFAULTS


def test_with_overrides_skips_none():
    base = Config()
    assert base.with_overrides(min_markup_pct=None, z=None, default_cover_days=None) == base
    assert base.with_overrides(today=pd.Timestamp("2026-08-13")).today == pd.Timestamp("2026-08-13")


def test_with_overrides_rejects_unknown_field():
    with pytest.raises(TypeError):
        Config().with_overrides(unknown_threshold=1)


def test_status_constants():
    assert DEAD == "МЁРТВЫЙ"
    assert LOW_MARKUP == "НИЗКАЯ НАЦЕНКА"
    assert NO_MARKUP == "НЕТ НАЦЕНКИ"
    assert SHORTAGE == "РИСК ДЕФИЦИТА"
    assert ORDER_NOW == "ЗАКАЗАТЬ СЕЙЧАС"
    assert EXCESS == "ИЗБЫТОК"
    assert PLAN == "ЗАПЛАНИРОВАТЬ"
    assert OK == "ЗАПАС В НОРМЕ"
    assert SHORT_HISTORY == "МАЛО ИСТОРИИ"
    assert OVERDUE_PO == "ПРОСРОЧЕН ЗАКАЗ"
    assert STATUS_ORDER == [DEAD, LOW_MARKUP, NO_MARKUP, SHORTAGE, ORDER_NOW, EXCESS, PLAN, OK]
