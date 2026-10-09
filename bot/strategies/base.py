"""Базовый класс стратегии.

Стратегия получает DataFrame со свечами (timestamp, open, high, low, close, volume)
и добавляет к нему колонки:
    enter_long, exit_long, enter_short, exit_short  — bool-сигналы на закрытии свечи
    atr                                             — для расчёта стоп-лосса и размера позиции
Один и тот же метод populate() используется и в бэктесте, и в реальной торговле,
поэтому результаты бэктеста соответствуют поведению живого бота.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .. import indicators as ta


class Strategy(ABC):
    name: str = "base"
    default_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        unknown = set(params or {}) - set(self.default_params)
        if unknown:
            raise ValueError(f"Неизвестные параметры стратегии {self.name}: {sorted(unknown)}")
        self.params = {**self.default_params, **(params or {})}

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Сколько свечей нужно, чтобы индикаторы стали валидными."""

    @abstractmethod
    def _signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить колонки сигналов к df (df уже копия)."""

    def populate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["atr"] = ta.atr(out, self.params.get("atr_period", 14))
        out = self._signals(out)
        for col in ("enter_long", "exit_long", "enter_short", "exit_short"):
            out[col] = out[col].fillna(False).astype(bool)
        # До окончания разогрева индикаторов сигналов нет.
        out.loc[out.index[: self.warmup], ["enter_long", "enter_short"]] = False
        return out
