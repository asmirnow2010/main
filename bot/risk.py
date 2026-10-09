"""Риск-менеджмент: позиция, стоп-лосс/тейк-профит/трейлинг-стоп, расчёт размера.

Эта логика общая для бэктеста, paper trading и реальной торговли.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LONG = "long"
SHORT = "short"


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01       # доля капитала, которой рискуем до стоп-лосса
    max_position_pct: float = 0.25     # макс. доля капитала (маржи) в одной позиции
    max_open_positions: int = 3
    stop_loss_atr: float = 2.0         # стоп = entry -/+ ATR * k
    take_profit_atr: float | None = 3.0
    trailing_stop_atr: float | None = None
    max_daily_loss_pct: float = 0.05   # при дневном убытке больше — новые сделки не открываются
    cooldown_bars: int = 2             # пауза (в свечах) после закрытия сделки по символу
    leverage: int = 1                  # только для фьючерсов
    allow_short: bool = False          # только для фьючерсов

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RiskConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Неизвестные параметры risk: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0 < self.risk_per_trade <= 0.1:
            raise ValueError("risk.risk_per_trade должен быть в диапазоне (0; 0.1]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("risk.max_position_pct должен быть в диапазоне (0; 1]")
        if self.stop_loss_atr <= 0:
            raise ValueError("risk.stop_loss_atr должен быть > 0 (бот всегда торгует со стоп-лоссом)")
        if not 1 <= int(self.leverage) <= 125:
            raise ValueError("risk.leverage должен быть от 1 до 125")
        if self.max_open_positions < 1:
            raise ValueError("risk.max_open_positions должен быть >= 1")


@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    opened_at: int                       # timestamp (ms)
    entry_fee: float = 0.0
    trailing_atr_dist: float | None = None
    extreme_price: float = 0.0           # лучшая цена с момента входа (для трейлинга)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.extreme_price:
            self.extreme_price = self.entry_price

    @property
    def is_long(self) -> bool:
        return self.side == LONG

    def unrealized_pnl(self, price: float) -> float:
        diff = price - self.entry_price if self.is_long else self.entry_price - price
        return diff * self.qty

    def update_trailing(self, high: float, low: float) -> None:
        """Подтянуть стоп за ценой (только в сторону уменьшения риска)."""
        if not self.trailing_atr_dist:
            return
        if self.is_long:
            self.extreme_price = max(self.extreme_price, high)
            self.stop_loss = max(self.stop_loss, self.extreme_price - self.trailing_atr_dist)
        else:
            self.extreme_price = min(self.extreme_price, low)
            self.stop_loss = min(self.stop_loss, self.extreme_price + self.trailing_atr_dist)

    def check_stops(self, open_: float, high: float, low: float) -> tuple[str, float] | None:
        """Проверить срабатывание SL/TP внутри свечи. Возвращает (причина, цена исполнения).

        Если в одной свече задеты и SL, и TP — консервативно считаем, что первым сработал SL.
        При гэпе через уровень исполнение по цене открытия.
        """
        if self.is_long:
            if low <= self.stop_loss:
                return "stop_loss", min(open_, self.stop_loss)
            if self.take_profit is not None and high >= self.take_profit:
                return "take_profit", max(open_, self.take_profit)
        else:
            if high >= self.stop_loss:
                return "stop_loss", max(open_, self.stop_loss)
            if self.take_profit is not None and low <= self.take_profit:
                return "take_profit", min(open_, self.take_profit)
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        return cls(**data)


def build_position(symbol: str, side: str, qty: float, price: float, atr_value: float,
                   cfg: RiskConfig, ts: int, fee: float = 0.0) -> Position:
    sl_dist = atr_value * cfg.stop_loss_atr
    tp_dist = atr_value * cfg.take_profit_atr if cfg.take_profit_atr else None
    sign = 1 if side == LONG else -1
    return Position(
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=price,
        stop_loss=price - sign * sl_dist,
        take_profit=price + sign * tp_dist if tp_dist else None,
        opened_at=ts,
        entry_fee=fee,
        trailing_atr_dist=atr_value * cfg.trailing_stop_atr if cfg.trailing_stop_atr else None,
    )


def position_size(equity: float, free_balance: float, price: float, atr_value: float,
                  cfg: RiskConfig, leverage: int = 1) -> float:
    """Размер позиции (в базовой валюте) по фиксированному риску.

    qty = (equity * risk_per_trade) / (ATR * stop_loss_atr)
    ограничено: max_position_pct от капитала (с учётом плеча) и свободной маржой.
    """
    if price <= 0 or atr_value <= 0 or equity <= 0:
        return 0.0
    stop_dist = atr_value * cfg.stop_loss_atr
    qty = equity * cfg.risk_per_trade / stop_dist
    max_notional = equity * cfg.max_position_pct * leverage
    qty = min(qty, max_notional / price)
    # Запас 2% под комиссию и проскальзывание.
    qty = min(qty, max(free_balance, 0.0) * 0.98 * leverage / price)
    return max(qty, 0.0)
