#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  ТОРГОВЫЙ БОТ ДЛЯ BINGX — ВСЁ В ОДНОМ ФАЙЛЕ
=============================================================================
Спот и бессрочные USDT-фьючерсы. Две стратегии (возврат к среднему и
голосование индикаторов), риск-менеджмент (стоп-лосс/тейк-профит по ATR,
трейлинг, дневной лимит убытка), бэктест, paper trading, Telegram.

УСТАНОВКА:
    pip install ccxt pandas numpy requests python-dotenv pyyaml
    (необязательно для графиков бэктеста: pip install matplotlib)

НАСТРОЙКА:
    1. Отредактируйте блок SETTINGS ниже (или передайте свой YAML: -c config.yaml).
    2. Для реальной торговли создайте рядом файл .env:
           BINGX_API_KEY=ваш_ключ
           BINGX_API_SECRET=ваш_секрет
           TELEGRAM_BOT_TOKEN=...      (необязательно)
           TELEGRAM_CHAT_ID=...        (необязательно)
       Права ключа: чтение + торговля. ВЫВОД СРЕДСТВ НЕ ВКЛЮЧАТЬ.
       Привяжите ключ к IP вашего сервера.

КОМАНДЫ:
    python trading_bot.py backtest --days 180   # проверить стратегию на истории
    python trading_bot.py run                   # запустить (режим из SETTINGS["mode"])
    python trading_bot.py check                 # проверить биржу, ключи, Telegram
    python trading_bot.py status                # открытые позиции и сделки
    python trading_bot.py close-all             # закрыть все позиции бота

Порядок: бэктест -> paper (1-2 недели) -> live с небольшой суммой.
Торговля криптовалютой связана с высоким риском. Прибыль не гарантируется.
Подробная инструкция — в README.md.
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import time
import types
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

import ccxt
import numpy as np
import pandas as pd
import requests

try:
    import yaml
except ImportError:  # YAML нужен только при запуске с -c config.yaml
    yaml = None
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


# =============================================================================
# НАСТРОЙКИ — редактируйте здесь
# =============================================================================
SETTINGS: dict[str, Any] = {
    # "paper" — симуляция на живых ценах без реальных ордеров; "live" — реальная торговля
    "mode": "paper",
    "exchange": {
        "name": "bingx",
        "market_type": "spot",       # "spot" — спот; "swap" — бессрочные USDT-фьючерсы
        "quote_currency": "USDT",
    },
    # Спот: "BTC/USDT"; фьючерсы: "BTC/USDT:USDT"
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframe": "15m",              # 1m 5m 15m 30m 1h 4h 1d ...
    "candles_limit": 500,            # свечей для расчёта индикаторов
    "poll_interval_sec": 30,         # как часто проверять рынок и стопы
    "strategy": {
        "name": "voting",            # "voting" или "mean_reversion"
        "params": {},                # пусто — параметры по умолчанию (см. классы стратегий)
    },
    "risk": {
        "risk_per_trade": 0.01,      # риск на сделку — 1% капитала
        "max_position_pct": 0.25,    # не более 25% капитала (маржи) в одной позиции
        "max_open_positions": 2,
        "stop_loss_atr": 2.0,        # стоп-лосс = 2 x ATR
        "take_profit_atr": 3.0,      # тейк-профит = 3 x ATR (None — выключить)
        "trailing_stop_atr": None,   # трейлинг-стоп в ATR (например 2.5)
        "max_daily_loss_pct": 0.05,  # дневной убыток > 5% — новые сделки не открываются
        "cooldown_bars": 2,          # пауза в свечах после сделки по паре
        "leverage": 2,               # только swap
        "allow_short": True,         # только swap
    },
    "fees": {"taker": 0.001, "slippage": 0.0005},   # для paper и бэктеста
    "paper": {"initial_balance": 1000},
    "live": {
        "exchange_stop_loss": True,          # swap: дублировать стоп-лосс ордером на бирже
        "i_understand_the_risks": False,     # True — разрешить режим live
    },
    "telegram": {"enabled": False, "notify_errors": True},
    "storage": {
        "state_file": "state/state.json",
        "trades_file": "state/trades.csv",
        "log_file": "logs/bot.log",
    },
}

log = logging.getLogger("bot")


# ============================================================================
# ИНДИКАТОРЫ
# ============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder(-delta.clip(upper=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Нет убытков за период -> RSI = 100; нет движения вообще -> 50.
    out = out.where(loss != 0.0, np.where(gain > 0.0, 100.0, 50.0))
    return out.where(gain.notna())


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Возвращает (middle, upper, lower)."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(df), period)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Возвращает (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — сила тренда (0..100). <20-25 — флэт."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = atr(df, period)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return _wilder(dx.fillna(0.0), period)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


# Пространство имён индикаторов: ta.rsi(...), ta.ema(...) и т.д.
ta = types.SimpleNamespace(sma=sma, ema=ema, rsi=rsi, bollinger=bollinger, true_range=true_range,
                           atr=atr, macd=macd, adx=adx, obv=obv)


# ============================================================================
# РИСК-МЕНЕДЖМЕНТ
# ============================================================================

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


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    mode: str
    exchange_name: str
    market_type: str
    quote_currency: str
    symbols: list[str]
    timeframe: str
    candles_limit: int
    poll_interval_sec: int
    strategy_name: str
    strategy_params: dict[str, Any]
    risk: RiskConfig
    taker_fee: float
    slippage: float
    paper_initial_balance: float
    exchange_stop_loss: bool
    live_confirmed: bool
    telegram_enabled: bool
    telegram_notify_errors: bool
    state_file: Path
    trades_file: Path
    log_file: Path
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    telegram_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""

    @property
    def is_swap(self) -> bool:
        return self.market_type == "swap"

    @property
    def leverage(self) -> int:
        return int(self.risk.leverage) if self.is_swap else 1


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None and default is not None else cur


def load_config(path: str | Path | None = None, env_file: str | Path | None = ".env") -> Config:
    """Без path — настройки из блока SETTINGS в начале файла; с path — из YAML-файла."""
    if env_file:
        load_dotenv(env_file)
    if path is None:
        return build_config(SETTINGS)
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Файл конфигурации {path} не найден")
    if yaml is None:
        raise ConfigError("Для чтения YAML установите pyyaml: pip install pyyaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return build_config(raw)


def build_config(raw: dict[str, Any]) -> Config:
    mode = str(_get(raw, "mode", "paper")).lower()
    if mode not in ("paper", "live"):
        raise ConfigError("mode должен быть 'paper' или 'live'")

    market_type = str(_get(raw, "exchange.market_type", "spot")).lower()
    if market_type not in ("spot", "swap"):
        raise ConfigError("exchange.market_type должен быть 'spot' или 'swap'")
    quote = str(_get(raw, "exchange.quote_currency", "USDT")).upper()

    symbols = [str(s).upper() for s in (_get(raw, "symbols") or [])]
    if not symbols:
        raise ConfigError("Укажите хотя бы одну торговую пару в symbols")
    for s in symbols:
        if "/" not in s:
            raise ConfigError(f"Неверный формат пары '{s}'. Пример: BTC/USDT (spot) или BTC/USDT:USDT (swap)")
        is_swap_symbol = ":" in s
        if market_type == "swap" and not is_swap_symbol:
            raise ConfigError(f"Для swap пары пишутся как BTC/USDT:USDT, а не '{s}'")
        if market_type == "spot" and is_swap_symbol:
            raise ConfigError(f"Для spot пары пишутся как BTC/USDT, а не '{s}'")
        if s.split("/")[1].split(":")[0] != quote:
            raise ConfigError(f"Пара {s} не котируется в {quote}")

    timeframe = str(_get(raw, "timeframe", "15m"))
    if timeframe not in VALID_TIMEFRAMES:
        raise ConfigError(f"Неизвестный таймфрейм {timeframe}. Доступны: {sorted(VALID_TIMEFRAMES)}")

    risk = RiskConfig.from_dict(_get(raw, "risk", {}))
    if market_type == "spot":
        risk.allow_short = False
        risk.leverage = 1

    cfg = Config(
        mode=mode,
        exchange_name=str(_get(raw, "exchange.name", "bingx")).lower(),
        market_type=market_type,
        quote_currency=quote,
        symbols=symbols,
        timeframe=timeframe,
        candles_limit=int(_get(raw, "candles_limit", 500)),
        poll_interval_sec=max(5, int(_get(raw, "poll_interval_sec", 30))),
        strategy_name=str(_get(raw, "strategy.name", "voting")),
        strategy_params=dict(_get(raw, "strategy.params", {}) or {}),
        risk=risk,
        taker_fee=float(_get(raw, "fees.taker", 0.001)),
        slippage=float(_get(raw, "fees.slippage", 0.0005)),
        paper_initial_balance=float(_get(raw, "paper.initial_balance", 1000)),
        exchange_stop_loss=bool(_get(raw, "live.exchange_stop_loss", True)),
        live_confirmed=bool(_get(raw, "live.i_understand_the_risks", False)),
        telegram_enabled=bool(_get(raw, "telegram.enabled", False)),
        telegram_notify_errors=bool(_get(raw, "telegram.notify_errors", True)),
        state_file=Path(_get(raw, "storage.state_file", "state/state.json")),
        trades_file=Path(_get(raw, "storage.trades_file", "state/trades.csv")),
        log_file=Path(_get(raw, "storage.log_file", "logs/bot.log")),
        api_key=os.getenv("BINGX_API_KEY", "").strip(),
        api_secret=os.getenv("BINGX_API_SECRET", "").strip(),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )

    if cfg.candles_limit < 100:
        raise ConfigError("candles_limit должен быть не меньше 100")
    if cfg.telegram_enabled and not (cfg.telegram_token and cfg.telegram_chat_id):
        raise ConfigError("telegram.enabled=true, но TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы в .env")
    return cfg


def validate_for_live(cfg: Config) -> None:
    if not (cfg.api_key and cfg.api_secret):
        raise ConfigError("Для режима live задайте BINGX_API_KEY и BINGX_API_SECRET в файле .env")
    if not cfg.live_confirmed:
        raise ConfigError(
            "Режим live заблокирован: установите SETTINGS['live']['i_understand_the_risks'] = True, "
            "предварительно проверив стратегию в бэктесте и в режиме paper."
        )


# ============================================================================
# СТРАТЕГИИ: база
# ============================================================================

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


# ============================================================================
# СТРАТЕГИЯ: возврат к среднему
# ============================================================================

class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "rsi_exit_long": 60.0,
        "rsi_exit_short": 40.0,
        "adx_period": 14,
        "max_adx": 30.0,  # 0 или null — отключить фильтр
        "atr_period": 14,
    }

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["bb_period"], p["rsi_period"], p["adx_period"] * 2, p["atr_period"]) + 1

    def _signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        mid, upper, lower = ta.bollinger(df["close"], p["bb_period"], p["bb_std"])
        df["bb_mid"], df["bb_upper"], df["bb_lower"] = mid, upper, lower
        df["rsi"] = ta.rsi(df["close"], p["rsi_period"])
        df["adx"] = ta.adx(df, p["adx_period"])

        flat = df["adx"] < p["max_adx"] if p["max_adx"] else True
        close = df["close"]
        df["enter_long"] = (close < lower) & (df["rsi"] < p["rsi_oversold"]) & flat
        df["exit_long"] = (close >= mid) | (df["rsi"] > p["rsi_exit_long"])
        df["enter_short"] = (close > upper) & (df["rsi"] > p["rsi_overbought"]) & flat
        df["exit_short"] = (close <= mid) | (df["rsi"] < p["rsi_exit_short"])
        return df


# ============================================================================
# СТРАТЕГИЯ: голосование
# ============================================================================

VOTERS = ("trend", "long_trend", "macd", "rsi", "bollinger", "volume")


class VotingStrategy(Strategy):
    name = "voting"
    default_params = {
        "ema_fast": 12,
        "ema_slow": 26,
        "ema_trend": 200,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_bull": 55.0,
        "rsi_bear": 45.0,
        "bb_period": 20,
        "bb_std": 2.0,
        "obv_ema": 20,
        "atr_period": 14,
        "weights": {"trend": 1.0, "long_trend": 1.0, "macd": 1.0, "rsi": 1.0, "bollinger": 0.5, "volume": 0.5},
        "entry_threshold": 0.6,
        "exit_threshold": 0.0,
    }

    def __init__(self, params=None):
        super().__init__(params)
        weights = {**self.default_params["weights"], **(self.params.get("weights") or {})}
        unknown = set(weights) - set(VOTERS)
        if unknown:
            raise ValueError(f"Неизвестные голосующие индикаторы: {sorted(unknown)}")
        if sum(abs(w) for w in weights.values()) <= 0:
            raise ValueError("Сумма весов должна быть больше нуля")
        self.params["weights"] = weights
        if not 0 < self.params["entry_threshold"] <= 1:
            raise ValueError("entry_threshold должен быть в диапазоне (0; 1]")
        if self.params["exit_threshold"] >= self.params["entry_threshold"]:
            raise ValueError("exit_threshold должен быть меньше entry_threshold")

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["ema_trend"], p["ema_slow"], p["macd_slow"] + p["macd_signal"], p["bb_period"], p["obv_ema"]) + 1

    def _signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]

        def sign(cond_up, cond_down):
            return pd.Series(np.where(cond_up, 1.0, np.where(cond_down, -1.0, 0.0)), index=df.index)

        ema_fast, ema_slow = ta.ema(close, p["ema_fast"]), ta.ema(close, p["ema_slow"])
        ema_trend = ta.ema(close, p["ema_trend"])
        _, _, hist = ta.macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        rsi = ta.rsi(close, p["rsi_period"])
        _, upper, lower = ta.bollinger(close, p["bb_period"], p["bb_std"])
        obv = ta.obv(df)
        obv_ema = ta.ema(obv, p["obv_ema"])

        votes = {
            "trend": sign(ema_fast > ema_slow, ema_fast < ema_slow),
            "long_trend": sign(close > ema_trend, close < ema_trend),
            "macd": sign(hist > 0, hist < 0),
            "rsi": sign(rsi > p["rsi_bull"], rsi < p["rsi_bear"]),
            "bollinger": sign(close < lower, close > upper),
            "volume": sign(obv > obv_ema, obv < obv_ema),
        }
        w = p["weights"]
        total_w = sum(abs(v) for v in w.values())
        score = sum(votes[k] * w.get(k, 0.0) for k in VOTERS) / total_w
        df["score"] = score
        df["rsi"] = rsi

        prev = score.shift(1)
        th, ex = p["entry_threshold"], p["exit_threshold"]
        df["enter_long"] = (score >= th) & (prev < th)
        df["exit_long"] = score < ex
        df["enter_short"] = (score <= -th) & (prev > -th)
        df["exit_short"] = score > -ex
        return df


# ============================================================================
# РЕЕСТР СТРАТЕГИЙ
# ============================================================================

STRATEGIES: dict[str, type[Strategy]] = {
    MeanReversionStrategy.name: MeanReversionStrategy,
    VotingStrategy.name: VotingStrategy,
}


def create_strategy(name: str, params: dict | None = None) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступны: {', '.join(STRATEGIES)}") from None
    return cls(params)


# ============================================================================
# БИРЖА (ccxt): данные, live- и paper-брокеры
# ============================================================================

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class Fill:
    price: float
    qty: float
    fee: float
    order_id: str | None = None


def timeframe_ms(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def ohlcv_to_df(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in OHLCV_COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df


def retry(fn, *args, attempts: int = 3, delay: float = 2.0, **kwargs):
    """Повтор запроса при сетевых ошибках/лимитах биржи."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
            if i == attempts - 1:
                raise
            log.warning("Сетевая ошибка (%s), повтор через %.0f c", exc, delay * (i + 1))
            time.sleep(delay * (i + 1))


class ExchangeClient:
    """Тонкая обёртка над ccxt для BingX."""

    def __init__(self, cfg: Config, authenticated: bool = False):
        self.cfg = cfg
        if not hasattr(ccxt, cfg.exchange_name):
            raise ValueError(f"Биржа '{cfg.exchange_name}' не поддерживается ccxt")
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": cfg.market_type},
        }
        if authenticated:
            params["apiKey"] = cfg.api_key
            params["secret"] = cfg.api_secret
        self.ex: ccxt.Exchange = getattr(ccxt, cfg.exchange_name)(params)
        self._markets_loaded = False

    def load_markets(self) -> dict:
        if not self._markets_loaded:
            retry(self.ex.load_markets)
            self._markets_loaded = True
        return self.ex.markets

    def market(self, symbol: str) -> dict:
        self.load_markets()
        if symbol not in self.ex.markets:
            raise ValueError(f"Пара {symbol} не найдена на {self.cfg.exchange_name} ({self.cfg.market_type})")
        return self.ex.markets[symbol]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = retry(self.ex.fetch_ohlcv, symbol, timeframe, limit=limit)
        return ohlcv_to_df(rows)

    def fetch_closed_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Только закрытые свечи (последняя, ещё формирующаяся, отбрасывается)."""
        df = self.fetch_ohlcv(symbol, timeframe, limit + 1)
        now_ms = self.ex.milliseconds()
        tf = timeframe_ms(timeframe)
        return df[df["timestamp"] + tf <= now_ms].tail(limit).reset_index(drop=True)

    def fetch_history(self, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
        """Загрузка длинной истории постранично (для бэктеста)."""
        until_ms = until_ms or self.ex.milliseconds()
        tf = timeframe_ms(timeframe)
        rows: list[list[float]] = []
        cursor = since_ms
        while cursor < until_ms:
            batch = retry(self.ex.fetch_ohlcv, symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1][0]
            if last < cursor:
                break
            cursor = last + tf
            log.info("Загружено %d свечей %s (до %s)", len(rows), symbol, pd.to_datetime(last, unit="ms"))
        df = ohlcv_to_df(rows) if rows else pd.DataFrame(columns=OHLCV_COLUMNS)
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] + tf <= until_ms)]
        return df.reset_index(drop=True)

    def fetch_price(self, symbol: str) -> float:
        ticker = retry(self.ex.fetch_ticker, symbol)
        price = ticker.get("last") or ticker.get("close")
        if not price:
            raise ccxt.ExchangeError(f"Нет цены для {symbol}")
        return float(price)

    def round_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            return 0.0
        try:
            return float(self.ex.amount_to_precision(symbol, qty))
        except ccxt.InvalidOrder:  # меньше минимального шага
            return 0.0

    def min_order_ok(self, symbol: str, qty: float, price: float) -> bool:
        limits = self.market(symbol).get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        return qty > 0 and qty >= min_amount and qty * price >= min_cost


class Broker(ABC):
    """Единый интерфейс исполнения для paper и live."""

    def __init__(self, client: ExchangeClient, cfg: Config):
        self.client = client
        self.cfg = cfg

    def prepare(self) -> None:
        """Подготовка перед стартом (проверка пар, установка плеча и т.п.)."""
        for symbol in self.cfg.symbols:
            self.client.market(symbol)

    @abstractmethod
    def equity(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def free_balance(self) -> float: ...

    @abstractmethod
    def open_position(self, symbol: str, side: str, qty: float, price: float,
                      stop_loss: float | None = None) -> Fill: ...

    @abstractmethod
    def close_position(self, symbol: str, side: str, qty: float, price: float) -> Fill: ...

    def position_exists(self, symbol: str, side: str) -> bool | None:
        """Есть ли позиция на бирже. None — проверка не поддерживается."""
        return None

    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, data: dict[str, Any]) -> None:
        pass


class PaperBroker(Broker):
    """Виртуальное исполнение по рыночной цене с комиссией и проскальзыванием.

    Учёт: cash = стартовый баланс + реализованный PnL - комиссии;
    equity = cash + нереализованный PnL. Маржа = notional / leverage.
    """

    def __init__(self, client: ExchangeClient, cfg: Config):
        super().__init__(client, cfg)
        self.cash = cfg.paper_initial_balance
        self.margin: dict[str, float] = {}
        self.open_qty: dict[str, tuple[str, float, float]] = {}  # symbol -> (side, qty, entry)

    def equity(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for symbol, (side, qty, entry) in self.open_qty.items():
            price = prices.get(symbol, entry)
            upnl += (price - entry) * qty if side == LONG else (entry - price) * qty
        return self.cash + upnl

    def free_balance(self) -> float:
        return self.cash - sum(self.margin.values())

    def _fill_price(self, price: float, buy: bool) -> float:
        return price * (1 + self.cfg.slippage) if buy else price * (1 - self.cfg.slippage)

    def open_position(self, symbol, side, qty, price, stop_loss=None) -> Fill:
        fill = self._fill_price(price, buy=side == LONG)
        fee = fill * qty * self.cfg.taker_fee
        self.cash -= fee
        self.margin[symbol] = fill * qty / self.cfg.leverage
        self.open_qty[symbol] = (side, qty, fill)
        return Fill(fill, qty, fee, order_id=f"paper-{int(time.time() * 1000)}")

    def close_position(self, symbol, side, qty, price) -> Fill:
        fill = self._fill_price(price, buy=side != LONG)
        _, _, entry = self.open_qty.pop(symbol, (side, qty, fill))
        pnl = (fill - entry) * qty if side == LONG else (entry - fill) * qty
        fee = fill * qty * self.cfg.taker_fee
        self.cash += pnl - fee
        self.margin.pop(symbol, None)
        return Fill(fill, qty, fee, order_id=f"paper-{int(time.time() * 1000)}")

    def state(self) -> dict[str, Any]:
        return {"cash": self.cash, "margin": self.margin,
                "open_qty": {k: list(v) for k, v in self.open_qty.items()}}

    def load_state(self, data: dict[str, Any]) -> None:
        if not data:
            return
        self.cash = float(data.get("cash", self.cash))
        self.margin = {k: float(v) for k, v in (data.get("margin") or {}).items()}
        self.open_qty = {k: (v[0], float(v[1]), float(v[2])) for k, v in (data.get("open_qty") or {}).items()}


class LiveBroker(Broker):
    """Реальные рыночные ордера на BingX."""

    def __init__(self, client: ExchangeClient, cfg: Config):
        super().__init__(client, cfg)
        self.ex = client.ex
        self.hedged = False

    # ---------- подготовка ----------
    def prepare(self) -> None:
        super().prepare()
        retry(self.ex.fetch_balance)  # проверка ключей
        if not self.cfg.is_swap:
            return
        try:
            mode = retry(self.ex.fetch_position_mode, self.cfg.symbols[0])
            self.hedged = bool(mode.get("hedged"))
        except ccxt.BaseError as exc:
            log.warning("Не удалось определить режим позиций (%s), считаю one-way", exc)
        log.info("Режим позиций на фьючерсах: %s", "hedge" if self.hedged else "one-way")
        sides = ["LONG", "SHORT"] if self.hedged else ["BOTH"]
        for symbol in self.cfg.symbols:
            for s in sides:
                try:
                    retry(self.ex.set_leverage, self.cfg.leverage, symbol, {"side": s})
                except ccxt.BaseError as exc:
                    log.warning("Не удалось установить плечо %sx для %s (%s): %s", self.cfg.leverage, symbol, s, exc)
            log.info("Плечо %s: %dx", symbol, self.cfg.leverage)

    # ---------- баланс ----------
    def _quote_balance(self) -> dict[str, float]:
        bal = retry(self.ex.fetch_balance)
        q = bal.get(self.cfg.quote_currency) or {}
        return {"free": float(q.get("free") or 0), "total": float(q.get("total") or 0)}

    def free_balance(self) -> float:
        return self._quote_balance()["free"]

    def equity(self, prices: dict[str, float]) -> float:
        total = self._quote_balance()["total"]
        if self.cfg.is_swap:
            try:
                positions = retry(self.ex.fetch_positions, self.cfg.symbols)
                total += sum(float(p.get("unrealizedPnl") or 0) for p in positions)
            except ccxt.BaseError as exc:
                log.warning("Не удалось получить нереализованный PnL: %s", exc)
            return total
        # Спот: USDT + стоимость монет, которыми торгует бот.
        bal = retry(self.ex.fetch_balance)
        for symbol in self.cfg.symbols:
            base = self.client.market(symbol)["base"]
            amount = float((bal.get(base) or {}).get("total") or 0)
            if amount and symbol in prices:
                total += amount * prices[symbol]
        return total

    # ---------- ордера ----------
    def _finalize(self, order: dict, symbol: str, fallback_price: float, fallback_qty: float) -> Fill:
        order_id = order.get("id")
        if order_id and (not order.get("average") or not order.get("filled")):
            time.sleep(0.5)
            try:
                order = retry(self.ex.fetch_order, order_id, symbol)
            except ccxt.BaseError as exc:
                log.warning("Не удалось получить детали ордера %s: %s", order_id, exc)
        price = float(order.get("average") or order.get("price") or fallback_price)
        qty = float(order.get("filled") or fallback_qty)
        fee_info = order.get("fee") or {}
        fee = fee_info.get("cost")
        if fee is None or fee_info.get("currency") not in (None, self.cfg.quote_currency):
            fee = price * qty * self.cfg.taker_fee  # оценка, если комиссия не в USDT
        return Fill(price, qty, float(fee), order_id)

    def open_position(self, symbol, side, qty, price, stop_loss=None) -> Fill:
        buy = side == LONG
        if self.cfg.is_swap:
            params: dict[str, Any] = {"hedged": self.hedged}
            if stop_loss and self.cfg.exchange_stop_loss:
                params["stopLoss"] = {"triggerPrice": stop_loss, "type": "STOP_MARKET", "workingType": "MARK_PRICE"}
            order = self.ex.create_order(symbol, "market", "buy" if buy else "sell", qty, None, params)
        else:
            if not buy:
                raise ValueError("Шорт на споте невозможен")
            cost = float(self.ex.cost_to_precision(symbol, qty * price))
            order = self.ex.create_order(symbol, "market", "buy", qty, None, {"cost": cost})
        log.info("Ордер на открытие отправлен: %s %s %s qty=%s", symbol, side, order.get("id"), qty)
        return self._finalize(order, symbol, price, qty)

    def close_position(self, symbol, side, qty, price) -> Fill:
        close_side = "sell" if side == LONG else "buy"
        if self.cfg.is_swap:
            params = {"reduceOnly": True, "hedged": self.hedged}
            order = self.ex.create_order(symbol, "market", close_side, qty, None, params)
            try:  # снять оставшиеся условные ордера (биржевой стоп-лосс)
                self.ex.cancel_all_orders(symbol)
            except ccxt.BaseError as exc:
                log.warning("Не удалось отменить условные ордера %s: %s", symbol, exc)
        else:
            # На споте комиссия при покупке списывается в монете, поэтому продаём не больше свободного остатка.
            base = self.client.market(symbol)["base"]
            free = float((retry(self.ex.fetch_balance).get(base) or {}).get("free") or 0)
            qty = self.client.round_qty(symbol, min(qty, free))
            if qty <= 0:
                raise ccxt.InsufficientFunds(f"Нет свободных {base} для продажи")
            order = self.ex.create_order(symbol, "market", "sell", qty)
        log.info("Ордер на закрытие отправлен: %s %s %s qty=%s", symbol, side, order.get("id"), qty)
        return self._finalize(order, symbol, price, qty)

    def position_exists(self, symbol: str, side: str) -> bool | None:
        if not self.cfg.is_swap:
            return None
        positions = retry(self.ex.fetch_positions, [symbol])
        for p in positions:
            if p.get("symbol") != symbol or not float(p.get("contracts") or 0):
                continue
            if self.hedged and p.get("side") and p["side"] != side:
                continue
            return True
        return False


def create_broker(cfg: Config) -> Broker:
    if cfg.mode == "live":
        return LiveBroker(ExchangeClient(cfg, authenticated=True), cfg)
    return PaperBroker(ExchangeClient(cfg, authenticated=False), cfg)


# ============================================================================
# СОСТОЯНИЕ
# ============================================================================

TRADE_FIELDS = ["symbol", "side", "qty", "entry_time", "entry_price", "exit_time", "exit_price",
                "reason", "pnl", "pnl_pct", "fees"]


@dataclass
class BotState:
    positions: dict[str, Position] = field(default_factory=dict)
    last_candle_ts: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, int] = field(default_factory=dict)
    day: str = ""
    day_start_equity: float = 0.0
    broker: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {k: p.to_dict() for k, p in self.positions.items()},
            "last_candle_ts": self.last_candle_ts,
            "cooldown_until": self.cooldown_until,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "broker": self.broker,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BotState":
        return cls(
            positions={k: Position.from_dict(v) for k, v in (d.get("positions") or {}).items()},
            last_candle_ts={k: int(v) for k, v in (d.get("last_candle_ts") or {}).items()},
            cooldown_until={k: int(v) for k, v in (d.get("cooldown_until") or {}).items()},
            day=d.get("day", ""),
            day_start_equity=float(d.get("day_start_equity") or 0),
            broker=d.get("broker") or {},
        )


class StateStore:
    def __init__(self, state_file: Path, trades_file: Path):
        self.state_file = Path(state_file)
        self.trades_file = Path(trades_file)

    def load(self) -> BotState:
        if not self.state_file.exists():
            return BotState()
        return BotState.from_dict(json.loads(self.state_file.read_text(encoding="utf-8")))

    def save(self, state: BotState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_file)  # атомарная запись

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        new = not self.trades_file.exists()
        with self.trades_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(trade)


# ============================================================================
# TELEGRAM
# ============================================================================

class Notifier:
    def __init__(self, token: str = "", chat_id: str = "", enabled: bool = False, prefix: str = ""):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token and chat_id)
        self.prefix = prefix

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": f"{self.prefix}{text}"[:4000]},
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning("Telegram ответил %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except requests.RequestException as exc:
            # Ошибка уведомления никогда не должна останавливать торговлю.
            log.warning("Не удалось отправить сообщение в Telegram: %s", exc)
            return False


# ============================================================================
# ТОРГОВЫЙ ЦИКЛ
# ============================================================================

def fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class Trader:
    def __init__(self, cfg: Config, broker: Broker, strategy: Strategy, store: StateStore,
                 notifier: Notifier, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.broker = broker
        self.client = broker.client
        self.strategy = strategy
        self.store = store
        self.notifier = notifier
        self.clock = clock
        self.state = BotState()
        self.running = False
        self.tf_ms = timeframe_ms(cfg.timeframe)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.state = self.store.load()
        self.broker.load_state(self.state.broker)
        self.broker.prepare()
        if self.state.positions:
            log.info("Восстановлены открытые позиции: %s", ", ".join(self.state.positions))
        self._reconcile(self._fetch_prices())
        self._save()
        msg = (f"🤖 Бот запущен [{self.cfg.mode.upper()}] {self.cfg.exchange_name} {self.cfg.market_type}\n"
               f"Стратегия: {self.strategy.name}, ТФ {self.cfg.timeframe}\n"
               f"Пары: {', '.join(self.cfg.symbols)}")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def run_forever(self) -> None:
        self.start()
        self.running = True
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        errors_in_row = 0
        try:
            while self.running:
                try:
                    self.tick()
                    errors_in_row = 0
                except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
                    errors_in_row += 1
                    log.warning("Сетевая ошибка: %s", exc)
                    if errors_in_row == 5:
                        self._notify_error(f"5 сетевых ошибок подряд: {exc}")
                except Exception as exc:  # бот не должен падать из-за единичной ошибки
                    errors_in_row += 1
                    log.exception("Ошибка в торговом цикле")
                    self._notify_error(f"Ошибка: {type(exc).__name__}: {exc}")
                self._sleep(self.cfg.poll_interval_sec)
        except KeyboardInterrupt:
            pass
        finally:
            self._save()
            log.info("Бот остановлен. Открытые позиции сохранены и будут подхвачены при следующем запуске.")
            self.notifier.send("⏹ Бот остановлен")

    def stop(self) -> None:
        self.running = False

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    # ------------------------------------------------------------------ main step
    def tick(self) -> None:
        prices = self._fetch_prices()
        self._roll_day(prices)
        self._reconcile(prices)
        self._check_stops(prices)
        for symbol in self.cfg.symbols:
            self._process_symbol(symbol, prices)
        self._save()

    def _fetch_prices(self) -> dict[str, float]:
        return {s: self.client.fetch_price(s) for s in self.cfg.symbols}

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def _roll_day(self, prices: dict[str, float]) -> None:
        today = datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")
        if self.state.day != today:
            equity = self.broker.equity(prices)
            if self.state.day:
                change = equity - self.state.day_start_equity
                self.notifier.send(f"📅 Итог дня {self.state.day}: капитал {equity:.2f} "
                                   f"{self.cfg.quote_currency} ({change:+.2f})")
            self.state.day = today
            self.state.day_start_equity = equity
            log.info("Новый торговый день %s, капитал %.2f", today, equity)

    def _daily_loss_hit(self, equity: float) -> bool:
        start = self.state.day_start_equity
        return start > 0 and equity < start * (1 - self.cfg.risk.max_daily_loss_pct)

    def _reconcile(self, prices: dict[str, float]) -> None:
        """Позиция закрыта на бирже (сработал биржевой стоп / закрыли вручную) — убрать её из состояния."""
        for symbol, pos in list(self.state.positions.items()):
            try:
                exists = self.broker.position_exists(symbol, pos.side)
            except ccxt.BaseError as exc:
                log.warning("Не удалось сверить позицию %s: %s", symbol, exc)
                continue
            if exists is False:
                price = prices.get(symbol, pos.entry_price)
                stop_hit = price <= pos.stop_loss if pos.is_long else price >= pos.stop_loss
                exit_price = pos.stop_loss if stop_hit else price
                log.warning("Позиция %s не найдена на бирже — считаю закрытой (~%.6g)", symbol, exit_price)
                self._record_close(pos, exit_price, pos.qty * exit_price * self.cfg.taker_fee, "closed_on_exchange")

    def _check_stops(self, prices: dict[str, float]) -> None:
        for symbol, pos in list(self.state.positions.items()):
            price = prices[symbol]
            pos.update_trailing(price, price)
            hit = pos.check_stops(price, price, price)
            if hit:
                self._close(pos, price, hit[0])

    def _process_symbol(self, symbol: str, prices: dict[str, float]) -> None:
        candles = self.client.fetch_closed_candles(symbol, self.cfg.timeframe, self.cfg.candles_limit)
        if len(candles) < self.strategy.warmup + 2:
            log.warning("%s: мало свечей (%d < %d), пропуск", symbol, len(candles), self.strategy.warmup + 2)
            return
        last_ts = int(candles["timestamp"].iloc[-1])
        if self.state.last_candle_ts.get(symbol) == last_ts:
            return  # новая свеча ещё не закрылась
        self.state.last_candle_ts[symbol] = last_ts

        row = self.strategy.populate(candles).iloc[-1]
        extra = f" score={row['score']:+.2f}" if "score" in row else ""
        log.info("%s свеча %s close=%.6g rsi=%.1f%s", symbol, fmt_ts(last_ts), row["close"],
                 row.get("rsi", float("nan")), extra)

        pos = self.state.positions.get(symbol)
        if pos is not None:
            if (pos.is_long and row["exit_long"]) or (not pos.is_long and row["exit_short"]):
                self._close(pos, prices[symbol], "signal")
            return

        side = None
        if row["enter_long"]:
            side = LONG
        elif row["enter_short"] and self.cfg.risk.allow_short:
            side = SHORT
        if side:
            self._try_open(symbol, side, prices, float(row["atr"]))

    # ------------------------------------------------------------------ orders
    def _try_open(self, symbol: str, side: str, prices: dict[str, float], atr_value: float) -> None:
        risk = self.cfg.risk
        if len(self.state.positions) >= risk.max_open_positions:
            log.info("%s: сигнал %s пропущен — достигнут лимит позиций", symbol, side)
            return
        if self._now_ms() < self.state.cooldown_until.get(symbol, 0):
            log.info("%s: сигнал %s пропущен — пауза после сделки", symbol, side)
            return
        if not atr_value or atr_value != atr_value:
            return
        price = prices[symbol]
        equity = self.broker.equity(prices)
        if self._daily_loss_hit(equity):
            log.warning("Дневной лимит убытка достигнут — новые сделки не открываются до 00:00 UTC")
            return
        qty = position_size(equity, self.broker.free_balance(), price, atr_value, risk, self.cfg.leverage)
        qty = self.client.round_qty(symbol, qty)
        if not self.client.min_order_ok(symbol, qty, price):
            log.warning("%s: размер позиции %.8f ниже минимального ордера биржи — пропуск", symbol, qty)
            return

        sign = 1 if side == LONG else -1
        planned_sl = price - sign * atr_value * risk.stop_loss_atr
        try:
            fill = self.broker.open_position(symbol, side, qty, price, stop_loss=planned_sl)
        except ccxt.BaseError as exc:
            log.error("Не удалось открыть %s %s: %s", side, symbol, exc)
            self._notify_error(f"Не удалось открыть {side} {symbol}: {exc}")
            return
        pos = build_position(symbol, side, fill.qty, fill.price, atr_value, risk, self._now_ms(), fill.fee)
        self.state.positions[symbol] = pos
        self._save()
        tp = f"{pos.take_profit:.6g}" if pos.take_profit else "—"
        msg = (f"{'🟢 LONG' if side == LONG else '🔴 SHORT'} {symbol}\n"
               f"Цена: {fill.price:.6g}, объём: {fill.qty:.8g} (~{fill.price * fill.qty:.2f} {self.cfg.quote_currency})\n"
               f"SL: {pos.stop_loss:.6g}  TP: {tp}")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def _close(self, pos: Position, price: float, reason: str) -> None:
        try:
            fill = self.broker.close_position(pos.symbol, pos.side, pos.qty, price)
        except ccxt.InsufficientFunds as exc:
            log.error("%s: монет для продажи нет (%s) — позиция удалена из состояния", pos.symbol, exc)
            self._notify_error(f"{pos.symbol}: позиция не найдена на балансе, удалена из учёта")
            self.state.positions.pop(pos.symbol, None)
            return
        except ccxt.BaseError as exc:
            log.error("Не удалось закрыть %s: %s (повтор на следующем цикле)", pos.symbol, exc)
            self._notify_error(f"Не удалось закрыть {pos.symbol}: {exc}")
            return
        self._record_close(pos, fill.price, fill.fee, reason)

    def _record_close(self, pos: Position, exit_price: float, exit_fee: float, reason: str) -> None:
        gross = pos.unrealized_pnl(exit_price)
        fees = pos.entry_fee + exit_fee
        pnl = gross - fees
        notional = pos.entry_price * pos.qty
        now = self._now_ms()
        self.store.append_trade({
            "symbol": pos.symbol, "side": pos.side, "qty": pos.qty,
            "entry_time": fmt_ts(pos.opened_at), "entry_price": pos.entry_price,
            "exit_time": fmt_ts(now), "exit_price": exit_price, "reason": reason,
            "pnl": round(pnl, 6), "pnl_pct": round(pnl / notional * 100, 3) if notional else 0,
            "fees": round(fees, 6),
        })
        self.state.positions.pop(pos.symbol, None)
        self.state.cooldown_until[pos.symbol] = now + self.cfg.risk.cooldown_bars * self.tf_ms
        self._save()
        reasons = {"stop_loss": "стоп-лосс", "take_profit": "тейк-профит", "signal": "сигнал стратегии",
                   "closed_on_exchange": "закрыта на бирже", "manual": "вручную"}
        msg = (f"{'✅' if pnl >= 0 else '❌'} Закрыта {pos.side.upper()} {pos.symbol} ({reasons.get(reason, reason)})\n"
               f"Вход: {pos.entry_price:.6g} → выход: {exit_price:.6g}\n"
               f"PnL: {pnl:+.2f} {self.cfg.quote_currency} ({pnl / notional * 100 if notional else 0:+.2f}%)")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def close_all(self) -> None:
        """Закрыть все позиции бота по рынку (команда close-all)."""
        self.state = self.store.load()
        self.broker.load_state(self.state.broker)
        self.broker.prepare()
        prices = self._fetch_prices()
        for pos in list(self.state.positions.values()):
            self._close(pos, prices[pos.symbol], "manual")
        self._save()

    # ------------------------------------------------------------------ helpers
    def _save(self) -> None:
        self.state.broker = self.broker.state()
        self.store.save(self.state)

    def _notify_error(self, text: str) -> None:
        if self.cfg.telegram_notify_errors:
            self.notifier.send(f"⚠️ {text}")

    def status(self) -> pd.DataFrame:
        rows = [p.to_dict() for p in self.state.positions.values()]
        return pd.DataFrame(rows)


# ============================================================================
# БЭКТЕСТ
# ============================================================================

BARS_PER_YEAR_MS = 365 * 24 * 3600 * 1000


@dataclass
class BacktestResult:
    symbol: str
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskConfig, *, symbol: str = "",
                 initial_balance: float = 1000.0, fee: float = 0.001, slippage: float = 0.0005,
                 leverage: int = 1, allow_short: bool = False) -> BacktestResult:
    data = strategy.populate(df.reset_index(drop=True))
    ts = data["timestamp"].to_numpy(dtype="int64")
    o, h, l, c = (data[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = data["atr"].to_numpy(dtype=float)
    enter_long, exit_long = data["enter_long"].to_numpy(), data["exit_long"].to_numpy()
    enter_short, exit_short = data["enter_short"].to_numpy(), data["exit_short"].to_numpy()

    cash = initial_balance
    pos: Position | None = None
    pending: str | None = None      # LONG / SHORT / "close" — исполнить на открытии следующей свечи
    pending_atr = 0.0
    cooldown_until = -1
    day, day_start_equity = None, initial_balance
    trades: list[dict] = []
    equity_curve = np.empty(len(data))

    def close(i: int, raw_price: float, reason: str, apply_slip: bool = True) -> None:
        nonlocal cash, pos, cooldown_until
        assert pos is not None
        if apply_slip:
            raw_price *= (1 - slippage) if pos.is_long else (1 + slippage)
        exit_fee = raw_price * pos.qty * fee
        gross = pos.unrealized_pnl(raw_price)
        cash += gross - exit_fee
        pnl = gross - exit_fee - pos.entry_fee
        notional = pos.entry_price * pos.qty
        trades.append({
            "symbol": symbol, "side": pos.side, "qty": pos.qty,
            "entry_time": _ts(pos.opened_at), "entry_price": pos.entry_price,
            "exit_time": _ts(int(ts[i])), "exit_price": raw_price, "reason": reason,
            "pnl": pnl, "pnl_pct": pnl / notional * 100, "fees": pos.entry_fee + exit_fee,
            "bars": i - int(pos.meta.get("bar", i)),
        })
        pos = None
        cooldown_until = i + risk.cooldown_bars

    for i in range(len(data)):
        # Дневной лимит убытка (по UTC).
        cur_day = int(ts[i]) // 86_400_000
        if cur_day != day:
            day = cur_day
            day_start_equity = cash + (pos.unrealized_pnl(o[i]) if pos else 0.0)

        # 1. Исполнение отложенного сигнала по открытию свечи.
        if pending == "close" and pos is not None:
            close(i, o[i], "signal")
        elif pending in (LONG, SHORT) and pos is None:
            equity_now = cash
            halted = equity_now < day_start_equity * (1 - risk.max_daily_loss_pct)
            if not halted:
                price = o[i] * ((1 + slippage) if pending == LONG else (1 - slippage))
                qty = position_size(equity_now, cash, price, pending_atr, risk, leverage)
                if qty > 0:
                    entry_fee = price * qty * fee
                    cash -= entry_fee
                    pos = build_position(symbol, pending, qty, price, pending_atr, risk, int(ts[i]), entry_fee)
                    pos.meta["bar"] = i
        pending = None

        # 2. Стоп-лосс / тейк-профит внутри свечи.
        if pos is not None:
            hit = pos.check_stops(o[i], h[i], l[i])
            if hit:
                reason, price = hit
                close(i, price, reason, apply_slip=reason == "stop_loss")
            else:
                pos.update_trailing(h[i], l[i])

        # 3. Сигналы на закрытии свечи.
        if pos is not None:
            if (pos.is_long and exit_long[i]) or (not pos.is_long and exit_short[i]):
                pending = "close"
        elif i >= cooldown_until and i < len(data) - 1 and not math.isnan(atr[i]) and atr[i] > 0:
            if enter_long[i]:
                pending, pending_atr = LONG, atr[i]
            elif allow_short and enter_short[i]:
                pending, pending_atr = SHORT, atr[i]

        equity_curve[i] = cash + (pos.unrealized_pnl(c[i]) if pos else 0.0)

    if pos is not None:
        close(len(data) - 1, c[-1], "end_of_data")
        equity_curve[-1] = cash

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"timestamp": ts, "equity": equity_curve, "close": c})
    result = BacktestResult(symbol, trades_df, equity_df)
    result.metrics = compute_metrics(equity_df, trades_df, initial_balance, strategy.warmup)
    return result


def compute_metrics(equity_df: pd.DataFrame, trades: pd.DataFrame, initial: float, warmup: int = 0) -> dict:
    eq = equity_df["equity"]
    close = equity_df["close"]
    final = float(eq.iloc[-1]) if len(eq) else initial
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    start_idx = min(warmup, len(close) - 1) if len(close) else 0
    buy_hold = float((close.iloc[-1] / close.iloc[start_idx] - 1) * 100) if len(close) else 0.0

    rets = eq.pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 2 and rets.std() > 0:
        bar_ms = float(np.median(np.diff(equity_df["timestamp"].to_numpy())))
        sharpe = float(rets.mean() / rets.std() * math.sqrt(BARS_PER_YEAR_MS / bar_ms))

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_loss = float(-losses["pnl"].sum()) if n else 0.0
    profit_factor = float(wins["pnl"].sum() / gross_loss) if n and gross_loss > 0 else (float("inf") if n and len(wins) else 0.0)
    return {
        "initial_balance": initial,
        "final_balance": round(final, 2),
        "total_return_pct": round((final / initial - 1) * 100, 2),
        "buy_and_hold_pct": round(buy_hold, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "trades": n,
        "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0.0,
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else profit_factor,
        "avg_trade_pct": round(float(trades["pnl_pct"].mean()), 3) if n else 0.0,
        "fees_paid": round(float(trades["fees"].sum()), 2) if n else 0.0,
    }


METRIC_NAMES = {
    "initial_balance": "Начальный баланс",
    "final_balance": "Итоговый баланс",
    "total_return_pct": "Доходность, %",
    "buy_and_hold_pct": "Купить и держать, %",
    "max_drawdown_pct": "Макс. просадка, %",
    "sharpe": "Коэф. Шарпа (год.)",
    "trades": "Сделок",
    "win_rate_pct": "Прибыльных сделок, %",
    "profit_factor": "Profit factor",
    "avg_trade_pct": "Средняя сделка, %",
    "fees_paid": "Комиссии",
}


def format_report(result: BacktestResult) -> str:
    lines = [f"===== Бэктест {result.symbol} ====="]
    if len(result.equity):
        lines.append(f"Период: {_ts(int(result.equity['timestamp'].iloc[0]))} — "
                     f"{_ts(int(result.equity['timestamp'].iloc[-1]))} ({len(result.equity)} свечей)")
    for key, name in METRIC_NAMES.items():
        lines.append(f"  {name:<24} {result.metrics.get(key)}")
    if len(result.trades):
        by_reason = result.trades.groupby("reason")["pnl"].agg(["count", "sum"]).round(2)
        lines.append("  Закрытия по причинам:")
        for reason, r in by_reason.iterrows():
            lines.append(f"    {reason:<16} {int(r['count']):>4} сделок, PnL {r['sum']:+.2f}")
    return "\n".join(lines)


def save_results(result: BacktestResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result.symbol.replace("/", "_").replace(":", "_") or "backtest"
    paths = [out_dir / f"{name}_trades.csv", out_dir / f"{name}_equity.csv"]
    result.trades.to_csv(paths[0], index=False)
    result.equity.to_csv(paths[1], index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    times = pd.to_datetime(result.equity["timestamp"], unit="ms")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    ax1.plot(times, result.equity["close"], color="#555", lw=0.8)
    ax1.set_title(f"{result.symbol}: цена и сделки")
    if len(result.trades):
        t = result.trades
        colors = np.where(t["pnl"] > 0, "#2a9d4b", "#d1495b")
        ax1.scatter(pd.to_datetime(t["entry_time"]), t["entry_price"], marker="^", c="#1f6feb", s=25, label="вход")
        ax1.scatter(pd.to_datetime(t["exit_time"]), t["exit_price"], marker="v", c=colors, s=25, label="выход")
        ax1.legend()
    ax2.plot(times, result.equity["equity"], color="#1f6feb")
    ax2.set_title("Капитал")
    fig.tight_layout()
    png = out_dir / f"{name}_chart.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    paths.append(png)
    return paths


# ============================================================================
# КОМАНДЫ (CLI)
# ============================================================================

def setup_logging(log_file: Path | None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build_trader(cfg: Config):

    strategy = create_strategy(cfg.strategy_name, cfg.strategy_params)
    broker = create_broker(cfg)
    store = StateStore(cfg.state_file, cfg.trades_file)
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id, cfg.telegram_enabled,
                        prefix="[PAPER] " if cfg.mode == "paper" else "")
    return Trader(cfg, broker, strategy, store, notifier)


def cmd_run(cfg: Config, args) -> int:
    if cfg.mode == "live":
        validate_for_live(cfg)
        log.warning("ВНИМАНИЕ: режим LIVE — бот торгует реальными деньгами!")
    build_trader(cfg).run_forever()
    return 0


def cmd_close_all(cfg: Config, args) -> int:
    if cfg.mode == "live":
        validate_for_live(cfg)
    if not args.yes and input("Закрыть ВСЕ позиции бота по рынку? [y/N] ").strip().lower() != "y":
        print("Отменено")
        return 1
    build_trader(cfg).close_all()
    return 0


def cmd_status(cfg: Config, args) -> int:
    state = StateStore(cfg.state_file, cfg.trades_file).load()
    print(f"Режим: {cfg.mode}, рынок: {cfg.market_type}, стратегия: {cfg.strategy_name}")
    if state.broker.get("cash") is not None:
        print(f"Виртуальный баланс (без открытых позиций): {state.broker['cash']:.2f} {cfg.quote_currency}")
    if state.positions:
        print("\nОткрытые позиции:")
        for p in state.positions.values():
            tp = f"{p.take_profit:.6g}" if p.take_profit else "—"
            print(f"  {p.symbol:<16} {p.side:<5} qty={p.qty:.8g} вход={p.entry_price:.6g} SL={p.stop_loss:.6g} TP={tp}")
    else:
        print("\nОткрытых позиций нет")
    if cfg.trades_file.exists():
        trades = pd.read_csv(cfg.trades_file)
        print(f"\nСделок всего: {len(trades)}, суммарный PnL: {trades['pnl'].sum():+.2f} {cfg.quote_currency}")
        if len(trades):
            print(f"Прибыльных: {(trades['pnl'] > 0).mean() * 100:.1f}%")
            print("\nПоследние сделки:")
            print(trades.tail(args.last).to_string(index=False))
    return 0


def cmd_check(cfg: Config, args) -> int:
    print(f"Конфигурация OK: режим={cfg.mode}, рынок={cfg.market_type}, пары={cfg.symbols}, "
          f"стратегия={cfg.strategy_name}, ТФ={cfg.timeframe}")
    strategy = create_strategy(cfg.strategy_name, cfg.strategy_params)
    print(f"Стратегия создана, параметры: {strategy.params}")
    client = ExchangeClient(cfg, authenticated=bool(cfg.api_key))
    client.load_markets()
    for symbol in cfg.symbols:
        m = client.market(symbol)
        price = client.fetch_price(symbol)
        limits = m.get("limits") or {}
        print(f"  {symbol}: цена {price}, мин. объём {(limits.get('amount') or {}).get('min')}, "
              f"мин. сумма {(limits.get('cost') or {}).get('min')}")
    if cfg.api_key:
        bal = client.ex.fetch_balance()
        q = bal.get(cfg.quote_currency) or {}
        print(f"API-ключи работают. Баланс {cfg.quote_currency} ({cfg.market_type}): "
              f"свободно {q.get('free')}, всего {q.get('total')}")
    else:
        print("API-ключи не заданы (для paper и backtest они не нужны)")
    if cfg.telegram_enabled:
        ok = Notifier(cfg.telegram_token, cfg.telegram_chat_id, True).send("✅ Проверка связи с ботом")
        print("Telegram: сообщение отправлено" if ok else "Telegram: ОШИБКА отправки (см. лог)")
    return 0


def cmd_backtest(cfg: Config, args) -> int:

    strategy = create_strategy(args.strategy or cfg.strategy_name, cfg.strategy_params if not args.strategy else {})
    symbols = [args.symbol.upper()] if args.symbol else cfg.symbols
    timeframe = args.timeframe or cfg.timeframe
    out_dir = Path(args.out)
    summary = []
    for symbol in symbols:
        if args.csv:
            df = pd.read_csv(args.csv)
            missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(df.columns)
            if missing:
                print(f"В CSV нет колонок: {missing}")
                return 1
            if not pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 1_000_000
        else:
            client = ExchangeClient(cfg)
            until = datetime.now(timezone.utc)
            since = until - timedelta(days=args.days)
            log.info("Загрузка истории %s %s за %d дней...", symbol, timeframe, args.days)
            df = client.fetch_history(symbol, timeframe, int(since.timestamp() * 1000), int(until.timestamp() * 1000))
            if args.save_data:
                Path("data").mkdir(exist_ok=True)
                df.to_csv(Path("data") / f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv", index=False)
        if len(df) < strategy.warmup + 50:
            print(f"{symbol}: слишком мало данных ({len(df)} свечей), нужно минимум {strategy.warmup + 50}")
            continue
        result = run_backtest(
            df, strategy, cfg.risk, symbol=symbol, initial_balance=args.balance or cfg.paper_initial_balance,
            fee=cfg.taker_fee, slippage=cfg.slippage, leverage=cfg.leverage, allow_short=cfg.risk.allow_short,
        )
        print(format_report(result))
        paths = save_results(result, out_dir)
        print("  Файлы: " + ", ".join(str(p) for p in paths) + "\n")
        summary.append({"symbol": symbol, **result.metrics})
        if args.csv:
            break
    if len(summary) > 1:
        print("===== Сводка =====")
        print(pd.DataFrame(summary)[["symbol", "total_return_pct", "buy_and_hold_pct", "max_drawdown_pct",
                                     "trades", "win_rate_pct", "profit_factor"]].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python trading_bot.py", description="Торговый бот для BingX")
    parser.add_argument("-c", "--config", default=None,
                        help="YAML-файл настроек (по умолчанию — блок SETTINGS в этом файле)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="запустить торговлю")
    sub.add_parser("check", help="проверить конфиг, биржу и ключи")
    p_status = sub.add_parser("status", help="открытые позиции и сделки")
    p_status.add_argument("--last", type=int, default=10, help="сколько последних сделок показать")
    p_close = sub.add_parser("close-all", help="закрыть все позиции бота")
    p_close.add_argument("-y", "--yes", action="store_true", help="без подтверждения")

    p_bt = sub.add_parser("backtest", help="бэктест на истории")
    p_bt.add_argument("--symbol", help="пара (по умолчанию — все из конфига)")
    p_bt.add_argument("--timeframe", help="таймфрейм (по умолчанию — из конфига)")
    p_bt.add_argument("--days", type=int, default=180, help="глубина истории в днях")
    p_bt.add_argument("--strategy", help="стратегия (по умолчанию — из конфига, с параметрами по умолчанию)")
    p_bt.add_argument("--balance", type=float, help="стартовый баланс")
    p_bt.add_argument("--csv", help="взять свечи из CSV (timestamp,open,high,low,close,volume)")
    p_bt.add_argument("--save-data", action="store_true", help="сохранить загруженные свечи в data/")
    p_bt.add_argument("--out", default="backtest_results", help="папка для результатов")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2
    setup_logging(cfg.log_file if args.command == "run" else None, args.verbose)

    commands = {"run": cmd_run, "check": cmd_check, "status": cmd_status,
                "close-all": cmd_close_all, "backtest": cmd_backtest}
    try:
        return commands[args.command](cfg, args)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2
    except ccxt.AuthenticationError as exc:
        print(f"Ошибка авторизации на бирже: {exc}\nПроверьте BINGX_API_KEY / BINGX_API_SECRET, "
              "права ключа и белый список IP.")
        return 3
    except ccxt.NetworkError as exc:
        print(f"Нет связи с биржей: {exc}\nПроверьте интернет / VPN / доступность open-api.bingx.com.")
        return 3
    except ccxt.BaseError as exc:
        print(f"Ошибка биржи: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
