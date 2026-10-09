"""Стратегия «возврат к среднему» (Bollinger Bands + RSI + фильтр ADX).

Идея: во флэте цена, вышедшая за границу полос Боллинджера при перепроданности
(перекупленности) по RSI, с высокой вероятностью вернётся к средней линии.

Long:  close < нижней полосы  И  RSI < rsi_oversold  И  ADX < max_adx (нет сильного тренда)
Выход: close >= средней линии  ИЛИ  RSI > rsi_exit_long
Short — зеркально (только для фьючерсов).
"""
from __future__ import annotations

import pandas as pd

from .. import indicators as ta
from .base import Strategy


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
