"""Комбинированная стратегия с голосованием индикаторов.

Каждый индикатор голосует +1 (за рост), -1 (за падение) или 0 (нейтрально).
Голоса умножаются на веса, итоговый score нормируется в диапазон [-1; 1]:

    trend      : EMA(fast) выше/ниже EMA(slow)
    long_trend : цена выше/ниже EMA(trend)
    macd       : гистограмма MACD > 0 / < 0
    rsi        : RSI > rsi_bull (+1), RSI < rsi_bear (-1)
    bollinger  : цена ниже нижней полосы (+1, перепроданность) / выше верхней (-1)
    volume     : OBV выше/ниже своей EMA (подтверждение объёмом)

Вход в long, когда score пересекает вверх entry_threshold; выход — когда score
опускается ниже exit_threshold. Short — зеркально.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from .base import Strategy

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
