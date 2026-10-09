import numpy as np
import pandas as pd

from bot import indicators as ta


def test_rsi_bounds_and_extremes(candles):
    r = ta.rsi(candles["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ta.rsi(up).iloc[-1] == 100.0
    flat = pd.Series(np.full(50, 10.0))
    assert ta.rsi(flat).iloc[-1] == 50.0


def test_rsi_matches_wilder_reference(candles):
    """Сравнение с классическим RSI Уайлдера (SMA-затравка + рекурсивное сглаживание)."""
    close = candles["close"].to_numpy()[:400]
    n = 14
    d = np.diff(close)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:n].mean(), l[:n].mean()
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
    expected = 100 - 100 / (1 + ag / al)
    assert abs(ta.rsi(pd.Series(close), n).iloc[-1] - expected) < 1e-6


def test_bollinger_order(candles):
    mid, up, low = ta.bollinger(candles["close"])
    ok = mid.notna()
    assert (up[ok] >= mid[ok]).all() and (mid[ok] >= low[ok]).all()


def test_atr_positive(candles):
    a = ta.atr(candles).dropna()
    assert (a > 0).all()


def test_adx_range(candles):
    a = ta.adx(candles).dropna()
    assert ((a >= 0) & (a <= 100)).all()


def test_ema_converges_to_constant():
    s = pd.Series(np.full(100, 5.0))
    assert abs(ta.ema(s, 10).iloc[-1] - 5.0) < 1e-12
