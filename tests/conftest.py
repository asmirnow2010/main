import numpy as np
import pandas as pd
import pytest

from bot.config import build_config


def make_candles(n: int = 1500, seed: int = 1, start_price: float = 30000.0, tf_ms: int = 900_000) -> pd.DataFrame:
    """Синтетические свечи: чередование флэта и трендов."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0008, 0.0, 0.0, 0.0008], size=n // 100 + 1), 100)[:n]
    rets = drift + rng.normal(0, 0.004, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.uniform(10, 100, n)
    ts = 1_699_999_980_000 + np.arange(n) * tf_ms
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


@pytest.fixture
def candles():
    return make_candles()


@pytest.fixture
def raw_config(tmp_path):
    return {
        "mode": "paper",
        "exchange": {"name": "bingx", "market_type": "swap", "quote_currency": "USDT"},
        "symbols": ["BTC/USDT:USDT"],
        "timeframe": "15m",
        "candles_limit": 300,
        "poll_interval_sec": 5,
        "strategy": {"name": "voting", "params": {}},
        "risk": {"risk_per_trade": 0.01, "max_position_pct": 0.5, "leverage": 3, "allow_short": True,
                 "stop_loss_atr": 2.0, "take_profit_atr": 3.0, "cooldown_bars": 0},
        "paper": {"initial_balance": 1000},
        "storage": {"state_file": str(tmp_path / "state.json"), "trades_file": str(tmp_path / "trades.csv"),
                    "log_file": str(tmp_path / "bot.log")},
    }


@pytest.fixture
def cfg(raw_config):
    return build_config(raw_config)
