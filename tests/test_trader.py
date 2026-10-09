"""Полный цикл Trader на фейковой бирже (paper-брокер, без сети)."""
import pandas as pd
import pytest

from bot.exchange import PaperBroker, timeframe_ms
from bot.notifier import Notifier
from bot.state import StateStore
from bot.strategies import create_strategy
from bot.trader import Trader

from .conftest import make_candles


class FakeExchange:
    def milliseconds(self):
        return self.now


class FakeClient:
    """Отдаёт свечи «по одной», имитируя течение времени."""

    def __init__(self, df: pd.DataFrame, tf: str = "15m"):
        self.df = df
        self.tf_ms = timeframe_ms(tf)
        self.i = 300
        self.ex = FakeExchange()

    def advance(self):
        self.i += 1

    @property
    def now_ms(self):
        return int(self.df["timestamp"].iloc[self.i]) + self.tf_ms

    def market(self, symbol):
        return {"base": "BTC", "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5}}}

    def load_markets(self):
        return {}

    def fetch_price(self, symbol):
        return float(self.df["close"].iloc[self.i])

    def fetch_closed_candles(self, symbol, timeframe, limit):
        return self.df.iloc[: self.i + 1].tail(limit).reset_index(drop=True)

    def round_qty(self, symbol, qty):
        return float(f"{qty:.4f}") if qty >= 0.0001 else 0.0

    def min_order_ok(self, symbol, qty, price):
        return qty >= 0.0001 and qty * price >= 5


def make_trader(cfg, client, strategy="mean_reversion"):
    broker = PaperBroker(client, cfg)
    store = StateStore(cfg.state_file, cfg.trades_file)
    trader = Trader(cfg, broker, create_strategy(strategy), store, Notifier(), clock=lambda: client.now_ms / 1000)
    return trader, broker, store


@pytest.mark.parametrize("strategy", ["mean_reversion", "voting"])
def test_trader_full_cycle_and_restart(cfg, strategy):
    client = FakeClient(make_candles(1200, seed=3))
    trader, broker, store = make_trader(cfg, client, strategy)
    trader.start()
    for _ in range(700):
        trader.tick()
        client.advance()

    trades = pd.read_csv(cfg.trades_file)
    assert len(trades) > 3
    assert set(trades["reason"]) <= {"stop_loss", "take_profit", "signal"}
    # Бухгалтерия: cash = старт + PnL закрытых сделок - комиссия входа открытой позиции.
    open_fees = sum(p.entry_fee for p in trader.state.positions.values())
    assert broker.cash == pytest.approx(1000 + trades["pnl"].sum() - open_fees, abs=1e-3)

    # Перезапуск: состояние восстанавливается из файла.
    saved_positions = set(trader.state.positions)
    trader2, broker2, _ = make_trader(cfg, client, strategy)
    trader2.start()
    assert set(trader2.state.positions) == saved_positions
    assert broker2.cash == pytest.approx(broker.cash)


def test_max_positions_and_daily_loss(cfg):
    client = FakeClient(make_candles(1200, seed=3))
    trader, broker, _ = make_trader(cfg, client)
    trader.start()
    trader.state.day_start_equity = 10_000  # имитируем большой дневной убыток
    for _ in range(300):
        trader.tick()
        trader.state.day_start_equity = 10_000
        client.advance()
    assert not trader.state.positions
    assert not cfg.trades_file.exists()


def test_close_all(cfg):
    client = FakeClient(make_candles(1200, seed=3))
    trader, broker, store = make_trader(cfg, client)
    trader.start()
    for _ in range(700):
        trader.tick()
        if trader.state.positions:
            break
        client.advance()
    assert trader.state.positions
    trader.close_all = Trader.close_all.__get__(trader)
    # close_all перечитывает состояние с диска и закрывает всё
    trader.broker.prepare = lambda: None
    trader.close_all()
    assert not trader.state.positions
    assert not store.load().positions
