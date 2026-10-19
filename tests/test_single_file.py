"""Проверка однофайловой версии trading_bot.py."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

import trading_bot as tb

from .conftest import make_candles
from .test_trader import FakeClient


def test_default_settings_valid():
    cfg = tb.load_config(env_file=None)
    assert cfg.mode == "paper" and cfg.strategy_name == "voting"


@pytest.mark.parametrize("name", ["mean_reversion", "voting"])
def test_backtest(name):
    res = tb.run_backtest(make_candles(3000, seed=5), tb.create_strategy(name), tb.RiskConfig(),
                          symbol="BTC/USDT", initial_balance=1000)
    assert res.metrics["trades"] > 0
    assert res.metrics["final_balance"] == pytest.approx(1000 + res.trades["pnl"].sum(), abs=0.01)


def test_trader_cycle(raw_config):
    cfg = tb.build_config(raw_config)
    client = FakeClient(make_candles(1200, seed=3))
    broker = tb.PaperBroker(client, cfg)
    trader = tb.Trader(cfg, broker, tb.create_strategy("mean_reversion"),
                       tb.StateStore(cfg.state_file, cfg.trades_file), tb.Notifier(),
                       clock=lambda: client.now_ms / 1000)
    trader.start()
    for _ in range(600):
        trader.tick()
        client.advance()
    trades = pd.read_csv(cfg.trades_file)
    assert len(trades) > 3
    open_fees = sum(p.entry_fee for p in trader.state.positions.values())
    assert broker.cash == pytest.approx(1000 + trades["pnl"].sum() - open_fees, abs=1e-3)


def test_live_swap_orders(raw_config):
    raw = {**raw_config, "mode": "live"}
    cfg = tb.build_config(raw)
    client = tb.ExchangeClient.__new__(tb.ExchangeClient)
    client.cfg, client._markets_loaded = cfg, True
    ex = MagicMock()
    ex.markets = {"BTC/USDT:USDT": {"base": "BTC", "limits": {}}}
    ex.fetch_position_mode.return_value = {"hedged": False}
    ex.create_order.return_value = {"id": "1", "average": 100.0, "filled": 0.01, "fee": {"cost": 0.001, "currency": "USDT"}}
    client.ex = ex
    broker = tb.LiveBroker(client, cfg)
    broker.prepare()
    broker.open_position("BTC/USDT:USDT", "short", 0.01, 100.0, stop_loss=105.0)
    args = ex.create_order.call_args.args
    assert args[2] == "sell" and args[5]["stopLoss"]["triggerPrice"] == 105.0
    broker.close_position("BTC/USDT:USDT", "short", 0.01, 99.0)
    assert ex.create_order.call_args.args[5]["reduceOnly"] is True


def test_cli_backtest_csv(tmp_path, capsys):
    csv_path = tmp_path / "c.csv"
    make_candles(2000, seed=2).to_csv(csv_path, index=False)
    code = tb.main(["backtest", "--csv", str(csv_path), "--symbol", "BTC/USDT", "--out", str(tmp_path / "out")])
    assert code == 0
    assert "Доходность" in capsys.readouterr().out
