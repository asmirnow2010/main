"""Проверка параметров реальных ордеров (ccxt замокан, сеть не используется)."""
from unittest.mock import MagicMock

import ccxt
import pytest

from bot.config import build_config
from bot.exchange import ExchangeClient, LiveBroker


def make_live(raw_config, market_type="swap", hedged=False):
    raw = dict(raw_config)
    raw["mode"] = "live"
    raw["exchange"] = {**raw["exchange"], "market_type": market_type}
    raw["symbols"] = ["BTC/USDT:USDT"] if market_type == "swap" else ["BTC/USDT"]
    cfg = build_config(raw)
    client = ExchangeClient.__new__(ExchangeClient)
    client.cfg = cfg
    client._markets_loaded = True
    ex = MagicMock()
    ex.markets = {raw["symbols"][0]: {"base": "BTC", "limits": {}}}
    ex.fetch_position_mode.return_value = {"hedged": hedged}
    ex.fetch_balance.return_value = {"USDT": {"free": 500, "total": 1000}, "BTC": {"free": 0.0098, "total": 0.0098}}
    ex.create_order.return_value = {"id": "1", "average": 100.0, "filled": 0.01,
                                    "fee": {"cost": 0.001, "currency": "USDT"}}
    ex.cost_to_precision.side_effect = lambda s, c: f"{c:.2f}"
    ex.amount_to_precision.side_effect = lambda s, a: f"{a:.4f}"
    client.ex = ex
    return LiveBroker(client, cfg), ex


def test_swap_one_way_open_close(raw_config):
    broker, ex = make_live(raw_config)
    broker.prepare()
    ex.set_leverage.assert_called_with(3, "BTC/USDT:USDT", {"side": "BOTH"})
    fill = broker.open_position("BTC/USDT:USDT", "long", 0.01, 100.0, stop_loss=95.0)
    args = ex.create_order.call_args.args
    assert args[:4] == ("BTC/USDT:USDT", "market", "buy", 0.01)
    assert args[5]["hedged"] is False and args[5]["stopLoss"]["triggerPrice"] == 95.0
    assert fill.price == 100.0 and fill.fee == 0.001
    broker.close_position("BTC/USDT:USDT", "long", 0.01, 101.0)
    args = ex.create_order.call_args.args
    assert args[2] == "sell" and args[5]["reduceOnly"] is True
    ex.cancel_all_orders.assert_called_once()


def test_swap_hedged_short(raw_config):
    broker, ex = make_live(raw_config, hedged=True)
    broker.prepare()
    sides = {c.args[2]["side"] for c in ex.set_leverage.call_args_list}
    assert sides == {"LONG", "SHORT"}
    broker.open_position("BTC/USDT:USDT", "short", 0.01, 100.0)
    assert ex.create_order.call_args.args[2] == "sell"
    assert ex.create_order.call_args.args[5]["hedged"] is True


def test_spot_buy_uses_cost_and_sell_uses_free_balance(raw_config):
    broker, ex = make_live(raw_config, market_type="spot")
    broker.prepare()
    ex.set_leverage.assert_not_called()
    broker.open_position("BTC/USDT", "long", 0.01, 100.0)
    args = ex.create_order.call_args.args
    assert args[2] == "buy" and args[4] is None and args[5] == {"cost": 1.0}
    broker.close_position("BTC/USDT", "long", 0.01, 100.0)
    args = ex.create_order.call_args.args
    # куплено 0.01, но комиссия списана в BTC -> продаём свободные 0.0098
    assert args[2] == "sell" and args[3] == pytest.approx(0.0098)
    with pytest.raises(ValueError):
        broker.open_position("BTC/USDT", "short", 0.01, 100.0)


def test_spot_sell_without_coins_raises(raw_config):
    broker, ex = make_live(raw_config, market_type="spot")
    ex.fetch_balance.return_value = {"USDT": {"free": 1, "total": 1}, "BTC": {"free": 0, "total": 0}}
    with pytest.raises(ccxt.InsufficientFunds):
        broker.close_position("BTC/USDT", "long", 0.01, 100.0)


def test_position_exists(raw_config):
    broker, ex = make_live(raw_config)
    ex.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "both"}]
    assert broker.position_exists("BTC/USDT:USDT", "long") is True
    ex.fetch_positions.return_value = []
    assert broker.position_exists("BTC/USDT:USDT", "long") is False
