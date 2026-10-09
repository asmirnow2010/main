import pytest

from bot.risk import LONG, SHORT, RiskConfig, build_position, position_size


def test_position_size_risk_based():
    cfg = RiskConfig(risk_per_trade=0.01, max_position_pct=1.0, stop_loss_atr=2.0)
    # equity 1000, риск 10 USDT, стоп 2*50=100 -> 0.1 BTC (notional 1000*... ограничен free)
    qty = position_size(1000, 10_000, 1000, 50, cfg)
    assert qty == pytest.approx(0.1)


def test_position_size_capped():
    cfg = RiskConfig(risk_per_trade=0.05, max_position_pct=0.2, stop_loss_atr=1.0)
    qty = position_size(1000, 1000, 100, 0.1, cfg, leverage=1)
    assert qty * 100 == pytest.approx(200)  # 20% капитала
    qty = position_size(1000, 1000, 100, 0.1, cfg, leverage=3)
    assert qty * 100 == pytest.approx(600)
    assert position_size(1000, 0, 100, 1, cfg) == 0


def test_stops_long_and_short():
    cfg = RiskConfig(stop_loss_atr=2, take_profit_atr=3)
    p = build_position("X", LONG, 1, 100, 1, cfg, 0)
    assert p.stop_loss == 98 and p.take_profit == 103
    assert p.check_stops(100, 101, 99) is None
    assert p.check_stops(100, 104, 97) == ("stop_loss", 98)       # оба уровня — считаем стоп
    assert p.check_stops(97, 97.5, 96) == ("stop_loss", 97)       # гэп вниз — по открытию
    assert p.check_stops(100, 103.5, 99) == ("take_profit", 103)
    s = build_position("X", SHORT, 1, 100, 1, cfg, 0)
    assert s.stop_loss == 102 and s.take_profit == 97
    assert s.check_stops(100, 102.5, 99) == ("stop_loss", 102)
    assert s.check_stops(100, 101, 96.5) == ("take_profit", 97)
    assert s.unrealized_pnl(90) == 10


def test_trailing_only_tightens():
    cfg = RiskConfig(stop_loss_atr=2, take_profit_atr=None, trailing_stop_atr=1.5)
    p = build_position("X", LONG, 1, 100, 1, cfg, 0)
    p.update_trailing(105, 104)
    assert p.stop_loss == 103.5
    p.update_trailing(104, 100)
    assert p.stop_loss == 103.5


def test_risk_validation():
    with pytest.raises(ValueError):
        RiskConfig.from_dict({"risk_per_trade": 0.5})
    with pytest.raises(ValueError):
        RiskConfig.from_dict({"stop_loss_atr": 0})
    with pytest.raises(ValueError):
        RiskConfig.from_dict({"unknown": 1})
