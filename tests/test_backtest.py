import pytest

from bot.backtest import format_report, run_backtest, save_results
from bot.risk import RiskConfig
from bot.strategies import create_strategy

from .conftest import make_candles


@pytest.mark.parametrize("name", ["mean_reversion", "voting"])
@pytest.mark.parametrize("allow_short", [False, True])
def test_backtest_runs_and_is_consistent(name, allow_short, candles, tmp_path):
    risk = RiskConfig(cooldown_bars=1, trailing_stop_atr=2.5)
    res = run_backtest(candles, create_strategy(name), risk, symbol="BTC/USDT", initial_balance=1000,
                       leverage=2 if allow_short else 1, allow_short=allow_short)
    m = res.metrics
    assert m["trades"] > 0
    # Итоговый капитал = старт + сумма PnL сделок (бухгалтерия сходится).
    assert m["final_balance"] == pytest.approx(1000 + res.trades["pnl"].sum(), abs=0.01)
    assert (res.trades["side"] == "short").any() == allow_short
    assert m["max_drawdown_pct"] <= 0
    assert "Доходность" in format_report(res)
    paths = save_results(res, tmp_path)
    assert all(p.exists() for p in paths)


def test_risk_per_trade_limits_stop_losses():
    """Убыток по стоп-лоссу не должен сильно превышать risk_per_trade (кроме гэпов и комиссий)."""
    df = make_candles(3000, seed=7)
    risk = RiskConfig(risk_per_trade=0.01, max_position_pct=1.0, take_profit_atr=None, cooldown_bars=0)
    res = run_backtest(df, create_strategy("mean_reversion"), risk, initial_balance=1000, fee=0.0, slippage=0.0)
    stops = res.trades[res.trades["reason"] == "stop_loss"]
    assert len(stops) > 0
    # риск 1% от капитала на момент входа, капитал в тесте не превышает ~1500
    assert stops["pnl"].min() > -1500 * 0.01 * 1.6


def test_entries_execute_on_next_open(candles):
    s = create_strategy("voting")
    data = s.populate(candles)
    res = run_backtest(candles, s, RiskConfig(), fee=0, slippage=0)
    first = res.trades.iloc[0]
    from datetime import datetime, timezone
    ts = int(datetime.strptime(first["entry_time"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
    idx = data.index[data["timestamp"] == ts][0]
    assert data["enter_long"].iloc[idx - 1] or data["enter_short"].iloc[idx - 1]
    assert first["entry_price"] == pytest.approx(data["open"].iloc[idx])
