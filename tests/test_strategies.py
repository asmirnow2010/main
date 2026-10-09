import pytest

from bot.strategies import create_strategy


@pytest.mark.parametrize("name", ["mean_reversion", "voting"])
def test_strategy_produces_signals(name, candles):
    s = create_strategy(name)
    out = s.populate(candles)
    for col in ("enter_long", "exit_long", "enter_short", "exit_short", "atr"):
        assert col in out
    assert out["enter_long"].sum() > 0
    assert out["enter_short"].sum() > 0
    assert not out["enter_long"].iloc[: s.warmup].any()


@pytest.mark.parametrize("name", ["mean_reversion", "voting"])
def test_no_lookahead(name, candles):
    """Сигнал на свече i не должен меняться от добавления будущих свечей."""
    s = create_strategy(name)
    full = s.populate(candles)
    part = s.populate(candles.iloc[:900])
    cols = ["enter_long", "exit_long", "enter_short", "exit_short"]
    assert (full[cols].iloc[:900].values == part[cols].values).all()


def test_unknown_params_rejected():
    with pytest.raises(ValueError):
        create_strategy("voting", {"bogus": 1})
    with pytest.raises(ValueError):
        create_strategy("nope")
    with pytest.raises(ValueError):
        create_strategy("voting", {"entry_threshold": 0.2, "exit_threshold": 0.5})


def test_voting_partial_weights_merge():
    s = create_strategy("voting", {"weights": {"volume": 0}})
    assert s.params["weights"]["volume"] == 0
    assert s.params["weights"]["trend"] == 1.0
