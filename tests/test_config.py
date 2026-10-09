import pytest

from bot.config import ConfigError, build_config, validate_for_live


def test_spot_forces_no_short_no_leverage(raw_config):
    raw_config["exchange"]["market_type"] = "spot"
    raw_config["symbols"] = ["BTC/USDT"]
    cfg = build_config(raw_config)
    assert cfg.risk.allow_short is False and cfg.leverage == 1


def test_symbol_format_checked(raw_config):
    raw_config["symbols"] = ["BTC/USDT"]  # swap требует BTC/USDT:USDT
    with pytest.raises(ConfigError):
        build_config(raw_config)


def test_live_requires_keys_and_confirmation(cfg):
    with pytest.raises(ConfigError):
        validate_for_live(cfg)
    cfg.api_key, cfg.api_secret = "k", "s"
    with pytest.raises(ConfigError):
        validate_for_live(cfg)
    cfg.live_confirmed = True
    validate_for_live(cfg)


def test_example_config_is_valid(tmp_path):
    from pathlib import Path

    from bot.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "config.example.yaml", env_file=None)
    assert cfg.mode == "paper"
