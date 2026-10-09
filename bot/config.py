"""Загрузка и проверка конфигурации (config.yaml + переменные окружения из .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .risk import RiskConfig

VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    mode: str
    exchange_name: str
    market_type: str
    quote_currency: str
    symbols: list[str]
    timeframe: str
    candles_limit: int
    poll_interval_sec: int
    strategy_name: str
    strategy_params: dict[str, Any]
    risk: RiskConfig
    taker_fee: float
    slippage: float
    paper_initial_balance: float
    exchange_stop_loss: bool
    live_confirmed: bool
    telegram_enabled: bool
    telegram_notify_errors: bool
    state_file: Path
    trades_file: Path
    log_file: Path
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    telegram_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""

    @property
    def is_swap(self) -> bool:
        return self.market_type == "swap"

    @property
    def leverage(self) -> int:
        return int(self.risk.leverage) if self.is_swap else 1


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None and default is not None else cur


def load_config(path: str | Path = "config.yaml", env_file: str | Path | None = ".env") -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Файл конфигурации {path} не найден. Скопируйте config.example.yaml в {path}.")
    if env_file:
        load_dotenv(env_file)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return build_config(raw)


def build_config(raw: dict[str, Any]) -> Config:
    mode = str(_get(raw, "mode", "paper")).lower()
    if mode not in ("paper", "live"):
        raise ConfigError("mode должен быть 'paper' или 'live'")

    market_type = str(_get(raw, "exchange.market_type", "spot")).lower()
    if market_type not in ("spot", "swap"):
        raise ConfigError("exchange.market_type должен быть 'spot' или 'swap'")
    quote = str(_get(raw, "exchange.quote_currency", "USDT")).upper()

    symbols = [str(s).upper() for s in (_get(raw, "symbols") or [])]
    if not symbols:
        raise ConfigError("Укажите хотя бы одну торговую пару в symbols")
    for s in symbols:
        if "/" not in s:
            raise ConfigError(f"Неверный формат пары '{s}'. Пример: BTC/USDT (spot) или BTC/USDT:USDT (swap)")
        is_swap_symbol = ":" in s
        if market_type == "swap" and not is_swap_symbol:
            raise ConfigError(f"Для swap пары пишутся как BTC/USDT:USDT, а не '{s}'")
        if market_type == "spot" and is_swap_symbol:
            raise ConfigError(f"Для spot пары пишутся как BTC/USDT, а не '{s}'")
        if s.split("/")[1].split(":")[0] != quote:
            raise ConfigError(f"Пара {s} не котируется в {quote}")

    timeframe = str(_get(raw, "timeframe", "15m"))
    if timeframe not in VALID_TIMEFRAMES:
        raise ConfigError(f"Неизвестный таймфрейм {timeframe}. Доступны: {sorted(VALID_TIMEFRAMES)}")

    risk = RiskConfig.from_dict(_get(raw, "risk", {}))
    if market_type == "spot":
        risk.allow_short = False
        risk.leverage = 1

    cfg = Config(
        mode=mode,
        exchange_name=str(_get(raw, "exchange.name", "bingx")).lower(),
        market_type=market_type,
        quote_currency=quote,
        symbols=symbols,
        timeframe=timeframe,
        candles_limit=int(_get(raw, "candles_limit", 500)),
        poll_interval_sec=max(5, int(_get(raw, "poll_interval_sec", 30))),
        strategy_name=str(_get(raw, "strategy.name", "voting")),
        strategy_params=dict(_get(raw, "strategy.params", {}) or {}),
        risk=risk,
        taker_fee=float(_get(raw, "fees.taker", 0.001)),
        slippage=float(_get(raw, "fees.slippage", 0.0005)),
        paper_initial_balance=float(_get(raw, "paper.initial_balance", 1000)),
        exchange_stop_loss=bool(_get(raw, "live.exchange_stop_loss", True)),
        live_confirmed=bool(_get(raw, "live.i_understand_the_risks", False)),
        telegram_enabled=bool(_get(raw, "telegram.enabled", False)),
        telegram_notify_errors=bool(_get(raw, "telegram.notify_errors", True)),
        state_file=Path(_get(raw, "storage.state_file", "state/state.json")),
        trades_file=Path(_get(raw, "storage.trades_file", "state/trades.csv")),
        log_file=Path(_get(raw, "storage.log_file", "logs/bot.log")),
        api_key=os.getenv("BINGX_API_KEY", "").strip(),
        api_secret=os.getenv("BINGX_API_SECRET", "").strip(),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )

    if cfg.candles_limit < 100:
        raise ConfigError("candles_limit должен быть не меньше 100")
    if cfg.telegram_enabled and not (cfg.telegram_token and cfg.telegram_chat_id):
        raise ConfigError("telegram.enabled=true, но TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы в .env")
    return cfg


def validate_for_live(cfg: Config) -> None:
    if not (cfg.api_key and cfg.api_secret):
        raise ConfigError("Для режима live задайте BINGX_API_KEY и BINGX_API_SECRET в файле .env")
    if not cfg.live_confirmed:
        raise ConfigError(
            "Режим live заблокирован: установите live.i_understand_the_risks: true в config.yaml, "
            "предварительно проверив стратегию в бэктесте и в режиме paper."
        )
