"""Точка входа: python -m bot <команда>

Команды:
  run        — запустить торговлю (paper или live — по config.yaml)
  backtest   — проверить стратегию на истории
  check      — проверить конфиг, подключение к бирже и API-ключи
  status     — показать открытые позиции и последние сделки
  close-all  — закрыть все позиции бота по рынку
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import ccxt
import pandas as pd

from .config import Config, ConfigError, load_config, validate_for_live
from .exchange import ExchangeClient, create_broker
from .notifier import Notifier
from .state import StateStore
from .strategies import create_strategy

log = logging.getLogger("bot")


def setup_logging(log_file: Path | None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build_trader(cfg: Config):
    from .trader import Trader

    strategy = create_strategy(cfg.strategy_name, cfg.strategy_params)
    broker = create_broker(cfg)
    store = StateStore(cfg.state_file, cfg.trades_file)
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id, cfg.telegram_enabled,
                        prefix="[PAPER] " if cfg.mode == "paper" else "")
    return Trader(cfg, broker, strategy, store, notifier)


def cmd_run(cfg: Config, args) -> int:
    if cfg.mode == "live":
        validate_for_live(cfg)
        log.warning("ВНИМАНИЕ: режим LIVE — бот торгует реальными деньгами!")
    build_trader(cfg).run_forever()
    return 0


def cmd_close_all(cfg: Config, args) -> int:
    if cfg.mode == "live":
        validate_for_live(cfg)
    if not args.yes and input("Закрыть ВСЕ позиции бота по рынку? [y/N] ").strip().lower() != "y":
        print("Отменено")
        return 1
    build_trader(cfg).close_all()
    return 0


def cmd_status(cfg: Config, args) -> int:
    state = StateStore(cfg.state_file, cfg.trades_file).load()
    print(f"Режим: {cfg.mode}, рынок: {cfg.market_type}, стратегия: {cfg.strategy_name}")
    if state.broker.get("cash") is not None:
        print(f"Виртуальный баланс (без открытых позиций): {state.broker['cash']:.2f} {cfg.quote_currency}")
    if state.positions:
        print("\nОткрытые позиции:")
        for p in state.positions.values():
            tp = f"{p.take_profit:.6g}" if p.take_profit else "—"
            print(f"  {p.symbol:<16} {p.side:<5} qty={p.qty:.8g} вход={p.entry_price:.6g} SL={p.stop_loss:.6g} TP={tp}")
    else:
        print("\nОткрытых позиций нет")
    if cfg.trades_file.exists():
        trades = pd.read_csv(cfg.trades_file)
        print(f"\nСделок всего: {len(trades)}, суммарный PnL: {trades['pnl'].sum():+.2f} {cfg.quote_currency}")
        if len(trades):
            print(f"Прибыльных: {(trades['pnl'] > 0).mean() * 100:.1f}%")
            print("\nПоследние сделки:")
            print(trades.tail(args.last).to_string(index=False))
    return 0


def cmd_check(cfg: Config, args) -> int:
    print(f"Конфигурация OK: режим={cfg.mode}, рынок={cfg.market_type}, пары={cfg.symbols}, "
          f"стратегия={cfg.strategy_name}, ТФ={cfg.timeframe}")
    strategy = create_strategy(cfg.strategy_name, cfg.strategy_params)
    print(f"Стратегия создана, параметры: {strategy.params}")
    client = ExchangeClient(cfg, authenticated=bool(cfg.api_key))
    client.load_markets()
    for symbol in cfg.symbols:
        m = client.market(symbol)
        price = client.fetch_price(symbol)
        limits = m.get("limits") or {}
        print(f"  {symbol}: цена {price}, мин. объём {(limits.get('amount') or {}).get('min')}, "
              f"мин. сумма {(limits.get('cost') or {}).get('min')}")
    if cfg.api_key:
        bal = client.ex.fetch_balance()
        q = bal.get(cfg.quote_currency) or {}
        print(f"API-ключи работают. Баланс {cfg.quote_currency} ({cfg.market_type}): "
              f"свободно {q.get('free')}, всего {q.get('total')}")
    else:
        print("API-ключи не заданы (для paper и backtest они не нужны)")
    if cfg.telegram_enabled:
        ok = Notifier(cfg.telegram_token, cfg.telegram_chat_id, True).send("✅ Проверка связи с ботом")
        print("Telegram: сообщение отправлено" if ok else "Telegram: ОШИБКА отправки (см. лог)")
    return 0


def cmd_backtest(cfg: Config, args) -> int:
    from .backtest import format_report, run_backtest, save_results

    strategy = create_strategy(args.strategy or cfg.strategy_name, cfg.strategy_params if not args.strategy else {})
    symbols = [args.symbol.upper()] if args.symbol else cfg.symbols
    timeframe = args.timeframe or cfg.timeframe
    out_dir = Path(args.out)
    summary = []
    for symbol in symbols:
        if args.csv:
            df = pd.read_csv(args.csv)
            missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(df.columns)
            if missing:
                print(f"В CSV нет колонок: {missing}")
                return 1
            if not pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 1_000_000
        else:
            client = ExchangeClient(cfg)
            until = datetime.now(timezone.utc)
            since = until - timedelta(days=args.days)
            log.info("Загрузка истории %s %s за %d дней...", symbol, timeframe, args.days)
            df = client.fetch_history(symbol, timeframe, int(since.timestamp() * 1000), int(until.timestamp() * 1000))
            if args.save_data:
                Path("data").mkdir(exist_ok=True)
                df.to_csv(Path("data") / f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv", index=False)
        if len(df) < strategy.warmup + 50:
            print(f"{symbol}: слишком мало данных ({len(df)} свечей), нужно минимум {strategy.warmup + 50}")
            continue
        result = run_backtest(
            df, strategy, cfg.risk, symbol=symbol, initial_balance=args.balance or cfg.paper_initial_balance,
            fee=cfg.taker_fee, slippage=cfg.slippage, leverage=cfg.leverage, allow_short=cfg.risk.allow_short,
        )
        print(format_report(result))
        paths = save_results(result, out_dir)
        print("  Файлы: " + ", ".join(str(p) for p in paths) + "\n")
        summary.append({"symbol": symbol, **result.metrics})
        if args.csv:
            break
    if len(summary) > 1:
        print("===== Сводка =====")
        print(pd.DataFrame(summary)[["symbol", "total_return_pct", "buy_and_hold_pct", "max_drawdown_pct",
                                     "trades", "win_rate_pct", "profit_factor"]].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bot", description="Торговый бот для BingX")
    parser.add_argument("-c", "--config", default="config.yaml", help="путь к config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="запустить торговлю")
    sub.add_parser("check", help="проверить конфиг, биржу и ключи")
    p_status = sub.add_parser("status", help="открытые позиции и сделки")
    p_status.add_argument("--last", type=int, default=10, help="сколько последних сделок показать")
    p_close = sub.add_parser("close-all", help="закрыть все позиции бота")
    p_close.add_argument("-y", "--yes", action="store_true", help="без подтверждения")

    p_bt = sub.add_parser("backtest", help="бэктест на истории")
    p_bt.add_argument("--symbol", help="пара (по умолчанию — все из конфига)")
    p_bt.add_argument("--timeframe", help="таймфрейм (по умолчанию — из конфига)")
    p_bt.add_argument("--days", type=int, default=180, help="глубина истории в днях")
    p_bt.add_argument("--strategy", help="стратегия (по умолчанию — из конфига, с параметрами по умолчанию)")
    p_bt.add_argument("--balance", type=float, help="стартовый баланс")
    p_bt.add_argument("--csv", help="взять свечи из CSV (timestamp,open,high,low,close,volume)")
    p_bt.add_argument("--save-data", action="store_true", help="сохранить загруженные свечи в data/")
    p_bt.add_argument("--out", default="backtest_results", help="папка для результатов")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2
    setup_logging(cfg.log_file if args.command == "run" else None, args.verbose)

    commands = {"run": cmd_run, "check": cmd_check, "status": cmd_status,
                "close-all": cmd_close_all, "backtest": cmd_backtest}
    try:
        return commands[args.command](cfg, args)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2
    except ccxt.AuthenticationError as exc:
        print(f"Ошибка авторизации на бирже: {exc}\nПроверьте BINGX_API_KEY / BINGX_API_SECRET, "
              "права ключа и белый список IP.")
        return 3
    except ccxt.NetworkError as exc:
        print(f"Нет связи с биржей: {exc}\nПроверьте интернет / VPN / доступность open-api.bingx.com.")
        return 3
    except ccxt.BaseError as exc:
        print(f"Ошибка биржи: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
