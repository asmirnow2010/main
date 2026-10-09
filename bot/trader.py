"""Основной торговый цикл (paper и live)."""
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from typing import Callable

import ccxt
import pandas as pd

from .config import Config
from .exchange import Broker, timeframe_ms
from .notifier import Notifier
from .risk import LONG, SHORT, Position, build_position, position_size
from .state import BotState, StateStore
from .strategies import Strategy

log = logging.getLogger(__name__)


def fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class Trader:
    def __init__(self, cfg: Config, broker: Broker, strategy: Strategy, store: StateStore,
                 notifier: Notifier, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.broker = broker
        self.client = broker.client
        self.strategy = strategy
        self.store = store
        self.notifier = notifier
        self.clock = clock
        self.state = BotState()
        self.running = False
        self.tf_ms = timeframe_ms(cfg.timeframe)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.state = self.store.load()
        self.broker.load_state(self.state.broker)
        self.broker.prepare()
        if self.state.positions:
            log.info("Восстановлены открытые позиции: %s", ", ".join(self.state.positions))
        self._reconcile(self._fetch_prices())
        self._save()
        msg = (f"🤖 Бот запущен [{self.cfg.mode.upper()}] {self.cfg.exchange_name} {self.cfg.market_type}\n"
               f"Стратегия: {self.strategy.name}, ТФ {self.cfg.timeframe}\n"
               f"Пары: {', '.join(self.cfg.symbols)}")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def run_forever(self) -> None:
        self.start()
        self.running = True
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        errors_in_row = 0
        try:
            while self.running:
                try:
                    self.tick()
                    errors_in_row = 0
                except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
                    errors_in_row += 1
                    log.warning("Сетевая ошибка: %s", exc)
                    if errors_in_row == 5:
                        self._notify_error(f"5 сетевых ошибок подряд: {exc}")
                except Exception as exc:  # бот не должен падать из-за единичной ошибки
                    errors_in_row += 1
                    log.exception("Ошибка в торговом цикле")
                    self._notify_error(f"Ошибка: {type(exc).__name__}: {exc}")
                self._sleep(self.cfg.poll_interval_sec)
        except KeyboardInterrupt:
            pass
        finally:
            self._save()
            log.info("Бот остановлен. Открытые позиции сохранены и будут подхвачены при следующем запуске.")
            self.notifier.send("⏹ Бот остановлен")

    def stop(self) -> None:
        self.running = False

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    # ------------------------------------------------------------------ main step
    def tick(self) -> None:
        prices = self._fetch_prices()
        self._roll_day(prices)
        self._reconcile(prices)
        self._check_stops(prices)
        for symbol in self.cfg.symbols:
            self._process_symbol(symbol, prices)
        self._save()

    def _fetch_prices(self) -> dict[str, float]:
        return {s: self.client.fetch_price(s) for s in self.cfg.symbols}

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def _roll_day(self, prices: dict[str, float]) -> None:
        today = datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")
        if self.state.day != today:
            equity = self.broker.equity(prices)
            if self.state.day:
                change = equity - self.state.day_start_equity
                self.notifier.send(f"📅 Итог дня {self.state.day}: капитал {equity:.2f} "
                                   f"{self.cfg.quote_currency} ({change:+.2f})")
            self.state.day = today
            self.state.day_start_equity = equity
            log.info("Новый торговый день %s, капитал %.2f", today, equity)

    def _daily_loss_hit(self, equity: float) -> bool:
        start = self.state.day_start_equity
        return start > 0 and equity < start * (1 - self.cfg.risk.max_daily_loss_pct)

    def _reconcile(self, prices: dict[str, float]) -> None:
        """Позиция закрыта на бирже (сработал биржевой стоп / закрыли вручную) — убрать её из состояния."""
        for symbol, pos in list(self.state.positions.items()):
            try:
                exists = self.broker.position_exists(symbol, pos.side)
            except ccxt.BaseError as exc:
                log.warning("Не удалось сверить позицию %s: %s", symbol, exc)
                continue
            if exists is False:
                price = prices.get(symbol, pos.entry_price)
                stop_hit = price <= pos.stop_loss if pos.is_long else price >= pos.stop_loss
                exit_price = pos.stop_loss if stop_hit else price
                log.warning("Позиция %s не найдена на бирже — считаю закрытой (~%.6g)", symbol, exit_price)
                self._record_close(pos, exit_price, pos.qty * exit_price * self.cfg.taker_fee, "closed_on_exchange")

    def _check_stops(self, prices: dict[str, float]) -> None:
        for symbol, pos in list(self.state.positions.items()):
            price = prices[symbol]
            pos.update_trailing(price, price)
            hit = pos.check_stops(price, price, price)
            if hit:
                self._close(pos, price, hit[0])

    def _process_symbol(self, symbol: str, prices: dict[str, float]) -> None:
        candles = self.client.fetch_closed_candles(symbol, self.cfg.timeframe, self.cfg.candles_limit)
        if len(candles) < self.strategy.warmup + 2:
            log.warning("%s: мало свечей (%d < %d), пропуск", symbol, len(candles), self.strategy.warmup + 2)
            return
        last_ts = int(candles["timestamp"].iloc[-1])
        if self.state.last_candle_ts.get(symbol) == last_ts:
            return  # новая свеча ещё не закрылась
        self.state.last_candle_ts[symbol] = last_ts

        row = self.strategy.populate(candles).iloc[-1]
        extra = f" score={row['score']:+.2f}" if "score" in row else ""
        log.info("%s свеча %s close=%.6g rsi=%.1f%s", symbol, fmt_ts(last_ts), row["close"],
                 row.get("rsi", float("nan")), extra)

        pos = self.state.positions.get(symbol)
        if pos is not None:
            if (pos.is_long and row["exit_long"]) or (not pos.is_long and row["exit_short"]):
                self._close(pos, prices[symbol], "signal")
            return

        side = None
        if row["enter_long"]:
            side = LONG
        elif row["enter_short"] and self.cfg.risk.allow_short:
            side = SHORT
        if side:
            self._try_open(symbol, side, prices, float(row["atr"]))

    # ------------------------------------------------------------------ orders
    def _try_open(self, symbol: str, side: str, prices: dict[str, float], atr_value: float) -> None:
        risk = self.cfg.risk
        if len(self.state.positions) >= risk.max_open_positions:
            log.info("%s: сигнал %s пропущен — достигнут лимит позиций", symbol, side)
            return
        if self._now_ms() < self.state.cooldown_until.get(symbol, 0):
            log.info("%s: сигнал %s пропущен — пауза после сделки", symbol, side)
            return
        if not atr_value or atr_value != atr_value:
            return
        price = prices[symbol]
        equity = self.broker.equity(prices)
        if self._daily_loss_hit(equity):
            log.warning("Дневной лимит убытка достигнут — новые сделки не открываются до 00:00 UTC")
            return
        qty = position_size(equity, self.broker.free_balance(), price, atr_value, risk, self.cfg.leverage)
        qty = self.client.round_qty(symbol, qty)
        if not self.client.min_order_ok(symbol, qty, price):
            log.warning("%s: размер позиции %.8f ниже минимального ордера биржи — пропуск", symbol, qty)
            return

        sign = 1 if side == LONG else -1
        planned_sl = price - sign * atr_value * risk.stop_loss_atr
        try:
            fill = self.broker.open_position(symbol, side, qty, price, stop_loss=planned_sl)
        except ccxt.BaseError as exc:
            log.error("Не удалось открыть %s %s: %s", side, symbol, exc)
            self._notify_error(f"Не удалось открыть {side} {symbol}: {exc}")
            return
        pos = build_position(symbol, side, fill.qty, fill.price, atr_value, risk, self._now_ms(), fill.fee)
        self.state.positions[symbol] = pos
        self._save()
        tp = f"{pos.take_profit:.6g}" if pos.take_profit else "—"
        msg = (f"{'🟢 LONG' if side == LONG else '🔴 SHORT'} {symbol}\n"
               f"Цена: {fill.price:.6g}, объём: {fill.qty:.8g} (~{fill.price * fill.qty:.2f} {self.cfg.quote_currency})\n"
               f"SL: {pos.stop_loss:.6g}  TP: {tp}")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def _close(self, pos: Position, price: float, reason: str) -> None:
        try:
            fill = self.broker.close_position(pos.symbol, pos.side, pos.qty, price)
        except ccxt.InsufficientFunds as exc:
            log.error("%s: монет для продажи нет (%s) — позиция удалена из состояния", pos.symbol, exc)
            self._notify_error(f"{pos.symbol}: позиция не найдена на балансе, удалена из учёта")
            self.state.positions.pop(pos.symbol, None)
            return
        except ccxt.BaseError as exc:
            log.error("Не удалось закрыть %s: %s (повтор на следующем цикле)", pos.symbol, exc)
            self._notify_error(f"Не удалось закрыть {pos.symbol}: {exc}")
            return
        self._record_close(pos, fill.price, fill.fee, reason)

    def _record_close(self, pos: Position, exit_price: float, exit_fee: float, reason: str) -> None:
        gross = pos.unrealized_pnl(exit_price)
        fees = pos.entry_fee + exit_fee
        pnl = gross - fees
        notional = pos.entry_price * pos.qty
        now = self._now_ms()
        self.store.append_trade({
            "symbol": pos.symbol, "side": pos.side, "qty": pos.qty,
            "entry_time": fmt_ts(pos.opened_at), "entry_price": pos.entry_price,
            "exit_time": fmt_ts(now), "exit_price": exit_price, "reason": reason,
            "pnl": round(pnl, 6), "pnl_pct": round(pnl / notional * 100, 3) if notional else 0,
            "fees": round(fees, 6),
        })
        self.state.positions.pop(pos.symbol, None)
        self.state.cooldown_until[pos.symbol] = now + self.cfg.risk.cooldown_bars * self.tf_ms
        self._save()
        reasons = {"stop_loss": "стоп-лосс", "take_profit": "тейк-профит", "signal": "сигнал стратегии",
                   "closed_on_exchange": "закрыта на бирже", "manual": "вручную"}
        msg = (f"{'✅' if pnl >= 0 else '❌'} Закрыта {pos.side.upper()} {pos.symbol} ({reasons.get(reason, reason)})\n"
               f"Вход: {pos.entry_price:.6g} → выход: {exit_price:.6g}\n"
               f"PnL: {pnl:+.2f} {self.cfg.quote_currency} ({pnl / notional * 100 if notional else 0:+.2f}%)")
        log.info(msg.replace("\n", " | "))
        self.notifier.send(msg)

    def close_all(self) -> None:
        """Закрыть все позиции бота по рынку (команда close-all)."""
        self.state = self.store.load()
        self.broker.load_state(self.state.broker)
        self.broker.prepare()
        prices = self._fetch_prices()
        for pos in list(self.state.positions.values()):
            self._close(pos, prices[pos.symbol], "manual")
        self._save()

    # ------------------------------------------------------------------ helpers
    def _save(self) -> None:
        self.state.broker = self.broker.state()
        self.store.save(self.state)

    def _notify_error(self, text: str) -> None:
        if self.cfg.telegram_notify_errors:
            self.notifier.send(f"⚠️ {text}")

    def status(self) -> pd.DataFrame:
        rows = [p.to_dict() for p in self.state.positions.values()]
        return pd.DataFrame(rows)
