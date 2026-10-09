"""Работа с биржей через ccxt: рыночные данные, реальные ордера (LiveBroker)
и симуляция исполнения (PaperBroker)."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from .config import Config
from .risk import LONG

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class Fill:
    price: float
    qty: float
    fee: float
    order_id: str | None = None


def timeframe_ms(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def ohlcv_to_df(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in OHLCV_COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df


def retry(fn, *args, attempts: int = 3, delay: float = 2.0, **kwargs):
    """Повтор запроса при сетевых ошибках/лимитах биржи."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
            if i == attempts - 1:
                raise
            log.warning("Сетевая ошибка (%s), повтор через %.0f c", exc, delay * (i + 1))
            time.sleep(delay * (i + 1))


class ExchangeClient:
    """Тонкая обёртка над ccxt для BingX."""

    def __init__(self, cfg: Config, authenticated: bool = False):
        self.cfg = cfg
        if not hasattr(ccxt, cfg.exchange_name):
            raise ValueError(f"Биржа '{cfg.exchange_name}' не поддерживается ccxt")
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": cfg.market_type},
        }
        if authenticated:
            params["apiKey"] = cfg.api_key
            params["secret"] = cfg.api_secret
        self.ex: ccxt.Exchange = getattr(ccxt, cfg.exchange_name)(params)
        self._markets_loaded = False

    def load_markets(self) -> dict:
        if not self._markets_loaded:
            retry(self.ex.load_markets)
            self._markets_loaded = True
        return self.ex.markets

    def market(self, symbol: str) -> dict:
        self.load_markets()
        if symbol not in self.ex.markets:
            raise ValueError(f"Пара {symbol} не найдена на {self.cfg.exchange_name} ({self.cfg.market_type})")
        return self.ex.markets[symbol]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = retry(self.ex.fetch_ohlcv, symbol, timeframe, limit=limit)
        return ohlcv_to_df(rows)

    def fetch_closed_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Только закрытые свечи (последняя, ещё формирующаяся, отбрасывается)."""
        df = self.fetch_ohlcv(symbol, timeframe, limit + 1)
        now_ms = self.ex.milliseconds()
        tf = timeframe_ms(timeframe)
        return df[df["timestamp"] + tf <= now_ms].tail(limit).reset_index(drop=True)

    def fetch_history(self, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
        """Загрузка длинной истории постранично (для бэктеста)."""
        until_ms = until_ms or self.ex.milliseconds()
        tf = timeframe_ms(timeframe)
        rows: list[list[float]] = []
        cursor = since_ms
        while cursor < until_ms:
            batch = retry(self.ex.fetch_ohlcv, symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1][0]
            if last < cursor:
                break
            cursor = last + tf
            log.info("Загружено %d свечей %s (до %s)", len(rows), symbol, pd.to_datetime(last, unit="ms"))
        df = ohlcv_to_df(rows) if rows else pd.DataFrame(columns=OHLCV_COLUMNS)
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] + tf <= until_ms)]
        return df.reset_index(drop=True)

    def fetch_price(self, symbol: str) -> float:
        ticker = retry(self.ex.fetch_ticker, symbol)
        price = ticker.get("last") or ticker.get("close")
        if not price:
            raise ccxt.ExchangeError(f"Нет цены для {symbol}")
        return float(price)

    def round_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            return 0.0
        try:
            return float(self.ex.amount_to_precision(symbol, qty))
        except ccxt.InvalidOrder:  # меньше минимального шага
            return 0.0

    def min_order_ok(self, symbol: str, qty: float, price: float) -> bool:
        limits = self.market(symbol).get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        return qty > 0 and qty >= min_amount and qty * price >= min_cost


class Broker(ABC):
    """Единый интерфейс исполнения для paper и live."""

    def __init__(self, client: ExchangeClient, cfg: Config):
        self.client = client
        self.cfg = cfg

    def prepare(self) -> None:
        """Подготовка перед стартом (проверка пар, установка плеча и т.п.)."""
        for symbol in self.cfg.symbols:
            self.client.market(symbol)

    @abstractmethod
    def equity(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def free_balance(self) -> float: ...

    @abstractmethod
    def open_position(self, symbol: str, side: str, qty: float, price: float,
                      stop_loss: float | None = None) -> Fill: ...

    @abstractmethod
    def close_position(self, symbol: str, side: str, qty: float, price: float) -> Fill: ...

    def position_exists(self, symbol: str, side: str) -> bool | None:
        """Есть ли позиция на бирже. None — проверка не поддерживается."""
        return None

    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, data: dict[str, Any]) -> None:
        pass


class PaperBroker(Broker):
    """Виртуальное исполнение по рыночной цене с комиссией и проскальзыванием.

    Учёт: cash = стартовый баланс + реализованный PnL - комиссии;
    equity = cash + нереализованный PnL. Маржа = notional / leverage.
    """

    def __init__(self, client: ExchangeClient, cfg: Config):
        super().__init__(client, cfg)
        self.cash = cfg.paper_initial_balance
        self.margin: dict[str, float] = {}
        self.open_qty: dict[str, tuple[str, float, float]] = {}  # symbol -> (side, qty, entry)

    def equity(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for symbol, (side, qty, entry) in self.open_qty.items():
            price = prices.get(symbol, entry)
            upnl += (price - entry) * qty if side == LONG else (entry - price) * qty
        return self.cash + upnl

    def free_balance(self) -> float:
        return self.cash - sum(self.margin.values())

    def _fill_price(self, price: float, buy: bool) -> float:
        return price * (1 + self.cfg.slippage) if buy else price * (1 - self.cfg.slippage)

    def open_position(self, symbol, side, qty, price, stop_loss=None) -> Fill:
        fill = self._fill_price(price, buy=side == LONG)
        fee = fill * qty * self.cfg.taker_fee
        self.cash -= fee
        self.margin[symbol] = fill * qty / self.cfg.leverage
        self.open_qty[symbol] = (side, qty, fill)
        return Fill(fill, qty, fee, order_id=f"paper-{int(time.time() * 1000)}")

    def close_position(self, symbol, side, qty, price) -> Fill:
        fill = self._fill_price(price, buy=side != LONG)
        _, _, entry = self.open_qty.pop(symbol, (side, qty, fill))
        pnl = (fill - entry) * qty if side == LONG else (entry - fill) * qty
        fee = fill * qty * self.cfg.taker_fee
        self.cash += pnl - fee
        self.margin.pop(symbol, None)
        return Fill(fill, qty, fee, order_id=f"paper-{int(time.time() * 1000)}")

    def state(self) -> dict[str, Any]:
        return {"cash": self.cash, "margin": self.margin,
                "open_qty": {k: list(v) for k, v in self.open_qty.items()}}

    def load_state(self, data: dict[str, Any]) -> None:
        if not data:
            return
        self.cash = float(data.get("cash", self.cash))
        self.margin = {k: float(v) for k, v in (data.get("margin") or {}).items()}
        self.open_qty = {k: (v[0], float(v[1]), float(v[2])) for k, v in (data.get("open_qty") or {}).items()}


class LiveBroker(Broker):
    """Реальные рыночные ордера на BingX."""

    def __init__(self, client: ExchangeClient, cfg: Config):
        super().__init__(client, cfg)
        self.ex = client.ex
        self.hedged = False

    # ---------- подготовка ----------
    def prepare(self) -> None:
        super().prepare()
        retry(self.ex.fetch_balance)  # проверка ключей
        if not self.cfg.is_swap:
            return
        try:
            mode = retry(self.ex.fetch_position_mode, self.cfg.symbols[0])
            self.hedged = bool(mode.get("hedged"))
        except ccxt.BaseError as exc:
            log.warning("Не удалось определить режим позиций (%s), считаю one-way", exc)
        log.info("Режим позиций на фьючерсах: %s", "hedge" if self.hedged else "one-way")
        sides = ["LONG", "SHORT"] if self.hedged else ["BOTH"]
        for symbol in self.cfg.symbols:
            for s in sides:
                try:
                    retry(self.ex.set_leverage, self.cfg.leverage, symbol, {"side": s})
                except ccxt.BaseError as exc:
                    log.warning("Не удалось установить плечо %sx для %s (%s): %s", self.cfg.leverage, symbol, s, exc)
            log.info("Плечо %s: %dx", symbol, self.cfg.leverage)

    # ---------- баланс ----------
    def _quote_balance(self) -> dict[str, float]:
        bal = retry(self.ex.fetch_balance)
        q = bal.get(self.cfg.quote_currency) or {}
        return {"free": float(q.get("free") or 0), "total": float(q.get("total") or 0)}

    def free_balance(self) -> float:
        return self._quote_balance()["free"]

    def equity(self, prices: dict[str, float]) -> float:
        total = self._quote_balance()["total"]
        if self.cfg.is_swap:
            try:
                positions = retry(self.ex.fetch_positions, self.cfg.symbols)
                total += sum(float(p.get("unrealizedPnl") or 0) for p in positions)
            except ccxt.BaseError as exc:
                log.warning("Не удалось получить нереализованный PnL: %s", exc)
            return total
        # Спот: USDT + стоимость монет, которыми торгует бот.
        bal = retry(self.ex.fetch_balance)
        for symbol in self.cfg.symbols:
            base = self.client.market(symbol)["base"]
            amount = float((bal.get(base) or {}).get("total") or 0)
            if amount and symbol in prices:
                total += amount * prices[symbol]
        return total

    # ---------- ордера ----------
    def _finalize(self, order: dict, symbol: str, fallback_price: float, fallback_qty: float) -> Fill:
        order_id = order.get("id")
        if order_id and (not order.get("average") or not order.get("filled")):
            time.sleep(0.5)
            try:
                order = retry(self.ex.fetch_order, order_id, symbol)
            except ccxt.BaseError as exc:
                log.warning("Не удалось получить детали ордера %s: %s", order_id, exc)
        price = float(order.get("average") or order.get("price") or fallback_price)
        qty = float(order.get("filled") or fallback_qty)
        fee_info = order.get("fee") or {}
        fee = fee_info.get("cost")
        if fee is None or fee_info.get("currency") not in (None, self.cfg.quote_currency):
            fee = price * qty * self.cfg.taker_fee  # оценка, если комиссия не в USDT
        return Fill(price, qty, float(fee), order_id)

    def open_position(self, symbol, side, qty, price, stop_loss=None) -> Fill:
        buy = side == LONG
        if self.cfg.is_swap:
            params: dict[str, Any] = {"hedged": self.hedged}
            if stop_loss and self.cfg.exchange_stop_loss:
                params["stopLoss"] = {"triggerPrice": stop_loss, "type": "STOP_MARKET", "workingType": "MARK_PRICE"}
            order = self.ex.create_order(symbol, "market", "buy" if buy else "sell", qty, None, params)
        else:
            if not buy:
                raise ValueError("Шорт на споте невозможен")
            cost = float(self.ex.cost_to_precision(symbol, qty * price))
            order = self.ex.create_order(symbol, "market", "buy", qty, None, {"cost": cost})
        log.info("Ордер на открытие отправлен: %s %s %s qty=%s", symbol, side, order.get("id"), qty)
        return self._finalize(order, symbol, price, qty)

    def close_position(self, symbol, side, qty, price) -> Fill:
        close_side = "sell" if side == LONG else "buy"
        if self.cfg.is_swap:
            params = {"reduceOnly": True, "hedged": self.hedged}
            order = self.ex.create_order(symbol, "market", close_side, qty, None, params)
            try:  # снять оставшиеся условные ордера (биржевой стоп-лосс)
                self.ex.cancel_all_orders(symbol)
            except ccxt.BaseError as exc:
                log.warning("Не удалось отменить условные ордера %s: %s", symbol, exc)
        else:
            # На споте комиссия при покупке списывается в монете, поэтому продаём не больше свободного остатка.
            base = self.client.market(symbol)["base"]
            free = float((retry(self.ex.fetch_balance).get(base) or {}).get("free") or 0)
            qty = self.client.round_qty(symbol, min(qty, free))
            if qty <= 0:
                raise ccxt.InsufficientFunds(f"Нет свободных {base} для продажи")
            order = self.ex.create_order(symbol, "market", "sell", qty)
        log.info("Ордер на закрытие отправлен: %s %s %s qty=%s", symbol, side, order.get("id"), qty)
        return self._finalize(order, symbol, price, qty)

    def position_exists(self, symbol: str, side: str) -> bool | None:
        if not self.cfg.is_swap:
            return None
        positions = retry(self.ex.fetch_positions, [symbol])
        for p in positions:
            if p.get("symbol") != symbol or not float(p.get("contracts") or 0):
                continue
            if self.hedged and p.get("side") and p["side"] != side:
                continue
            return True
        return False


def create_broker(cfg: Config) -> Broker:
    if cfg.mode == "live":
        return LiveBroker(ExchangeClient(cfg, authenticated=True), cfg)
    return PaperBroker(ExchangeClient(cfg, authenticated=False), cfg)
