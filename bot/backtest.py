"""Бэктест стратегии на исторических свечах.

Правила симуляции (без «заглядывания в будущее»):
  * сигнал считается по закрытию свечи i, исполнение — по цене открытия свечи i+1;
  * стоп-лосс/тейк-профит проверяются внутри каждой свечи по high/low;
    если в одной свече задеты оба — считается, что сработал стоп (консервативно);
  * учитываются комиссия и проскальзывание; размер позиции — по тем же правилам риска, что и в боте.
Не учитываются: funding-платежи на фьючерсах, ликвидация, минимальный размер ордера.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import LONG, SHORT, Position, RiskConfig, build_position, position_size
from .strategies import Strategy

BARS_PER_YEAR_MS = 365 * 24 * 3600 * 1000


@dataclass
class BacktestResult:
    symbol: str
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskConfig, *, symbol: str = "",
                 initial_balance: float = 1000.0, fee: float = 0.001, slippage: float = 0.0005,
                 leverage: int = 1, allow_short: bool = False) -> BacktestResult:
    data = strategy.populate(df.reset_index(drop=True))
    ts = data["timestamp"].to_numpy(dtype="int64")
    o, h, l, c = (data[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = data["atr"].to_numpy(dtype=float)
    enter_long, exit_long = data["enter_long"].to_numpy(), data["exit_long"].to_numpy()
    enter_short, exit_short = data["enter_short"].to_numpy(), data["exit_short"].to_numpy()

    cash = initial_balance
    pos: Position | None = None
    pending: str | None = None      # LONG / SHORT / "close" — исполнить на открытии следующей свечи
    pending_atr = 0.0
    cooldown_until = -1
    day, day_start_equity = None, initial_balance
    trades: list[dict] = []
    equity_curve = np.empty(len(data))

    def close(i: int, raw_price: float, reason: str, apply_slip: bool = True) -> None:
        nonlocal cash, pos, cooldown_until
        assert pos is not None
        if apply_slip:
            raw_price *= (1 - slippage) if pos.is_long else (1 + slippage)
        exit_fee = raw_price * pos.qty * fee
        gross = pos.unrealized_pnl(raw_price)
        cash += gross - exit_fee
        pnl = gross - exit_fee - pos.entry_fee
        notional = pos.entry_price * pos.qty
        trades.append({
            "symbol": symbol, "side": pos.side, "qty": pos.qty,
            "entry_time": _ts(pos.opened_at), "entry_price": pos.entry_price,
            "exit_time": _ts(int(ts[i])), "exit_price": raw_price, "reason": reason,
            "pnl": pnl, "pnl_pct": pnl / notional * 100, "fees": pos.entry_fee + exit_fee,
            "bars": i - int(pos.meta.get("bar", i)),
        })
        pos = None
        cooldown_until = i + risk.cooldown_bars

    for i in range(len(data)):
        # Дневной лимит убытка (по UTC).
        cur_day = int(ts[i]) // 86_400_000
        if cur_day != day:
            day = cur_day
            day_start_equity = cash + (pos.unrealized_pnl(o[i]) if pos else 0.0)

        # 1. Исполнение отложенного сигнала по открытию свечи.
        if pending == "close" and pos is not None:
            close(i, o[i], "signal")
        elif pending in (LONG, SHORT) and pos is None:
            equity_now = cash
            halted = equity_now < day_start_equity * (1 - risk.max_daily_loss_pct)
            if not halted:
                price = o[i] * ((1 + slippage) if pending == LONG else (1 - slippage))
                qty = position_size(equity_now, cash, price, pending_atr, risk, leverage)
                if qty > 0:
                    entry_fee = price * qty * fee
                    cash -= entry_fee
                    pos = build_position(symbol, pending, qty, price, pending_atr, risk, int(ts[i]), entry_fee)
                    pos.meta["bar"] = i
        pending = None

        # 2. Стоп-лосс / тейк-профит внутри свечи.
        if pos is not None:
            hit = pos.check_stops(o[i], h[i], l[i])
            if hit:
                reason, price = hit
                close(i, price, reason, apply_slip=reason == "stop_loss")
            else:
                pos.update_trailing(h[i], l[i])

        # 3. Сигналы на закрытии свечи.
        if pos is not None:
            if (pos.is_long and exit_long[i]) or (not pos.is_long and exit_short[i]):
                pending = "close"
        elif i >= cooldown_until and i < len(data) - 1 and not math.isnan(atr[i]) and atr[i] > 0:
            if enter_long[i]:
                pending, pending_atr = LONG, atr[i]
            elif allow_short and enter_short[i]:
                pending, pending_atr = SHORT, atr[i]

        equity_curve[i] = cash + (pos.unrealized_pnl(c[i]) if pos else 0.0)

    if pos is not None:
        close(len(data) - 1, c[-1], "end_of_data")
        equity_curve[-1] = cash

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"timestamp": ts, "equity": equity_curve, "close": c})
    result = BacktestResult(symbol, trades_df, equity_df)
    result.metrics = compute_metrics(equity_df, trades_df, initial_balance, strategy.warmup)
    return result


def compute_metrics(equity_df: pd.DataFrame, trades: pd.DataFrame, initial: float, warmup: int = 0) -> dict:
    eq = equity_df["equity"]
    close = equity_df["close"]
    final = float(eq.iloc[-1]) if len(eq) else initial
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    start_idx = min(warmup, len(close) - 1) if len(close) else 0
    buy_hold = float((close.iloc[-1] / close.iloc[start_idx] - 1) * 100) if len(close) else 0.0

    rets = eq.pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 2 and rets.std() > 0:
        bar_ms = float(np.median(np.diff(equity_df["timestamp"].to_numpy())))
        sharpe = float(rets.mean() / rets.std() * math.sqrt(BARS_PER_YEAR_MS / bar_ms))

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_loss = float(-losses["pnl"].sum()) if n else 0.0
    profit_factor = float(wins["pnl"].sum() / gross_loss) if n and gross_loss > 0 else (float("inf") if n and len(wins) else 0.0)
    return {
        "initial_balance": initial,
        "final_balance": round(final, 2),
        "total_return_pct": round((final / initial - 1) * 100, 2),
        "buy_and_hold_pct": round(buy_hold, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "trades": n,
        "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0.0,
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else profit_factor,
        "avg_trade_pct": round(float(trades["pnl_pct"].mean()), 3) if n else 0.0,
        "fees_paid": round(float(trades["fees"].sum()), 2) if n else 0.0,
    }


METRIC_NAMES = {
    "initial_balance": "Начальный баланс",
    "final_balance": "Итоговый баланс",
    "total_return_pct": "Доходность, %",
    "buy_and_hold_pct": "Купить и держать, %",
    "max_drawdown_pct": "Макс. просадка, %",
    "sharpe": "Коэф. Шарпа (год.)",
    "trades": "Сделок",
    "win_rate_pct": "Прибыльных сделок, %",
    "profit_factor": "Profit factor",
    "avg_trade_pct": "Средняя сделка, %",
    "fees_paid": "Комиссии",
}


def format_report(result: BacktestResult) -> str:
    lines = [f"===== Бэктест {result.symbol} ====="]
    if len(result.equity):
        lines.append(f"Период: {_ts(int(result.equity['timestamp'].iloc[0]))} — "
                     f"{_ts(int(result.equity['timestamp'].iloc[-1]))} ({len(result.equity)} свечей)")
    for key, name in METRIC_NAMES.items():
        lines.append(f"  {name:<24} {result.metrics.get(key)}")
    if len(result.trades):
        by_reason = result.trades.groupby("reason")["pnl"].agg(["count", "sum"]).round(2)
        lines.append("  Закрытия по причинам:")
        for reason, r in by_reason.iterrows():
            lines.append(f"    {reason:<16} {int(r['count']):>4} сделок, PnL {r['sum']:+.2f}")
    return "\n".join(lines)


def save_results(result: BacktestResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result.symbol.replace("/", "_").replace(":", "_") or "backtest"
    paths = [out_dir / f"{name}_trades.csv", out_dir / f"{name}_equity.csv"]
    result.trades.to_csv(paths[0], index=False)
    result.equity.to_csv(paths[1], index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    times = pd.to_datetime(result.equity["timestamp"], unit="ms")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    ax1.plot(times, result.equity["close"], color="#555", lw=0.8)
    ax1.set_title(f"{result.symbol}: цена и сделки")
    if len(result.trades):
        t = result.trades
        colors = np.where(t["pnl"] > 0, "#2a9d4b", "#d1495b")
        ax1.scatter(pd.to_datetime(t["entry_time"]), t["entry_price"], marker="^", c="#1f6feb", s=25, label="вход")
        ax1.scatter(pd.to_datetime(t["exit_time"]), t["exit_price"], marker="v", c=colors, s=25, label="выход")
        ax1.legend()
    ax2.plot(times, result.equity["equity"], color="#1f6feb")
    ax2.set_title("Капитал")
    fig.tight_layout()
    png = out_dir / f"{name}_chart.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    paths.append(png)
    return paths
