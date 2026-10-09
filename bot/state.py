"""Сохранение состояния бота на диск (переживает перезапуск) и журнал сделок."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk import Position

TRADE_FIELDS = ["symbol", "side", "qty", "entry_time", "entry_price", "exit_time", "exit_price",
                "reason", "pnl", "pnl_pct", "fees"]


@dataclass
class BotState:
    positions: dict[str, Position] = field(default_factory=dict)
    last_candle_ts: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, int] = field(default_factory=dict)
    day: str = ""
    day_start_equity: float = 0.0
    broker: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {k: p.to_dict() for k, p in self.positions.items()},
            "last_candle_ts": self.last_candle_ts,
            "cooldown_until": self.cooldown_until,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "broker": self.broker,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BotState":
        return cls(
            positions={k: Position.from_dict(v) for k, v in (d.get("positions") or {}).items()},
            last_candle_ts={k: int(v) for k, v in (d.get("last_candle_ts") or {}).items()},
            cooldown_until={k: int(v) for k, v in (d.get("cooldown_until") or {}).items()},
            day=d.get("day", ""),
            day_start_equity=float(d.get("day_start_equity") or 0),
            broker=d.get("broker") or {},
        )


class StateStore:
    def __init__(self, state_file: Path, trades_file: Path):
        self.state_file = Path(state_file)
        self.trades_file = Path(trades_file)

    def load(self) -> BotState:
        if not self.state_file.exists():
            return BotState()
        return BotState.from_dict(json.loads(self.state_file.read_text(encoding="utf-8")))

    def save(self, state: BotState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_file)  # атомарная запись

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        new = not self.trades_file.exists()
        with self.trades_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(trade)
