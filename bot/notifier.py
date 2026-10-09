"""Уведомления в Telegram."""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, token: str = "", chat_id: str = "", enabled: bool = False, prefix: str = ""):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token and chat_id)
        self.prefix = prefix

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": f"{self.prefix}{text}"[:4000]},
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning("Telegram ответил %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except requests.RequestException as exc:
            # Ошибка уведомления никогда не должна останавливать торговлю.
            log.warning("Не удалось отправить сообщение в Telegram: %s", exc)
            return False
