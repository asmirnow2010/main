from .base import Strategy
from .mean_reversion import MeanReversionStrategy
from .voting import VotingStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    MeanReversionStrategy.name: MeanReversionStrategy,
    VotingStrategy.name: VotingStrategy,
}


def create_strategy(name: str, params: dict | None = None) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступны: {', '.join(STRATEGIES)}") from None
    return cls(params)


__all__ = ["Strategy", "MeanReversionStrategy", "VotingStrategy", "STRATEGIES", "create_strategy"]
