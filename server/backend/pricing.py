"""What a model call cost, from the price table in utils/config.py (ROADMAP 1.7)."""

from utils import config


def call_cost(usage: dict) -> float | None:
    """Dollars for one call, or None if its model has no price.

    Unpriced is not free: a missing price makes the question's cost unknown,
    so a forgotten table entry can never pass for $0.
    """
    prices = config.PRICES.get(usage.get("model"))
    if prices is None:
        return None
    cached = usage.get("cached_tokens", 0)
    fresh = usage["prompt_tokens"] - cached
    return (fresh * prices["input"] + cached * prices["cached_input"]
            + usage["completion_tokens"] * prices["output"]) / 1_000_000
