"""
StockTiming/simulation.py — Moving average crossover strategy simulator.

The domain: given a simulated price series scenario, score a strategy's
entry/exit decisions based on moving average signals.

This is the example domain — run it out of the box:
  python run.py run --domain StockTiming --batches 5 --rounds 100
  python run.py run --domain StockTiming --brain --batches 5 --rounds 100
"""

import random
import math

METRIC_NAME = "pnl"

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "fast_window":    {"type": "integer", "minimum": 2,  "maximum": 20,
                           "description": "Fast MA period in bars"},
        "slow_window":    {"type": "integer", "minimum": 10, "maximum": 60,
                           "description": "Slow MA period in bars"},
        "entry_threshold":{"type": "number",  "minimum": 0.0, "maximum": 0.05,
                           "description": "Min fast/slow ratio gap to trigger entry"},
        "exit_threshold": {"type": "number",  "minimum": 0.0, "maximum": 0.03,
                           "description": "Max fast/slow ratio gap below which to exit"},
        "position_size":  {"type": "number",  "minimum": 0.1, "maximum": 1.0,
                           "description": "Fraction of capital to deploy when long"},
        "regime_filter":  {"type": "string",  "enum": ["none", "trend", "volatility"],
                           "description": "Pre-filter before taking signals"},
    },
    "required": ["fast_window", "slow_window", "entry_threshold",
                 "exit_threshold", "position_size", "regime_filter"],
    "additionalProperties": False,
}


def random_state() -> dict:
    """
    Simulate a price series scenario. Returns scenario metadata — the
    actual price series is generated deterministically from these seeds
    inside simulate() so results are reproducible.
    """
    market_regime = random.choice(["trending_up", "trending_down", "ranging", "volatile"])
    return {
        "regime":        market_regime,
        "volatility":    round(random.uniform(0.005, 0.04), 4),   # daily vol
        "trend_strength":round(random.uniform(-0.002, 0.002), 4), # daily drift
        "n_bars":        random.choice([60, 90, 120, 180]),        # scenario length
        "noise_seed":    random.randint(0, 9999),
        "is_event":      random.random() < 0.1,  # sharp move event
    }


def _generate_prices(state: dict) -> list:
    """Generate a deterministic price series from state parameters."""
    rng = random.Random(state["noise_seed"])
    prices = [100.0]
    drift = state["trend_strength"]
    vol   = state["volatility"]

    if state["is_event"]:
        event_bar = rng.randint(20, state["n_bars"] - 10)
        event_mag = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
    else:
        event_bar = -1
        event_mag = 0.0

    for i in range(state["n_bars"] - 1):
        shock = rng.gauss(drift, vol)
        if i == event_bar:
            shock += event_mag
        prices.append(round(prices[-1] * (1 + shock), 4))

    return prices


def _moving_average(prices: list, window: int, end: int) -> float:
    start = max(0, end - window + 1)
    return sum(prices[start:end + 1]) / (end - start + 1)


def simulate(candidate: dict, state: dict) -> float:
    """
    Score a strategy against a price scenario.
    Returns PnL as percentage of initial capital (higher is better).
    """
    fast  = max(2, candidate["fast_window"])
    slow  = max(fast + 1, candidate["slow_window"])
    entry = candidate["entry_threshold"]
    exit_ = candidate["exit_threshold"]
    size  = candidate["position_size"]
    filt  = candidate["regime_filter"]

    prices    = _generate_prices(state)
    n         = len(prices)
    capital   = 1.0
    position  = 0.0
    entry_price = 0.0

    # Regime filter: skip signals if filter condition not met
    regime = state["regime"]
    if filt == "trend" and regime == "ranging":
        return 0.0   # filter blocks all signals in ranging market
    if filt == "volatility" and state["volatility"] > 0.025:
        return 0.0   # filter blocks high-vol scenarios

    for i in range(slow, n):
        fast_ma = _moving_average(prices, fast, i)
        slow_ma = _moving_average(prices, slow, i)
        ratio   = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0

        if position == 0.0 and ratio > entry:
            # Enter long
            position    = size
            entry_price = prices[i]
        elif position > 0.0 and ratio < exit_:
            # Exit long
            pnl     = position * (prices[i] - entry_price) / entry_price
            capital += pnl
            position = 0.0

    # Close any open position at end
    if position > 0.0:
        pnl     = position * (prices[-1] - entry_price) / entry_price
        capital += pnl

    return round((capital - 1.0) * 100, 4)  # return as % PnL
