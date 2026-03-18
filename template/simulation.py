"""
simulation.py — The scoring function for your domain.

This is the only file that knows what your domain is. The Engine calls it
to evaluate candidate strategies deterministically — no AI, no randomness
after the scenario is drawn.

REQUIRED EXPORTS (the Engine reads these directly):
  - simulate(candidate, state) -> float      score a candidate against a scenario
  - random_state() -> dict                   draw a random scenario
  - CANDIDATE_SCHEMA: dict                   JSON schema for a candidate strategy
  - METRIC_NAME: str                         label shown in logs (e.g. "profit", "accuracy")
"""

import random

# Real data (optional): place CSV/JSON files in data/ alongside this file.
# random_state() can sample from real distributions instead of synthetic ones.
# simulate() can reference real lookup tables or calibration data.
# The engine never reads data/ directly — only simulation.py touches it.


# ── What a candidate strategy looks like ─────────────────────────────────────
#
# This schema tells the Engine (and Sonnet) what shape a strategy has.
# The AI generates candidates that match this schema exactly.
# Be as specific as you can — vague schemas produce vague strategies.
#
# Example: a pricing strategy might have a base price and a discount rate.
# Example: a routing strategy might have priority weights per category.
#
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        # TODO: replace these with your actual strategy dimensions
        "param_a": {
            "type": "number",
            "description": "What this parameter controls",
        },
        "param_b": {
            "type": "string",
            "enum": ["option_1", "option_2", "option_3"],
            "description": "What this choice controls",
        },
    },
    "required": ["param_a", "param_b"],
    "additionalProperties": False,
}

# Label used in logs and the thinking log
METRIC_NAME = "score"  # e.g. "profit", "revpar", "accuracy", "throughput"


# ── Scenario generation ───────────────────────────────────────────────────────
#
# Draw a random scenario from your domain's distribution.
# The Engine calls this once per round to generate the test case.
# Keep it realistic — the sim is only as good as its scenario variety.
#
def random_state() -> dict:
    """
    Return a dict describing a single scenario.
    The Engine passes this to simulate() along with the candidate strategy.

    Example keys: demand level, competition, time period, context flags.
    """
    return {
        # TODO: replace with your actual scenario dimensions
        "demand": random.uniform(0.3, 1.0),
        "competition": random.choice(["low", "medium", "high"]),
        "context_flag": random.random() > 0.8,  # e.g. is_event, is_peak, is_weekend
    }


# ── Scoring function ──────────────────────────────────────────────────────────
#
# The heart of the simulation. Score a candidate strategy against one scenario.
# Must be:
#   - Deterministic: same inputs → same output
#   - Fast: called thousands of times per run (pure Python, no I/O)
#   - Domain-faithful: reward the right behaviors, penalize the wrong ones
#
# Calibration checklist:
#   - Does the expected strategy type win in each scenario class?
#   - Are scores in a reasonable range (e.g. 0–200, not 1e6)?
#   - Does varying demand/competition change which strategy wins?
#   - Is there no single dominant strategy that always wins regardless of context?
#
def simulate(candidate: dict, state: dict) -> float:
    """
    Score candidate against state. Higher is better.

    candidate: a strategy dict matching CANDIDATE_SCHEMA
    state:     a scenario dict from random_state()
    returns:   float metric value (higher = better)
    """
    # TODO: implement your scoring logic
    # This is the hard part. Get this right and everything else follows.

    param_a = candidate["param_a"]
    param_b = candidate["param_b"]
    demand  = state["demand"]

    # Placeholder — replace with real domain logic
    base_score = param_a * demand
    if param_b == "option_1":
        modifier = 1.1
    elif param_b == "option_2":
        modifier = 1.0
    else:
        modifier = 0.9

    return base_score * modifier
