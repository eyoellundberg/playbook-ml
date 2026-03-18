You are extracting conditional principles from a stock timing tournament.

Each round tests moving average crossover strategies against a simulated price scenario.
The winner is the strategy with the highest PnL% for that scenario.

Extract 0, 1, or 2 principles. Only extract when the result reveals something genuinely useful:
- A clear pattern between scenario type and winning strategy parameters
- A surprising winner that suggests a non-obvious approach
- A consistent failure mode across losers

Good principles for this domain:
- "In trending_up regimes with low volatility, fast_window < 8 beats slow crossovers"
- "When is_event=True, strategies with position_size < 0.4 outperform"
- "volatility filter eliminates noise in high-vol scenarios — keeps capital flat"

Bad principles:
- Generic statements true of all MA strategies
- Single-round observations with no clear mechanism
- Principles contradicted by existing high-confidence entries

RETIRED TOPICS (never regenerate): {{RETIRED_TOPICS}}

Return 0 principles if this round reveals nothing new.
