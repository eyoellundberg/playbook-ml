You are designing a library of 16 moving average crossover strategy archetypes for a stock/asset timing system.

The domain: a strategy observes a price series and decides when to be long (invested) vs flat (cash) based on moving average signals. It scores on PnL% over the scenario.

Strategy parameters:
- fast_window (2-20 bars): the faster moving average period
- slow_window (10-60 bars): the slower moving average period
- entry_threshold (0-0.05): minimum fast/slow ratio gap to enter a long position
- exit_threshold (0-0.03): ratio gap below which to exit the position
- position_size (0.1-1.0): fraction of capital deployed when long
- regime_filter ("none", "trend", "volatility"): pre-filter before taking signals

Scenario types:
- trending_up: clear upward drift — reward fast entries, high position size
- trending_down: downward drift — reward conservative filters, small size
- ranging: mean-reverting — MA crossovers are noisy, reward tight exits
- volatile: high volatility — reward caution, volatility filter, small size
- is_event: sharp price shock — reward either quick exit or staying flat

Generate exactly 16 archetypes. Cover the full space:
- Include aggressive (fast windows, large size, no filter)
- Include conservative (slow windows, small size, regime filter)
- Include trend-follower (wide windows, trend filter)
- Include volatility-aware (tight thresholds, volatility filter)
- Include at least one contrarian (very tight fast window vs slow)
- Name each archetype distinctively — the name should convey its philosophy
