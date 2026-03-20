"""
autoforge — public Python API.

    from autoforge import run

    champion = run(
        simulate=my_score_fn,
        state=my_random_state_fn,
        schema=MY_SCHEMA,
    )
"""

from sdk import run, Champion

__all__ = ["run", "Champion"]
