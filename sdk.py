"""
sdk.py — AutoForge Python SDK.

Lets you call the engine inline without a domain folder:

    from autoforge import run

    champion = run(
        simulate=my_score_fn,
        state=my_random_state_fn,
        schema=MY_CANDIDATE_SCHEMA,
        description="retail pricing — maximise margin across demand levels",
    )

    print(champion.strategy)
    print(champion.playbook)
    print(champion.score)
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Callable


class Champion:
    """Result returned by autoforge.run()."""

    def __init__(self, data: dict, result: dict):
        self.strategy   = data.get("strategy", {})
        self.name       = data.get("name", "")
        self.philosophy = data.get("philosophy", "")
        self.best_for   = data.get("best_for", "")
        self.playbook   = data.get("playbook", [])
        self.score      = result.get("avg_score", 0)
        self.best_score = result.get("best_score", 0)

    def __repr__(self):
        label = self.name or "Champion"
        return f"{label}(score={self.score:.3f})"


def run(
    simulate: Callable,
    state: Callable,
    schema: dict,
    description: str = "",
    metric: str = "score",
    batches: int = 8,
    rounds: int = 200,
    brain: bool = True,
    workers: int = 1,
    verbose: bool = True,
    api_key: str | None = None,
) -> Champion:
    """
    Run AutoForge with inline Python functions. No domain folder needed.

    Parameters
    ----------
    simulate    : callable(candidate: dict, state: dict) -> float
        Your scoring function. Must be deterministic and fast.
    state       : callable() -> dict
        Draws a random scenario. Called once per round.
    schema      : dict
        JSON schema describing the candidate strategy shape.
    description : str, optional
        Plain-English description of your domain. Helps the brain
        generate better archetypes. E.g. "fraud detection for card payments".
    metric      : str
        Label for what you're optimising (e.g. "profit", "accuracy").
    batches     : int
        How many batches to run (default 8). More = smarter playbook.
    rounds      : int
        Rounds per batch (default 200). More = better signal per batch.
    brain       : bool
        True (default) = Stage 2, AI-generated archetypes + playbook.
        False = Stage 1, evolutionary only. No API key needed.
    workers     : int
        Parallel simulation workers (default 1).
    verbose     : bool
        Print progress per batch (default True). Set False for scripts.
    api_key     : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns
    -------
    Champion
        .strategy   — the best strategy dict found
        .name       — archetype name (brain mode only)
        .philosophy — why this strategy works (brain mode only)
        .playbook   — conditional principles learned (brain mode only)
        .score      — average score across the final batch
        .best_score — peak score seen
    """
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    # Keep engine root on sys.path — tournament.py does os.chdir(domain_path)
    # which would otherwise lose the engine modules
    engine_root = str(Path(__file__).parent)
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)

    # Inject a synthetic simulation module — engine imports this by name
    sim = types.ModuleType("simulation")
    sim.simulate         = simulate
    sim.random_state     = state
    sim.CANDIDATE_SCHEMA = schema
    sim.METRIC_NAME      = metric
    sim._synthetic       = True   # tells tournament.py not to reload from disk
    sys.modules["simulation"] = sim

    from tournament import run_batch

    hints  = []
    result = {}

    with tempfile.TemporaryDirectory(prefix="autoforge_") as tmp:
        domain_path = Path(tmp)

        # Write mission so the brain has context
        params = ", ".join(schema.get("properties", {}).keys())
        mission = description or f"Optimise {metric} across diverse scenarios."
        (domain_path / "mission.md").write_text(
            f"{mission}\nStrategy parameters: {params}."
        )

        if verbose:
            mode = "brain" if brain else "evolutionary"
            print(f"autoforge  {batches} batches × {rounds} rounds  [{mode}]")

        for batch_num in range(1, batches + 1):
            result = run_batch(
                domain_path,
                n_rounds=rounds,
                generation_offset=(batch_num - 1) * rounds,
                hints=hints,
                use_brain=brain,
                workers=workers,
            )

            if verbose:
                trend = f"{result['trend_pct']:+.1f}%"
                print(f"  [{batch_num}/{batches}]  avg {result['avg_score']:.3f}  best {result['best_score']:.3f}  {trend}")

            # Run director between batches for hints (brain mode only)
            if brain and batch_num < batches:
                try:
                    from director import call_director
                    analysis = call_director(batch_num, result, None, [], domain_path, [])
                    hints = analysis.get("hints", [])
                except Exception:
                    hints = []

        # Load champion and playbook
        champ_path = domain_path / "champion_archetype.json"
        champ = json.loads(champ_path.read_text()) if champ_path.exists() else {}

        pb_path = domain_path / "playbook.jsonl"
        playbook = []
        if pb_path.exists():
            for line in pb_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        playbook.append(json.loads(line))
                    except Exception:
                        pass

        champ["playbook"] = playbook

        if verbose and champ.get("name"):
            print(f"\nchampion   {champ['name']}")
            if champ.get("philosophy"):
                print(f"           {champ['philosophy']}")

    # Clean up synthetic module
    sys.modules.pop("simulation", None)

    return Champion(champ, result)
