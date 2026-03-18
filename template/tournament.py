"""
tournament.py — The tournament loop for your domain.

Runs rounds: generate a scenario, score all archetypes, log the winner.
Called by run.py via run_batch().

Copy this file into your domain folder and adapt the context dict
(the key-value pairs that describe each scenario for the AI's extraction step).
Everything else can stay as-is.
"""

import json
import sys
from pathlib import Path

# Engine root is one level up
ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

from engine_extract import extract_principles_ai
from simulation import simulate, random_state, CANDIDATE_SCHEMA, METRIC_NAME


DOMAIN_PATH = Path(__file__).parent


def _load_playbook() -> list:
    p = DOMAIN_PATH / "playbook.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _save_playbook(playbook: list):
    with open(DOMAIN_PATH / "playbook.jsonl", "w") as f:
        for entry in playbook:
            f.write(json.dumps(entry) + "\n")


def _log_round(log_path: Path, round_num: int, winner: dict, score: float, state: dict, archetype_name: str = None):
    entry = {
        "round":  round_num,
        "score":  score,
        "metric": METRIC_NAME,
        "winner": winner,
        "state":  state,
    }
    if archetype_name:
        entry["archetype"] = archetype_name
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _build_context(state: dict) -> dict:
    """
    Build the context dict passed to the AI extractor.

    These key-value pairs describe the scenario in plain terms — the AI reads
    them to extract conditional principles ("IF demand is high AND competition is low...").

    Make these human-readable. The AI's extraction quality depends on this.
    """
    # TODO: adapt this to your actual state dict structure
    return {
        "demand":      state.get("demand", "unknown"),
        "competition": state.get("competition", "unknown"),
        "context_flag": state.get("context_flag", False),
        # Add more domain-relevant context here
    }


def run_batch(
    n_rounds: int,
    generation_offset: int = 0,
    hints: list = None,
    use_brain: bool = False,
) -> dict:
    """
    Run one batch of the tournament.

    n_rounds:          how many rounds to run
    generation_offset: global round counter offset (for logging)
    hints:             director hints from the prior batch (Stage 2)
    use_brain:         True → Stage 2 (Sonnet generates archetype library)

    Returns a result dict consumed by run.py's director call.
    """
    hints = hints or []
    playbook = _load_playbook()
    log_path = DOMAIN_PATH / "tournament_log.jsonl"

    # ── Stage 2: generate archetype library via Sonnet ────────────────────────
    archetypes = []
    if use_brain:
        from engine_brain import call_library
        archetypes = call_library(
            playbook=playbook,
            hints=hints,
            domain_path=DOMAIN_PATH,
            candidate_schema=CANDIDATE_SCHEMA,
        )

    # ── Stage 1: generate procedural candidates once per batch ───────────────
    # Generating once per batch (not per round) means wins accumulate against
    # the same fixed set — so archetype_wins reliably identifies the top performers
    # for the next batch's evolution step.
    stage1_candidates = []
    stage1_names = []
    if not archetypes:
        stage1_candidates = _generate_procedural_candidates(n=16)
        stage1_names = [f"procedural_{j}" for j in range(len(stage1_candidates))]

    # ── Tournament loop ───────────────────────────────────────────────────────
    scores = []
    archetype_wins         = {}
    archetype_wins_event   = {}
    archetype_wins_nonevent = {}
    context_mix            = {}

    for i in range(n_rounds):
        round_num = generation_offset + i + 1
        state = random_state()
        context = _build_context(state)

        # Determine candidates: archetype library (Stage 2) or procedural (Stage 1)
        if archetypes:
            candidates = [a["strategy"] for a in archetypes]
            candidate_names = [a["name"] for a in archetypes]
        else:
            # Stage 1: use the fixed batch candidates generated above
            candidates = stage1_candidates
            candidate_names = stage1_names

        # Score all candidates
        scored = [(simulate(c, state), c, n) for c, n in zip(candidates, candidate_names)]
        scored.sort(key=lambda x: x[0], reverse=True)

        winner_score, winner, winner_name = scored[0]
        losers = [c for _, c, _ in scored[1:4]]

        scores.append(winner_score)
        _log_round(log_path, round_num, winner, winner_score, state, winner_name)

        # Track archetype win counts
        if winner_name not in archetype_wins:
            archetype_wins[winner_name]         = 0
            archetype_wins_event[winner_name]   = 0
            archetype_wins_nonevent[winner_name] = 0
        archetype_wins[winner_name] += 1
        # TODO: update event/nonevent split if your domain has event scenarios
        # is_event = state.get("context_flag", False)
        # if is_event:
        #     archetype_wins_event[winner_name] += 1
        # else:
        #     archetype_wins_nonevent[winner_name] += 1
        archetype_wins_nonevent[winner_name] += 1

        # Track context distribution
        for k, v in context.items():
            key = f"{k}:{v}"
            context_mix[key] = context_mix.get(key, 0) + 1

        # Extract principles every 10 rounds
        if round_num % 10 == 0:
            playbook = extract_principles_ai(
                winner=winner,
                losers=losers,
                context=context,
                score=winner_score,
                generation=round_num,
                existing=playbook,
                domain_path=DOMAIN_PATH,
            )
            _save_playbook(playbook)

    # ── Champion propagation ──────────────────────────────────────────────────
    if archetypes and archetype_wins_nonevent:
        champion_name = max(archetype_wins_nonevent, key=archetype_wins_nonevent.get)
        champion = next((a for a in archetypes if a["name"] == champion_name), None)
        if champion:
            champion_data = {
                **champion,
                "nonevent_wins": archetype_wins_nonevent[champion_name],
                "total_wins":    archetype_wins[champion_name],
            }
            (DOMAIN_PATH / "champion_archetype.json").write_text(json.dumps(champion_data, indent=2))

    # ── Save top candidates for Stage 1 evolution ─────────────────────────────
    # Works for both Stage 1 (procedural) and Stage 2 (archetypes).
    # The next Stage 1 batch reads top_candidates.json and evolves from these winners.
    if archetypes:
        # Stage 2: rank archetypes by total wins (nonevent + event)
        top_n = sorted(
            [(archetype_wins_nonevent.get(a["name"], 0) + archetype_wins.get(a["name"], 0), a["strategy"], a["name"])
             for a in archetypes],
            key=lambda x: x[0],
            reverse=True,
        )[:4]
        top_candidates = [{"name": n, "strategy": c, "wins": w} for w, c, n in top_n]
        (DOMAIN_PATH / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))
    elif stage1_candidates:
        # Stage 1: rank the fixed batch candidates by accumulated wins
        top_n = sorted(
            [(archetype_wins_nonevent.get(stage1_names[j], 0) + archetype_wins.get(stage1_names[j], 0),
              stage1_candidates[j], stage1_names[j])
             for j in range(len(stage1_candidates))],
            reverse=True,
        )[:4]
        top_candidates = [{"name": n, "strategy": c, "wins": w} for w, c, n in top_n]
        (DOMAIN_PATH / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))

    # ── Build result dict for director ───────────────────────────────────────
    n = len(scores)
    quarter = max(1, n // 4)
    first_q_avg = sum(scores[:quarter]) / quarter
    last_q_avg  = sum(scores[-quarter:]) / quarter
    trend_pct   = (last_q_avg - first_q_avg) / abs(first_q_avg) * 100 if first_q_avg else 0.0

    top_principles = sorted(playbook, key=lambda p: p.get("confidence", 0), reverse=True)[:5]

    return {
        "n_rounds":    n,
        "avg_score":   round(sum(scores) / n, 2) if n else 0,
        "best_score":  round(max(scores), 2) if scores else 0,
        "worst_score": round(min(scores), 2) if scores else 0,
        "trend_pct":   round(trend_pct, 1),
        "score_last_10": [round(s, 2) for s in scores[-10:]],
        "context_mix": {k: v for k, v in sorted(context_mix.items(), key=lambda x: -x[1])[:10]},
        "playbook_size":          len(playbook),
        "top_principles":         top_principles,
        "archetype_wins":         archetype_wins,
        "archetype_wins_event":   archetype_wins_event,
        "archetype_wins_nonevent": archetype_wins_nonevent,
    }


# ── Stage 1: evolutionary candidate generation ───────────────────────────────

def _generate_procedural_candidates(n: int = 16) -> list:
    """
    Evolutionary Stage 1 candidate generation. No API calls.

    Reads top_candidates.json from the prior batch and evolves from them:
    - Elitism: keep top 2 winners unchanged
    - Mutation: gaussian noise on numeric params, occasional enum flip
    - Crossover: mix parameters from two winners
    - Exploration: random candidates to prevent premature convergence

    First batch (no prior winners): purely random.

    Reads CANDIDATE_SCHEMA to know parameter types, ranges, and enums —
    no hardcoded domain knowledge in this function.
    """
    import random

    prior = []
    top_path = DOMAIN_PATH / "top_candidates.json"
    if top_path.exists():
        try:
            prior = [e["strategy"] for e in json.loads(top_path.read_text())]
        except Exception:
            pass

    schema_props = CANDIDATE_SCHEMA.get("properties", {})

    def random_candidate() -> dict:
        c = {}
        for key, spec in schema_props.items():
            if spec.get("type") == "number":
                lo = spec.get("minimum", 0.0)
                hi = spec.get("maximum", 1.0)
                c[key] = round(random.uniform(lo, hi), 4)
            elif spec.get("type") == "integer":
                lo = spec.get("minimum", 0)
                hi = spec.get("maximum", 10)
                c[key] = random.randint(lo, hi)
            elif "enum" in spec:
                c[key] = random.choice(spec["enum"])
            else:
                c[key] = None
        return c

    def mutate(base: dict) -> dict:
        child = dict(base)
        for key, spec in schema_props.items():
            if random.random() < 0.3:  # mutate each param with 30% probability
                if spec.get("type") == "number":
                    lo = spec.get("minimum", 0.0)
                    hi = spec.get("maximum", 1.0)
                    noise = (hi - lo) * 0.15 * random.gauss(0, 1)
                    child[key] = round(max(lo, min(hi, base[key] + noise)), 4)
                elif spec.get("type") == "integer":
                    lo = spec.get("minimum", 0)
                    hi = spec.get("maximum", 10)
                    delta = random.choice([-2, -1, 1, 2])
                    child[key] = max(lo, min(hi, base[key] + delta))
                elif "enum" in spec:
                    child[key] = random.choice(spec["enum"])
        return child

    def crossover(a: dict, b: dict) -> dict:
        return {k: (a[k] if random.random() < 0.5 else b[k]) for k in schema_props}

    if not prior:
        return [random_candidate() for _ in range(n)]

    candidates = []
    # Elitism: keep top 2 winners unchanged
    candidates.extend(prior[:2])
    # Mutations of top winners
    for p in prior[:3]:
        candidates.extend([mutate(p) for _ in range(2)])
    # Crossovers between top winners
    if len(prior) >= 2:
        candidates.extend([crossover(prior[0], prior[i % len(prior)]) for i in range(1, 3)])
    # Fill remainder with random exploration
    while len(candidates) < n:
        candidates.append(random_candidate())

    return candidates[:n]
