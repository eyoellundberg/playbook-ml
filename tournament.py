"""
tournament.py — Core tournament engine.

Domains expose:
    simulate(candidate, state) -> float       required
    random_state() -> dict                    required
    CANDIDATE_SCHEMA: dict                    required
    METRIC_NAME: str                          required
    build_context(state) -> dict              optional
    prepare_state(state) -> dict              optional
    is_event(state) -> bool                   optional

Usage:
    from tournament import run_batch
    result = run_batch(domain_path, n_rounds=200, use_brain=True, workers=4)
"""

import heapq
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from utils import normalize_playbook_entry, random_candidate_from_schema

_SIM = None


def _init_worker(domain_path_str: str):
    """Initialize a worker process with the domain's simulation module."""
    global _SIM
    if domain_path_str not in sys.path:
        sys.path.insert(0, domain_path_str)
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    _SIM = importlib.import_module("simulation")


def _score_round(args):
    """Score all candidates against one state. Module-level for multiprocessing pickling."""
    state, candidates, candidate_names = args
    prepare_state = getattr(_SIM, "prepare_state", None)
    if prepare_state is not None:
        state = prepare_state(state)
    all_results = [(_SIM.simulate(c, state), c, n) for c, n in zip(candidates, candidate_names)]
    return heapq.nlargest(4, all_results, key=lambda x: x[0])


# ── Playbook helpers ──────────────────────────────────────────────────────────

def _load_playbook(domain_path: Path) -> list:
    p = domain_path / "playbook.jsonl"
    if not p.exists():
        return []
    return [normalize_playbook_entry(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def _save_playbook(domain_path: Path, playbook: list):
    with open(domain_path / "playbook.jsonl", "w") as f:
        for entry in playbook:
            f.write(json.dumps(entry) + "\n")


def _context_tags(context: dict) -> list[str]:
    """Extract coarse categorical tags from a context dict."""
    tags = []
    for key, value in context.items():
        if isinstance(value, bool):
            tags.append(f"{key}:{value}")
        elif isinstance(value, str) and value.strip():
            tags.append(f"{key}:{value.strip()}")
    return tags


def _select_top_candidates(
    candidate_strategies: dict,
    selection_scores: dict,
    tag_candidate_wins: dict,
    tag_totals: dict,
    limit: int = 4,
) -> list:
    """
    Preserve both overall winners and categorical specialists.
    Prevents the engine from collapsing to one average performer.
    """
    selected = []
    selected_names = set()

    def add_candidate(name: str, score: int, *, specialist_for: str | None = None, support: int | None = None):
        if name in selected_names or name not in candidate_strategies:
            return
        row = {"name": name, "strategy": candidate_strategies[name], "wins": score}
        if specialist_for:
            row["specialist_for"] = specialist_for
        if support is not None:
            row["support"] = support
        selected.append(row)
        selected_names.add(name)

    overall_sorted = sorted(selection_scores.items(), key=lambda item: item[1], reverse=True)
    for name, score in overall_sorted[:2]:
        add_candidate(name, score)

    tag_rank = sorted(tag_totals.items(), key=lambda item: (-item[1], item[0]))
    for tag, total in tag_rank:
        if len(selected) >= limit:
            break
        if total < 3:
            continue
        wins_by_name = tag_candidate_wins.get(tag, {})
        if not wins_by_name:
            continue
        specialist_name, specialist_support = max(
            wins_by_name.items(),
            key=lambda item: (item[1], selection_scores.get(item[0], 0)),
        )
        if specialist_name in selected_names:
            continue
        if specialist_support < 2:
            continue
        if specialist_support / total < 0.35:
            continue
        add_candidate(
            specialist_name,
            selection_scores.get(specialist_name, specialist_support),
            specialist_for=tag,
            support=specialist_support,
        )

    for name, score in overall_sorted:
        if len(selected) >= limit:
            break
        add_candidate(name, score)

    return selected[:limit]


# ── Stage 1: evolutionary candidate generation ────────────────────────────────

def _generate_procedural_candidates(domain_path: Path, n: int = 16) -> list:
    """
    Evolutionary Stage 1 candidate generation. No API calls.

    Reads top_candidates.json from the prior batch and evolves from them:
    - Elitism: keep top 2 winners unchanged
    - Mutation: gaussian noise on numeric params, occasional enum flip
    - Crossover: mix parameters from two winners
    - Exploration: random candidates to prevent premature convergence

    First batch (no prior winners): purely random.
    """
    import random

    schema = _SIM.CANDIDATE_SCHEMA
    schema_props = schema.get("properties", {})

    def random_candidate() -> dict:
        return random_candidate_from_schema(schema)

    def mutate(base: dict) -> dict:
        child = dict(base)
        for key, spec in schema_props.items():
            if random.random() < 0.3:
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

    prior = []
    top_path = domain_path / "top_candidates.json"
    if top_path.exists():
        try:
            prior = [e["strategy"] for e in json.loads(top_path.read_text())]
        except Exception:
            pass

    if not prior:
        return [random_candidate() for _ in range(n)]

    candidates = []
    candidates.extend(prior[:2])
    for p in prior[:3]:
        candidates.extend([mutate(p) for _ in range(2)])
    if len(prior) >= 2:
        candidates.extend([crossover(prior[0], prior[i % len(prior)]) for i in range(1, 3)])
    while len(candidates) < n:
        candidates.append(random_candidate())
    return candidates[:n]


# ── Main batch runner ─────────────────────────────────────────────────────────

def run_batch(
    domain_path,
    n_rounds: int,
    generation_offset: int = 0,
    hints: list = None,
    use_brain: bool = False,
    workers: int = 1,
    run_id: str = "",
    adversarial_states: list = None,
) -> dict:
    """
    Run one batch of the tournament.

    domain_path:       path to the domain folder
    n_rounds:          how many rounds to run
    generation_offset: global round counter offset (for logging)
    hints:             director hints from the prior batch (Stage 2)
    use_brain:         True → Stage 2 (Sonnet generates archetype library)
    workers:           parallel simulation workers

    Returns a result dict consumed by the director call.
    """
    global _SIM

    hints = hints or []
    domain_path = Path(domain_path)
    os.chdir(domain_path)

    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    existing = sys.modules.get("simulation")
    if existing and getattr(existing, "_synthetic", False):
        _SIM = existing  # SDK-injected module — don't reload from disk
    else:
        if "simulation" in sys.modules:
            del sys.modules["simulation"]
        _SIM = importlib.import_module("simulation")

    build_context = getattr(_SIM, "build_context", None) or (lambda s: dict(s))
    is_event      = getattr(_SIM, "is_event", None)

    playbook  = _load_playbook(domain_path)
    log_path  = domain_path / "tournament_log.jsonl"

    # ── Stage 2: generate archetype library via Sonnet ────────────────────────
    archetypes = []
    if use_brain:
        from brain import call_library
        archetypes = call_library(
            playbook=playbook,
            hints=hints,
            domain_path=domain_path,
            candidate_schema=_SIM.CANDIDATE_SCHEMA,
        )

    # ── Stage 1: generate procedural candidates once per batch ───────────────
    stage1_candidates = []
    stage1_names      = []
    if not archetypes:
        stage1_candidates = _generate_procedural_candidates(domain_path, n=16)
        stage1_names      = [f"procedural_{j}" for j in range(len(stage1_candidates))]

    if archetypes:
        candidates      = [a["strategy"] for a in archetypes]
        candidate_names = [a["name"]     for a in archetypes]
    else:
        candidates      = stage1_candidates
        candidate_names = stage1_names

    # ── Generate all states and contexts ─────────────────────────────────────
    scores                   = []
    archetype_wins           = {}
    archetype_wins_event     = {}
    archetype_wins_nonevent  = {}
    context_mix              = {}
    tag_candidate_wins       = defaultdict(lambda: defaultdict(int))
    tag_totals               = defaultdict(int)

    adversarial = adversarial_states or []
    n_adv = min(len(adversarial), n_rounds // 5)
    if n_adv > 0:
        import random as _random
        states = adversarial[:n_adv] + [_SIM.random_state() for _ in range(n_rounds - n_adv)]
        _random.shuffle(states)
    else:
        states = [_SIM.random_state() for _ in range(n_rounds)]
    contexts = [build_context(s) for s in states]

    # ── Score all rounds ──────────────────────────────────────────────────────
    score_args = [(state, candidates, candidate_names) for state in states]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(domain_path),),
        ) as pool:
            all_scored = list(pool.map(_score_round, score_args))
    else:
        all_scored = [_score_round(a) for a in score_args]

    # ── Process results ───────────────────────────────────────────────────────
    from brain import extract_principles_ai

    with open(log_path, "a") as log_file:
        for i, (scored, state, context) in enumerate(zip(all_scored, states, contexts)):
            round_num    = generation_offset + i + 1
            winner_score, winner, winner_name = scored[0]
            losers       = [c for _, c, _ in scored[1:4]]
            score_margin = round(scored[0][0] - scored[1][0], 4) if len(scored) > 1 else 0.0

            scores.append(winner_score)

            entry = {
                "round":        round_num,
                "score":        winner_score,
                "score_margin": score_margin,
                "metric":       _SIM.METRIC_NAME,
                "winner":       winner,
                "state":        state,
                "archetype":    winner_name,
                "run_id":       run_id,
                "contenders": [
                    {
                        "name": name,
                        "score": round(score, 4),
                        "strategy": candidate,
                    }
                    for score, candidate, name in scored
                ],
            }
            log_file.write(json.dumps(entry) + "\n")

            if winner_name not in archetype_wins:
                archetype_wins[winner_name]          = 0
                archetype_wins_event[winner_name]    = 0
                archetype_wins_nonevent[winner_name] = 0
            archetype_wins[winner_name] += 1
            if is_event is not None and is_event(state):
                archetype_wins_event[winner_name] += 1
            else:
                archetype_wins_nonevent[winner_name] += 1

            for k, v in context.items():
                key = f"{k}:{v}"
                context_mix[key] = context_mix.get(key, 0) + 1

            for tag in _context_tags(context):
                tag_totals[tag] += 1
                tag_candidate_wins[tag][winner_name] += 1

            if use_brain and round_num % 10 == 0:
                playbook = extract_principles_ai(
                    winner=winner,
                    losers=losers,
                    context=context,
                    score=winner_score,
                    generation=round_num,
                    existing=playbook,
                    domain_path=domain_path,
                )
                _save_playbook(domain_path, playbook)

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
            (domain_path / "champion_archetype.json").write_text(
                json.dumps(champion_data, indent=2)
            )

    # ── Save top candidates for Stage 1 evolution ────────────────────────────
    if archetypes:
        selection_scores = {
            a["name"]: archetype_wins_nonevent.get(a["name"], 0) * 2
            + archetype_wins_event.get(a["name"], 0)
            for a in archetypes
        }
        top_candidates = _select_top_candidates(
            candidate_strategies={a["name"]: a["strategy"] for a in archetypes},
            selection_scores=selection_scores,
            tag_candidate_wins=tag_candidate_wins,
            tag_totals=tag_totals,
            limit=4,
        )
        (domain_path / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))
    elif stage1_candidates:
        selection_scores = {
            stage1_names[j]: archetype_wins_nonevent.get(stage1_names[j], 0)
            + archetype_wins.get(stage1_names[j], 0)
            for j in range(len(stage1_candidates))
        }
        top_candidates = _select_top_candidates(
            candidate_strategies={stage1_names[j]: stage1_candidates[j] for j in range(len(stage1_candidates))},
            selection_scores=selection_scores,
            tag_candidate_wins=tag_candidate_wins,
            tag_totals=tag_totals,
            limit=4,
        )
        (domain_path / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))

    # ── Build result dict for director ────────────────────────────────────────
    n       = len(scores)
    quarter = max(1, n // 4)
    first_q_avg = sum(scores[:quarter]) / quarter
    last_q_avg  = sum(scores[-quarter:]) / quarter
    trend_pct   = (
        (last_q_avg - first_q_avg) / abs(first_q_avg) * 100 if first_q_avg else 0.0
    )
    top_principles = sorted(
        playbook, key=lambda p: p.get("confidence", 0), reverse=True
    )[:5]

    return {
        "n_rounds":               n,
        "avg_score":              round(sum(scores) / n, 2) if n else 0,
        "best_score":             round(max(scores), 2)     if scores else 0,
        "worst_score":            round(min(scores), 2)     if scores else 0,
        "trend_pct":              round(trend_pct, 1),
        "score_last_10":          [round(s, 2) for s in scores[-10:]],
        "context_mix":            {
            k: v
            for k, v in sorted(context_mix.items(), key=lambda x: -x[1])[:10]
        },
        "playbook_size":          len(playbook),
        "top_principles":         top_principles,
        "archetype_wins":         archetype_wins,
        "archetype_wins_event":   archetype_wins_event,
        "archetype_wins_nonevent": archetype_wins_nonevent,
    }
