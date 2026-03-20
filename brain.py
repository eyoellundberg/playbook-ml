"""
brain.py — Archetype library, adversarial scenarios, principle extraction.

call_library():          Generate 16 named strategy archetypes for Stage 2.
call_adversarial():      Generate targeted scenarios for the champion's weak spots.
extract_principles_ai(): Extract conditional principles from a tournament round.
"""

import json
import os
import sys
import importlib
from collections import defaultdict
from pathlib import Path

from api import structured_ai_call, ai_backend_available
from utils import (
    load_env, load_mission, load_world_model, load_hypotheses,
    normalize_confidence, normalize_playbook_entry, load_jsonl,
)

MODEL_LIBRARY  = os.environ.get("AUTOFORGE_LIBRARY_MODEL", "claude-sonnet-4-6")
MODEL_EXTRACT  = os.environ.get("AUTOFORGE_EXTRACT_MODEL", "claude-haiku-4-5-20251001")


def _sanitize_for_api(schema: dict) -> dict:
    """Strip constraints unsupported by Anthropic's structured output."""
    import copy
    schema = copy.deepcopy(schema)
    props = schema.get("properties", {})
    for spec in props.values():
        if spec.get("type") in ("integer", "number"):
            spec.pop("minimum", None)
            spec.pop("maximum", None)
    return schema


def build_library_schema(candidate_schema: dict) -> dict:
    """Wrap a domain's CANDIDATE_SCHEMA into the archetype library schema."""
    clean_schema = _sanitize_for_api(candidate_schema)
    return {
        "type": "object",
        "properties": {
            "archetypes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "philosophy": {"type": "string"},
                        "best_for":   {"type": "string"},
                        "strategy":   clean_schema,
                    },
                    "required": ["name", "philosophy", "best_for", "strategy"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["archetypes"],
        "additionalProperties": False,
    }


def _load_champion(domain_path: Path) -> dict:
    p = domain_path / "champion_archetype.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_reference_strategies(domain_path: Path) -> list:
    strategies = []
    seen = set()

    champion = _load_champion(domain_path)
    if champion.get("strategy"):
        serialized = json.dumps(champion["strategy"], sort_keys=True)
        seen.add(serialized)
        strategies.append({"name": champion.get("name", "champion"), "strategy": champion["strategy"]})

    top_path = domain_path / "top_candidates.json"
    if top_path.exists():
        try:
            top = json.loads(top_path.read_text())
            for item in top[:4]:
                strategy = item.get("strategy")
                if not strategy:
                    continue
                serialized = json.dumps(strategy, sort_keys=True)
                if serialized in seen:
                    continue
                seen.add(serialized)
                strategies.append({"name": item.get("name", "candidate"), "strategy": strategy})
        except Exception:
            pass

    return strategies


def _load_eval_feedback(domain_path: Path) -> list[str]:
    """Summarize current eval failures so the next library can target them."""
    evals_path = domain_path / "evals" / "scenarios.jsonl"
    if not evals_path.exists():
        return []

    strategies = _load_reference_strategies(domain_path)
    if not strategies:
        return []

    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))

    try:
        sim = importlib.import_module("simulation")
    except Exception:
        return []

    scenarios = load_jsonl(evals_path, skip_comments=True)
    if not scenarios:
        return []

    failures = []
    passed = 0
    for scenario in scenarios:
        state = scenario.get("state", {})
        if not state:
            continue
        scored = [
            (sim.simulate(item["strategy"], state), item.get("name", "candidate"))
            for item in strategies
        ]
        best_score, best_name = max(scored, key=lambda item: item[0])
        min_score = scenario.get("min_score", 0)
        if best_score >= min_score:
            passed += 1
            continue
        failures.append(
            f"- {scenario.get('id', '?')}: best current strategy `{best_name}` scored "
            f"{best_score:.2f} vs required {min_score:.2f} "
            f"({scenario.get('description', '')[:80]})"
        )

    if not failures:
        return [f"Current eval status: passing {passed}/{len(scenarios)} scenarios. Maintain breadth."]

    lines = [f"Current eval status: passing {passed}/{len(scenarios)} scenarios.", "Failed evals:"]
    lines.extend(failures[:6])
    return lines


def generate_library_prompt(playbook: list, hints: list, domain_path: Path) -> str:
    """Build the user-turn prompt for library generation."""
    champion = _load_champion(domain_path)
    lines = []

    world_model = load_world_model(domain_path)
    if world_model:
        lines += ["WORLD MODEL:", world_model, ""]
    else:
        mission_text = load_mission(domain_path)
        if mission_text:
            lines += ["MISSION:", mission_text, ""]

    if champion:
        lines += [
            f"CHAMPION ARCHETYPE FROM PRIOR BATCH "
            f"({champion.get('nonevent_wins', 0)} non-event wins / {champion.get('total_wins', 0)} total):",
            f"  Name: {champion['name']}",
            f"  Philosophy: {champion['philosophy']}",
        ]
        strat = champion.get("strategy", {})
        first_key = next(iter(strat), None)
        if first_key and isinstance(strat[first_key], dict):
            vals = strat[first_key]
            lines.append("  Strategy (first dimension): " + "  ".join(
                f"{k}:{v}" for k, v in vals.items()
            ))
        lines += [
            "This archetype consistently won on non-event rounds. Include it or a direct variant.",
            "",
        ]

    eval_feedback = _load_eval_feedback(domain_path)
    if eval_feedback:
        lines.append("EVAL FEEDBACK:")
        lines.extend(eval_feedback)
        lines.append("")

    if playbook:
        lines.append("CURRENT PLAYBOOK PRINCIPLES:")
        for p in playbook:
            cond = f"  IF {p['condition']}" if p.get("condition") else ""
            lines.append(
                f"  [{normalize_confidence(p.get('confidence', 0)):.0%}] [{p.get('context', '')}] "
                f"{p.get('principle', '')}{cond}"
            )
        lines.append("")

    if hints:
        lines.append("HINTS FROM PRIOR DIRECTOR BATCH:")
        for h in hints:
            lines.append(f"  - {h}")
        lines.append("")

    hypotheses = load_hypotheses(domain_path)
    if hypotheses.get("open"):
        lines.append("OPEN HYPOTHESES TO TEST (design archetypes that test these):")
        for h in hypotheses["open"]:
            lines.append(f"  - {h}")
        lines.append("")
    if hypotheses.get("confirmed"):
        lines.append("CONFIRMED HYPOTHESES (build on these):")
        for h in hypotheses["confirmed"][-5:]:
            lines.append(f"  - {h}")
        lines.append("")

    return "\n".join(lines)


def _clamp_strategy(strategy: dict, schema: dict) -> dict:
    """Clamp strategy values to schema bounds after API generation."""
    props = schema.get("properties", {})
    result = {}
    for key, spec in props.items():
        val = strategy.get(key)
        if val is None:
            result[key] = val
            continue
        t = spec.get("type")
        if t == "number":
            lo, hi = spec.get("minimum", float("-inf")), spec.get("maximum", float("inf"))
            result[key] = max(lo, min(hi, float(val)))
        elif t == "integer":
            lo, hi = spec.get("minimum", float("-inf")), spec.get("maximum", float("inf"))
            result[key] = max(int(lo), min(int(hi), int(round(float(val)))))
        elif "enum" in spec:
            result[key] = val if val in spec["enum"] else spec["enum"][0]
        else:
            result[key] = val
    return result


_ADVERSARIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["scenarios", "rationale"],
    "additionalProperties": False,
}


def call_adversarial(
    domain_path: Path,
    n: int = 20,
    context_mix: dict = None,
    champion: dict = None,
) -> list:
    """
    Generate adversarial scenarios targeting the champion's weak spots.
    Returns a list of state dicts to inject into the next batch.
    """
    load_env(domain_path)
    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    try:
        sim = importlib.import_module("simulation")
        sample_states = [sim.random_state() for _ in range(3)]
    except Exception:
        return []

    mission_text = load_mission(domain_path)
    rare_contexts = []
    if context_mix:
        rare_contexts = [k for k, _ in sorted(context_mix.items(), key=lambda x: x[1])[:5]]

    n_crisis = max(2, n // 5)
    n_edge   = n - n_crisis

    user_prompt = f"""Generate {n} adversarial state scenarios for this domain.

Mission: {mission_text or "(none)"}
Champion philosophy: {champion.get("philosophy", "(unknown)") if champion else "(none yet)"}

Sample states — your output must match this exact schema:
{json.dumps(sample_states, indent=2)}

REQUIRED — generate exactly {n_crisis} CRISIS / DOOMSDAY scenarios (regardless of what the \
tournament has seen so far). These must represent the worst imaginable conditions for this domain: \
catastrophic collapses, coordinated attacks, cascading failures, pandemic-level shocks, \
adversarial extremes — whatever "everything goes wrong at once" looks like here. \
Push numeric values to their worst extremes. Do not generate average or edge-case states for these slots.

REMAINING {n_edge} scenarios — target the champion's weak spots and underrepresented contexts:
{chr(10).join(f"  - {c}" for c in rare_contexts) if rare_contexts else "  (none identified — generate edge cases)"}

Return each scenario as a JSON-encoded string in the 'scenarios' array.
Each string must be valid JSON matching the schema above exactly."""

    try:
        data = structured_ai_call(
            task_name="adversarial",
            domain_path=domain_path,
            model=MODEL_LIBRARY,
            max_tokens=4096,
            system_prompt="You are generating adversarial test scenarios for a strategy evaluation engine.",
            user_prompt=user_prompt,
            schema=_ADVERSARIAL_SCHEMA,
            metadata={"n": n},
        )
        expected_keys = set(sample_states[0].keys()) if sample_states else set()
        valid = []
        for s in data.get("scenarios", []):
            try:
                state = json.loads(s)
                if isinstance(state, dict) and set(state.keys()) >= expected_keys:
                    valid.append(state)
            except Exception:
                continue
        return valid[:n]
    except Exception:
        return []


def call_library(
    playbook: list,
    hints: list,
    domain_path: Path,
    candidate_schema: dict,
) -> list:
    """
    Call Sonnet once to generate a library of 16 strategy archetypes.
    Returns list of archetype dicts: [{name, philosophy, best_for, strategy}, ...]
    """
    load_env(domain_path)

    system_text = (
        "You are an expert strategy architect. Generate a diverse, opinionated library of "
        "strategy archetypes based on the domain instructions and playbook context provided."
    )

    library_schema = build_library_schema(candidate_schema)
    prompt = generate_library_prompt(playbook, hints, domain_path)

    for attempt in range(2):
        try:
            data = structured_ai_call(
                task_name="brain",
                domain_path=domain_path,
                model=MODEL_LIBRARY,
                max_tokens=8192,
                system_prompt=system_text,
                user_prompt=prompt,
                schema=library_schema,
                metadata={"playbook_size": len(playbook), "hint_count": len(hints)},
            )
            archetypes = data["archetypes"]
            for arch in archetypes:
                arch["strategy"] = _clamp_strategy(arch["strategy"], candidate_schema)
            return archetypes
        except Exception as e:
            if attempt == 0:
                continue
            raise RuntimeError(f"Library generation failed: {e}") from e

    return []


# ── Principle extraction (Haiku, every 10 rounds) ─────────────────────────────

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "principles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic":         {"type": "string"},
                    "context":       {"type": "string"},
                    "principle":     {"type": "string"},
                    "condition":     {"type": ["string", "null"]},
                    "confidence":    {"type": "number"},
                    "rounds_tested": {"type": "integer"},
                },
                "required": ["topic", "context", "principle", "condition",
                             "confidence", "rounds_tested"],
                "additionalProperties": False,
            }
        }
    },
    "required": ["principles"],
    "additionalProperties": False,
}


def _merge_principles(new_principles: list, existing: list) -> list:
    """Merge new into existing playbook. Max 2 per topic, max 60 total."""
    by_topic: dict = defaultdict(list)
    for p in existing:
        p = normalize_playbook_entry(p)
        by_topic[p.get("topic", p["principle"][:30])].append(p)
    for p in new_principles:
        p = normalize_playbook_entry(p)
        by_topic[p["topic"]].append(p)

    kept = []
    for entries in by_topic.values():
        entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        kept.extend(entries[:2])

    kept.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return kept[:60]


def extract_principles_ai(
    winner: dict,
    losers: list,
    context: dict,
    score: float,
    generation: int,
    existing: list,
    domain_path: Path = None,
) -> list:
    """
    Call Haiku to extract 0-2 principles from a tournament round.
    Falls back to returning existing playbook unchanged on any error.
    """
    if domain_path is None:
        domain_path = Path(__file__).parent

    load_env(domain_path)

    rt_path = domain_path / "retired_topics.json"
    retired = json.loads(rt_path.read_text()) if rt_path.exists() else []

    world_model = load_world_model(domain_path)
    if world_model:
        system_prompt = (
            "You are extracting conditional principles from tournament results.\n\n"
            "WORLD MODEL:\n" + world_model
        )
    else:
        extract_prompt_path = domain_path / "prompts" / "extract.md"
        if not extract_prompt_path.exists():
            return existing
        system_prompt = extract_prompt_path.read_text().strip()
        mission_text = load_mission(domain_path)
        if mission_text:
            system_prompt = "MISSION:\n" + mission_text + "\n\n" + system_prompt

    system_prompt = system_prompt.replace(
        "{{RETIRED_TOPICS}}",
        json.dumps(retired) if retired else "(none yet)"
    )

    def compact(strategy: dict) -> str:
        return json.dumps(strategy, separators=(",", ":"))

    loser_lines   = "\n".join(f"LOSER {i+1}: {compact(l)}" for i, l in enumerate(losers[:3]))
    context_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
    playbook_summary = "\n".join(
        f"  [{e.get('topic')}] {e.get('principle', '')[:70]}  (conf {normalize_confidence(e.get('confidence', 0)):.0%})"
        for e in existing
    ) or "  (empty)"

    user_content = f"""ROUND {generation}

WINNER: {compact(winner)}
{loser_lines}

SCORE: {score}

CONTEXT:
{context_lines}

EXISTING PLAYBOOK:
{playbook_summary}
"""

    if not ai_backend_available():
        return existing

    for attempt in range(2):
        try:
            data = structured_ai_call(
                task_name="extract",
                domain_path=domain_path,
                model=MODEL_EXTRACT,
                max_tokens=512,
                system_prompt=system_prompt,
                user_prompt=user_content,
                schema=_EXTRACT_SCHEMA,
                metadata={"generation": generation, "score": score},
            )
            new_principles = data.get("principles", [])
            new_principles = [p for p in new_principles if p["topic"] not in retired]
            return _merge_principles(new_principles, existing)

        except Exception as e:
            if attempt == 0:
                continue
            print(f"  [extract warning] Haiku call failed: {e}")
            return existing

    return existing
