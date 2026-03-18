"""
engine_brain.py — Generic archetype library generator for Autoforge.

Any domain with a prompts/brain.md and simulation.CANDIDATE_SCHEMA gets a
fully functional strategy library generator. Replaces hardcoded brain.py
files in individual domains.

Uses Sonnet for creative, opinionated archetype generation.
"""

import json
import os
import sys
import importlib
from pathlib import Path

from commands.shared import load_env, load_mission, normalize_confidence, structured_ai_call

MODEL_LIBRARY = os.environ.get("AUTOFORGE_LIBRARY_MODEL", "claude-sonnet-4-6")


def _sanitize_for_api(schema: dict) -> dict:
    """
    Strip constraints unsupported by Anthropic's structured output from a JSON schema.
    Currently: 'integer' type does not allow minimum/maximum.
    """
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
    """Load the champion archetype from the prior batch, if one exists."""
    p = domain_path / "champion_archetype.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_reference_strategies(domain_path: Path) -> list:
    """Load current champion/top candidates for eval feedback."""
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
    """
    Summarize current eval failures so the next library can target them.
    """
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

    scenarios = []
    for line in evals_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            scenarios.append(json.loads(line))
        except json.JSONDecodeError:
            continue

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
    mission_text = load_mission(domain_path)

    if mission_text:
        lines += [
            "MISSION:",
            mission_text,
            "",
        ]

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

    brain_md = domain_path / "prompts" / "brain.md"
    if brain_md.exists():
        lines.append(brain_md.read_text().strip())

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

    user_prompt = f"""Generate {n} adversarial state scenarios for this domain.

Mission: {mission_text or "(none)"}
Champion philosophy: {champion.get("philosophy", "(unknown)") if champion else "(none yet)"}

Sample states — your output must match this exact schema:
{json.dumps(sample_states, indent=2)}

Underrepresented contexts from recent tournament (generate more of these):
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
    Call Sonnet once to generate a library of strategy archetypes.

    playbook:         current playbook list
    hints:            director hints from prior batch
    domain_path:      path to domain folder (reads prompts/brain.md + champion)
    candidate_schema: domain's CANDIDATE_SCHEMA (what a strategy looks like)

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
