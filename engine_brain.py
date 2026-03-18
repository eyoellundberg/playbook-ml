"""
engine_brain.py — Generic archetype library generator for The Engine.

Any domain with a prompts/brain.md and simulation.CANDIDATE_SCHEMA gets a
fully functional strategy library generator. Replaces hardcoded brain.py
files in individual domains.

Uses Sonnet for creative, opinionated archetype generation.
"""

import json
import os
from pathlib import Path


MODEL_LIBRARY = os.environ.get("ENGINE_LIBRARY_MODEL", "claude-sonnet-4-6")


def _load_env(domain_path: Path):
    """Load .env from domain folder, then engine root as fallback."""
    for env in [domain_path / ".env", domain_path.parent / ".env"]:
        if env.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
            for line in env.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


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


def generate_library_prompt(playbook: list, hints: list, domain_path: Path) -> str:
    """Build the user-turn prompt for library generation."""
    champion = _load_champion(domain_path)
    lines = []

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

    if playbook:
        lines.append("CURRENT PLAYBOOK PRINCIPLES:")
        for p in playbook:
            cond = f"  IF {p['condition']}" if p.get("condition") else ""
            lines.append(
                f"  [{p.get('confidence', 0):.0%}] [{p.get('context', '')}] "
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
    _load_env(domain_path)

    system_text = (
        "You are an expert strategy architect. Generate a diverse, opinionated library of "
        "strategy archetypes based on the domain instructions and playbook context provided."
    )

    library_schema = build_library_schema(candidate_schema)
    prompt = generate_library_prompt(playbook, hints, domain_path)

    for attempt in range(2):
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=MODEL_LIBRARY,
                max_tokens=8192,
                system=system_text,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": library_schema,
                    }
                },
            )
            data = json.loads(response.content[0].text)
            return data["archetypes"]
        except Exception as e:
            if attempt == 0:
                continue
            raise RuntimeError(f"Library generation failed: {e}") from e

    return []
