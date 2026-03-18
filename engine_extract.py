"""
engine_extract.py — Generic AI-driven principle extractor for The Engine.

Any domain with a prompts/extract.md gets intelligent principle extraction.
Called every 10 rounds by the domain's tournament.py. ~$0.002/call at Haiku pricing.

The domain is responsible for building the context dict and serializing strategies.
This file knows nothing about hotels, trading, crops, or any other domain.
"""

import json
import os
from collections import defaultdict
from pathlib import Path


MODEL_EXTRACT = os.environ.get("ENGINE_EXTRACT_MODEL", "claude-haiku-4-5-20251001")


def _load_env(domain_path: Path):
    """Load .env from domain folder, then engine root as fallback."""
    for env in [domain_path / ".env", domain_path.parent / ".env"]:
        if env.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
            for line in env.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ── Structured output schema ───────────────────────────────────────────────────

EXTRACT_SCHEMA = {
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


# ── Playbook merge / dedup / cap ───────────────────────────────────────────────

def _merge(new_principles: list, existing: list) -> list:
    """
    Merge new principles into existing playbook.
    Key: topic — same topic updates rather than duplicates.
    Max 2 per topic, max 60 total. Higher confidence wins on collision.
    """
    by_topic: dict = defaultdict(list)
    for p in existing:
        by_topic[p.get("topic", p["principle"][:30])].append(p)
    for p in new_principles:
        by_topic[p["topic"]].append(p)

    kept = []
    for entries in by_topic.values():
        entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        kept.extend(entries[:2])

    kept.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return kept[:60]


# ── Core extractor ─────────────────────────────────────────────────────────────

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

    winner:      candidate strategy dict (domain-specific structure)
    losers:      list of candidate strategy dicts
    context:     flat key-value pairs describing the scenario — domain builds this
    score:       the metric value for the winning candidate (higher = better)
    generation:  round number (used as rounds_tested seed)
    existing:    current playbook list
    domain_path: path to the domain folder (reads prompts/extract.md + retired_topics.json)
    """
    if domain_path is None:
        domain_path = Path(__file__).parent

    _load_env(domain_path)

    rt_path = domain_path / "retired_topics.json"
    retired = json.loads(rt_path.read_text()) if rt_path.exists() else []

    extract_prompt_path = domain_path / "prompts" / "extract.md"
    if not extract_prompt_path.exists():
        return existing

    system_prompt = extract_prompt_path.read_text().strip()
    system_prompt = system_prompt.replace(
        "{{RETIRED_TOPICS}}",
        json.dumps(retired) if retired else "(none yet)"
    )

    def compact(strategy: dict) -> str:
        return json.dumps(strategy, separators=(",", ":"))

    top_losers  = losers[:3]
    loser_lines = "\n".join(
        f"LOSER {i+1}: {compact(l)}" for i, l in enumerate(top_losers)
    )

    context_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())

    playbook_summary = "\n".join(
        f"  [{e.get('topic')}] {e.get('principle', '')[:70]}  (conf {e.get('confidence', 0):.0%})"
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

    # Skip silently if no API key or anthropic not installed
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return existing

    for attempt in range(2):
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=MODEL_EXTRACT,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": EXTRACT_SCHEMA,
                    }
                },
            )
            data = json.loads(response.content[0].text)
            new_principles = data.get("principles", [])

            new_principles = [p for p in new_principles if p["topic"] not in retired]

            return _merge(new_principles, existing)

        except Exception as e:
            if attempt == 0:
                continue
            print(f"  [extract warning] Haiku call failed: {e}")
            return existing

    return existing
