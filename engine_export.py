"""
engine_export.py — Stage 3 training data exporter.

Reads tournament_log.jsonl and formats it as instruction-tuning JSONL
for fine-tuning a local model. The local model learns:
  given scenario + playbook context -> propose strategy parameters

Output format: messages JSONL (compatible with OpenAI fine-tuning + MLX-LM)
For numerical domains: also exports training_features.csv (XGBoost/sklearn)

Usage (via run.py):
  python run.py export --domain MyDomain
"""

import json
import csv
import statistics
from pathlib import Path


# Keys that are never the score field — used for backward-compat detection
_NON_SCORE_KEYS = {"round", "winner", "state", "archetype", "metric"}


def _detect_score(entry: dict) -> float:
    """
    Return the score from a log entry.
    Tries 'score' first (new format), then falls back to detecting the score
    key by excluding known non-score keys (old format used METRIC_NAME as key).
    """
    if "score" in entry:
        return float(entry["score"])
    for key, value in entry.items():
        if key not in _NON_SCORE_KEYS:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    raise KeyError(f"Cannot find score in log entry: {list(entry.keys())}")


def _format_scenario(state: dict) -> str:
    """Format a state dict as a readable scenario block."""
    lines = []
    for k, v in state.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _format_playbook_context(playbook: list) -> str:
    """Format playbook as bullet points for system context."""
    if not playbook:
        return "(no principles yet)"
    lines = []
    for p in playbook:
        conf = p.get("confidence", 0)
        ctx  = p.get("context", "")
        text = p.get("principle", "")
        lines.append(f"- [{conf:.0%}] [{ctx}] {text}")
    return "\n".join(lines)


def _detect_domain_type(domain_path: Path) -> str:
    """
    Returns "numerical" if all CANDIDATE_SCHEMA params are numbers/integers/enums,
    "language" if any params are free-form strings.
    Defaults to "numerical" if simulation.py can't be imported.
    """
    import sys as _sys
    _sys.path.insert(0, str(domain_path))
    try:
        import importlib
        sim = importlib.import_module("simulation")
        schema = getattr(sim, "CANDIDATE_SCHEMA", {})
        props  = schema.get("properties", {})
        for spec in props.values():
            t = spec.get("type")
            if t == "string" and "enum" not in spec:
                return "language"
        return "numerical"
    except Exception:
        return "numerical"
    finally:
        _sys.path.pop(0)


def export_training_data(domain_path: Path):
    """
    Export tournament_log.jsonl as instruction-tuning JSONL for fine-tuning.

    Quality filter: only keep rounds where score >= median score across all rounds.
    Each kept round becomes one training example:
      system: domain context + playbook principles
      user:   scenario (state dict)
      assistant: winner strategy (JSON)

    For numerical domains: also exports training_features.csv (XGBoost/sklearn ready).

    Output: domain_path/training_data.jsonl (and optionally training_features.csv)
    """
    log_path = domain_path / "tournament_log.jsonl"
    pb_path  = domain_path / "playbook.jsonl"
    out_path = domain_path / "training_data.jsonl"

    if not log_path.exists():
        print(f"No tournament_log.jsonl found in {domain_path}")
        print("Run some batches first before exporting.")
        return

    # Load all log entries
    raw_lines = [l.strip() for l in log_path.read_text().splitlines() if l.strip()]
    if not raw_lines:
        print("tournament_log.jsonl is empty — nothing to export.")
        return

    entries = []
    for line in raw_lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    total_rounds = len(entries)

    # Detect scores and filter
    scored = []
    for entry in entries:
        try:
            score = _detect_score(entry)
            scored.append((score, entry))
        except (KeyError, ValueError):
            continue

    if not scored:
        print("Could not find score values in tournament_log.jsonl — check log format.")
        return

    all_scores = [s for s, _ in scored]
    median_score = statistics.median(all_scores)

    kept = [(s, e) for s, e in scored if s >= median_score]
    print(f"Total rounds: {total_rounds}")
    print(f"After quality filter (score >= median {median_score:.2f}): {len(kept)} kept")

    # Detect domain type from CANDIDATE_SCHEMA
    domain_type = _detect_domain_type(domain_path)

    # Load playbook for context
    playbook = []
    if pb_path.exists():
        playbook = [json.loads(l) for l in pb_path.read_text().splitlines() if l.strip()]
    playbook_context = _format_playbook_context(playbook)

    system_content = (
        f"You are a strategy expert for this domain.\n\n"
        f"Playbook principles:\n{playbook_context}"
    )

    # Write messages JSONL (always written for both domain types)
    written = 0
    with open(out_path, "w") as f:
        for score, entry in kept:
            state  = entry.get("state", {})
            winner = entry.get("winner", {})

            if not state or not winner:
                continue

            example = {
                "messages": [
                    {"role": "system",    "content": system_content},
                    {"role": "user",      "content": f"Scenario:\n{_format_scenario(state)}"},
                    {"role": "assistant", "content": json.dumps(winner)},
                ]
            }
            f.write(json.dumps(example) + "\n")
            written += 1

    if domain_type == "numerical":
        # Build CSV: state keys + winner keys + score column
        # Collect all state and winner keys from kept entries
        state_keys  = []
        winner_keys = []
        for _, entry in kept:
            state  = entry.get("state", {})
            winner = entry.get("winner", {})
            for k in state:
                if k not in state_keys:
                    state_keys.append(k)
            for k in winner:
                if k not in winner_keys:
                    winner_keys.append(k)

        csv_path = domain_path / "training_features.csv"
        fieldnames = state_keys + winner_keys + ["score"]
        n_cols = len(fieldnames)

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            csv_rows = 0
            for score, entry in kept:
                state  = entry.get("state", {})
                winner = entry.get("winner", {})
                if not state or not winner:
                    continue
                row = {}
                for k in state_keys:
                    row[k] = state.get(k, "")
                for k in winner_keys:
                    row[k] = winner.get(k, "")
                row["score"] = score
                writer.writerow(row)
                csv_rows += 1

        domain_name = domain_path.name
        print(f"\nDomain type: numerical")
        print(f"  training_features.csv  — {csv_rows} rows, {n_cols} columns (use with XGBoost/sklearn)")
        print(f"  training_data.jsonl    — {written} examples (use with MLX-LM if preferred)")
        print(f"""
For XGBoost:
  import pandas as pd, xgboost as xgb
  df = pd.read_csv('{domain_name}/training_features.csv')
  # features: all columns except the last (score)
  # label: last column (score)""")

    else:
        domain_name = domain_path.name
        print(f"\nDomain type: language (free-form string params detected)")
        print(f"  training_data.jsonl — {written} examples")
        print(f"  Fine-tune: mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --data {domain_name}/ --train")
