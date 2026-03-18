"""
run.py — Autoforge CLI.

Five commands that cover the full lifecycle: scaffold a domain, run the
tournament, inspect state, export training data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bootstrap <Domain> --description "..."
    Generate a complete domain from a natural language description.
    Sonnet writes simulation.py, all three prompt files, and adapts
    tournament.py's _build_context() automatically. One-time cost ~$0.05.
    Always review simulation.py before running — it is the only thing
    the engine ever learns from.

    python run.py bootstrap GrainMarketing \
      --description "corn/soy marketing, midwest US farms"

new <Domain>
    Scaffold a new domain by copying template/ verbatim.
    Use this when you want to write simulation.py yourself from scratch.
    Produces an empty skeleton with comments explaining each required export.

    python run.py new GrainMarketing

calibrate --domain <Domain> [--n N]
    Sample N random scenarios, show distributions for each state key, score
    8 random candidates, and flag calibration issues (dominant strategies,
    zero-heavy scores, low variance). Run this after editing simulation.py
    before committing to a long run.

    python run.py calibrate --domain GrainMarketing
    python run.py calibrate --domain GrainMarketing --n 1000

run --domain <Domain> [options]
    Run the tournament. Loads tournament.py from the domain folder and
    calls run_batch() for each batch. Between batches, the director AI
    (Sonnet) reads results, appends to thinking_log.md, and sets hints
    for the next batch's archetype generation.

    Modes:
      (default)   Stage 1 — evolutionary mutation, no API calls.
                  _generate_procedural_candidates() evolves from prior batch
                  winners using elitism, mutation, crossover, and random fill.
                  Playbook grows from Haiku extraction every 10 rounds.

      --brain     Stage 2 — Sonnet generates 16 named strategy archetypes
                  per batch. Each has a philosophy, not just parameters.
                  Haiku extracts principles every 10 rounds. Director reads
                  results and retires losers between batches. Champion
                  archetype propagates to seed the next batch. ~$0.50/run.

      --auto      Stage 1 until playbook flatlines for 2 consecutive batches,
                  then auto-promotes to Stage 2. Fully autonomous overnight run.

    Options:
      --batches N   batches to run per year (default: 8)
      --rounds N    rounds per batch (default: 200)
      --years N     years to run — playbook carries across years (default: 1)

    Examples:
      python run.py run --domain GrainMarketing --batches 10 --rounds 150
      python run.py run --domain GrainMarketing --brain --batches 8 --rounds 150
      python run.py run --domain GrainMarketing --auto --batches 20 --rounds 150
      python run.py run --domain GrainMarketing --brain --years 2 --batches 5

    Stops early if the director returns verdict "saturated" (success) or
    "reward_hacking" (stop and fix the sim). Writes last_run.json on exit.

export --domain <Domain>
    Export Stage 3 training data from tournament_log.jsonl.
    Quality filter: keeps rounds scoring >= median across all rounds.
    Output: training_data.jsonl — messages-format JSONL (system/user/assistant).

      system:    domain context + playbook principles
      user:      scenario state dict
      assistant: winning strategy JSON

    Compatible with MLX-LM (fine-tune Qwen on Apple Silicon) and
    OpenAI-compatible fine-tuning endpoints.

    python run.py export --domain GrainMarketing

status --domain <Domain>
    Show current domain state in a Rich terminal table:
      - Top 5 playbook principles by confidence
      - Champion archetype name and philosophy
      - Retired topics (permanent blocklist)
      - Last run timestamp, round count, final verdict
      - tournament_log.jsonl size + training_data.jsonl status

    python run.py status --domain GrainMarketing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTOR VERDICTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  converging        score and playbook improving — keep running
  exploring         mixed results, still searching — keep running
  stalled           no improvement — adjust sim or prompts
  reward_hacking    score rising for wrong reason — stop, fix the sim
  needs_calibration sim rewarding wrong behavior — fix simulation.py
  saturated         playbook full, score stable — export and train

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CONFIGURATION (optional, set in MyDomain/.env)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ENGINE_DIRECTOR_MODEL   between-batch director  (default: claude-sonnet-4-6)
  ENGINE_LIBRARY_MODEL    archetype generation    (default: claude-sonnet-4-6)
  ENGINE_EXTRACT_MODEL    principle extraction    (default: claude-haiku-4-5-20251001)

  Swap Sonnet for Haiku everywhere to cut cost ~10x.
  Use Opus for the director for more rigorous analysis.
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

ENGINE_ROOT = Path(__file__).parent


# ── Director schema ────────────────────────────────────────────────────────────

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["converging", "exploring", "stalled", "reward_hacking", "needs_calibration", "saturated"],
        },
        "observations": {
            "type": "array", "items": {"type": "string"},
        },
        "principles_gaining_confidence": {
            "type": "array", "items": {"type": "string"},
        },
        "concerns": {
            "type": "array", "items": {"type": "string"},
        },
        "mistakes_to_note": {
            "type": "array", "items": {"type": "string"},
        },
        "next_batch_focus": {"type": "string"},
        "hints": {
            "type": "array", "items": {"type": "string"},
        },
        "retire_principles": {
            "type": "array", "items": {"type": "string"},
            "description": "Topic names to remove from the playbook before the next batch.",
        },
    },
    "required": [
        "verdict", "observations", "principles_gaining_confidence",
        "concerns", "mistakes_to_note", "next_batch_focus", "hints",
        "retire_principles",
    ],
    "additionalProperties": False,
}


# ── Bootstrap schema ───────────────────────────────────────────────────────────

BOOTSTRAP_SCHEMA = {
    "type": "object",
    "properties": {
        "domain_summary":    {"type": "string"},
        "metric_name":       {"type": "string"},
        "simulation_py":     {"type": "string"},
        "brain_md":          {"type": "string"},
        "extract_md":        {"type": "string"},
        "director_md":       {"type": "string"},
        "context_keys":      {"type": "array", "items": {"type": "string"}},
        "calibration_notes": {"type": "string"},
    },
    "required": ["domain_summary", "metric_name", "simulation_py", "brain_md",
                 "extract_md", "director_md", "context_keys", "calibration_notes"],
    "additionalProperties": False,
}


# ── Director call ──────────────────────────────────────────────────────────────

def call_director(
    batch_num: int,
    result: dict,
    prior_analysis: dict,
    all_results: list,
    domain_path: Path,
    playbook_sizes: list,
) -> dict:
    """Ask the director AI to analyze the batch and direct the next one."""

    system_prompt = (
        "You are an expert analyst directing an autonomous learning engine. "
        "Be specific and actionable. Identify real learning vs sim artifacts. "
        "Keep observations concise — 2-4 items each."
    )

    # Trend across all batches so far
    batch_avgs = [r["avg_score"] for r in all_results]
    overall_trend = ""
    if len(batch_avgs) >= 2:
        delta = batch_avgs[-1] - batch_avgs[0]
        overall_trend = f"{delta:+.2f} from batch 1 to batch {batch_num}"

    top_principles = "\n".join(
        f"  [{p.get('confidence', 0):.0%}] [{p.get('context', '')}] {p.get('principle', '')}"
        for p in result["top_principles"]
    ) or "  none yet"

    archetype_wins          = result.get("archetype_wins", {})
    archetype_wins_event    = result.get("archetype_wins_event", {})
    archetype_wins_nonevent = result.get("archetype_wins_nonevent", {})
    top_archetypes_str = ""
    if archetype_wins:
        top = sorted(archetype_wins.items(), key=lambda x: x[1], reverse=True)[:6]
        top_archetypes_str = "\nTOP WINNING ARCHETYPES THIS BATCH:\n" + "\n".join(
            f"  {name}: {count} total  ({archetype_wins_event.get(name, 0)} event / {archetype_wins_nonevent.get(name, 0)} non-event)"
            for name, count in top
        )
        top_archetypes_str += (
            "\nNOTE: wins marked 'event' may be sim artifacts (demand is guaranteed when event fires)."
            " Weight non-event wins more heavily when evaluating archetype quality."
        )

    prior_focus   = prior_analysis.get("next_batch_focus", "none — this is the first batch") if prior_analysis else "none — this is the first batch"
    prior_hints   = prior_analysis.get("hints", []) if prior_analysis else []
    prior_mistakes = prior_analysis.get("mistakes_to_note", []) if prior_analysis else []

    pb_path = domain_path / "playbook.jsonl"
    playbook_topics = "none"
    if pb_path.exists():
        topics = sorted({json.loads(l).get("topic", "") for l in pb_path.read_text().splitlines() if l.strip()})
        playbook_topics = ", ".join(topics) if topics else "none"

    # Playbook growth deltas
    growth_deltas = []
    if len(playbook_sizes) >= 2:
        growth_deltas = [playbook_sizes[i] - playbook_sizes[i - 1] for i in range(1, len(playbook_sizes))]
    growth_str = f"Playbook growth per batch: {growth_deltas}" if growth_deltas else ""
    saturation_note = ""
    if len(growth_deltas) >= 2 and growth_deltas[-1] == 0 and growth_deltas[-2] == 0:
        saturation_note = '\nNOTE: if growth is 0 for 2+ batches and score is stable, use verdict "saturated"'

    prompt = f"""You are directing an autonomous learning engine.

BATCH {batch_num} RESULTS:
  Rounds:        {result['n_rounds']}
  Avg score:     {result['avg_score']}
  Best score:    {result['best_score']}
  Worst score:   {result['worst_score']}
  Trend (first→last quarter): {result['trend_pct']:+.1f}%
  Overall across batches: {overall_trend or 'n/a'}
  Last 10 rounds scores: {result['score_last_10']}
  Context mix: {json.dumps(result['context_mix'])}
  Playbook size: {result['playbook_size']} principles
{growth_str}{saturation_note}

TOP PLAYBOOK PRINCIPLES:
{top_principles}
{top_archetypes_str}
PRIOR BATCH FOCUS: {prior_focus}
PRIOR HINTS APPLIED: {json.dumps(prior_hints)}
MISTAKES NOTED PREVIOUSLY: {json.dumps(prior_mistakes)}

Analyze this batch. Is the engine learning real principles or gaming the sim?
Is the score trending up meaningfully or flat? Are the playbook principles plausible?
What should the next batch focus on to push learning further?

Verdicts:
  converging         = score and playbook are improving steadily
  exploring          = mixed results, still searching
  stalled            = no improvement for multiple batches
  reward_hacking     = sim artifact — score rising but for the wrong reason
  needs_calibration  = sim or prompts need adjustment before more runs
  saturated          = playbook full, score stable, nothing left to learn from this sim

Hints are short strings that bias strategy generation toward certain approaches.
Good hints are specific and actionable — name the thing to explore and why.

retire_principles is a list of playbook topic names to DELETE before the next batch.
Use it when a principle is a confirmed sim artifact, suppressing real exploration, or contradicted by evidence.
Available topics in the current playbook: {playbook_topics}
GUARDRAIL: principles with confidence >=88% are protected and cannot be retired — do not list them.
Cap retirements at 2 per batch. Only retire if you have clear evidence the principle is wrong or harmful.
"""

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": DIRECTOR_SCHEMA,
            }
        },
    )

    return json.loads(response.content[0].text)


# ── Thinking log ───────────────────────────────────────────────────────────────

def append_thinking_log(log_path: Path, batch_num: int, result: dict, analysis: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    section = f"""
## Batch {batch_num} — {timestamp}  |  {result['n_rounds']} rounds  |  avg score {result['avg_score']}  |  trend {result['trend_pct']:+.1f}%

**Verdict:** {analysis['verdict']}

**Observations:**
{chr(10).join(f"- {o}" for o in analysis['observations'])}

**Principles gaining confidence:**
{chr(10).join(f"- {p}" for p in analysis['principles_gaining_confidence']) or "- none yet"}

**Concerns:**
{chr(10).join(f"- {c}" for c in analysis['concerns']) or "- none"}

**Mistakes to not repeat:**
{chr(10).join(f"- {m}" for m in analysis['mistakes_to_note']) or "- none"}

**Next batch focus:** {analysis['next_batch_focus']}

**Hints injected for next batch:**
{chr(10).join(f"- {h}" for h in analysis['hints']) or "- none"}

---"""

    with open(log_path, "a") as f:
        f.write(section + "\n")


# ── Git commit helper ──────────────────────────────────────────────────────────

def git_commit_batch(domain: str, domain_path: Path, global_batch: int, result: dict, analysis: dict):
    """
    Auto-commit domain state after each batch. Silently skips if git unavailable.

    Commits: playbook.jsonl, retired_topics.json, champion_archetype.json,
             top_candidates.json, simulation.py, tournament.py, prompts/

    Commit message: "{domain} batch {N}: avg {score} [{verdict}]"
    Each kept commit = one experiment. git log = full experiment history.
    """
    import subprocess

    git_dir = ENGINE_ROOT / ".git"
    if not git_dir.exists():
        return  # not a git repo — skip silently

    # Files to stage: learned state + domain code (not artifacts)
    stage_targets = [
        domain_path / "playbook.jsonl",
        domain_path / "retired_topics.json",
        domain_path / "champion_archetype.json",
        domain_path / "top_candidates.json",
        domain_path / "simulation.py",
        domain_path / "tournament.py",
        domain_path / "prompts",
    ]
    existing = [str(p.relative_to(ENGINE_ROOT)) for p in stage_targets if p.exists()]
    if not existing:
        return

    verdict   = analysis["verdict"]
    avg_score = result["avg_score"]
    focus     = analysis.get("next_batch_focus", "")[:60]

    msg_lines = [
        f"{domain} batch {global_batch}: avg {avg_score} [{verdict}]",
        focus,
    ]
    if analysis.get("concerns"):
        msg_lines.append(f"concern: {analysis['concerns'][0][:80]}")
    msg = "\n".join(l for l in msg_lines if l)

    try:
        subprocess.run(
            ["git", "add"] + existing,
            cwd=ENGINE_ROOT, check=True, capture_output=True,
        )
        result_proc = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ENGINE_ROOT, capture_output=True,
        )
        if result_proc.returncode != 0:  # staged changes exist
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=ENGINE_ROOT, check=True, capture_output=True,
            )
    except Exception:
        pass  # git failure never interrupts a run


# ── Retire principles helper ───────────────────────────────────────────────────

def retire_principles(analysis: dict, domain_path: Path):
    """Retire flagged playbook topics. Protects high-confidence principles (>=88%), caps at 2/batch."""
    to_retire = analysis.get("retire_principles", [])
    if not to_retire:
        return

    pb_path = domain_path / "playbook.jsonl"
    if not pb_path.exists():
        return

    entries = [json.loads(l) for l in pb_path.read_text().strip().splitlines() if l.strip()]
    protected = {e.get("topic") for e in entries if e.get("confidence", 0) >= 0.88}
    safe_to_retire = [t for t in to_retire if t not in protected][:2]

    if safe_to_retire:
        kept = [e for e in entries if e.get("topic") not in safe_to_retire]
        retired_count = len(entries) - len(kept)
        with open(pb_path, "w") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
        if retired_count:
            print(f"  retired {retired_count} principle(s): {', '.join(safe_to_retire)}")

        rt_path = domain_path / "retired_topics.json"
        rt_list = json.loads(rt_path.read_text()) if rt_path.exists() else []
        for t in safe_to_retire:
            if t not in rt_list:
                rt_list.append(t)
        rt_path.write_text(json.dumps(rt_list))

    blocked = [t for t in to_retire if t in protected]
    if blocked:
        print(f"  protected (>=88% conf): {', '.join(blocked)}")


# ── Subcommands ────────────────────────────────────────────────────────────────

def cmd_bootstrap(args):
    """Generate a new domain from a natural language description using Sonnet."""
    template_path = ENGINE_ROOT / "template"
    domain_path   = ENGINE_ROOT / args.domain

    if domain_path.exists():
        print(f"Error: {domain_path} already exists.")
        sys.exit(1)
    if not template_path.exists():
        print("Error: template/ folder not found.")
        sys.exit(1)

    print(f"\nBootstrapping {args.domain} from description...")
    print("Calling Sonnet to generate domain files...\n")

    system_prompt = """You are designing a simulation domain for Autoforge, an autonomous strategy learning system.

Autoforge works by:
1. Generating a library of 16 named strategy archetypes (via Sonnet)
2. Running a deterministic simulation that scores each archetype against random scenarios
3. Extracting conditional principles from what wins
4. A director AI reading results and improving the next archetype library

You will generate all files needed to define a new domain. Each file must be complete and runnable.

SIMULATION.PY REQUIREMENTS:
- simulate(candidate: dict, state: dict) -> float   # deterministic, fast, no I/O
- random_state() -> dict                             # draws one scenario from the domain distribution
- CANDIDATE_SCHEMA: dict                             # JSON schema describing a strategy's parameters
- METRIC_NAME: str                                   # e.g. "profit", "score", "accuracy"

CALIBRATION RULES:
- simulate() must be deterministic — same inputs always produce same output
- random_state() must cover diverse scenarios — different scenario types should favor different strategies
- The score range should be reasonable (e.g. 0-200, not 1e9)
- No single strategy should dominate all scenarios — variety is essential for learning
- Include an is_event or equivalent flag for unusual/extreme scenarios

CANDIDATE_SCHEMA RULES:
- Must be a valid JSON schema object
- Parameters should be numeric ranges or small enums
- Include 3-6 meaningful dimensions that a strategy can vary
- Each dimension should meaningfully affect the score in simulate()

PROMPT RULES:
- brain.md: instructions for generating 16 diverse strategy archetypes. Name each archetype distinctively. Cover the full strategy space including contrarian approaches.
- extract.md: instructions for extracting 0-2 conditional principles per round. Include {{RETIRED_TOPICS}} placeholder.
- director.md: context about the domain for the between-batch director. What to watch for, what failure modes exist.
"""

    user_prompt = f"""Generate all domain files for Autoforge based on this description:

{args.description}

Requirements:
- simulation_py: complete, runnable Python file with all required exports
- brain_md: complete prompt for generating 16 strategy archetypes for this domain
- extract_md: complete prompt for principle extraction (include {{{{RETIRED_TOPICS}}}} placeholder)
- director_md: domain context for the batch director
- context_keys: list of key names from random_state() that should appear in _build_context()
- calibration_notes: specific things to check/tune in the simulation before running the full tournament
- metric_name: the metric being optimized
- domain_summary: 1-2 sentence description of what this domain learns
"""

    import anthropic
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": BOOTSTRAP_SCHEMA,
                }
            },
        )
        data = json.loads(response.content[0].text)
    except Exception as e:
        print(f"Error: Anthropic API call failed: {e}")
        # Clean up partial domain folder if it was created
        if domain_path.exists():
            shutil.rmtree(domain_path)
        sys.exit(1)

    # 1. Copy template/ to domain_path/
    shutil.copytree(template_path, domain_path)

    try:
        # 2. Write the generated simulation.py (overwrite template skeleton)
        (domain_path / "simulation.py").write_text(data["simulation_py"])

        # 3. Write brain_md, extract_md, director_md to prompts/
        prompts_path = domain_path / "prompts"
        prompts_path.mkdir(exist_ok=True)
        (prompts_path / "brain.md").write_text(data["brain_md"])
        (prompts_path / "extract.md").write_text(data["extract_md"])
        (prompts_path / "director.md").write_text(data["director_md"])

        # 4. Generate _build_context() body from context_keys — edit tournament.py
        context_keys = data["context_keys"]
        tournament_path = domain_path / "tournament.py"
        tournament_text = tournament_path.read_text()

        # Build the new _build_context() body lines
        return_lines = []
        for key in context_keys:
            return_lines.append(f'        "{key}": state.get("{key}"),')
        return_body = "\n".join(return_lines)

        old_build_context = '''def _build_context(state: dict) -> dict:
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
    }'''

        new_build_context = f'''def _build_context(state: dict) -> dict:
    """
    Build the context dict passed to the AI extractor.

    These key-value pairs describe the scenario in plain terms — the AI reads
    them to extract conditional principles ("IF demand is high AND competition is low...").

    Make these human-readable. The AI's extraction quality depends on this.
    """
    return {{
{return_body}
    }}'''

        tournament_text = tournament_text.replace(old_build_context, new_build_context)
        tournament_path.write_text(tournament_text)

        # 5. Write .env with ANTHROPIC_API_KEY placeholder
        (domain_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-...\n")

        # 6. Create data/ directory inside domain_path
        (domain_path / "data").mkdir(exist_ok=True)
        (domain_path / "data" / ".gitkeep").touch()

    except Exception as e:
        print(f"Error: Failed to write domain files: {e}")
        shutil.rmtree(domain_path)
        sys.exit(1)

    domain_summary    = data["domain_summary"]
    calibration_notes = data["calibration_notes"]

    print(f"Created {args.domain}/\n")
    print(domain_summary)
    print(f"\nCalibration notes:\n{calibration_notes}")
    print(f"""
IMPORTANT: Review {args.domain}/simulation.py before running.
The simulation is the only thing the engine learns from — get it right.

Next:
  python run.py run --domain {args.domain} --batches 3 --rounds 50   # sanity check
  python run.py run --domain {args.domain} --brain --batches 5 --rounds 150""")


def cmd_new(args):
    """Scaffold a new domain from the template."""
    template_path = ENGINE_ROOT / "template"
    if not template_path.exists():
        print(f"Error: template/ folder not found at {template_path}")
        sys.exit(1)

    domain_path = ENGINE_ROOT / args.domain
    if domain_path.exists():
        print(f"Error: {domain_path} already exists.")
        sys.exit(1)

    shutil.copytree(template_path, domain_path)

    env_file = domain_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-...\n")

    print(f"\nCreated {args.domain}/")
    print()
    print("Next steps:")
    print(f"  1. Edit {args.domain}/simulation.py — implement simulate(), random_state(), CANDIDATE_SCHEMA, METRIC_NAME")
    print(f"  2. Edit {args.domain}/prompts/brain.md — tell Sonnet what archetypes to generate")
    print(f"  3. Edit {args.domain}/prompts/extract.md — tell Haiku what principles to extract")
    print(f"  4. Edit {args.domain}/prompts/director.md — context for the batch director")
    print(f"  5. Set your API key in {args.domain}/.env")
    print()
    print("Then run:")
    print(f"  python run.py run --domain {args.domain}                  # Stage 1, no API calls")
    print(f"  python run.py run --domain {args.domain} --brain          # Stage 2, AI archetypes")
    print(f"  python run.py run --domain {args.domain} --auto           # Stage 1 → Stage 2 auto")


def cmd_run(args):
    """Run the tournament in batches with optional AI direction."""
    domain_path = ENGINE_ROOT / args.domain

    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    # Load .env: domain folder first, then engine root as fallback
    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)

    from tournament import run_batch

    thinking_log = domain_path / "thinking_log.md"
    if not thinking_log.exists():
        thinking_log.write_text(
            f"# Thinking Log — {args.domain}\n\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
            f"|  {args.batches} batches × {args.rounds} rounds\n\n---\n"
        )

    use_brain  = args.brain
    auto_mode  = getattr(args, "auto", False)
    brain_label = "Stage 2 — AI archetypes" if use_brain else ("auto" if auto_mode else "Stage 1 — procedural")

    print(f"\n{'='*60}")
    print(f"Autoforge — {args.domain}  [{brain_label}]")
    print(f"{args.years} year(s) x {args.batches} batches x {args.rounds} rounds = {args.years * args.batches * args.rounds} total")
    print(f"{'='*60}\n")

    generation_offset = 0
    year_avg_scores   = []
    prior_analysis    = None
    playbook_sizes    = []
    stop_reason       = None
    all_batch_rows    = []  # for Rich summary table

    # Auto-mode saturation tracking: count consecutive batches with no playbook growth
    auto_consecutive_flat = 0

    for year_num in range(1, args.years + 1):
        hints       = []
        all_results = []

        if args.years > 1:
            print(f"\n{'─'*60}")
            print(f"Year {year_num} / {args.years}  — playbook carries over from prior year")
            print(f"{'─'*60}\n")
            with open(thinking_log, "a") as f:
                f.write(f"\n# -- Year {year_num} -- {datetime.now().strftime('%Y-%m-%d %H:%M')} --\n\n")

        for batch_num in range(1, args.batches + 1):
            global_batch = (year_num - 1) * args.batches + batch_num

            # Auto-mode: promote to Stage 2 if playbook flatlined for 2 consecutive batches
            if auto_mode and not use_brain and auto_consecutive_flat >= 2:
                use_brain = True
                print(f"[auto] Playbook flat for 2 batches — promoting to Stage 2 (AI archetypes)\n")
                with open(thinking_log, "a") as f:
                    f.write(f"\n*[Auto] Stage 1 saturated at batch {global_batch} — promoted to Stage 2 (AI archetypes).*\n\n")

            print(f"[Y{year_num} Batch {batch_num}/{args.batches}] running {args.rounds} rounds", end="", flush=True)

            result = run_batch(args.rounds, generation_offset=generation_offset, hints=hints, use_brain=use_brain)
            all_results.append(result)
            generation_offset += args.rounds

            # Track playbook sizes for saturation detection
            current_size = result["playbook_size"]
            playbook_sizes.append(current_size)

            # Auto-mode: track consecutive flat batches
            if auto_mode and not use_brain:
                if len(playbook_sizes) >= 2 and playbook_sizes[-1] == playbook_sizes[-2]:
                    auto_consecutive_flat += 1
                else:
                    auto_consecutive_flat = 0

            trend_char = "+" if result["trend_pct"] >= 0 else ""
            print(f"  ->  avg {result['avg_score']}  best {result['best_score']}  trend {trend_char}{result['trend_pct']:.1f}%")

            has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if has_api_key:
                print(f"  director analyzing...", end="", flush=True)
                analysis = call_director(global_batch, result, prior_analysis, all_results, domain_path, playbook_sizes)
                append_thinking_log(thinking_log, global_batch, result, analysis)
                verdict = analysis["verdict"]
                print(f"  [{verdict}]")
                print(f"  -> {analysis['next_batch_focus'][:80]}")
                if analysis["concerns"]:
                    print(f"  ! {analysis['concerns'][0]}")
            else:
                # Stage 1 free mode — no director without API key
                analysis = {"verdict": "exploring", "hints": [], "retire_principles": [],
                            "next_batch_focus": "Stage 1 — set ANTHROPIC_API_KEY to enable director",
                            "observations": [], "principles_gaining_confidence": [],
                            "concerns": [], "mistakes_to_note": []}
                verdict = "exploring"
                print(f"  [Stage 1 — no director]")
            print()

            # Store row for summary table
            if len(all_batch_rows) > 0:
                prev_avg = all_batch_rows[-1]["avg"]
                trend_sym = "up" if result["avg_score"] > prev_avg else ("down" if result["avg_score"] < prev_avg else "flat")
            else:
                trend_sym = "--"
            all_batch_rows.append({
                "batch": global_batch,
                "avg":   result["avg_score"],
                "best":  result["best_score"],
                "trend": trend_sym,
            })

            hints          = analysis["hints"]
            prior_analysis = analysis

            retire_principles(analysis, domain_path)
            git_commit_batch(args.domain, domain_path, global_batch, result, analysis)

            if verdict == "reward_hacking":
                print("! Reward hacking flagged — stopping run. Check thinking_log.md.")
                stop_reason = "reward_hacking"
                year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))
                break

            if verdict == "saturated":
                print("Playbook saturated — engine has learned everything this sim can teach.")
                print("Run:  python run.py export --domain " + args.domain)
                stop_reason = "saturated"
                year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))
                break

        else:
            year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))

        if stop_reason in ("reward_hacking", "saturated"):
            break

    # Write last_run.json
    final_verdict = prior_analysis["verdict"] if prior_analysis else "unknown"
    last_run = {
        "timestamp":    datetime.now().isoformat(),
        "domain":       args.domain,
        "total_rounds": generation_offset,
        "year_avg_scores": year_avg_scores,
        "final_verdict":   final_verdict,
        "stop_reason":     stop_reason,
    }
    (domain_path / "last_run.json").write_text(json.dumps(last_run, indent=2))

    # Terminal bell
    print("\a", end="", flush=True)

    # Rich summary
    table = Table(title=f"Run Summary — {args.domain}", show_header=True)
    table.add_column("Batch", style="dim", width=6)
    table.add_column("Avg", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Trend", justify="center")
    for row in all_batch_rows:
        trend_display = {"up": "[green]up[/green]", "down": "[red]down[/red]", "flat": "flat", "--": "--"}.get(row["trend"], row["trend"])
        table.add_row(str(row["batch"]), str(row["avg"]), str(row["best"]), trend_display)
    console.print()
    console.print(table)

    verdict_color = {
        "converging": "green", "saturated": "green", "exploring": "yellow",
        "stalled": "yellow", "reward_hacking": "red", "needs_calibration": "red",
    }.get(final_verdict, "white")
    panel_content = f"[{verdict_color}]Verdict: {final_verdict}[/{verdict_color}]"
    if stop_reason:
        panel_content += f"\nStop reason: {stop_reason}"
    if prior_analysis and prior_analysis.get("concerns"):
        panel_content += f"\nTop concern: {prior_analysis['concerns'][0]}"
    panel_content += f"\nThinking log: {thinking_log}"
    console.print(Panel(panel_content, title="Final State"))

    # Print champion archetype if it exists
    champion_path = domain_path / "champion_archetype.json"
    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            print(f"Champion: {champ.get('name', 'unknown')}")
            print(f"  {champ.get('philosophy', '')}")
        except Exception:
            pass


def _print_validation(ok: list, warnings: list, errors: list):
    for msg in ok:
        print(f"  [ok] {msg}")
    for msg in warnings:
        print(f"  [warn] {msg}")
    for msg in errors:
        print(f"  [error] {msg}")


def cmd_calibrate(args):
    """Show scenario distributions, score stats, and dominance check."""
    import importlib
    import statistics
    import collections

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)

    try:
        sim = importlib.import_module("simulation")
    except Exception as e:
        print(f"simulation.py import failed: {e}")
        sys.exit(1)

    n = args.n
    print(f"\nCalibrating {args.domain} — sampling {n} scenarios...\n")

    states = [sim.random_state() for _ in range(n)]

    import random as _random
    schema_props = sim.CANDIDATE_SCHEMA.get("properties", {})

    def random_candidate():
        c = {}
        for key, spec in schema_props.items():
            if spec.get("type") == "number":
                lo, hi = spec.get("minimum", 0.0), spec.get("maximum", 1.0)
                c[key] = _random.uniform(lo, hi)
            elif spec.get("type") == "integer":
                lo, hi = spec.get("minimum", 0), spec.get("maximum", 10)
                c[key] = _random.randint(lo, hi)
            elif "enum" in spec:
                c[key] = _random.choice(spec["enum"])
            else:
                c[key] = None
        return c

    candidates = [random_candidate() for _ in range(8)]
    win_counts = collections.Counter()
    all_scores = []

    for state in states:
        scored = [(sim.simulate(c, state), i) for i, c in enumerate(candidates)]
        scored.sort(reverse=True)
        all_scores.append(scored[0][0])
        win_counts[scored[0][1]] += 1

    # ── Scenario distribution ─────────────────────────────────────────────────
    tbl = Table(title=f"Scenario Distribution ({n} samples)", show_header=True)
    tbl.add_column("Key")
    tbl.add_column("Type")
    tbl.add_column("Distribution")

    for k in list(states[0].keys()):
        vals = [s[k] for s in states if k in s]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            true_pct = sum(1 for v in vals if v) / len(vals) * 100
            tbl.add_row(k, "bool", f"{true_pct:.0f}% True  /  {100-true_pct:.0f}% False")
        elif isinstance(vals[0], (int, float)):
            tbl.add_row(k, "number",
                f"min {min(vals):.2f}  median {statistics.median(vals):.2f}  max {max(vals):.2f}")
        else:
            counts = collections.Counter(vals)
            top = sorted(counts.items(), key=lambda x: -x[1])[:6]
            tbl.add_row(k, "categorical",
                "  ".join(f"{v}: {c/len(vals)*100:.0f}%" for v, c in top))

    console.print(tbl)

    # ── Score distribution ────────────────────────────────────────────────────
    s = sorted(all_scores)
    q1  = s[len(s) // 4]
    q3  = s[3 * len(s) // 4]
    stbl = Table(title=f"Score Distribution ({sim.METRIC_NAME})", show_header=False)
    stbl.add_column("Metric")
    stbl.add_column("Value", justify="right")
    for label, val in [("min", min(all_scores)), ("p25", q1),
                       ("median", statistics.median(all_scores)),
                       ("p75", q3), ("max", max(all_scores)),
                       ("stdev", statistics.stdev(all_scores))]:
        stbl.add_row(label, f"{val:.2f}")
    console.print(stbl)

    # ── Dominance check ───────────────────────────────────────────────────────
    wtbl = Table(title="Candidate Win Distribution (8 random strategies)", show_header=True)
    wtbl.add_column("Candidate")
    wtbl.add_column("Wins", justify="right")
    wtbl.add_column("Win %", justify="right")
    for i in range(len(candidates)):
        wins = win_counts.get(i, 0)
        wtbl.add_row(f"candidate_{i}", str(wins), f"{wins/n*100:.0f}%")
    console.print(wtbl)

    # ── Verdict ───────────────────────────────────────────────────────────────
    issues = []
    if max(all_scores) == min(all_scores):
        issues.append("All scores identical — simulate() may not depend on scenario state")
    elif statistics.stdev(all_scores) < abs(statistics.mean(all_scores)) * 0.05:
        issues.append("Very low score variance — check that scenario factors affect outcomes")
    top_pct = win_counts.most_common(1)[0][1] / n * 100
    if top_pct > 70:
        issues.append(f"One candidate wins {top_pct:.0f}% of rounds — likely a dominant strategy")
    zero_pct = sum(1 for sc in all_scores if sc == 0) / len(all_scores) * 100
    if zero_pct > 30:
        issues.append(f"{zero_pct:.0f}% of rounds score 0 — check simulate() edge cases")

    if issues:
        console.print("\n[yellow]Calibration warnings:[/yellow]")
        for issue in issues:
            console.print(f"  [yellow]! {issue}[/yellow]")
        console.print(f"\n[dim]Edit {args.domain}/simulation.py and re-run calibrate.[/dim]")
    else:
        console.print(f"\n[green]Calibration looks good.[/green]")
        console.print(f"  python run.py run --domain {args.domain} --batches 3 --rounds 50")


def cmd_validate(args):
    """Sanity-check simulation.py: imports, required exports, schema, sim output."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)

    errors   = []
    warnings = []
    ok       = []

    # 1. Import
    try:
        import importlib
        sim = importlib.import_module("simulation")
        ok.append("simulation.py imports cleanly")
    except Exception as e:
        errors.append(f"simulation.py import failed: {e}")
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 2. Required exports
    for name in ["simulate", "random_state", "CANDIDATE_SCHEMA", "METRIC_NAME"]:
        if hasattr(sim, name):
            ok.append(f"{name} exists")
        else:
            errors.append(f"Missing required export: {name}")

    if errors:
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 3. METRIC_NAME is a non-empty string
    if isinstance(sim.METRIC_NAME, str) and sim.METRIC_NAME.strip():
        ok.append(f"METRIC_NAME = '{sim.METRIC_NAME}'")
    else:
        errors.append("METRIC_NAME must be a non-empty string")

    # 4. CANDIDATE_SCHEMA is a valid dict with properties
    schema = sim.CANDIDATE_SCHEMA
    if isinstance(schema, dict) and "properties" in schema and schema["properties"]:
        n_params = len(schema["properties"])
        ok.append(f"CANDIDATE_SCHEMA has {n_params} parameter(s): {', '.join(schema['properties'].keys())}")
    else:
        errors.append("CANDIDATE_SCHEMA must be a dict with a non-empty 'properties' key")

    if errors:
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 5. random_state() returns a dict
    try:
        state = sim.random_state()
        if isinstance(state, dict) and state:
            ok.append(f"random_state() returns dict with {len(state)} key(s): {', '.join(list(state.keys())[:5])}")
        else:
            errors.append("random_state() must return a non-empty dict")
    except Exception as e:
        errors.append(f"random_state() raised: {e}")

    # 6. Build a minimal valid candidate from schema and call simulate()
    try:
        candidate = {}
        for key, spec in schema.get("properties", {}).items():
            t = spec.get("type")
            if t == "number":
                lo = spec.get("minimum", 0.0)
                hi = spec.get("maximum", 1.0)
                candidate[key] = (lo + hi) / 2
            elif t == "integer":
                lo = spec.get("minimum", 0)
                hi = spec.get("maximum", 10)
                candidate[key] = (lo + hi) // 2
            elif "enum" in spec:
                candidate[key] = spec["enum"][0]
            else:
                candidate[key] = None

        state    = sim.random_state()
        score    = sim.simulate(candidate, state)

        if isinstance(score, (int, float)):
            ok.append(f"simulate() returns {sim.METRIC_NAME} = {score}")
        else:
            errors.append(f"simulate() must return a number, got {type(score).__name__}")
    except Exception as e:
        errors.append(f"simulate() raised: {e}")

    # 7. Score variety check — run 20 scenarios, warn if all identical
    try:
        import random as _random
        scores = [sim.simulate(candidate, sim.random_state()) for _ in range(20)]
        unique = len(set(round(s, 4) for s in scores))
        if unique < 5:
            warnings.append(f"simulate() returned only {unique} unique scores across 20 random scenarios — check calibration")
        else:
            ok.append(f"simulate() produces varied scores across scenarios ({unique}/20 unique)")
    except Exception:
        pass

    # 8. Check prompts exist
    for fname in ["brain.md", "extract.md", "director.md"]:
        p = domain_path / "prompts" / fname
        if p.exists():
            ok.append(f"prompts/{fname} exists")
        else:
            warnings.append(f"prompts/{fname} missing — needed for Stage 2 (--brain)")

    _print_validation(ok, warnings, errors)

    if errors:
        sys.exit(1)
    else:
        print(f"\nDomain {args.domain} looks good. Run:")
        print(f"  python run.py run --domain {args.domain} --batches 3 --rounds 50")


def cmd_export(args):
    """Export Stage 3 training data from tournament_log.jsonl."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    from engine_export import export_training_data
    export_training_data(domain_path)


def cmd_status(args):
    """Show domain state: playbook, champion, last run, log stats."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    # Playbook
    pb_path = domain_path / "playbook.jsonl"
    if pb_path.exists():
        playbook = [json.loads(l) for l in pb_path.read_text().splitlines() if l.strip()]
        top5 = sorted(playbook, key=lambda p: p.get("confidence", 0), reverse=True)[:5]
        tbl = Table(title=f"Playbook — {len(playbook)} principles", show_header=True)
        tbl.add_column("Topic")
        tbl.add_column("Conf", justify="right")
        tbl.add_column("Principle")
        for p in top5:
            tbl.add_row(
                p.get("topic", ""),
                f"{p.get('confidence', 0):.0%}",
                p.get("principle", "")[:60],
            )
        console.print(tbl)
    else:
        console.print("Playbook: not found")

    # Champion
    champion_path = domain_path / "champion_archetype.json"
    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            line = f"Champion: {champ.get('name', 'unknown')} — {champ.get('philosophy', '')[:70]}"
            console.print(f"\n[bold]{line}[/bold]")
        except Exception:
            pass

    # Retired topics
    rt_path = domain_path / "retired_topics.json"
    if rt_path.exists():
        try:
            retired = json.loads(rt_path.read_text())
            if retired:
                msg = f"Retired topics ({len(retired)}): {', '.join(retired)}"
                console.print(f"\n[dim]{msg}[/dim]")
        except Exception:
            pass

    # Last run
    last_run_path = domain_path / "last_run.json"
    if last_run_path.exists():
        try:
            lr = json.loads(last_run_path.read_text())
            lines = [
                f"Last run: {lr.get('timestamp', 'unknown')[:19]}",
                f"  Rounds: {lr.get('total_rounds', 0)}   Verdict: {lr.get('final_verdict', 'unknown')}",
            ]
            if lr.get("stop_reason"):
                lines.append(f"  Stop reason: {lr['stop_reason']}")
            console.print("\n" + "\n".join(lines))
        except Exception:
            pass

    # Tournament log + training data
    log_path = domain_path / "tournament_log.jsonl"
    td_path  = domain_path / "training_data.jsonl"
    if log_path.exists():
        try:
            n_rounds = sum(1 for l in log_path.read_text().splitlines() if l.strip())
            td_exists = td_path.exists()
            msg = f"tournament_log.jsonl: {n_rounds} rounds"
            if td_exists:
                n_td = sum(1 for l in td_path.read_text().splitlines() if l.strip())
                msg += f"   training_data.jsonl: {n_td} examples (ready)"
            else:
                msg += "   training_data.jsonl: not yet exported"
            console.print(f"\n[dim]{msg}[/dim]")
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Autoforge — autonomous strategy learning system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_bootstrap = subparsers.add_parser("bootstrap", help="Generate a new domain from a description using AI")
    p_bootstrap.add_argument("domain", help="Domain name (e.g. GrainMarketing)")
    p_bootstrap.add_argument("--description", required=True, help="Natural language description of the domain")

    # new
    p_new = subparsers.add_parser("new", help="Scaffold a new domain from template")
    p_new.add_argument("domain", help="Domain name (e.g. GrainMarketing)")

    # run
    p_run = subparsers.add_parser("run", help="Run the tournament")
    p_run.add_argument("--domain",  required=True, help="Domain subfolder name")
    p_run.add_argument("--batches", type=int, default=8,   help="Batches per year (default 8)")
    p_run.add_argument("--rounds",  type=int, default=200, help="Rounds per batch (default 200)")
    p_run.add_argument("--years",   type=int, default=1,   help="Years to run (default 1)")
    p_run.add_argument("--brain",   action="store_true",   help="Stage 2: AI archetypes")
    p_run.add_argument("--auto",    action="store_true",   help="Stage 1 until saturated, then auto-promote to Stage 2")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Show scenario distributions and score stats")
    p_cal.add_argument("--domain", required=True, help="Domain subfolder name")
    p_cal.add_argument("--n", type=int, default=500, help="Scenarios to sample (default 500)")

    # validate
    p_validate = subparsers.add_parser("validate", help="Sanity-check a domain's simulation.py")
    p_validate.add_argument("--domain", required=True, help="Domain subfolder name")

    # export
    p_export = subparsers.add_parser("export", help="Export Stage 3 training data")
    p_export.add_argument("--domain", required=True, help="Domain subfolder name")

    # status
    p_status = subparsers.add_parser("status", help="Show domain state")
    p_status.add_argument("--domain", required=True, help="Domain subfolder name")

    args = parser.parse_args()

    if args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "new":
        cmd_new(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
