"""
commands/shared.py — Shared constants, helpers, and AI-call utilities.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

ENGINE_ROOT = Path(__file__).parent.parent
console = Console()


def load_sim(domain_path: Path):
    """Import the domain's simulation module. Raises ImportError on failure."""
    import importlib
    load_env(domain_path)
    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)
    try:
        return importlib.import_module("simulation")
    except Exception as e:
        raise ImportError(f"simulation.py import failed: {e}") from e


def load_env(domain_path: Path):
    """Load .env from domain folder, then engine root as fallback."""
    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_mission(domain_path: Path) -> str:
    """Return mission.md contents for a domain, if present."""
    mission_path = domain_path / "mission.md"
    if not mission_path.exists():
        return ""
    return mission_path.read_text().strip()


def normalize_confidence(value) -> float:
    """Normalize legacy confidence values like 73 into 0.73."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def normalize_playbook_entry(entry: dict) -> dict:
    """Return a playbook entry with normalized confidence."""
    return {**entry, "confidence": normalize_confidence(entry.get("confidence", 0))}


def get_ai_backend() -> str:
    """Return the configured AI backend."""
    return os.environ.get("AUTOFORGE_AI_BACKEND", "anthropic").strip().lower() or "anthropic"


def _valid_anthropic_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return bool(key) and "..." not in key and not key.lower().startswith("your_")


def _valid_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key) and "..." not in key and not key.lower().startswith("your_")


def ai_backend_available() -> bool:
    """Whether an AI backend is available for structured calls."""
    backend = get_ai_backend()
    if backend == "manual":
        return True
    if backend == "anthropic":
        if not _valid_anthropic_key():
            return False
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True
    if backend == "openai":
        if not _valid_openai_key():
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True
    return False


def _validate_top_level_required(data: dict, schema: dict, task_name: str):
    """Lightweight validation for manual responses."""
    required = schema.get("required", [])
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Manual {task_name} response missing required keys: {missing}")


def _manual_ai_roundtrip(
    *,
    task_name: str,
    domain_path: Path,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    metadata: dict | None = None,
) -> dict:
    """Write a manual AI request file and wait for a response file."""
    manual_dir = domain_path / "manual_ai"
    requests_dir = manual_dir / "requests"
    responses_dir = manual_dir / "responses"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{task_name}"
    request_path = requests_dir / f"{request_id}.json"
    response_path = responses_dir / f"{request_id}.json"

    payload = {
        "task": task_name,
        "request_id": request_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "schema": schema,
        "response_path": str(response_path),
        "metadata": metadata or {},
    }
    request_path.write_text(json.dumps(payload, indent=2) + "\n")

    console.print(
        f"[cyan]manual-ai[/cyan] Waiting for `{task_name}` response:\n"
        f"  request: {request_path}\n"
        f"  reply to: {response_path}"
    )

    timeout_s = int(os.environ.get("AUTOFORGE_MANUAL_TIMEOUT_SECONDS", "1800"))
    poll_s = float(os.environ.get("AUTOFORGE_MANUAL_POLL_SECONDS", "1.0"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if response_path.exists():
            data = json.loads(response_path.read_text())
            if isinstance(data, dict) and "response" in data and isinstance(data["response"], dict):
                data = data["response"]
            if not isinstance(data, dict):
                raise ValueError(f"Manual {task_name} response must be a JSON object")
            _validate_top_level_required(data, schema, task_name)
            return data
        time.sleep(poll_s)

    raise TimeoutError(f"Timed out waiting for manual {task_name} response at {response_path}")


def structured_ai_call(
    *,
    task_name: str,
    domain_path: Path,
    model: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    metadata: dict | None = None,
) -> dict:
    """Run a structured AI call via the configured backend."""
    backend = get_ai_backend()
    if backend == "manual":
        return _manual_ai_roundtrip(
            task_name=task_name,
            domain_path=domain_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            metadata=metadata,
        )

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return json.loads(response.content[0].text)

    if backend == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            },
        )
        return json.loads(response.choices[0].message.content)

    raise RuntimeError(f"Unsupported AI backend: '{backend}'. Set AUTOFORGE_AI_BACKEND=anthropic or openai.")


MODEL_DIRECTOR = os.environ.get("AUTOFORGE_DIRECTOR_MODEL", "claude-sonnet-4-6")


# ── Director schema ─────────────────────────────────────────────────────────

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
        "simulation_fix_suggestions": {
            "type": "array", "items": {"type": "string"},
            "description": "Concrete, actionable suggestions for fixing simulation.py when reward_hacking or needs_calibration is detected. Empty otherwise.",
        },
    },
    "required": [
        "verdict", "observations", "principles_gaining_confidence",
        "concerns", "mistakes_to_note", "next_batch_focus", "hints",
        "retire_principles", "simulation_fix_suggestions",
    ],
    "additionalProperties": False,
}


# ── Bootstrap schema ─────────────────────────────────────────────────────────

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


# ── Director call ────────────────────────────────────────────────────────────

def call_director(
    batch_num: int,
    result: dict,
    prior_analysis: dict,
    all_results: list,
    domain_path: Path,
    playbook_sizes: list,
) -> dict:
    """Ask the director AI to analyze the batch and direct the next one."""

    director_md = domain_path / "prompts" / "director.md"
    domain_context = director_md.read_text().strip() if director_md.exists() else ""
    mission_text = load_mission(domain_path)

    system_prompt = (
        "You are an expert analyst directing an autonomous learning engine. "
        "Be specific and actionable. Identify real learning vs sim artifacts. "
        "Keep observations concise — 2-4 items each."
    )
    if mission_text:
        system_prompt = "MISSION:\n" + mission_text + "\n\n" + system_prompt
    if domain_context:
        system_prompt = domain_context + "\n\n" + system_prompt

    # Trend across all batches so far
    batch_avgs = [r["avg_score"] for r in all_results]
    overall_trend = ""
    if len(batch_avgs) >= 2:
        delta = batch_avgs[-1] - batch_avgs[0]
        overall_trend = f"{delta:+.2f} from batch 1 to batch {batch_num}"

    top_principles = "\n".join(
        f"  [{normalize_confidence(p.get('confidence', 0)):.0%}] [{p.get('context', '')}] {p.get('principle', '')}"
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

simulation_fix_suggestions: REQUIRED when verdict is reward_hacking or needs_calibration.
The user may not know what is wrong with their simulation.py — you are their expert diagnosis.
Be specific: name the exact behavior that is broken and what to change in simulate() or random_state().
Examples of good suggestions:
  - "simulate() always returns positive scores even when the strategy should fail — add a penalty branch for [CONDITION]"
  - "random_state() never generates [SCENARIO TYPE] — add a case so strategies are tested under that condition"
  - "score range is 0.95–1.05 — too narrow for strategies to differentiate; widen the reward spread"
  - "one parameter (e.g. threshold) has no effect on score — simulate() may not be reading it"
Leave empty [] for converging / exploring / stalled / saturated.
"""

    return structured_ai_call(
        task_name="director",
        domain_path=domain_path,
        model=MODEL_DIRECTOR,
        max_tokens=2048,
        system_prompt=system_prompt,
        user_prompt=prompt,
        schema=DIRECTOR_SCHEMA,
        metadata={"batch_num": batch_num, "playbook_size": result["playbook_size"]},
    )


# ── Thinking log ─────────────────────────────────────────────────────────────

def append_thinking_log(log_path: Path, batch_num: int, result: dict, analysis: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    simulation_fix_section = ""
    if analysis.get("simulation_fix_suggestions"):
        fixes = chr(10).join(f"- {s}" for s in analysis["simulation_fix_suggestions"])
        simulation_fix_section = f"\n**Simulation fix suggestions:**\n{fixes}\n"

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
{simulation_fix_section}

---"""

    with open(log_path, "a") as f:
        f.write(section + "\n")


# ── Git commit helper ─────────────────────────────────────────────────────────

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


# ── Retire principles helper ──────────────────────────────────────────────────

def retire_principles(analysis: dict, domain_path: Path):
    """Retire flagged playbook topics. Protects high-confidence principles (>=88%), caps at 2/batch."""
    to_retire = analysis.get("retire_principles", [])
    if not to_retire:
        return

    pb_path = domain_path / "playbook.jsonl"
    if not pb_path.exists():
        return

    entries = [normalize_playbook_entry(json.loads(l)) for l in pb_path.read_text().strip().splitlines() if l.strip()]
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
