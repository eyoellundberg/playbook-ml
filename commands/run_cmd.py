"""
commands/run_cmd.py — cmd_run subcommand (tournament runner).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from commands.shared import (
    ENGINE_ROOT,
    console,
    call_director,
    append_thinking_log,
    git_commit_batch,
    retire_principles,
)


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

            try:
                result = run_batch(
                    args.rounds,
                    generation_offset=generation_offset,
                    hints=hints,
                    use_brain=use_brain,
                    workers=args.workers,
                )
            except Exception as e:
                print(f"\n  [batch error: {e}] — skipping to next batch")
                with open(thinking_log, "a") as f:
                    f.write(f"\n*Batch {global_batch} failed: {e}*\n\n")
                generation_offset += args.rounds
                continue

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
                try:
                    analysis = call_director(global_batch, result, prior_analysis, all_results, domain_path, playbook_sizes)
                    append_thinking_log(thinking_log, global_batch, result, analysis)
                    verdict = analysis["verdict"]
                    print(f"  [{verdict}]")
                    print(f"  -> {analysis['next_batch_focus'][:80]}")
                    if analysis["concerns"]:
                        print(f"  ! {analysis['concerns'][0]}")
                except Exception as e:
                    print(f"  [director error: {e}] — continuing")
                    analysis = {"verdict": "exploring", "hints": hints, "retire_principles": [],
                                "next_batch_focus": "director unavailable — continuing previous hints",
                                "observations": [], "principles_gaining_confidence": [],
                                "concerns": [], "mistakes_to_note": []}
                    verdict = "exploring"
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
