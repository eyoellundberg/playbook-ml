"""
commands/tools.py — cmd_calibrate, cmd_validate, cmd_status, cmd_export subcommands.
"""

import json
import sys

from rich.table import Table

from commands.shared import ENGINE_ROOT, console, load_env, load_sim, normalize_confidence


def _print_validation(ok: list, warnings: list, errors: list):
    for msg in ok:
        print(f"  [ok] {msg}")
    for msg in warnings:
        print(f"  [warn] {msg}")
    for msg in errors:
        print(f"  [error] {msg}")


def cmd_calibrate(args):
    """Show scenario distributions, score stats, and dominance check."""
    import statistics
    import collections

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
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

    # Build degenerate adversary candidates (min-all, max-all, midpoint-all)
    # These should consistently lose — if they win, the sim rewards the wrong thing
    def degenerate_candidates():
        defs = []
        for label, pick in [("min-all", "min"), ("max-all", "max"), ("mid-all", "mid")]:
            c = {}
            for key, spec in schema_props.items():
                if spec.get("type") == "number":
                    lo, hi = spec.get("minimum", 0.0), spec.get("maximum", 1.0)
                    c[key] = lo if pick == "min" else (hi if pick == "max" else (lo + hi) / 2)
                elif spec.get("type") == "integer":
                    lo, hi = spec.get("minimum", 0), spec.get("maximum", 10)
                    c[key] = lo if pick == "min" else (hi if pick == "max" else (lo + hi) // 2)
                elif "enum" in spec:
                    vals = spec["enum"]
                    c[key] = vals[0] if pick in ("min", "mid") else vals[-1]
                else:
                    c[key] = None
            defs.append((label, c))
        return defs

    adversaries = degenerate_candidates()
    candidates = [random_candidate() for _ in range(8)]
    all_candidates = candidates + [c for _, c in adversaries]

    win_counts = collections.Counter()
    adversary_wins = collections.Counter()
    all_scores = []

    for state in states:
        scored = [(sim.simulate(c, state), i) for i, c in enumerate(all_candidates)]
        scored.sort(reverse=True)
        all_scores.append(scored[0][0])
        winner_idx = scored[0][1]
        win_counts[winner_idx] += 1
        if winner_idx >= 8:  # adversary won
            adversary_wins[adversaries[winner_idx - 8][0]] += 1

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

    # ── Sanity adversary check ────────────────────────────────────────────────
    atbl = Table(title="Sanity Adversary Check (degenerate candidates — should lose)", show_header=True)
    atbl.add_column("Adversary")
    atbl.add_column("Wins", justify="right")
    atbl.add_column("Win %", justify="right")
    for label, _ in adversaries:
        wins = adversary_wins.get(label, 0)
        color = "red" if wins / n > 0.25 else ("yellow" if wins / n > 0.10 else "green")
        atbl.add_row(label, str(wins), f"[{color}]{wins/n*100:.0f}%[/{color}]")
    console.print(atbl)

    # ── Verdict ───────────────────────────────────────────────────────────────
    issues = []
    if max(all_scores) == min(all_scores):
        issues.append("All scores identical — simulate() may not depend on scenario state")
    elif statistics.stdev(all_scores) < abs(statistics.mean(all_scores)) * 0.05:
        issues.append("Very low score variance — check that scenario factors affect outcomes")
    random_wins_only = {k: v for k, v in win_counts.items() if k < 8}
    if random_wins_only:
        top_pct = max(random_wins_only.values()) / n * 100
        if top_pct > 70:
            issues.append(f"One random candidate wins {top_pct:.0f}% of rounds — likely a dominant strategy")
    for label, _ in adversaries:
        adv_pct = adversary_wins.get(label, 0) / n * 100
        if adv_pct > 25:
            issues.append(f"Sanity adversary '{label}' wins {adv_pct:.0f}% of rounds — sim may reward degenerate strategies")
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

    errors   = []
    warnings = []
    ok       = []

    # 1. Import
    try:
        sim = load_sim(domain_path)
        ok.append("simulation.py imports cleanly")
    except ImportError as e:
        errors.append(str(e))
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

    # 8. Multi-candidate dominance check — one candidate shouldn't always win
    try:
        import collections as _col
        _candidates = []
        for _ in range(4):
            c = {}
            for key, spec in schema.get("properties", {}).items():
                t = spec.get("type")
                if t == "number":
                    c[key] = _random.uniform(spec.get("minimum", 0.0), spec.get("maximum", 1.0))
                elif t == "integer":
                    c[key] = _random.randint(spec.get("minimum", 0), spec.get("maximum", 10))
                elif "enum" in spec:
                    c[key] = _random.choice(spec["enum"])
                else:
                    c[key] = None
            _candidates.append(c)
        win_counts = _col.Counter()
        for _ in range(50):
            state = sim.random_state()
            winner_idx = max(range(len(_candidates)), key=lambda j: sim.simulate(_candidates[j], state))
            win_counts[winner_idx] += 1
        top_pct = max(win_counts.values()) / 50
        if top_pct > 0.8:
            warnings.append(f"One random candidate wins {top_pct:.0%} of 50 rounds — possible dominant strategy, check simulation.py")
        else:
            ok.append(f"No single candidate dominates (top win rate {top_pct:.0%} across 4 candidates)")
    except Exception:
        pass

    # 9. build_context check — if present, verify it returns categorically diverse results
    if hasattr(sim, "build_context"):
        try:
            ctxs = [sim.build_context(sim.random_state()) for _ in range(30)]
            if ctxs and isinstance(ctxs[0], dict):
                categorical_keys = [k for k, v in ctxs[0].items() if isinstance(v, str)]
                diverse = sum(1 for k in categorical_keys if len(set(c.get(k) for c in ctxs)) >= 2)
                if categorical_keys and diverse == 0:
                    warnings.append("build_context() returns same categorical values every time — specialist preservation won't work effectively")
                else:
                    ok.append(f"build_context() present, {len(ctxs[0])} key(s), categorically diverse")
            else:
                warnings.append("build_context() returned non-dict — expected dict")
        except Exception as e:
            warnings.append(f"build_context() raised: {e}")

    # 10. Check prompts exist
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
        top5 = sorted(playbook, key=lambda p: normalize_confidence(p.get("confidence", 0)), reverse=True)[:5]
        tbl = Table(title=f"Playbook — {len(playbook)} principles", show_header=True)
        tbl.add_column("Topic")
        tbl.add_column("Conf", justify="right")
        tbl.add_column("Principle")
        for p in top5:
            tbl.add_row(
                p.get("topic", ""),
                f"{normalize_confidence(p.get('confidence', 0)):.0%}",
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
    pref_path = domain_path / "training_preferences.jsonl"
    threshold_path = domain_path / "abstain_threshold.json"
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
            if pref_path.exists():
                n_pref = sum(1 for l in pref_path.read_text().splitlines() if l.strip())
                msg += f"   training_preferences.jsonl: {n_pref} pairs"
            if threshold_path.exists():
                threshold = json.loads(threshold_path.read_text()).get("threshold")
                msg += f"   abstain_threshold: {threshold}"
            console.print(f"\n[dim]{msg}[/dim]")
        except Exception:
            pass


def cmd_pack(args):
    """Bundle a domain into a shareable .zip pack."""
    import zipfile

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    # Read or create pack.json
    pack_path = domain_path / "pack.json"
    if pack_path.exists():
        pack = json.loads(pack_path.read_text())
    else:
        # Generate minimal pack.json from available info
        sim_path = domain_path / "simulation.py"
        metric = "score"
        if sim_path.exists():
            for line in sim_path.read_text().splitlines():
                if "METRIC_NAME" in line and "=" in line:
                    metric = line.split("=", 1)[1].strip().strip('"\'')
                    break
        pack = {
            "name": args.domain,
            "version": "1.0.0",
            "author": "",
            "description": "",
            "metric": metric,
            "autoforge_version": "1.0",
            "evals": "evals/scenarios.jsonl",
        }
        pack_path.write_text(json.dumps(pack, indent=2) + "\n")
        print(f"  Created pack.json (edit version/author/description before sharing)")

    # Files to include in the pack
    include = [
        "simulation.py",
        "mission.md",
        "pack.json",
        "prompts/brain.md",
        "prompts/extract.md",
        "prompts/director.md",
        "playbook.jsonl",
        "champion_archetype.json",
        "top_candidates.json",
        "abstain_threshold.json",
    ]
    # Include evals/ folder if it exists
    evals_path = domain_path / "evals"
    if evals_path.exists():
        for f in evals_path.iterdir():
            include.append(f"evals/{f.name}")

    name    = pack.get("name", args.domain)
    version = pack.get("version", "1.0.0")
    zip_name = f"{name}-{version}.zip"
    zip_path = ENGINE_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            src = domain_path / rel
            if src.exists():
                zf.write(src, f"{name}/{rel}")

    size_kb = zip_path.stat().st_size // 1024
    print(f"\nPacked {args.domain} → {zip_name}  ({size_kb} KB)")
    files_included = [r for r in include if (domain_path / r).exists()]
    for f in files_included:
        print(f"  {f}")
    print(f"\nShare {zip_name} — install with: python run.py install {zip_name}")


def cmd_install(args):
    """Install a domain pack from a .zip file."""
    import zipfile

    # Handle both absolute and relative paths
    from pathlib import Path as _Path
    zip_path = _Path(args.pack) if _Path(args.pack).is_absolute() else ENGINE_ROOT / args.pack

    if not zip_path.exists():
        print(f"Pack file not found: {zip_path}")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            print("Empty zip file.")
            sys.exit(1)
        # Detect domain name from top-level folder in zip
        domain_name = names[0].split("/")[0]
        domain_path = ENGINE_ROOT / domain_name

        if domain_path.exists() and not getattr(args, "force", False):
            print(f"Domain '{domain_name}' already exists. Use --force to overwrite.")
            sys.exit(1)

        domain_path.mkdir(exist_ok=True)
        zf.extractall(ENGINE_ROOT)

    # Move files from domain_name/domain_name/ to domain_name/ if nested
    nested = domain_path / domain_name
    if nested.exists():
        import shutil
        for item in nested.iterdir():
            shutil.move(str(item), str(domain_path / item.name))
        nested.rmdir()

    pack_file = domain_path / "pack.json"
    if pack_file.exists():
        pack = json.loads(pack_file.read_text())
        print(f"\nInstalled: {pack.get('name', domain_name)} v{pack.get('version', '?')}")
        print(f"  {pack.get('description', '')}")
        print(f"  Metric: {pack.get('metric', '?')}")
    else:
        print(f"\nInstalled: {domain_name}")

    print(f"\nNext steps:")
    print(f"  python run.py calibrate --domain {domain_name}")
    print(f"  python run.py run --domain {domain_name} --batches 5 --rounds 100")
    print(f"  python run.py eval --domain {domain_name}   # if evals are included")


def cmd_eval(args):
    """Run the champion strategy against eval scenarios and report pass/fail."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    evals_path = domain_path / "evals" / "scenarios.jsonl"
    if not evals_path.exists():
        print(f"No evals found at {evals_path}")
        print("Add eval scenarios in evals/scenarios.jsonl:")
        print('  {"id": "test_1", "state": {...}, "description": "...", "min_score": 0}')
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
        sys.exit(1)

    # Load champion or top candidates
    champion_path = domain_path / "champion_archetype.json"
    top_path      = domain_path / "top_candidates.json"
    strategies = []

    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            strategies = [{"name": champ.get("name", "champion"), "strategy": champ["strategy"]}]
        except Exception:
            pass
    if not strategies and top_path.exists():
        try:
            top = json.loads(top_path.read_text())
            strategies = [{"name": t.get("name", f"top_{i}"), "strategy": t["strategy"]} for i, t in enumerate(top[:4])]
        except Exception:
            pass
    if not strategies:
        print("No champion_archetype.json or top_candidates.json — run some batches first.")
        sys.exit(1)

    # Load eval scenarios (skip comment lines)
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
        print("evals/scenarios.jsonl is empty or has no valid JSON lines.")
        sys.exit(1)

    tbl = Table(title=f"Eval Results — {args.domain} ({len(scenarios)} scenarios)", show_header=True)
    tbl.add_column("ID")
    tbl.add_column("Description")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Min", justify="right")
    tbl.add_column("Pass", justify="center")

    passed = 0
    for scen in scenarios:
        state       = scen.get("state", {})
        min_score   = scen.get("min_score", 0)
        description = scen.get("description", "")[:40]
        scen_id     = scen.get("id", "?")

        # Score with each strategy, take best
        best_score = max(sim.simulate(s["strategy"], state) for s in strategies)
        ok = best_score >= min_score
        if ok:
            passed += 1
        color = "green" if ok else "red"
        tbl.add_row(
            scen_id,
            description,
            f"[{color}]{best_score:.2f}[/{color}]",
            f"{min_score:.2f}",
            f"[{color}]{'✓' if ok else '✗'}[/{color}]",
        )

    console.print(tbl)
    pct = passed / len(scenarios) * 100
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    console.print(f"\n[{color}]{passed}/{len(scenarios)} passed ({pct:.0f}%)[/{color}]")
    if pct < 80:
        console.print("[dim]Run more batches to improve — then re-eval.[/dim]")
    if passed < len(scenarios):
        sys.exit(1)


def cmd_tail(args):
    """Live snapshot of an in-progress run."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    checkpoint_path = domain_path / "run_checkpoint.json"
    ckpt = None
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
        except Exception:
            pass

    log_path = domain_path / "tournament_log.jsonl"
    recent_scores = []
    total_rounds = 0
    if log_path.exists():
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        total_rounds = len(lines)
        for line in lines[-50:]:
            try:
                entry = json.loads(line)
                score = entry.get("score")
                if score is not None:
                    recent_scores.append(float(score))
            except Exception:
                pass

    print(f"\nTail — {args.domain}")
    print(f"{'─'*40}")

    if ckpt:
        stage = "Stage 2 (AI)" if ckpt.get("use_brain") else "Stage 1 (evolutionary)"
        pb_sizes = ckpt.get("playbook_sizes", [])
        print(f"Batch:    {ckpt['global_batch']}  (year {ckpt['year_num']})")
        print(f"Stage:    {stage}")
        print(f"Playbook: {pb_sizes[-1] if pb_sizes else '?'} principles")
        pa = ckpt.get("prior_analysis") or {}
        if pa.get("verdict"):
            print(f"Verdict:  {pa['verdict']}")
        if pa.get("next_batch_focus"):
            print(f"Focus:    {pa['next_batch_focus'][:72]}")
        rows = ckpt.get("all_batch_rows", [])[-6:]
        if rows:
            print()
            for row in rows:
                arrow = "↑" if row["trend"] == "up" else ("↓" if row["trend"] == "down" else "─")
                print(f"  batch {row['batch']:3d}  avg {row['avg']:6.2f}  best {row['best']:6.2f}  {arrow}")
    else:
        print("No active run checkpoint.")

    print(f"\nRounds logged: {total_rounds}")
    if recent_scores:
        avg = sum(recent_scores) / len(recent_scores)
        print(f"Last 50:  avg {avg:.2f}  min {min(recent_scores):.2f}  max {max(recent_scores):.2f}")
    print()


def cmd_train(args):
    """Train a Stage 3 model from exported training data."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    from engine_export import _detect_domain_type, export_training_data
    domain_type = _detect_domain_type(domain_path)

    csv_path   = domain_path / "training_features.csv"
    jsonl_path = domain_path / "training_data.jsonl"

    # Auto-export if nothing exists yet
    if not csv_path.exists() and not jsonl_path.exists():
        print("No training data found — running export first...\n")
        export_training_data(domain_path)

    if domain_type == "numerical":
        if not csv_path.exists():
            print(f"No training_features.csv — run: python run.py export --domain {args.domain}")
            sys.exit(1)

        try:
            import xgboost as xgb  # type: ignore
            import pandas as pd    # type: ignore
            import numpy as np     # type: ignore
        except ImportError as e:
            print(f"Missing package: {e}")
            print("Install with: pip install xgboost pandas numpy")
            sys.exit(1)

        print(f"Training XGBoost on {csv_path.name}...")
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ("score", "uncertain")]
        X = df[feature_cols].values
        y = df["score"].values

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        params  = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
        model   = xgb.train(params, dtrain, num_boost_round=200,
                            evals=[(dtrain, "train")], verbose_eval=50)

        model_path = domain_path / "model.json"
        model.save_model(str(model_path))

        preds  = model.predict(dtrain)
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"\nSaved: {args.domain}/model.json")
        print(f"R²: {r2:.3f}  ({len(y)} examples, {len(feature_cols)} features)")
        print(f"\nLoad at inference:")
        print(f"  m = xgb.Booster(); m.load_model('{args.domain}/model.json')")

    else:
        if not jsonl_path.exists():
            print(f"No training_data.jsonl — run: python run.py export --domain {args.domain}")
            sys.exit(1)

        n = sum(1 for l in jsonl_path.read_text().splitlines() if l.strip())
        dn = args.domain
        print(f"Language domain — {n} training examples")
        print(f"\nFine-tune (Apple Silicon):")
        print(f"  mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \\")
        print(f"               --data {dn}/ --train --iters 1000")
        print(f"\nFuse + Ollama:")
        print(f"  mlx_lm.fuse --model mlx-community/Qwen3-4B-4bit \\")
        print(f"               --adapter-path {dn}/adapters --save-path {dn}/fused-model")
        print(f'  ollama create {dn.lower()} -f - <<EOF\nFROM ./{dn}/fused-model\nEOF')


def cmd_import(args):
    """
    Import real production decisions into the domain's training data.

    Reads a JSONL file of real decisions:
      {"state": {...}, "decision": {...}, "outcome": float}

    - Validates state keys match random_state() output structure
    - Validates decision keys match CANDIDATE_SCHEMA
    - Appends valid records to production_log.jsonl
    - Appends to tournament_log.jsonl so Stage 3 export includes real decisions
    - Reports distribution comparison vs simulated training data
    """
    import statistics
    from pathlib import Path as _Path

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    input_path = _Path(args.file)
    if not input_path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
        sys.exit(1)

    # Get expected structure from simulation
    expected_state_keys    = set(sim.random_state().keys())
    expected_decision_keys = set(sim.CANDIDATE_SCHEMA.get("properties", {}).keys())
    metric_name = sim.METRIC_NAME

    # Load existing tournament log to find max round number
    log_path  = domain_path / "tournament_log.jsonl"
    prod_path = domain_path / "production_log.jsonl"

    max_round = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                e = json.loads(line)
                max_round = max(max_round, e.get("round", 0))
            except Exception:
                pass

    # Parse and validate input records
    raw_lines = [l.strip() for l in input_path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    records = []
    skipped = []
    for i, line in enumerate(raw_lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped.append(f"line {i+1}: invalid JSON")
            continue

        state    = rec.get("state", {})
        decision = rec.get("decision", {})
        outcome  = rec.get("outcome")

        # Validate keys
        missing_state    = expected_state_keys - set(state.keys())
        missing_decision = expected_decision_keys - set(decision.keys())

        if missing_state:
            skipped.append(f"line {i+1}: missing state keys: {missing_state}")
            continue
        if missing_decision:
            skipped.append(f"line {i+1}: missing decision keys: {missing_decision}")
            continue
        if outcome is None:
            skipped.append(f"line {i+1}: missing 'outcome' field")
            continue
        try:
            outcome = float(outcome)
        except (TypeError, ValueError):
            skipped.append(f"line {i+1}: outcome must be a number")
            continue

        records.append({"state": state, "decision": decision, "outcome": outcome})

    print(f"\nImport: {args.file} → {args.domain}")
    print(f"  {len(records)} valid  /  {len(skipped)} skipped")
    if skipped:
        for msg in skipped[:5]:
            print(f"  [skip] {msg}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    if not records:
        print("Nothing to import.")
        return

    # Compute score stats
    outcomes = [r["outcome"] for r in records]
    median_outcome = statistics.median(outcomes)
    print(f"  Score range: {min(outcomes):.2f} – {max(outcomes):.2f}  median {median_outcome:.2f}")

    # Compare vs simulation distribution
    sim_states = [sim.random_state() for _ in range(min(len(records) * 2, 200))]
    print(f"\n  Distribution comparison (imported vs simulated):")
    for key in sorted(expected_state_keys):
        imported_vals = [r["state"].get(key) for r in records if isinstance(r["state"].get(key), (int, float))]
        sim_vals      = [s.get(key) for s in sim_states if isinstance(s.get(key), (int, float))]
        if imported_vals and sim_vals:
            print(f"    {key}:  imported median {statistics.median(imported_vals):.2f}  |  sim median {statistics.median(sim_vals):.2f}")

    # Write to production_log.jsonl
    with open(prod_path, "a") as f:
        for rec in records:
            entry = {
                "source":   "production",
                "state":    rec["state"],
                "decision": rec["decision"],
                "outcome":  rec["outcome"],
            }
            f.write(json.dumps(entry) + "\n")

    # Append to tournament_log.jsonl so Stage 3 export includes real decisions
    with open(log_path, "a") as f:
        for i, rec in enumerate(records):
            entry = {
                "round":  max_round + i + 1,
                "score":  rec["outcome"],
                "metric": metric_name,
                "winner": rec["decision"],
                "state":  rec["state"],
                "source": "production",
            }
            f.write(json.dumps(entry) + "\n")

    console.print(f"\n[green]Imported {len(records)} records.[/green]")
    console.print(f"  production_log.jsonl — provenance trail")
    console.print(f"  tournament_log.jsonl — {len(records)} records added to Stage 3 training data")
    console.print(f"\n[dim]Run 'python run.py export --domain {args.domain}' to include in Stage 3.[/dim]")
