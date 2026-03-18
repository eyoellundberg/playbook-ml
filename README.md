# The Engine

An autonomous strategy learning system. Describe a domain, run overnight, wake up to a trained local model that makes expert decisions for free.

---

## Installation

```bash
git clone https://github.com/eyoellundberg/engine
cd engine
pip install -e .   # installs anthropic + rich, adds 'engine' CLI alias
```

Or without installing:
```bash
pip install anthropic rich
python run.py <command>
```

Set your API key (only needed for Stage 2):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or add to MyDomain/.env — engine reads it automatically
```

---

## The Core Idea

**The simulation is the teacher, not the AI.** You build a deterministic scoring function for your domain. The engine runs thousands of strategy candidates against it. What consistently wins becomes the playbook.

**Frontier models (Sonnet/Haiku) design the strategy space and extract conditional principles.** They generate the best possible training signal — named archetypes with philosophy, conditional rules from what won. They are never used at inference time.

**A local model is trained on what the simulation proved.** XGBoost for numerical domains (grain pricing, demand scoring, rates). A fine-tuned small LLM for language domains (classification, routing, document scoring). Either way: runs locally, zero API cost, forever.

**Inspired by Karpathy's autoresearch — but generalized.** autoresearch autonomously improves a language model training loop. The Engine autonomously learns any structured decision domain.

---

## Two Filters

If both are true, The Engine can learn it:

1. **Can the job be described in a markdown file?** If someone can write what the job is, what good output looks like, and what the inputs are — the Engine can learn it.

2. **Is the output structured?** Pricing decisions, scoring, triage, routing, forecasting — any job that is: read inputs → produce structured output.

---

## Quickstart

```bash
git clone https://github.com/eyoellundberg/engine
cd engine
pip install anthropic rich

# Generate a domain from a description (Sonnet writes simulation.py + prompts)
python run.py bootstrap GrainMarketing \
  --description "corn/soy marketing for midwest US farms, optimizing \
  timing of cash grain sales against local elevator basis"

# Review GrainMarketing/simulation.py — calibrate until the right strategy wins
# in each scenario type. This is the only manual step.

# Validate domain before running (catches bad simulation.py early)
python run.py validate --domain MyDomain

# Stage 1 — free, evolutionary, no API calls
python run.py run --domain GrainMarketing --batches 10 --rounds 150

# Stage 2 — AI archetypes + director (~$0.50/run)
export ANTHROPIC_API_KEY=sk-ant-...
python run.py run --domain GrainMarketing --brain --batches 8 --rounds 150

# Fully autonomous: Stage 1 → Stage 2 auto-promotion on saturation
python run.py run --domain GrainMarketing --auto --batches 20 --rounds 150

# Check state anytime
python run.py status --domain GrainMarketing

# Export Stage 3 training data when saturated
python run.py export --domain GrainMarketing
```

---

## The Three Stages

### Stage 1 — Evolutionary mutation (free)

No API calls. `_generate_procedural_candidates()` generates 16 candidates per batch. Each batch evolves from the last: top winners are kept (elitism), mutated (gaussian noise on parameters), crossed over, with random candidates filling the rest for exploration. The playbook grows from Haiku extraction every 10 rounds.

After 10 batches the parameter space has narrowed toward what actually works — a warm start for Stage 2.

### Stage 2 — Frontier model direction (~$0.50/run)

Sonnet reads the warm playbook and generates 16 named strategy archetypes with philosophy, not just parameter values. "Drought Patience" — hold 60%+ through July in confirmed drought years. Each archetype is a full strategic position.

The sim tests all 16 across 150 random scenarios per batch. Haiku extracts 0-2 conditional principles every 10 rounds. The director (Sonnet) reads results between batches, retires losers, sharpens the next library. The champion archetype (most non-event wins) propagates to seed the next batch.

By batch 5, Sonnet is refining its own prior designs based on what the sim proved.

### Stage 3 — Local model (free forever)

After saturation, the tournament log contains thousands of labeled examples: (scenario state → winning strategy parameters). `run.py export` formats them as training data.

For numerical domains (grain, pricing, scoring): train XGBoost on the scenario features — tiny, fast, explainable, runs anywhere. For language domains (classification, routing based on text): fine-tune a small LLM (Qwen 1.5b via MLX-LM). The sim can still validate outputs. No more API calls.

---

## The Loop (precise)

```
bootstrap:
  Sonnet reads description → writes simulation.py + all prompts
  (one-time, ~$0.05, then never again)

Stage 1 batch:
  _generate_evolved_candidates(): elites + mutations + crossovers + random
  → 16 candidates tested across N rounds (no API call)
  → every 10 rounds: Haiku extracts 0-2 principles → playbook
  → batch end: director analyzes, sets hints for next batch
  → top 4 winners saved for next batch's evolution

Stage 2 batch:
  Sonnet reads: playbook + director hints + champion archetype
  → generates 16 named archetypes with philosophy (1 API call)
  → sim tests all 16 across N rounds (no API call per round)
  → every 10 rounds: Haiku extracts principles (1 API call per 10 rounds)
  → batch end: director analyzes (1 API call)
  → champion propagates to next batch

Saturation:
  director returns "saturated" → run stops
  → export training data
  → train local model
  → deploy: sim validates, local model decides
```

---

## Adding Your Domain

Four things to provide. Everything else is the engine.

```
MyDomain/
├── simulation.py      # the scoring function — the only hard part
│                      # exports: simulate(), random_state(),
│                      #          CANDIDATE_SCHEMA, METRIC_NAME
├── tournament.py      # copy from template/, adapt _build_context()
└── prompts/
    ├── brain.md       # tell Sonnet what 16 archetypes to generate
    ├── extract.md     # tell Haiku what principles to extract
    └── director.md    # give the director domain context
```

**`simulation.py` is the investment.** The engine only learns what the simulation teaches. If it rewards the wrong behavior, the whole engine optimizes toward garbage. Calibration checklist:

- Does the expected strategy type win in each scenario class?
- Does varying scenario factors change which strategy wins?
- Is the score range reasonable?
- No single strategy dominates regardless of scenario?

Real data is optional: place CSVs in `MyDomain/data/`. Only `simulation.py` reads it. `random_state()` samples from real distributions instead of synthetic ones. The engine doesn't change.

---

## Try It Now — StockTiming Example

A complete working domain is included. No configuration needed:

```bash
# Stage 1 — free, no API key
python run.py run --domain StockTiming --batches 5 --rounds 100

# Stage 2 — AI archetypes (needs API key)
python run.py run --domain StockTiming --brain --batches 5 --rounds 100

# Validate before running
python run.py validate --domain StockTiming

# Check what it learned
python run.py status --domain StockTiming
```

StockTiming optimizes moving average crossover parameters across simulated market regimes (trending, ranging, volatile, event). It's a minimal but complete example of the engine loop — use it to understand the system before building your own domain.

---

## Architecture

```
engine/
├── run.py              CLI: bootstrap / new / validate / run / export / status
├── engine_brain.py     Sonnet archetype library generator
│                       reads: prompts/brain.md + playbook + champion + hints
│                       writes: 16 named archetypes with strategy parameters
├── engine_extract.py   Haiku principle extractor (called every 10 rounds)
│                       reads: prompts/extract.md + winner/losers/context
│                       writes: 0-2 conditional principles → playbook
├── engine_export.py    Stage 3 training data exporter
│                       reads: tournament_log.jsonl + playbook
│                       writes: training_data.jsonl (messages format)
└── template/           copy this to start a new domain
    ├── simulation.py   skeleton with all required exports + calibration guide
    ├── tournament.py   skeleton tournament loop with evolution built in
    └── prompts/        skeleton brain.md / extract.md / director.md

MyDomain/               your domain (one per problem)
    ├── simulation.py   scoring function — only file with domain knowledge
    ├── tournament.py   tournament loop (mostly template, adapt _build_context)
    ├── prompts/        domain-specific AI instructions
    ├── data/           optional: real historical/calibration data
    ├── playbook.jsonl          learned conditional principles (commit this)
    ├── retired_topics.json     permanent blocklist (commit this)
    ├── champion_archetype.json best archetype from last batch (commit this)
    ├── top_candidates.json     top Stage 1 winners for evolution (commit this)
    ├── thinking_log.md         director reasoning trail (do not commit)
    ├── tournament_log.jsonl    round-by-round scores (do not commit)
    ├── training_data.jsonl     Stage 3 export (do not commit)
    └── last_run.json           last run summary (do not commit)
```

The engine root has zero domain knowledge. `simulation.py` is the only file that knows what your domain is.

---

## Convergence & Stopping

The director issues a verdict after every batch:

| Verdict | Meaning | Action |
|---|---|---|
| `converging` | Score and playbook improving | Keep running |
| `exploring` | Mixed results, still searching | Keep running |
| `stalled` | No progress for multiple batches | Adjust sim or prompts |
| `reward_hacking` | Score rising for wrong reason | Stop — fix the sim |
| `needs_calibration` | Sim rewarding wrong behavior | Fix simulation.py |
| `saturated` | Playbook full, score stable | Export and train local model |

`saturated` is the success state. The engine has learned everything this simulation can teach.

---

## Stage 3: Choosing Your Local Model

The choice depends on what your domain's inputs look like:

**Numerical domain** (grain prices, basis, demand, scores, rates):

```bash
# export produces features + labels
python run.py export --domain GrainMarketing
# train XGBoost — tiny, fast, explainable
import xgboost as xgb
model = xgb.train(params, dtrain)  # <1MB model, microsecond inference
```

**Language domain** (text classification, routing, document scoring):

```bash
# export produces messages JSONL
python run.py export --domain TicketTriage
# fine-tune Qwen 1.5b via MLX-LM on Apple Silicon
mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
             --data TicketTriage/ --train --iters 1000
```

XGBoost is not "less AI" than Qwen — it's the right tool for a regression problem. A gradient boosted tree trained on 1,000 tournament examples will outperform a raw LLM on numerical decisions. Qwen earns its place when the inputs contain language that needs understanding.

---

## Cost

```
bootstrap              ~$0.05   one-time per domain
Stage 1 (procedural)   $0.00    evolutionary, no API
Stage 2 (AI)           ~$0.50   per 5-batch run to saturation
Stage 3 training       $0.00    local compute only
Stage 3 inference      $0.00    forever
```

Total to go from description → trained local model: **~$4**.

---

## BYOK — Model Configuration

All models are configurable. Set in `MyDomain/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Override any model (optional)
ENGINE_DIRECTOR_MODEL=claude-sonnet-4-6        # between-batch director
ENGINE_LIBRARY_MODEL=claude-sonnet-4-6         # archetype generation
ENGINE_EXTRACT_MODEL=claude-haiku-4-5-20251001 # principle extraction (runs every 10 rounds)
```

Tighter budget: swap Sonnet for Haiku everywhere. More quality: use Opus for the director. Same engine, different cost/quality point.

---

## Compared to autoresearch

Karpathy's autoresearch autonomously improves a language model training loop: one experiment at a time, agent modifies `train.py`, runs for 5 minutes, checks `val_bpb`, keeps or discards. Elegant and simple.

The Engine generalizes this to any structured decision domain:

| | autoresearch | The Engine |
|---|---|---|
| Strategy space | Open-ended (any code change) | Constrained (CANDIDATE_SCHEMA) |
| Experiments | One at a time, 5 min each | 16 simultaneously, milliseconds |
| Tracking | git commits | playbook.jsonl + thinking_log.md |
| Output | Improved train.py | Trained local model |
| Domain | Language model training | Any structured decision |

Both are valid. autoresearch goes deep on one idea. The Engine goes wide across many simultaneously, then distills what won into a deployable model.

---

## Hardware

Designed to run on a Mac Mini M4 overnight unattended. Simulation runs in pure Python — no GPU needed for scoring. Stage 3 fine-tuning (if LLM): MLX-LM on Apple Silicon. Stage 3 training (if XGBoost): any hardware, seconds to train.
