# Playbook ML

**Describe a decision. Get a deployed specialist. No labeled data.**

The idea: describe a business decision in plain English. Playbook ML writes a simulator for your domain, runs thousands of competing strategies against it, distills what wins into training data, and fine-tunes a local model you drop into your app. You get a specialist that reasons about your domain — freight quoting, grain marketing, fraud scoring, anything with a scorable outcome. No labeled data, no ML team, ~$0.80.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install playbook-ml
playbook-ml
```

```
  What do you want to call it?  > GrainExpert
  What does it do?              > grain marketing decisions for midwest corn farmers

  Planning GrainExpert...

  Metric:     expected_profit
  Parameters: basis_threshold, carry_weight, hedge_ratio, storage_capacity... (24 total)
  Scenarios:  SA crop status, seasonal timing, cash flow pressure, basis level...
  Tension:    wide carry favors holding, but cash flow and basis risk favor selling

  Generate GrainExpert? [Y/n]

Bootstrap   ✔  world model written                                      (~$0.30)
Validate    ✔  simulation looks good
Train       ████░░░░  batch 3/8...                                      (~$0.50)
Fine-tune   ████░░░░  Qwen3-4B LoRA on Apple Silicon...                 (free, ~20 min)

Done. Specialist lives at ~/.playbook-ml/GrainExpert/specialist/
```

Playbook ML disappears after this. GrainExpert is yours.

## How it works

- **Simulate.** Opus acts as a domain expert and writes a generative simulator — 20-35 variables that actually drive decisions in your domain. Not a toy with 5 parameters.
- **Compete.** Strategies fight across hundreds of thousands of random scenarios. What consistently wins becomes your training signal. You never label a single example.
- **Fine-tune.** Results are verbalized into natural language and used to fine-tune Qwen3-4B locally via LoRA. The specialist learns to reason, not just look up a score.

## The specialist

A standalone folder. No Playbook ML dependency, no API calls, no framework.

```bash
playbook-ml ask --domain GrainExpert "basis is -0.35, SA drought risk, October"
# → Hold. SA drought risk in October overrides the slightly negative basis...
```

```python
from specialist.ask import ask, record

result = ask({"basis": -0.35, "carry": 0.02, "south_america": "drought_risk"})
# → "Hold. SA drought risk overrides the slightly negative basis.
#    Holding 4-6 weeks improves basis by $0.06-0.11 in ~75% of cases..."

record(features, actual_outcome)  # log real outcomes for retraining
```

Retrains on real outcomes automatically:

```
0 2 * * * cd /your/app && python specialist/retrain.py
```

Build whatever you want on top — API, iOS app, Slack bot, agent tool.

## Design choices

- **Simulation is the teacher.** The simulator is the hardest part to get right, and it's the only part that matters. Opus spends most of its budget here. A shallow sim produces a shallow specialist.
- **Fixed domain, not fixed model.** The specialist is trained on *your* domain's logic, not a generic one. Grain marketing and fraud scoring need different reasoning, not the same model with a different prompt.
- **Runs unattended.** Training runs overnight on a Mac Mini. No GPU needed. The fine-tune is local, free, and private.
- **Playbook ML is scaffolding.** Once the specialist is deployed, you remove it. The building stands on its own.

## Cost

| | Cost |
|---|---|
| Bootstrap (one-time) | ~$0.30 |
| Training run | ~$0.50 |
| Fine-tune Qwen3-4B (local) | free, ~20 min on Apple Silicon |
| Specialist inference + retraining | free forever |

## License

MIT
