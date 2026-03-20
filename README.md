# Autoforge

**Describe a decision. Get a deployed specialist. No labeled data.**

One command turns a plain-English description into a trained local model that makes domain-specific decisions — freight quoting, fraud detection, loan approval, anything with a scorable outcome. Autoforge writes the simulator, runs hundreds of thousands of competing strategies, and ships a standalone specialist you deploy and retrain on real outcomes.

## Quick start

```bash
uv pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

autoforge FreightQuoting "freight brokerage load quoting, spot and contract lanes"
```

```
● [1/3] Bootstrap domain     — AI writes the simulator        (~$0.30)
● [2/3] Validate simulation  — checks schema, score range      (free)
● [3/3] Train                — Stage 1 → Stage 2 → specialist (~$0.50)

✔ Done in 12m 34s.  Deploy: cp -r FreightQuoting/specialist/ /your/app/
```

## Python SDK

If you already have a scoring function, skip the CLI entirely:

```python
pip install autoforge
```

```python
import random
from autoforge import run

# 1. Define what a scenario looks like
def my_state():
    return {
        "demand":      random.uniform(0.0, 1.0),
        "competition": random.choice(["low", "medium", "high"]),
        "is_peak":     random.random() > 0.8,
    }

# 2. Define how to score a strategy against a scenario
def my_simulate(candidate, state):
    base = candidate["price"] * state["demand"]
    if state["competition"] == "high":
        base *= (1 - candidate["discount"])
    return base

# 3. Define what a strategy looks like
SCHEMA = {
    "type": "object",
    "properties": {
        "price":    {"type": "number", "minimum": 0.5, "maximum": 2.0},
        "discount": {"type": "number", "minimum": 0.0, "maximum": 0.4},
    },
    "required": ["price", "discount"],
    "additionalProperties": False,
}

# 4. Run
champion = run(simulate=my_simulate, state=my_state, schema=SCHEMA)

print(champion.strategy)    # {"price": 1.42, "discount": 0.08}
print(champion.philosophy)  # "Moderate price with low discount wins medium competition"
print(champion.playbook)    # [{"principle": "...", "confidence": 0.87}, ...]
print(champion.score)       # 0.94
```

Set `ANTHROPIC_API_KEY` in your environment or pass `api_key=` directly. Run without a key using `brain=False` for evolutionary-only mode (free, no AI calls).

## How it works

```
describe → simulate → compete → extract → deploy → retrain on reality
```

1. You describe the domain in plain English
2. AI writes a generative simulator — `simulate()` returns `P(outcome | features) × magnitude`
3. 16 strategies compete across hundreds of thousands of scenarios
4. Conditional principles extracted — *"when volatility > 20% and lane is new, conservative margins win"*
5. A hypothesis-driven director tracks what's being learned, retires weak principles, steers exploration
6. XGBoost trained on winners — exported as a standalone specialist with abstention

The simulation is the teacher. What consistently wins becomes your training data. You never label a single example.

## The specialist

Training produces a standalone module — no framework, no API calls, no Autoforge dependency:

```python
from specialist.predict import predict, record

result = predict({"lane_distance": 450, "weight": 42000, "volatility": 0.12})
# → {"strategy": {"bid_margin": 0.12}, "score": 6.4}
# → {"action": "ABSTAIN", "reason": "strategies too close — escalate"}

record(scenario, result["strategy"], actual_profit)
```

The specialist retrains itself on real outcomes with `python retrain.py`. No Autoforge, no API calls.

Autoforge is scaffolding. You remove it when the building stands.

## Cost

| Stage | Cost |
|---|---|
| Bootstrap | ~$0.30 (one-time) |
| Stage 1 — evolutionary tournament | free |
| Stage 2 — AI-directed exploration | ~$0.50/run |
| Specialist inference + retraining | free forever |

Runs unattended on a Mac Mini M4. No GPU needed.

## Learn more

Full documentation in the [wiki](../../wiki): CLI reference, simulator contract, self-evolution, reality grounding, hypothesis tracking, domain packs, model configuration.

## License

MIT
