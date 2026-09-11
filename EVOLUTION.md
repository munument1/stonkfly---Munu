# EvoStonkFly MVP

This branch adds a **paper-only neuroevolution assay** around the existing Stonkfly MaleCNS connectome controller.

The connectome graph is not mutated. Each individual inherits a small bounded genome containing learning rate, reward/aversive stimulation sensitivity, decoder threshold, KC resting potential, and KC adaptation parameters. Every individual is evaluated against the same market sequence, then ranked by a risk-aware fitness score. Elites survive unchanged and generate mutated offspring for the next generation.

## Safety and scope

- Evolution never reads Coinbase account credentials.
- Evolution never invokes Coinbase AgentKit actions.
- No live-order path exists in the evolution module.
- Coinbase recording uses public market endpoints only.
- Replays are paper-only and deterministic: every individual receives the same recorded observations.
- Profitability is not claimed; this is an experimental selection framework.

## Prepare the original dataset

Use the normal Stonkfly preparation step first:

```bash
stonkfly prepare
stonkfly verify
```

## Run a tiny synthetic assay

Start small because every individual runs the full connectome simulation:

```bash
python -m stonkfly.evolution \
  --population 4 \
  --generations 1 \
  --steps 6 \
  --neural-ms 100 \
  --out runs/evolution-demo
```

## Record real Coinbase public observations

The recorder uses the public Coinbase market client with no API key, portfolio, AgentKit action, or order submission. It seeds the recording with completed one-minute closes, then records the public top-of-book quote at the requested interval.

```bash
python -m stonkfly.evolution.record \
  --product BTC-USDC \
  --steps 60 \
  --interval 60 \
  --out runs/market/btc-1h.jsonl
```

The recorder writes to a `.partial` file and only renames it to the requested filename after all observations are complete, so interrupted captures are not silently treated as valid replays.

## Evolve on the exact same real-market recording

```bash
python -m stonkfly.evolution \
  --product BTC-USDC \
  --replay runs/market/btc-1h.jsonl \
  --steps 0 \
  --population 4 \
  --generations 3 \
  --neural-ms 100 \
  --out runs/evolution-btc-replay
```

With `--replay`, `--steps 0` means use every observation in the recording. A positive value evaluates only that many observations. The experiment records the replay SHA-256 in `config.json`, making it possible to verify that every generation was scored against the same source file.

This recorder is an observation capture, not arbitrary historical backfill. For robust claims, collect multiple recordings from different market regimes and evaluate champions on held-out recordings they did not evolve on.

Results are written as:

- `config.json`
- `generation-0000.json`, `generation-0001.json`, ...
- `champion.json`

## Fitness

The MVP score is:

```text
score = return_pct - 0.5 * max_drawdown_pct - 0.02 * trade_count
```

Trading fees are already reflected in the equity curve. The drawdown and churn terms reduce the chance that a lucky high-risk or hyperactive individual dominates selection.

## Current genome

- `eta`
- `reward_current`
- `aversive_current`
- `decoder_threshold_hz`
- `kc_rest`
- `adaptation_jump`
- `adaptation_tau`

Inheritance is currently Darwinian: only bounded genome parameters are inherited. Learned KC/MBON memory is intentionally not inherited yet. A separate Lamarckian mode can be added next so the two inheritance schemes can be compared rather than mixed together.
