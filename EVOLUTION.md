# EvoStonkFly MVP

This branch adds a **paper-only neuroevolution assay** around the existing Stonkfly MaleCNS connectome controller.

The connectome graph is not mutated. Each individual inherits a small bounded genome containing learning rate, reward/aversive stimulation sensitivity, decoder threshold, KC resting potential, and KC adaptation parameters. Every individual is evaluated against the same deterministic synthetic market sequence, then ranked by a risk-aware fitness score. Elites survive unchanged and generate mutated offspring for the next generation.

## Safety and scope

- No Coinbase account credentials are read.
- No Coinbase AgentKit actions are invoked.
- No live order path exists in the evolution module.
- The current MVP uses `FixtureMarket`, not real market history.
- Profitability is not claimed; this is an experimental selection framework.

## Prepare the original dataset

Use the normal Stonkfly preparation step first:

```bash
stonkfly prepare
stonkfly verify
```

## Run a tiny assay

Start small because every individual runs the full connectome simulation:

```bash
python -m stonkfly.evolution \
  --population 4 \
  --generations 1 \
  --steps 6 \
  --neural-ms 100 \
  --out runs/evolution-demo
```

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

The next logical step is a recorded Coinbase-public-market replay so every individual sees the exact same real historical observation sequence. Lamarckian inheritance of learned KC/MBON memory can then be added as a separate experimental mode rather than silently mixing it with Darwinian parameter inheritance.
