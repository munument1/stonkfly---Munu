# EvoStonkFly MVP

This branch adds a **paper-only neuroevolution assay** around the existing Stonkfly MaleCNS connectome controller.

The connectome graph is not mutated. Each individual inherits a small bounded genome containing learning rate, reward/aversive stimulation sensitivity, decoder threshold, KC resting potential, and KC adaptation parameters. Every individual is evaluated against the same market sequence, then ranked by a risk-aware fitness score. Elites survive and generate mutated offspring for the next generation.

## Safety and scope

- No Coinbase account credentials are required for evolution or recording.
- No Coinbase AgentKit actions are invoked by the evolution module.
- No live order path exists in the evolution module.
- Coinbase recording uses public market endpoints only.
- Recorded market data can be replayed identically for every individual.
- Profitability is not claimed; this is an experimental selection framework.

## Prepare the original dataset

```bash
stonkfly prepare
stonkfly verify
```

## Check the machine before the first run

```bash
python -m stonkfly.evolution.doctor
```

To validate a Coinbase-public recording at the same time:

```bash
python -m stonkfly.evolution.doctor --replay recordings/btc-60m.jsonl
```

The doctor checks Python version, availability of a C++ compiler, checksum-verified MaleCNS data, and the optional replay file. On native Windows it emits a warning because the upstream neural kernel build currently uses Unix-style `c++`, `-fPIC`, and `.so` conventions. **WSL2/Linux is recommended for the first end-to-end run.**

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

For the very first smoke test, reduce the cost further:

```bash
python -m stonkfly.evolution \
  --inheritance both \
  --population 2 \
  --generations 2 \
  --steps 6 \
  --neural-ms 50 \
  --out runs/smoke
```

Results are written as:

- `config.json`
- `generation-0000.json`, `generation-0001.json`, ...
- `champion.json`

## Record Coinbase public market observations

This does not read account credentials and cannot submit orders:

```bash
python -m stonkfly.evolution.record \
  --product BTC-USDC \
  --steps 60 \
  --interval 60 \
  --out recordings/btc-60m.jsonl
```

The recorder stores the initial chart history plus timestamped public bid/ask observations. The completed JSONL file is then immutable input to an assay.

## Replay the same real market sequence for every fly

```bash
python -m stonkfly.evolution \
  --population 4 \
  --generations 5 \
  --steps 0 \
  --replay recordings/btc-60m.jsonl \
  --out runs/btc-replay
```

With `--replay`, `--steps 0` uses the full recording.

## Darwinian vs Lamarckian inheritance

The experiment has two explicit inheritance modes.

### Darwinian

```bash
python -m stonkfly.evolution \
  --inheritance darwinian \
  --population 4 \
  --generations 5 \
  --replay recordings/btc-60m.jsonl \
  --steps 0 \
  --out runs/darwin
```

Only the bounded physiology genome is inherited. Acquired KC/MBON memory is reset for offspring.

### Lamarckian

```bash
python -m stonkfly.evolution \
  --inheritance lamarckian \
  --population 4 \
  --generations 5 \
  --replay recordings/btc-60m.jsonl \
  --steps 0 \
  --out runs/lamarck
```

Offspring inherit both the physiology genome and the parent's acquired `memory_u` / `memory_w` plasticity state. Transient membrane voltage, spike counts, rate traces, and other momentary neural state are reset.

### Matched comparison

```bash
python -m stonkfly.evolution \
  --inheritance both \
  --population 4 \
  --generations 5 \
  --replay recordings/btc-60m.jsonl \
  --steps 0 \
  --out runs/inheritance-comparison
```

This runs Darwinian and Lamarckian populations with the same seed and market recording into separate subdirectories, making the comparison reproducible.

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

## What is and is not evolving

The current system evolves **parameters around an anatomically retained MaleCNS connectome**. It does not yet add/delete arbitrary synapses or neurons. That is deliberate: keeping the graph fixed makes it possible to ask whether selection can improve behavior while the original connectome remains recognizable.

A later structural-evolution phase can introduce tightly bounded synaptic mutations as a separate experiment rather than silently turning the fly connectome into a generic neural network.
