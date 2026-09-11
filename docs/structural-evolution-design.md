# Structural evolution design (proposal only)

This document proposes a separately gated experimental mode. It does not authorize or implement structural mutation.

## Objective and non-goals

The objective is to test whether tightly bounded changes to existing MaleCNS connections improve a prespecified paper-trading fitness measure under identical market replays. The checksum-verified MaleCNS v1.0 graph remains the baseline.

The first structural experiments must not add neurons, create arbitrary synapses, delete CSR entries, use an LLM policy, select actions from profit outside the fixed decoder, or connect to live execution.

## Immutable baseline

Each individual records the following baseline provenance:

- MaleCNS release and source checksums
- graph field and CSR hashes
- original edge count and endpoint hashes
- original weight hash
- eligible-edge-set definition and hash
- structural experiment schema version

The runtime retains every original CSR edge. A mutation changes only an overlay applied to eligible baseline edge weights. This keeps original topology recoverable and makes graph-distance accounting exact.

## Eligible edge subset

Phase A should use only the existing 7,835 KC-to-MBON07/11 edges already identified by the candidate memory circuit. Eligibility is defined by stable baseline edge indices plus pre/post cell IDs and is checksum locked.

Expanding eligibility requires a new experiment schema and explicit approval. Candidate expansions should be anatomy-motivated, bounded subcircuits rather than all 25,582,938 directed edges.

## Mutation representation

Each structural genome contains a sparse, sorted overlay:

- baseline edge index
- pre and post cell IDs
- inherited weight multiplier
- optional enabled mask in a later phase
- originating generation and parent
- mutation event identifier

Phase A permits weight multipliers only. Multipliers are composed relative to the original baseline weight, not repeatedly rounded from a parent's materialized float array.

Suggested initial bounds are 0.8 to 1.25. Sign changes are forbidden. Existing zero/nonzero structure is unchanged.

Phase B may add explicit weakening with a lower bound above zero. Phase C may test reversible edge masking, represented as an overlay flag while retaining the original CSR edge and weight. Each phase requires separate approval and analysis.

## Mutation budget

Start with a maximum of 8 changed eligible edges per offspring and a per-eligible-edge proposal rate no greater than 0.0001. At most one new structural event should be accepted for a given edge in one birth operation.

Every offspring records proposed, accepted, clipped, duplicate, and rejected mutations. Experiments fail closed if the accepted count exceeds the configured budget or an ineligible edge is referenced.

## Selection and inheritance

The physiology genome, structural overlay, and acquired plastic memory remain separate artifacts.

- Darwinian mode inherits the physiology genome and structural overlay, then resets acquired memory.
- Lamarckian mode inherits independent copies of the physiology genome, structural overlay, and only `memory_u` / `memory_w`.
- Transient voltage, conductance, refractory state, queues, spike counts, rate traces, adaptation state, and market/account state always reset.
- Elites retain an exact structural overlay; offspring receive a deep copy before mutation.

## Audit artifacts

Each generation should include:

- baseline and eligible-set hashes
- full parent/child lineage
- sparse mutation event log
- structural mutation budget and utilization
- changed-edge count and fraction of eligible and total graph edges
- multiplier and mask distributions
- champion overlay and materialized-weight hash
- fitness components and identical market-input sequence hash
- runtime and peak memory measurements

A replay tool should reconstruct any champion from the immutable baseline plus its overlay and verify all hashes before evaluation.

## Validation gates

Before enabling Phase A:

1. Existing fixed-topology synthetic and Coinbase-public replay tests remain green.
2. Structural mode is explicit and paper-only.
3. Zero mutation budget reproduces fixed-topology outputs bit-for-bit.
4. Same seed and replay reproduce identical overlays and results.
5. Mutation bounds, budget, eligibility, sign preservation, elite preservation, and deep-copy behavior have tests.
6. Darwinian and Lamarckian transient-reset tests pass on the full graph.
7. Evolution imports neither live broker nor AgentKit action modules.
8. Output directories cannot overwrite prior experiments without explicit opt-in.

Before interpreting fitness, run multiple seeds and held-out replay windows. Report selection pressure, variance, turnover, drawdown, and compute cost. A better in-sample score alone is not evidence of profitable learning or biological validity.

## Recommended first experiment

Use Phase A only: the 7,835 existing KC-to-MBON07/11 edges, weight multipliers in [0.8, 1.25], no masking, no sign changes, and an 8-edge mutation budget. Compare it against a zero-budget control with matched seeds and replay files. Do not proceed to broader subcircuits until the control is reproducible and the effect survives held-out replays.
