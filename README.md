# TensorSAT-Rust

High-performance Rust implementation of the **TensorSAT** planar tensor-network uncrossing decomposition and sub-quadratic linear attention engine.

## Features

- **Core Uncrossing Algebra (`uncrossing`)**:
  - `s_factor(p, q)`: Exact Lean 4 phase factor $s(p, q) = (-1)^p [p \ne q]$.
  - `eval_slice_a(n_bits, i, x, y, u, v)`: Exact evaluation of uncrossing slice $A_i$ without dense allocations.
- **Scientific High-Precision Tensor Contraction (`scientific`)**:
  - `IEEETensorContractor`: Lossless 4-index contraction in IEEE `FP64` and `Complex64/128` within machine epsilon ($\le 10^{-15}$).
- **CNF Graph Partitioner (`cnf`)**:
  - Parses DIMACS CNF files and isolates dense interaction cliques from sparse clauses.
- **Sub-quadratic Attention Engine (`attention`)**:
  - `SingletAttention`: $\mathcal{O}(N)$ memory-efficient chunked causal prefix accumulation.
  - `KVCache`: Recursive invariant $\mathcal{O}(1)$ step latency and constant memory traffic.
- **Hybrid Attention (`hybrid`)**:
  - `HybridSingletAttention`: Fuses exact local banded sliding-window SDPA with global singlet uncrossing.
- **Numerical Stabilizers (`stabilizer`)**:
  - `StateRMSNorm` & `DecayStabilizer` for dynamic range control over long contexts.
- **Pre-Trained Weight Adapter (`adapter`)**:
  - `SoftmaxToSingletAdapter`: Positive ELU kernel transduction mapping softmax geometries into linear singlet spaces.

## Building and Testing

```bash
cargo build --release
cargo test
```

## Running Benchmarks

```bash
cargo run --release --bin benchmark_tensorsat
```
