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

## Benchmarks (Measured on AMD Ryzen / Linux Bare Metal)

Executed with `cargo run --release --bin benchmark_tensorsat`:

### 1. Neural Sequence Scaling (Prefill Mode)
| Tokens ($N$) | Latency (ms) | Scaling Complexity |
| :--- | :--- | :--- |
| **1,024** | 11.43 ms | $\mathcal{O}(N)$ linear |
| **2,048** | 22.78 ms | $\mathcal{O}(N)$ linear |
| **4,096** | 45.26 ms | $\mathcal{O}(N)$ linear |
| **8,192** | 90.68 ms | $\mathcal{O}(N)$ linear |
| **16,384** | 185.81 ms | $\mathcal{O}(N)$ linear |

### 2. Autoregressive Incremental Generation Throughput (`KVCache`)
| Past Prompt Depth | Step Latency (ms / token) | Throughput (Tokens / s) | Step Complexity |
| :--- | :--- | :--- | :--- |
| **512 tokens** | 0.0134 ms | **74,810 tokens/s** | $\mathcal{O}(1)$ constant |
| **2,048 tokens** | 0.0132 ms | **75,503 tokens/s** | $\mathcal{O}(1)$ constant |
| **8,192 tokens** | 0.0133 ms | **75,353 tokens/s** | $\mathcal{O}(1)$ constant |

### 3. IEEE 754 High-Precision Double (`FP64`) Contraction
| Dimension ($2^\lambda$) | Contraction Time | Numerical Drift vs. Exact Naive |
| :--- | :--- | :--- |
| **$\lambda = 1$ (dim 2)** | 0.000 ms | $\le 10^{-15}$ (Machine Epsilon) |
| **$\lambda = 2$ (dim 4)** | 0.002 ms | $\le 10^{-15}$ (Machine Epsilon) |
| **$\lambda = 3$ (dim 8)** | 0.040 ms | $\le 10^{-15}$ (Machine Epsilon) |
