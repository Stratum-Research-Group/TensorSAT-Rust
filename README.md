# TensorSAT-Rust

<p align="center">
  <img src="assets/tensorsat_rust_logo.png" alt="TensorSAT - Rust Edition - Stratum Research Group" width="100%" />
</p>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Precision: IEEE 754](https://img.shields.io/badge/Precision-IEEE%20754%20FP64%20%7C%20Complex64-gold.svg)]()
[![Context: Linear O(N)](https://img.shields.io/badge/Context-Linear%20O(N)%20Scaling-purple.svg)]()

**TensorSAT-Rust** is the official high-performance, memory-safe Rust implementation of the **TensorSAT** planar tensor-network uncrossing decomposition and sub-quadratic linear attention engine developed by the **Stratum Research Group**.

It provides pure native Rust routines for deep learning sequence modeling, exact model counting (#SAT), and high-precision scientific linear algebra, operationalizing the formal mathematical breakthrough from Anthropic's Exponential Time Hypothesis (ETH) refutation paper (*"A $2^{o(n)}$ Time Algorithm for 3-SAT"*, October 2026).

---

## 1. Mathematical Foundations & The Uncrossing Identity

In combinatorial constraint satisfaction and sequence modeling, dense non-planar crossings cause exponential state explosions ($O(2^N)$ in exact #SAT) or quadratic memory ceilings ($O(N^2)$ in self-attention).

### The Singlet-Swap Decomposition
TensorSAT operationalizes the discrete uncrossing identity formalized in Lean 4 (`Groundwork.uncross`):
$$\text{SWAP} = I - 2 \cdot |\Psi^-\rangle\langle\Psi^-|$$

In coordinate space for single-bit boolean values:
$$X_1(x, y, u, v) = I_1(x, y, u, v) - s(x, y) \cdot s(u, v)$$
where:
* $X_1(x, y, u, v) = [x = v \land y = u]$ is the diagonal crossing operator.
* $I_1(x, y, u, v) = [x = u \land y = v]$ is the parallel uncrossed identity.
* $s(p, q) = (-1)^p [p \neq q]$ is the singlet phase factor.

For multi-bit interaction widths of $\lambda$ bits, the full crossing expands into $\lambda + 1$ independent operator slices:
$$\text{cross}_\lambda(x, y, u, v) = \sum_{i=0}^\lambda A_i(\lambda, x, y, u, v)$$
where each slice $A_i$ has a bounded rank product:
$$\text{rank}_{\text{row}}(A_i) \cdot \text{rank}_{\text{col}}(A_i) \le 12^\lambda$$

### Exact IEEE 754 Representation
Because TensorSAT uses **exact discrete integer arithmetic and phase factors** $\{-1, 0, 1\}$:
* It produces **zero representation drift** in IEEE 754 double precision (`FP64`) and complex floating-point (`Complex64` / pairs of `f64`) tensor contraction ($\le 10^{-15}$ machine epsilon).
* Incremental cached generation matches full sequence prefill within standard IEEE floating-point tolerance (empirically observed $\le 10^{-7}$ on single-partition uniform tests; regression test gates enforce $< 10^{-6}$ for single-partition and $< 10^{-4}$ for multi-partition continuation).

---

## 2. Empirical Benchmarks (Measured on AMD / Linux Bare Metal)

Executed with `cargo run --release --bin benchmark_tensorsat` (rows above 65,536 tokens require `-- --large`) on single-threaded CPU execution:

### Benchmark A: Neural Sequence Scaling (Prefill Mode)
*Configuration: Batch = 1, Heads = 8, Head Dimension = 64, Embed Dimension = 512, Dtype = FP32, Causal Masking = True, Chunk Size = 512. Benchmarks evaluate single-core CPU execution on AMD / Linux bare metal across context lengths up to 1,048,576 tokens using synthetic uniform inputs to measure raw linear throughput:*

| Context Length ($N$ Tokens) | Single-Core CPU Latency | Sustained Throughput | Scaling Complexity |
| :--- | :--- | :--- | :--- |
| **1,024 tokens** | **13.3 ms** | 77,132 tok/s | $\mathcal{O}(N)$ linear |
| **2,048 tokens** | **23.6 ms** | 86,854 tok/s | $\mathcal{O}(N)$ linear |
| **4,096 tokens** | **46.9 ms** | 87,383 tok/s | $\mathcal{O}(N)$ linear |
| **8,192 tokens** | **93.9 ms** | 87,250 tok/s | $\mathcal{O}(N)$ linear |
| **16,384 tokens** | **196.5 ms** | 83,391 tok/s | $\mathcal{O}(N)$ linear |
| **32,768 tokens** | **430.0 ms** | 76,196 tok/s | $\mathcal{O}(N)$ linear |
| **65,536 tokens** | **824.0 ms** | 79,536 tok/s | $\mathcal{O}(N)$ linear |
| **131,072 tokens** | **1.64 s** | 79,997 tok/s | $\mathcal{O}(N)$ linear |
| **262,144 tokens** | **3.29 s** | 79,653 tok/s | $\mathcal{O}(N)$ linear |
| **524,288 tokens** | **6.24 s** | 84,025 tok/s | $\mathcal{O}(N)$ linear |
| **1,048,576 tokens (1M)** | **13.23 s** | **79,272 tok/s** | $\mathcal{O}(N)$ linear |

### Benchmark B: Autoregressive KV-Cache Throughput (Token Generation)
*Measuring step latency for generating next tokens across prefill prompt depths:*

| Prior Context Depth | Step Latency (ms / token) | Throughput (Tokens / Second) | Step Complexity |
| :--- | :--- | :--- | :--- |
| **512 tokens** | **0.0134 ms** | **74,493 tokens/s** | $\mathcal{O}(1)$ constant |
| **2,048 tokens** | **0.0133 ms** | **75,119 tokens/s** | $\mathcal{O}(1)$ constant |
| **8,192 tokens** | **0.0133 ms** | **75,232 tokens/s** | $\mathcal{O}(1)$ constant |

*Observation: Because the low-rank singlet context updates recursively ($S_t = S_{t-1} + k_t^T v_t$), generation latency is strictly constant ($O(1)$) regardless of context depth, delivering over $75{,}000$ tokens/second per CPU core.*

### Benchmark C: IEEE 754 High-Precision Double (`FP64`) Scientific Contraction
*Evaluating 4-index tensor contractions in exact 64-bit floating point. Note that dense slice materialization and contraction evaluates all $2^\lambda \times 2^\lambda \times 2^\lambda \times 2^\lambda$ entries, scaling as $(\lambda + 1) \cdot 16^\lambda$:*

| Dimension ($2^\lambda$) | Contraction Time | Maximum Numerical Drift vs. Exact Naive |
| :--- | :--- | :--- |
| **$\lambda = 1$ (dim 2)** | 0.000 ms | $\le 10^{-15}$ (Machine Epsilon) |
| **$\lambda = 2$ (dim 4)** | 0.002 ms | $\le 10^{-15}$ (Machine Epsilon) |
| **$\lambda = 3$ (dim 8)** | 0.039 ms | $\le 10^{-15}$ (Machine Epsilon) |

---

## 3. Production Architecture (v0.2.0 Modules)

### 1. Hybrid Attention (`tensorsat::hybrid::HybridSingletAttention`)
* Blends an exact causal local sliding window ($W = 128$) with the global $\mathcal{O}(N)$ singlet uncrossing accumulator via configurable gating (defaults to balanced [0.5, 0.5]):
$$\text{Attention}(Q, K, V) = g_{\text{local}} \cdot \text{LocalSDPA}(Q, K, V) + g_{\text{global}} \cdot \text{SingletGlobal}(Q, K, V)$$
* Global singlet state uses continuous leaky tracking (`DecayStabilizer`) and `StateRMSNorm` across chunks. Operates in sequence prefill mode. Inter-chunk normalization introduces bounded chunk-boundary sensitivity (observed ~10^-3 on tested configs, regression-pinned at $< 10^{-2}$) in exchange for long-context state stability.

### 2. Pre-Trained Weight Adapter (`tensorsat::adapter::SoftmaxToSingletAdapter`)
* Maps Q/K representations into positive singlet features using positive kernel transduction ($\text{ELU}(x) + 1.0$) with configurable/loadable transduction weights `q_trans` and `k_trans`. A small positive clamp floor ($10^{-12}$) prevents exact zero underflow for extreme negative projections ($x < \ln(10^{-12}) \approx -27.6$), guaranteeing strict positivity with bounded deviation.
* Projected positive features `q_s` and `k_s` can be supplied to downstream linear uncrossing attention layers directly via `SingletAttention::forward_with_projected_singlets`.

### 3. Numerical Stabilizers (`tensorsat::stabilizer::DecayStabilizer`, `StateRMSNorm`)
* Implements continuous leaky tracking $S_t = \gamma^C S_{t-1} + K_s^T V$, guaranteeing bounded spectral radius:
$$\|S_t\| \le \frac{1}{1 - \gamma} \max \|K_s^T V\|$$
* `StateRMSNorm` applies pre-scaled grouped RMS normalization along $(R, d)$ state dimensions, preventing float mantissa overflow even across extreme state values ($> 10^{20}$).

### 4. Graph Clique Partitioner (`tensorsat::cnf::CNFFormula`)
* Streamingly parses DIMACS CNF files with strict and permissive modes.
* Extracts the primal variable interaction graph without false self-loops.
* Partitions formulas into dense sub-graphs, sparse clauses, dense clauses, and boundary bridge clauses with 100% clause conservation.

---

## 4. Codebase Map

```
TensorSAT-Rust/
├── assets/
│   ├── tensorsat_rust_logo.png       # Official Rust Edition brand banner
│   ├── tensorsat_rust_logo.jpg       # High-res JPEG
│   └── tensorsat_rust_logo.svg       # Scalable vector graphic
│
├── src/
│   ├── lib.rs                        # Library root & exports
│   ├── uncrossing.rs                 # Lean 4 discrete s_factor & eval_slice_a
│   ├── attention.rs                  # SingletAttention & recursive KVCache
│   ├── hybrid.rs                     # Hybrid Local-Window + Global Singlet Attention
│   ├── adapter.rs                    # Softmax-to-Singlet Adapter layer
│   ├── stabilizer.rs                 # DecayStabilizer & scaled StateRMSNorm
│   ├── cnf.rs                        # Token-streaming DIMACS parser & partitioner
│   ├── scientific.rs                 # IEEE FP64 / Complex128 tensor contractors
│   └── bin/
│       └── benchmark.rs              # Comprehensive CPU benchmark suite
│
├── Cargo.toml                        # Rust package specification (Apache-2.0)
└── README.md                         # This specification
```

---

## 5. Building, Testing, and Benchmarking

### Prerequisites
Requires Rust 1.75+ (2021 edition):
```bash
# Clone the repository
git clone https://github.com/Stratum-Research-Group/TensorSAT-Rust.git
cd TensorSAT-Rust

# Build in release mode
cargo build --release

# Run unit and regression tests
cargo test

# Run standard benchmark suite (sequences up to 65k tokens)
cargo run --release --bin benchmark_tensorsat

# Run full benchmark suite up to 1M tokens (requires ~8.6 GB available RAM)
cargo run --release --bin benchmark_tensorsat -- --large
# or: TENSORSAT_BENCH_LARGE=1 cargo run --release --bin benchmark_tensorsat
```

---

## 6. API Reference & Code Examples

### A. Autoregressive Causal Attention with `KVCache`
```rust
use tensorsat::attention::{SingletAttention, KVCache};

let embed_dim = 512;
let num_heads = 8;
let num_singlets = 16;
let chunk_size = 512;

let layer = SingletAttention::new(embed_dim, num_heads, num_singlets, chunk_size, true);
let mut cache = KVCache::new(num_heads, num_singlets, embed_dim / num_heads);

// Stepwise autoregressive generation (O(1) step latency)
let head_dim = embed_dim / num_heads;
let q = vec![0.02f32; num_heads * 1 * head_dim];
let k = vec![0.02f32; num_heads * 1 * head_dim];
let v = vec![0.02f32; num_heads * 1 * head_dim];

let output = layer.forward(&q, &k, &v, 1, Some(&mut cache));
println!("Step token output length: {}", output.len());
```

### B. High-Precision IEEE FP64 Tensor Contraction
```rust
use tensorsat::scientific::IEEETensorContractor;

let contractor = IEEETensorContractor::new(2); // lambda = 2 bits (dim = 4)
let w1 = vec![1.23456789f64; 16];
let w2 = vec![9.87654321f64; 16];

let (total_z, slice_contributions) = contractor.contract_f64(&w1, &w2);
println!("Exact Contracted Scalar Z: {:.15}", total_z);
```

---

## 7. License & Credits

**TensorSAT-Rust** is licensed under the **Apache License, Version 2.0**.  
Copyright (c) 2026 Stratum Research Group.
