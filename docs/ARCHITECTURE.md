# TensorSAT-Rust Architecture & Pipeline Specification

This document provides architectural diagrams, execution flowcharts, and component pipelines for the **TensorSAT-Rust** engine.

---

## 1. High-Level System Architecture

![TensorSAT-Rust Architecture](images/architecture.png)

TensorSAT decomposes SAT formulas, discrete tensor networks, and neural sequence representations into linear-complexity exact or bounded contractions.

```mermaid
graph TD
    subgraph Input ["Input Domains"]
        CNF["DIMACS CNF Formula (.cnf)"]
        TN["Discrete Planar Tensor Network"]
        SEQ["Autoregressive / Neural Sequence (Q, K, V)"]
    end

    subgraph CoreEngine ["TensorSAT Engine Core"]
        direction TB
        CNF_ENG["CNF Bounded Treewidth Parser & Dilution<br/>(src/cnf.rs)"]
        UNCROSS["Discrete Planar Uncrossing<br/>(src/uncrossing.rs)"]
        SCIENTIFIC["FP64 Contraction Contract<br/>(src/scientific.rs)"]
        STABILIZER["Decay & RMS State Norm<br/>(src/stabilizer.rs)"]
        ADAPTER["Softmax-to-Singlet Adapter<br/>(src/adapter.rs)"]
        ATTENTION["Singlet Attention & KVCache<br/>(src/attention.rs)"]
        HYBRID["Hybrid Gated Causal Sequence Layer<br/>(src/hybrid.rs)"]
    end

    subgraph Output ["Target Workloads"]
        SAT_SOLV["Linear Time SAT / Exact Counting (#SAT)"]
        FP64_CONT["Zero-Drift Scientific Simulation"]
        NEURAL_INF["O(N) Long-Context Inference & Generation"]
    end

    CNF --> CNF_ENG
    TN --> UNCROSS
    TN --> SCIENTIFIC
    SEQ --> ADAPTER
    SEQ --> ATTENTION
    SEQ --> HYBRID

    CNF_ENG --> UNCROSS
    UNCROSS --> SCIENTIFIC
    SCIENTIFIC --> SAT_SOLV
    SCIENTIFIC --> FP64_CONT

    ADAPTER --> ATTENTION
    ATTENTION --> HYBRID
    STABILIZER --> HYBRID
    HYBRID --> NEURAL_INF
```

---

## 2. Neural Attention & Hybrid Layer Pipeline

The neural sequence execution pipeline supports both full sequence prefill ($\mathcal{O}(N)$ chunked singlet attention) and recurrent incremental decoding ($\mathcal{O}(1)$ step latency via `SingletKVCache`).

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Client / Model Runner
    participant Adapter as SoftmaxToSingletAdapter
    participant Attn as SingletAttention
    participant Hybrid as HybridSequenceLayer
    participant Cache as SingletKVCache

    alt Prefill Mode (Sequence Length N)
        Caller->>Adapter: project_queries_and_keys(Q, K)
        Adapter-->>Caller: Positive Singlet Projections (Q_s, K_s >= 1e-12)
        Caller->>Hybrid: forward(Q, K, V, N, None)
        Hybrid->>Attn: forward_with_projected_singlets(Q, K, V, Q_s, K_s, N, None)
        Note over Attn: Chunked Causal Prefix Sums (O(N) Complexity)
        Attn-->>Hybrid: Global Singlet Output
        Note over Hybrid: Local Sliding Window + Gated Blend [g_local, g_global]
        Hybrid-->>Caller: Final Hidden States (N x EmbedDim)
    else Incremental Generation Mode (Step N+1)
        Caller->>Hybrid: forward(q_tok, k_tok, v_tok, 1, Some(&mut cache))
        Hybrid->>Attn: forward_stepwise(q_tok, k_tok, v_tok, Some(&mut cache))
        Attn->>Cache: update(k_s, v_tok)
        Cache-->>Attn: updated running state S (Heads x Singlets x HeadDim)
        Attn-->>Hybrid: Step Singlet Output
        Hybrid-->>Caller: Step Output Token (1 x EmbedDim)
    end
```

---

## 3. CNF Partitioning & Dilution Pipeline

The CNF pre-processor ingests DIMACS formulas and performs topological partitioning:

```mermaid
flowchart LR
    A["Raw DIMACS Input (.cnf)"] --> B["Streaming Parser (Strict / Permissive)"]
    B --> C["Clause Conservation & Variable Indexing"]
    C --> D{"Clause-to-Var Ratio Dilution Check"}
    D -- "Pendant Clauses Detected" --> E["Dilution & Treewidth Decoupling"]
    D -- "Normal Ratio" --> F["Clause Boundary Partitioning"]
    E --> G["Sparse / Dense Buckets"]
    F --> G
    G --> H["Planar Crossing Elimination (Uncrossing)"]
```

---

## 4. Discrete Planar Uncrossing Engine

Planar graphs and crossings are reduced via the deterministic $\{-1, 0, 1\}$ tensor algebra:

```mermaid
stateDiagram-v2
    [*] --> PlanarGridInit: Parse Crossing Coordinates
    PlanarGridInit --> SliceEvaluation: Evaluate Slice A_i (n_bits <= 4)
    SliceEvaluation --> FactorComputation: Compute s_factor(x, y, u, v)
    FactorComputation --> RankVerification: Verify rank_row(A_i) * rank_col(A_i) <= 12^lambda
    RankVerification --> Contraction: Exact Contraction (Zero-Drift FP64)
    Contraction --> [*]
```

---

## 5. Memory & Cache Hierarchy for Inference

```mermaid
classDiagram
    class SingletKVCache {
        +usize num_heads
        +usize head_dim
        +usize num_singlets
        +Vec~f32~ state
        +Vec~f32~ normalizer
        +usize current_seq_len
        +reset()
    }

    class SingletAttention {
        +usize embed_dim
        +usize num_heads
        +usize head_dim
        +usize chunk_size
        +bool is_causal
        +usize num_singlets
        +forward()
        +forward_with_projected_singlets()
    }

    class HybridSequenceLayer {
        +SingletAttention global_branch
        +usize local_window_size
        +Vec~f32~ gate_weights
        +Option~DecayStabilizer~ decay
        +Option~StateRMSNorm~ norm
        +forward()
    }

    HybridSequenceLayer *-- SingletAttention
    SingletAttention ..> SingletKVCache : reads and updates state
```


---

## 6. GPU Acceleration: Triton Fused Singlet Kernel

The low-rank projection $Q_{\text{singlet}} = Q \times U$ ($B \times H \times N \times D \to B \times H \times N \times R$) is accelerated on NVIDIA/AMD GPUs using fused Triton kernels (`triton_singlet_proj`).

![TensorSAT Triton Kernel Architecture](images/triton_kernel.png)

### Key Features:
- **2D Tiled Grid Mapping**: Thread blocks are dispatched across a 2D grid `(B * H, num_n_tiles * num_r_tiles)`.
- **SRAM Staging**: Blocks iterate across dimension $D$ in slices of `BLOCK_D = 32`, holding accumulator tiles of size `BLOCK_N = 64` $\times$ `BLOCK_R = 16` directly in on-chip registers.
- **Strict IEEE-Precision GEMM**: Uses `input_precision="ieee"` in `tl.dot` to avoid TF32 truncation artifacts, maintaining exact discrete low-rank properties.
