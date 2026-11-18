//! Standalone benchmark binary for TensorSAT-Rust.

use std::time::Instant;
use tensorsat::attention::{KVCache, SingletAttention};
use tensorsat::scientific::IEEETensorContractor;

fn main() {
    println!("==============================================================================");
    println!("TensorSAT-Rust: High-Performance Engine Benchmark");
    println!("==============================================================================");

    let embed_dim = 512;
    let num_heads = 8;
    let num_singlets = 16;
    let head_dim = embed_dim / num_heads;

    // Check if large-scale sequences (131k .. 1M) should be benchmarked.
    // By default, the benchmark runs up to 65,536 tokens. Set TENSORSAT_BENCH_LARGE=1
    // or pass `--large` on the command line to benchmark up to 1,048,576 tokens (~8.6 GB RAM requirement).
    let run_large = std::env::var("TENSORSAT_BENCH_LARGE").map(|v| v == "1" || v == "true").unwrap_or(false)
        || std::env::args().any(|arg| arg == "--large" || arg == "-l");

    let n_sizes: &[usize] = if run_large {
        &[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    } else {
        &[1024, 2048, 4096, 8192, 16384, 32768, 65536]
    };

    let layer = SingletAttention::new(embed_dim, num_heads, num_singlets, 512, true);

    println!("\n1. Neural Sequence Scaling (Prefill Mode):");
    if !run_large {
        println!("   [Note: Running standard suite <= 65k tokens. Use `--large` or TENSORSAT_BENCH_LARGE=1 for 131k-1M rows]");
    }
    println!("{:<15} | {:<20} | {:<18} | {:<15}", "Tokens (N)", "Latency (ms)", "Throughput (tok/s)", "Complexity");
    println!("-----------------------------------------------------------------------------");

    for &n in n_sizes {
        let size = num_heads * n * head_dim;
        let q = vec![0.05f32; size];
        let k = vec![0.05f32; size];
        let v = vec![0.05f32; size];

        // Warmup
        let _ = layer.forward(&q[..num_heads * 64 * head_dim], &k[..num_heads * 64 * head_dim], &v[..num_heads * 64 * head_dim], 64, None);

        let t0 = Instant::now();
        let iters = if n >= 524288 { 1 } else if n >= 65536 { 2 } else if n >= 8192 { 4 } else { 10 };
        for _ in 0..iters {
            let out = layer.forward(&q, &k, &v, n, None);
            std::hint::black_box(out);
        }
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let throughput = (n as f64) / (elapsed / 1000.0);
        println!("{:<15} | {:<20.2} | {:<18.1} | O(N) linear", n, elapsed, throughput);
    }

    println!("\n2. Incremental Autoregressive Generation Throughput (KVCache):");
    println!("{:<20} | {:<20} | {:<20}", "Past Prompt Depth", "Step Latency (ms)", "Throughput (Tokens/s)");
    println!("------------------------------------------------------------------");

    for &p in &[512, 2048, 8192] {
        let mut cache = KVCache::new(num_heads, num_singlets, head_dim);

        // Prime cache with past prompt
        let prompt_size = num_heads * p * head_dim;
        let prompt_q = vec![0.02f32; prompt_size];
        let prompt_k = vec![0.02f32; prompt_size];
        let prompt_v = vec![0.02f32; prompt_size];
        let _ = layer.forward(&prompt_q, &prompt_k, &prompt_v, p, Some(&mut cache));

        // Generate 1000 single steps
        let token_size = num_heads * head_dim;
        let tok_q = vec![0.02f32; token_size];
        let tok_k = vec![0.02f32; token_size];
        let tok_v = vec![0.02f32; token_size];

        let steps = 1000;
        let t0 = Instant::now();
        for _ in 0..steps {
            let _ = layer.forward(&tok_q, &tok_k, &tok_v, 1, Some(&mut cache));
        }
        let step_latency = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        let throughput = 1000.0 / step_latency;
        println!("{:<20} | {:<20.4} | {:<20.1}", p, step_latency, throughput);
    }

    println!("\n3. IEEE 754 High-Precision Double (FP64) Tensor Contraction:");
    for bits in [1, 2, 3] {
        let contractor = IEEETensorContractor::new(bits);
        let dim = 1 << bits;
        let w1 = vec![1.23456789f64; dim * dim];
        let w2 = vec![9.87654321f64; dim * dim];

        let t0 = Instant::now();
        let (val, _) = contractor.contract_f64(&w1, &w2);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        println!("Bits: {} (dim = {:<2}) | Contraction Time: {:<8.3} ms | Result = {:.6}", bits, dim, elapsed, val);
    }

    println!("\n==============================================================================");
    println!("ALL BENCHMARKS EXECUTED CLEANLY.");
    println!("==============================================================================");
}
