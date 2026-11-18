//! TensorSAT Hybrid Attention: Local Sliding-Window SDPA + Global Singlet Attention.
//!
//! Blends:
//! 1. Exact local banded sliding window of width W for 100% sharp local recall.
//! 2. Global singlet uncrossing linear accumulator for O(N) context scaling.

use crate::attention::SingletAttention;
use crate::stabilizer::{DecayStabilizer, StateRMSNorm};

/// Hybrid Attention combining exact local windowing with global planar singlet uncrossing.
pub struct HybridSingletAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub num_singlets: usize,
    pub chunk_size: usize,
    pub is_causal: bool,

    pub singlet_attn: SingletAttention,
    pub state_norm: StateRMSNorm,
    pub decay_stabilizer: DecayStabilizer,
}

impl HybridSingletAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        window_size: usize,
        num_singlets: usize,
        chunk_size: usize,
        is_causal: bool,
    ) -> Self {
        assert_eq!(embed_dim % num_heads, 0);
        let head_dim = embed_dim / num_heads;

        let singlet_attn = SingletAttention::new(embed_dim, num_heads, num_singlets, chunk_size, is_causal);
        let state_norm = StateRMSNorm::new(num_singlets, head_dim, 1e-6);
        let decay_stabilizer = DecayStabilizer::new(num_heads, 0.999);

        Self {
            embed_dim,
            num_heads,
            head_dim,
            window_size,
            num_singlets,
            chunk_size,
            is_causal,
            singlet_attn,
            state_norm,
            decay_stabilizer,
        }
    }

    /// Computes local sliding-window attention of width W.
    ///
    /// Each query token t attends to keys in [max(0, t - W + 1), t].
    pub fn local_window_attention(
        &self,
        q_heads: &[f32],
        k_heads: &[f32],
        v_heads: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let h_heads = self.num_heads;
        let d = self.head_dim;
        let w_size = self.window_size;
        let scale = 1.0f32 / (d as f32).sqrt();

        let mut out = vec![0.0f32; h_heads * n_tokens * d];

        for h in 0..h_heads {
            let h_offset = h * n_tokens * d;

            for t in 0..n_tokens {
                let start_key = if self.is_causal {
                    if t >= w_size { t - w_size + 1 } else { 0 }
                } else {
                    let half = w_size / 2;
                    if t >= half { t - half } else { 0 }
                };
                let end_key = if self.is_causal {
                    t + 1
                } else {
                    (t + w_size / 2 + 1).min(n_tokens)
                };

                let q_t = &q_heads[h_offset + t * d..h_offset + (t + 1) * d];

                // Compute attention logits
                let mut max_logit = f32::NEG_INFINITY;
                let mut logits = Vec::with_capacity(end_key - start_key);

                for s in start_key..end_key {
                    let k_s = &k_heads[h_offset + s * d..h_offset + (s + 1) * d];
                    let mut dot = 0.0f32;
                    for c in 0..d {
                        dot += q_t[c] * k_s[c];
                    }
                    let logit = dot * scale;
                    if logit > max_logit {
                        max_logit = logit;
                    }
                    logits.push(logit);
                }

                // Stable Softmax
                let mut sum_exp = 0.0f32;
                for l in &mut logits {
                    *l = (*l - max_logit).exp();
                    sum_exp += *l;
                }
                let inv_sum = 1.0f32 / sum_exp.max(1e-12);

                // Weighted sum over V
                let out_t = &mut out[h_offset + t * d..h_offset + (t + 1) * d];
                for (idx, s) in (start_key..end_key).enumerate() {
                    let weight = logits[idx] * inv_sum;
                    let v_s = &v_heads[h_offset + s * d..h_offset + (s + 1) * d];
                    for c in 0..d {
                        out_t[c] += weight * v_s[c];
                    }
                }
            }
        }

        out
    }

    /// Fused forward pass: 0.5 * Local_SDPA + 0.5 * Global_SingletAttention
    pub fn forward(
        &self,
        q_heads: &[f32],
        k_heads: &[f32],
        v_heads: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let local_out = self.local_window_attention(q_heads, k_heads, v_heads, n_tokens);
        let global_out = self.singlet_attn.forward(q_heads, k_heads, v_heads, n_tokens, None);

        let mut out = vec![0.0f32; local_out.len()];
        for i in 0..local_out.len() {
            out[i] = 0.5 * local_out[i] + 0.5 * global_out[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_local_window_causal() {
        let embed_dim = 32;
        let num_heads = 2;
        let window_size = 8;
        let n_tokens = 24;
        let hybrid = HybridSingletAttention::new(embed_dim, num_heads, window_size, 4, 16, true);

        let size = num_heads * n_tokens * (embed_dim / num_heads);
        let q = vec![0.1f32; size];
        let k = vec![0.1f32; size];
        let v = vec![0.1f32; size];

        let out = hybrid.local_window_attention(&q, &k, &v, n_tokens);
        assert_eq!(out.len(), size);
        for &val in &out {
            assert!(!val.is_nan());
        }
    }
}
