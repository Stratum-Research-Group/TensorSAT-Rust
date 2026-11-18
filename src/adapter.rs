//! TensorSAT Pre-Trained Weight Adapter: Softmax-to-Singlet Transduction.
//!
//! Uses ELU + 1.0 positive kernel transduction mapping pre-trained Q/K projections
//! into linear singlet uncrossing space.

/// Adapter mapping pre-trained Q, K representations into positive singlet features.
#[derive(Debug, Clone)]
pub struct SoftmaxToSingletAdapter {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_singlets: usize,

    pub q_trans: Vec<f32>, // shape (num_heads, head_dim, num_singlets)
    pub k_trans: Vec<f32>, // shape (num_heads, head_dim, num_singlets)
}

impl SoftmaxToSingletAdapter {
    pub fn new(embed_dim: usize, num_heads: usize, num_singlets: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0);
        let head_dim = embed_dim / num_heads;
        let size = num_heads * head_dim * num_singlets;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_trans = vec![0.0f32; size];
        let mut k_trans = vec![0.0f32; size];
        for i in 0..size {
            q_trans[i] = ((i as f32 * 0.19).sin()) * scale;
            k_trans[i] = ((i as f32 * 0.27).cos()) * scale;
        }

        Self {
            embed_dim,
            num_heads,
            head_dim,
            num_singlets,
            q_trans,
            k_trans,
        }
    }

    /// Creates an adapter loading transduction weights directly from raw IEEE 754 f32 slices.
    pub fn from_transduction_weights(
        embed_dim: usize,
        num_heads: usize,
        num_singlets: usize,
        q_weights: &[f32],
        k_weights: &[f32],
    ) -> Result<Self, String> {
        if num_heads == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err("embed_dim must be divisible by positive num_heads".to_string());
        }
        let head_dim = embed_dim / num_heads;
        let expected_size = num_heads * head_dim * num_singlets;
        if q_weights.len() != expected_size || k_weights.len() != expected_size {
            return Err(format!(
                "transduction weight size mismatch: expected {}, got q={}, k={}",
                expected_size,
                q_weights.len(),
                k_weights.len()
            ));
        }

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            num_singlets,
            q_trans: q_weights.to_vec(),
            k_trans: k_weights.to_vec(),
        })
    }

    /// Projects Q and K with unit-normalization along head_dim and ELU + 1.0 positive activation.
    pub fn forward(
        &self,
        q_heads: &[f32],
        k_heads: &[f32],
        n_tokens: usize,
        q_s_out: &mut [f32],
        k_s_out: &mut [f32],
    ) {
        let h_heads = self.num_heads;
        let d = self.head_dim;
        let r_singlets = self.num_singlets;

        assert_eq!(q_heads.len(), h_heads * n_tokens * d);
        assert_eq!(k_heads.len(), h_heads * n_tokens * d);
        assert_eq!(q_s_out.len(), h_heads * n_tokens * r_singlets);
        assert_eq!(k_s_out.len(), h_heads * n_tokens * r_singlets);

        for h in 0..h_heads {
            let h_x_offset = h * n_tokens * d;
            let h_basis_offset = h * d * r_singlets;
            let h_s_offset = h * n_tokens * r_singlets;

            for n in 0..n_tokens {
                let q_tok = &q_heads[h_x_offset + n * d..h_x_offset + (n + 1) * d];
                let k_tok = &k_heads[h_x_offset + n * d..h_x_offset + (n + 1) * d];

                // Unit normalize
                let mut q_norm_sq = 0.0f32;
                let mut k_norm_sq = 0.0f32;
                for c in 0..d {
                    q_norm_sq += q_tok[c] * q_tok[c];
                    k_norm_sq += k_tok[c] * k_tok[c];
                }
                let inv_q = 1.0f32 / (q_norm_sq.sqrt() + 1e-12);
                let inv_k = 1.0f32 / (k_norm_sq.sqrt() + 1e-12);

                let q_s = &mut q_s_out[h_s_offset + n * r_singlets..h_s_offset + (n + 1) * r_singlets];
                let k_s = &mut k_s_out[h_s_offset + n * r_singlets..h_s_offset + (n + 1) * r_singlets];

                for r in 0..r_singlets {
                    let mut q_proj = 0.0f32;
                    let mut k_proj = 0.0f32;
                    for c in 0..d {
                        q_proj += (q_tok[c] * inv_q) * self.q_trans[h_basis_offset + c * r_singlets + r];
                        k_proj += (k_tok[c] * inv_k) * self.k_trans[h_basis_offset + c * r_singlets + r];
                    }
                    // ELU(x) + 1.0; clamped to prevent exact zero from exp underflow
                    q_s[r] = if q_proj > 0.0 { q_proj + 1.0 } else { q_proj.exp().max(1e-12) };
                    k_s[r] = if k_proj > 0.0 { k_proj + 1.0 } else { k_proj.exp().max(1e-12) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_from_weights_and_strict_positivity() {
        let embed_dim = 16;
        let num_heads = 2;
        let num_singlets = 4;
        let head_dim = embed_dim / num_heads;
        let size = num_heads * head_dim * num_singlets;

        let q_weights = vec![100.0f32; size]; // Very large positive
        let k_weights = vec![-100.0f32; size]; // Very large negative (would underflow to 0 without clamp)

        let adapter = SoftmaxToSingletAdapter::from_transduction_weights(
            embed_dim,
            num_heads,
            num_singlets,
            &q_weights,
            &k_weights,
        )
        .expect("Failed to load adapter weights");

        let n_tokens = 2;
        let q = vec![1.0f32; num_heads * n_tokens * head_dim];
        let k = vec![1.0f32; num_heads * n_tokens * head_dim];
        let mut q_s = vec![0.0f32; num_heads * n_tokens * num_singlets];
        let mut k_s = vec![0.0f32; num_heads * n_tokens * num_singlets];

        adapter.forward(&q, &k, n_tokens, &mut q_s, &mut k_s);

        // Verify strictly positive
        for &val in &q_s {
            assert!(val > 0.0, "q_s must be strictly positive");
        }
        for &val in &k_s {
            assert!(val > 0.0, "k_s must be strictly positive (prevented underflow)");
        }
    }

    #[test]
    fn test_adapter_piping_to_singlet_attention() {
        use crate::attention::SingletAttention;

        let embed_dim = 16;
        let num_heads = 2;
        let num_singlets = 4;
        let head_dim = embed_dim / num_heads;
        let n_tokens = 3;

        let adapter = SoftmaxToSingletAdapter::new(embed_dim, num_heads, num_singlets);
        let layer = SingletAttention::new(embed_dim, num_heads, num_singlets, 4, true);

        let size = num_heads * n_tokens * head_dim;
        let q = vec![0.25f32; size];
        let k = vec![0.25f32; size];
        let v = vec![0.5f32; size];

        let mut q_s = vec![0.0f32; num_heads * n_tokens * num_singlets];
        let mut k_s = vec![0.0f32; num_heads * n_tokens * num_singlets];

        // 1. Project through adapter
        adapter.forward(&q, &k, n_tokens, &mut q_s, &mut k_s);

        // 2. Consume adapter features directly in downstream attention layer
        let out = layer.forward_with_projected_singlets(&q_s, &k_s, &v, n_tokens);

        assert_eq!(out.len(), size);
        for &val in &out {
            assert!(!val.is_nan());
            assert!(val.is_finite());
        }
    }
}
