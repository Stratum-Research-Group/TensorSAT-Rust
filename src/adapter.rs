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
                    // ELU(x) + 1.0
                    q_s[r] = if q_proj > 0.0 { q_proj + 1.0 } else { q_proj.exp() };
                    k_s[r] = if k_proj > 0.0 { k_proj + 1.0 } else { k_proj.exp() };
                }
            }
        }
    }
}
