//! Numerical stabilization routines for linear attention states.
//!
//! Includes RMS state normalization and continuous decay tracking.

/// RMS normalizer along (Rank, Dim) dimensions of the running state matrix.
#[derive(Debug, Clone)]
pub struct StateRMSNorm {
    pub rank: usize,
    pub dim: usize,
    pub eps: f32,
    pub scale: Vec<f32>, // shape (rank, dim)
}

impl StateRMSNorm {
    pub fn new(rank: usize, dim: usize, eps: f32) -> Self {
        Self {
            rank,
            dim,
            eps,
            scale: vec![1.0f32; rank * dim],
        }
    }

    /// Normalizes a state tensor slice of shape (rank, dim) in-place or into a new buffer.
    pub fn forward(&self, state: &[f32], out: &mut [f32]) {
        assert_eq!(state.len(), self.rank * self.dim);
        assert_eq!(out.len(), self.rank * self.dim);

        let mut sum_sq = 0.0f32;
        for &val in state {
            sum_sq += val * val;
        }
        let mean_sq = sum_sq / (state.len() as f32);
        let rsqrt = 1.0f32 / (mean_sq + self.eps).sqrt();

        for i in 0..state.len() {
            out[i] = state[i] * rsqrt * self.scale[i];
        }
    }
}

/// Continuous decay mechanism gamma in (0, 1] applied to running accumulator.
///
/// S_t = gamma^steps * S_{t-1} + K_s^T V
#[derive(Debug, Clone)]
pub struct DecayStabilizer {
    pub num_heads: usize,
    pub base_decay: f32,
}

impl DecayStabilizer {
    pub fn new(num_heads: usize, base_decay: f32) -> Self {
        Self {
            num_heads,
            base_decay,
        }
    }

    /// Applies continuous decay to a per-head accumulator state of shape (rank, dim).
    pub fn apply(
        &self,
        running_kv: &[f32],
        new_kv_chunk: &[f32],
        steps: usize,
        out: &mut [f32],
    ) {
        assert_eq!(running_kv.len(), new_kv_chunk.len());
        assert_eq!(running_kv.len(), out.len());

        let decay_factor = self.base_decay.powi(steps as i32);
        for i in 0..running_kv.len() {
            out[i] = running_kv[i] * decay_factor + new_kv_chunk[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_rms_norm() {
        let rank = 4;
        let dim = 8;
        let norm = StateRMSNorm::new(rank, dim, 1e-6);

        let input = vec![10.0f32; rank * dim];
        let mut output = vec![0.0f32; rank * dim];
        norm.forward(&input, &mut output);

        let mut sum_sq = 0.0f32;
        for &x in &output {
            sum_sq += x * x;
        }
        let var = sum_sq / output.len() as f32;
        assert!((var - 1.0).abs() < 1e-3);
    }
}
