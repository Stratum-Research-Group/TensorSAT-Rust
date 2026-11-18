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

        // Pre-scale by max amplitude to prevent square overflow in float32
        let mut max_abs = 0.0f32;
        for &val in state {
            let abs_v = val.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }
        let scale = if max_abs > 1.0 { max_abs } else { 1.0 };
        let inv_scale = 1.0 / scale;

        let mut sum_sq = 0.0f32;
        for &val in state {
            let scaled_val = val * inv_scale;
            sum_sq += scaled_val * scaled_val;
        }
        let mean_sq = sum_sq / (state.len() as f32);
        let rms = (mean_sq + self.eps * inv_scale * inv_scale).sqrt() * scale;
        let inv_rms = 1.0 / rms;

        for i in 0..state.len() {
            out[i] = state[i] * inv_rms * self.scale[i];
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

        // Test with massive magnitude (1e20) that would overflow naive squares
        let input_large = vec![1e20f32; rank * dim];
        norm.forward(&input_large, &mut output);
        assert!(!output.iter().any(|v| v.is_nan() || v.is_infinite()));
        assert!((output[0] - 1.0).abs() < 1e-3);
    }
}
