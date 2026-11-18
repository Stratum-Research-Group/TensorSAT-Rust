//! Exact IEEE 754 Floating-Point Tensor Contraction.
//!
//! Provides lossless 4-index tensor contractions in FP64 and Complex64/128.

use crate::uncrossing::eval_slice_a;
use num_complex::Complex64;

/// High-precision IEEE tensor contractor.
/// Computes Z = sum_{x,y,u,v} W1[x, y] * cross(x, y, u, v) * W2[u, v]
/// using exact discrete uncrossing slices A_i.
pub struct IEEETensorContractor {
    pub n_bits: usize,
    pub dim: usize,
    slices: Vec<Vec<f64>>,
}

impl IEEETensorContractor {
    pub fn new(n_bits: usize) -> Self {
        let dim = 1 << n_bits;
        let size = dim * dim * dim * dim;
        let mut slices = Vec::with_capacity(n_bits + 1);

        for i in 0..=n_bits {
            let mut slice = vec![0.0f64; size];
            for x in 0..dim {
                for y in 0..dim {
                    for u in 0..dim {
                        for v in 0..dim {
                            let val = eval_slice_a(n_bits, i, x, y, u, v);
                            if val != 0 {
                                let idx = ((x * dim + y) * dim + u) * dim + v;
                                slice[idx] = val as f64;
                            }
                        }
                    }
                }
            }
            slices.push(slice);
        }

        Self { n_bits, dim, slices }
    }

    /// Contracts boundary interaction matrices W1 and W2 in exact FP64.
    /// W1 and W2 must have shape (dim, dim) stored as row-major flat slices.
    pub fn contract_f64(&self, w1: &[f64], w2: &[f64]) -> (f64, Vec<f64>) {
        assert_eq!(w1.len(), self.dim * self.dim);
        assert_eq!(w2.len(), self.dim * self.dim);

        let dim = self.dim;
        let mut slice_contributions = Vec::with_capacity(self.slices.len());
        let mut total = 0.0f64;

        for slice in &self.slices {
            let mut slice_sum = 0.0f64;
            for x in 0..dim {
                for y in 0..dim {
                    let w1_val = w1[x * dim + y];
                    if w1_val == 0.0 {
                        continue;
                    }
                    for u in 0..dim {
                        for v in 0..dim {
                            let w2_val = w2[u * dim + v];
                            if w2_val == 0.0 {
                                continue;
                            }
                            let a_val = slice[((x * dim + y) * dim + u) * dim + v];
                            slice_sum += w1_val * a_val * w2_val;
                        }
                    }
                }
            }
            slice_contributions.push(slice_sum);
            total += slice_sum;
        }

        (total, slice_contributions)
    }

    /// Contracts boundary interaction matrices in Complex64 (FP64 real + imag).
    pub fn contract_complex(&self, w1: &[Complex64], w2: &[Complex64]) -> (Complex64, Vec<Complex64>) {
        assert_eq!(w1.len(), self.dim * self.dim);
        assert_eq!(w2.len(), self.dim * self.dim);

        let dim = self.dim;
        let mut slice_contributions = Vec::with_capacity(self.slices.len());
        let mut total = Complex64::new(0.0, 0.0);

        for slice in &self.slices {
            let mut slice_sum = Complex64::new(0.0, 0.0);
            for x in 0..dim {
                for y in 0..dim {
                    let w1_val = w1[x * dim + y];
                    for u in 0..dim {
                        for v in 0..dim {
                            let w2_val = w2[u * dim + v];
                            let a_val = slice[((x * dim + y) * dim + u) * dim + v];
                            if a_val != 0.0 {
                                slice_sum += w1_val * a_val * w2_val;
                            }
                        }
                    }
                }
            }
            slice_contributions.push(slice_sum);
            total += slice_sum;
        }

        (total, slice_contributions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_f64_matches_exact_naive() {
        for n_bits in [1, 2, 3] {
            let contractor = IEEETensorContractor::new(n_bits);
            let dim = 1 << n_bits;

            let mut w1 = vec![0.0f64; dim * dim];
            let mut w2 = vec![0.0f64; dim * dim];
            for i in 0..dim * dim {
                w1[i] = ((i * 7 + 3) % 17) as f64 - 8.0;
                w2[i] = ((i * 13 + 5) % 19) as f64 - 9.0;
            }

            let (contracted_total, _) = contractor.contract_f64(&w1, &w2);

            // Exact direct crossing naive reference:
            // cross(x, y, u, v) == [x == v && y == u]
            let mut expected = 0.0f64;
            for x in 0..dim {
                for y in 0..dim {
                    for u in 0..dim {
                        for v in 0..dim {
                            if x == v && y == u {
                                expected += w1[x * dim + y] * w2[u * dim + v];
                            }
                        }
                    }
                }
            }

            let diff = (contracted_total - expected).abs();
            assert!(
                diff <= 1e-14,
                "Drift too large at n_bits={n_bits}: {diff} vs machine epsilon"
            );
        }
    }
}
