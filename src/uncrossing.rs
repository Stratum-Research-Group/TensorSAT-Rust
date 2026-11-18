//! TensorSAT core uncrossing tensor algebra.
//!
//! Provides mathematically exact evaluation of the phase factor s(p, q)
//! and uncrossing tensor slice decomposition A_i(x, y, u, v).

/// Phase factor s(p, q) = (-1)^p [p != q].
///
/// Matches Lean 4 definition: Groundwork.s
#[inline]
pub fn s_factor(p: u32, q: u32) -> i32 {
    if p == q {
        0
    } else if p == 1 {
        -1
    } else {
        1
    }
}

/// Evaluates slice A_i(x, y, u, v) without dense matrix allocation.
///
/// sum over subsets S of size i:
///   term = prod_{k in S} (-s(x_k, y_k)*s(u_k, v_k)) * prod_{k not in S} [x_k == u_k and y_k == v_k]
pub fn eval_slice_a(n_bits: usize, i: usize, x: usize, y: usize, u: usize, v: usize) -> i32 {
    let mut total = 0i32;

    // Helper for combinations
    fn combinations(n: usize, k: usize, f: &mut impl FnMut(&[usize])) {
        let mut comb = vec![0; k];
        fn backtrack(
            start: usize,
            depth: usize,
            n: usize,
            k: usize,
            comb: &mut [usize],
            f: &mut impl FnMut(&[usize]),
        ) {
            if depth == k {
                f(comb);
                return;
            }
            for i in start..=n - (k - depth) {
                comb[depth] = i;
                backtrack(i + 1, depth + 1, n, k, comb, f);
            }
        }
        backtrack(0, 0, n, k, &mut comb, f);
    }

    combinations(n_bits, i, &mut |subset| {
        let mut prod_s = 1i32;
        let mut subset_mask = 0u64;

        for &k in subset {
            subset_mask |= 1u64 << k;
            let xk = ((x >> k) & 1) as u32;
            let yk = ((y >> k) & 1) as u32;
            let uk = ((u >> k) & 1) as u32;
            let vk = ((v >> k) & 1) as u32;

            let s1 = s_factor(xk, yk);
            let s2 = s_factor(uk, vk);
            prod_s *= -(s1 * s2);
            if prod_s == 0 {
                break;
            }
        }

        if prod_s == 0 {
            return;
        }

        let mut match_rest = true;
        for k in 0..n_bits {
            if (subset_mask & (1u64 << k)) == 0 {
                let xk = (x >> k) & 1;
                let yk = (y >> k) & 1;
                let uk = (u >> k) & 1;
                let vk = (v >> k) & 1;
                if !(xk == uk && yk == vk) {
                    match_rest = false;
                    break;
                }
            }
        }

        if match_rest {
            total += prod_s;
        }
    });

    total
}

/// Generates the (n_bits + 1) slice tensors A_0 ... A_n in row-major flat vectors.
/// Each slice has dimension (2^n, 2^n, 2^n, 2^n).
pub fn generate_uncrossing_tensor(n_bits: usize) -> Vec<Vec<f64>> {
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
    slices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_factor() {
        assert_eq!(s_factor(0, 0), 0);
        assert_eq!(s_factor(1, 1), 0);
        assert_eq!(s_factor(0, 1), 1);
        assert_eq!(s_factor(1, 0), -1);
    }

    #[test]
    fn test_uncrossing_identity_1bit() {
        for x in 0..=1 {
            for y in 0..=1 {
                for u in 0..=1 {
                    for v in 0..=1 {
                        let a0 = eval_slice_a(1, 0, x, y, u, v);
                        let a1 = eval_slice_a(1, 1, x, y, u, v);
                        let target = if x == v && y == u { 1 } else { 0 };
                        assert_eq!(a0 + a1, target, "Failed at x={x}, y={y}, u={u}, v={v}");
                    }
                }
            }
        }
    }
}
