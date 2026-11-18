//! TensorSAT: High-Performance Rust Implementation.
//!
//! Planar tensor-network uncrossing decomposition and sub-quadratic linear attention.

pub mod uncrossing;
pub mod scientific;
pub mod cnf;
pub mod stabilizer;
pub mod attention;
pub mod hybrid;
pub mod adapter;

pub use uncrossing::{s_factor, eval_slice_a, generate_uncrossing_tensor};
pub use scientific::IEEETensorContractor;
pub use cnf::CNFFormula;
pub use stabilizer::{StateRMSNorm, DecayStabilizer};
pub use attention::{SingletAttention, KVCache};
pub use hybrid::HybridSingletAttention;
pub use adapter::SoftmaxToSingletAdapter;
