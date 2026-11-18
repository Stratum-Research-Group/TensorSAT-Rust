//! DIMACS CNF Formula parser and graph partitioner.
//!
//! Separates dense variable interaction cliques from sparse clauses.

use std::collections::{HashMap, HashSet};

/// Representation of a Boolean formula in Conjunctive Normal Form (CNF).
#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub n_vars: usize,
    pub clauses: Vec<Vec<i32>>,
}

impl CNFFormula {
    pub fn new(n_vars: usize, clauses: Vec<Vec<i32>>) -> Self {
        Self { n_vars, clauses }
    }

    /// Parse DIMACS CNF format string.
    pub fn from_dimacs(dimacs: &str) -> Result<Self, String> {
        let mut clauses = Vec::new();
        let mut n_vars = 0;

        for line in dimacs.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') {
                continue;
            }
            if line.starts_with("p cnf") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    n_vars = parts[2].parse::<usize>().map_err(|e| e.to_string())?;
                }
                continue;
            }

            let mut current_clause = Vec::new();
            for part in line.split_whitespace() {
                if let Ok(lit) = part.parse::<i32>() {
                    if lit == 0 {
                        if !current_clause.is_empty() {
                            clauses.push(current_clause);
                            current_clause = Vec::new();
                        }
                    } else {
                        current_clause.push(lit);
                    }
                }
            }
            if !current_clause.is_empty() {
                clauses.push(current_clause);
            }
        }

        Ok(Self { n_vars, clauses })
    }

    /// Builds the primal variable interaction graph (edges between variables
    /// co-occurring in at least one clause).
    pub fn extract_variable_interaction_graph(&self) -> HashMap<usize, HashSet<usize>> {
        let mut adj: HashMap<usize, HashSet<usize>> = (1..=self.n_vars)
            .map(|i| (i, HashSet::new()))
            .collect();

        for clause in &self.clauses {
            let vars: Vec<usize> = clause.iter().map(|&lit| lit.unsigned_abs() as usize).collect();
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let u = vars[i];
                    let v = vars[j];
                    if u <= self.n_vars && v <= self.n_vars && u != v {
                        adj.get_mut(&u).unwrap().insert(v);
                        adj.get_mut(&v).unwrap().insert(u);
                    }
                }
            }
        }

        adj
    }

    /// Identifies dense variable cliques and separates them from sparse clauses.
    pub fn partition_sparse_and_dense_cliques(
        &self,
        density_threshold: f64,
    ) -> (Vec<HashSet<usize>>, Vec<Vec<i32>>) {
        let adj = self.extract_variable_interaction_graph();
        let mut dense_clusters: Vec<HashSet<usize>> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        for v in 1..=self.n_vars {
            if visited.contains(&v) {
                continue;
            }
            let neighbors = match adj.get(&v) {
                Some(n) if !n.is_empty() => n,
                _ => continue,
            };

            let k = neighbors.len() + 1;
            if k >= 4 {
                let mut cluster: HashSet<usize> = neighbors.clone();
                cluster.insert(v);

                let mut internal_edges = 0;
                for &u in &cluster {
                    if let Some(u_neighbors) = adj.get(&u) {
                        internal_edges += u_neighbors.intersection(&cluster).count();
                    }
                }
                let max_edges = k * (k - 1);
                let density = if max_edges > 0 {
                    internal_edges as f64 / max_edges as f64
                } else {
                    0.0
                };

                if density >= density_threshold {
                    visited.extend(&cluster);
                    dense_clusters.push(cluster);
                }
            }
        }

        let mut dense_vars: HashSet<usize> = HashSet::new();
        for cluster in &dense_clusters {
            dense_vars.extend(cluster);
        }

        let sparse_clauses = self
            .clauses
            .iter()
            .filter(|clause| !clause.iter().any(|&lit| dense_vars.contains(&(lit.unsigned_abs() as usize))))
            .cloned()
            .collect();

        (dense_clusters, sparse_clauses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnf_partitioning() {
        let dimacs = r#"
c Benchmark DIMACS CNF with dense 4-clique and sparse tree
p cnf 6 7
1 2 0
2 3 0
3 4 0
1 4 0
2 4 0
1 3 0
5 6 0
"#;
        let cnf = CNFFormula::from_dimacs(dimacs).expect("Failed to parse DIMACS");
        let (dense, sparse) = cnf.partition_sparse_and_dense_cliques(0.8);

        assert!(!dense.is_empty());
        let expected_clique: HashSet<usize> = [1, 2, 3, 4].into_iter().collect();
        assert!(dense.iter().any(|c| c == &expected_clique));
        assert!(sparse.iter().any(|c| c == &vec![5, 6]));
    }
}
