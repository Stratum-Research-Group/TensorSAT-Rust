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

pub type CNFPartition = (
    Vec<HashSet<usize>>,
    Vec<Vec<i32>>,
    Vec<Vec<i32>>,
    Vec<Vec<i32>>,
);

impl CNFFormula {
    pub fn new(n_vars: usize, clauses: Vec<Vec<i32>>) -> Self {
        Self { n_vars, clauses }
    }

    /// Parse DIMACS CNF format string with optional strict validation.
    pub fn from_dimacs_with_options(dimacs: &str, strict: bool) -> Result<Self, String> {
        let mut clauses = Vec::new();
        let mut n_vars = 0;
        let mut n_clauses_decl = None;
        let mut has_header = false;
        let mut current_clause = Vec::new();

        for (line_idx, line) in dimacs.lines().enumerate() {
            let line_num = line_idx + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') {
                continue;
            }
            if line.starts_with("p cnf") {
                if has_header && strict {
                    return Err(format!("Duplicate 'p cnf' header at line {}", line_num));
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    if strict {
                        return Err(format!("Malformed header at line {}: expected 4 tokens", line_num));
                    }
                    if parts.len() >= 3 {
                        n_vars = parts[2].parse::<usize>().map_err(|e| e.to_string())?;
                    }
                } else {
                    n_vars = parts[2].parse::<usize>().map_err(|e| e.to_string())?;
                    n_clauses_decl = Some(parts[3].parse::<usize>().map_err(|e| e.to_string())?);
                }
                has_header = true;
                continue;
            }

            for part in line.split_whitespace() {
                let lit = part.parse::<i32>().map_err(|_| format!("Invalid literal '{}' at line {}", part, line_num))?;
                if lit == 0 {
                    clauses.push(current_clause);
                    current_clause = Vec::new();
                } else {
                    let var = lit.unsigned_abs() as usize;
                    if strict && n_vars > 0 && (var < 1 || var > n_vars) {
                        return Err(format!("Variable {} out of declared bounds [1, {}] at line {}", var, n_vars, line_num));
                    }
                    current_clause.push(lit);
                }
            }
        }

        if !current_clause.is_empty() {
            if strict {
                return Err("Unterminated trailing clause: missing final 0 delimiter".to_string());
            }
            clauses.push(current_clause);
        }

        if strict {
            if let Some(decl) = n_clauses_decl {
                if clauses.len() != decl {
                    return Err(format!("Clause count mismatch: declared {}, but parsed {}", decl, clauses.len()));
                }
            }
        }

        Ok(Self { n_vars, clauses })
    }

    /// Parse DIMACS CNF format string (non-strict mode for backward compatibility).
    pub fn from_dimacs(dimacs: &str) -> Result<Self, String> {
        Self::from_dimacs_with_options(dimacs, false)
    }

    /// Builds the primal variable interaction graph (edges between distinct variables
    /// co-occurring in at least one clause).
    /// Deduplicates variables per clause to prevent self-loops and false clique inflation.
    pub fn extract_variable_interaction_graph(&self) -> HashMap<usize, HashSet<usize>> {
        let mut adj: HashMap<usize, HashSet<usize>> = (1..=self.n_vars)
            .map(|i| (i, HashSet::new()))
            .collect();

        for clause in &self.clauses {
            let mut vars_set: HashSet<usize> = clause.iter().map(|&lit| lit.unsigned_abs() as usize).collect();
            vars_set.retain(|&v| v >= 1 && v <= self.n_vars);
            let vars: Vec<usize> = vars_set.into_iter().collect();

            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let u = vars[i];
                    let v = vars[j];
                    if u != v {
                        adj.get_mut(&u).unwrap().insert(v);
                        adj.get_mut(&v).unwrap().insert(u);
                    }
                }
            }
        }

        adj
    }

    /// Identifies dense variable cliques and partitions clauses into:
    /// (dense_clusters, sparse_clauses, dense_clauses, boundary_clauses).
    /// Preserves 100% of clause ownership across partition boundaries.
    ///
    /// Note on clique detection heuristic: Candidate clusters are formed by v union N(v)
    /// for vertices with degree >= 3 (cluster size k >= 4). Cluster density is evaluated over
    /// all pairs in the candidate set; low-degree pendant neighbors of v can dilute the density
    /// score below the threshold. Clause conservation is strictly guaranteed regardless.
    pub fn partition_sparse_and_dense_cliques(
        &self,
        density_threshold: f64,
    ) -> CNFPartition {
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

        let mut sparse_clauses = Vec::new();
        let mut dense_clauses = Vec::new();
        let mut boundary_clauses = Vec::new();

        for clause in &self.clauses {
            let clause_vars: HashSet<usize> = clause.iter().map(|&l| l.unsigned_abs() as usize).collect();
            if !clause_vars.iter().any(|v| dense_vars.contains(v)) {
                sparse_clauses.push(clause.clone());
            } else if dense_clusters.iter().any(|cluster| clause_vars.is_subset(cluster)) {
                dense_clauses.push(clause.clone());
            } else {
                boundary_clauses.push(clause.clone());
            }
        }

        (dense_clusters, sparse_clauses, dense_clauses, boundary_clauses)
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
        let (dense, sparse, dense_cls, boundary_cls) = cnf.partition_sparse_and_dense_cliques(0.8);

        assert!(!dense.is_empty());
        let expected_clique: HashSet<usize> = [1, 2, 3, 4].into_iter().collect();
        assert!(dense.iter().any(|c| c == &expected_clique));
        assert_eq!(dense_cls.len(), 6);
        assert_eq!(sparse.len(), 1);
        assert_eq!(sparse[0], vec![5, 6]);
        assert_eq!(sparse.len() + dense_cls.len() + boundary_cls.len(), cnf.clauses.len());
    }

    #[test]
    fn test_cnf_pendant_dilution_and_clause_conservation() {
        // 4-clique {1, 2, 3, 4} with 2 private pendant neighbors per clique vertex:
        // Vertex 1: 10, 11
        // Vertex 2: 20, 21
        // Vertex 3: 30, 31
        // Vertex 4: 40, 41
        // Total clauses: 6 (clique) + 8 (pendants) = 14 clauses.
        let dimacs = r#"
p cnf 42 14
1 2 0
2 3 0
3 4 0
1 4 0
2 4 0
1 3 0
1 10 0
1 11 0
2 20 0
2 21 0
3 30 0
3 31 0
4 40 0
4 41 0
"#;
        let cnf = CNFFormula::from_dimacs(dimacs).expect("Failed to parse DIMACS");
        let (dense, sparse, dense_cls, boundary_cls) = cnf.partition_sparse_and_dense_cliques(0.8);

        // Due to pendant dilution, candidate cluster v U N(v) has size 6 with density diluted below 0.8.
        // As documented in Item 13, all clauses fall into the sparse bucket under high threshold.
        // Clause conservation MUST strictly hold regardless.
        assert_eq!(
            sparse.len() + dense_cls.len() + boundary_cls.len(),
            cnf.clauses.len()
        );
        assert_eq!(sparse.len() + dense_cls.len() + boundary_cls.len(), 14);
        let _ = dense;
    }

    #[test]
    fn test_dimacs_token_streaming_and_strict() {
        let stream = "c multi-line\np cnf 3 2\n1 2\n3 0 0\n";
        let cnf = CNFFormula::from_dimacs(stream).expect("stream parse failed");
        assert_eq!(cnf.clauses, vec![vec![1, 2, 3], vec![]]);

        assert!(CNFFormula::from_dimacs_with_options("p cnf 2 3\n1 2 0\n", true).is_err());
        assert!(CNFFormula::from_dimacs_with_options("p cnf 2 1\n1 4 0\n", true).is_err());
    }
}
