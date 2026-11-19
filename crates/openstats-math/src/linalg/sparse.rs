//! Sparse matrices: triplet (COO) builder, compressed sparse row (CSR) and
//! column (CSC) storage, a fill-reducing sparse Cholesky factorisation and
//! the preconditioned conjugate-gradient method.
//!
//! The sparse Cholesky follows the up-looking algorithm of Davis, *Direct
//! Methods for Sparse Linear Systems* (2006): minimum-degree ordering,
//! elimination tree, symbolic row patterns via `ereach`, then numeric
//! factorisation one row of L at a time.

use super::matrix::Matrix;
use crate::error::{MathError, Result};
use std::collections::{BTreeSet, HashSet};

/// Coordinate-format builder. Duplicate entries are summed on compression.
#[derive(Debug, Clone, Default)]
pub struct Triplet {
    pub rows: usize,
    pub cols: usize,
    entries: Vec<(usize, usize, f64)>,
}

impl Triplet {
    pub fn new(rows: usize, cols: usize) -> Self {
        Triplet { rows, cols, entries: Vec::new() }
    }

    pub fn push(&mut self, i: usize, j: usize, v: f64) -> Result<()> {
        if i >= self.rows || j >= self.cols {
            return Err(MathError::InvalidArgument(format!("entry ({i}, {j}) outside {}×{}", self.rows, self.cols)));
        }
        if !v.is_finite() {
            return Err(MathError::NonFinite { op: "sparse entry" });
        }
        self.entries.push((i, j, v));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn to_csr(&self) -> Csr {
        Csr::from_triplets(self.rows, self.cols, &self.entries)
    }

    pub fn to_csc(&self) -> Csc {
        let t: Vec<(usize, usize, f64)> = self.entries.iter().map(|&(i, j, v)| (j, i, v)).collect();
        let c = Csr::from_triplets(self.cols, self.rows, &t);
        Csc { rows: self.rows, cols: self.cols, colptr: c.rowptr, rowidx: c.colidx, values: c.values }
    }
}

/// Compressed sparse row matrix with sorted column indices and no duplicates.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    pub rows: usize,
    pub cols: usize,
    pub rowptr: Vec<usize>,
    pub colidx: Vec<usize>,
    pub values: Vec<f64>,
}

impl Csr {
    fn from_triplets(rows: usize, cols: usize, t: &[(usize, usize, f64)]) -> Csr {
        let mut count = vec![0usize; rows + 1];
        for &(i, _, _) in t {
            count[i + 1] += 1;
        }
        for i in 0..rows {
            count[i + 1] += count[i];
        }
        let mut next = count.clone();
        let mut ci = vec![0usize; t.len()];
        let mut cv = vec![0.0; t.len()];
        for &(i, j, v) in t {
            ci[next[i]] = j;
            cv[next[i]] = v;
            next[i] += 1;
        }
        // Sort each row by column and sum duplicates.
        let mut rowptr = vec![0usize; rows + 1];
        let mut colidx = Vec::with_capacity(t.len());
        let mut values = Vec::with_capacity(t.len());
        let mut buf: Vec<(usize, f64)> = Vec::new();
        for i in 0..rows {
            buf.clear();
            buf.extend((count[i]..count[i + 1]).map(|p| (ci[p], cv[p])));
            buf.sort_by_key(|e| e.0);
            for &(j, v) in &buf {
                if colidx.len() > rowptr[i] && *colidx.last().expect("nonempty") == j {
                    *values.last_mut().expect("nonempty") += v;
                } else {
                    colidx.push(j);
                    values.push(v);
                }
            }
            rowptr[i + 1] = colidx.len();
        }
        Csr { rows, cols, rowptr, colidx, values }
    }

    /// From a dense matrix (entries with |a| > `drop_tol` kept).
    pub fn from_dense(a: &Matrix, drop_tol: f64) -> Csr {
        let mut t = Vec::new();
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                if a[(i, j)].abs() > drop_tol {
                    t.push((i, j, a[(i, j)]));
                }
            }
        }
        Csr::from_triplets(a.rows(), a.cols(), &t)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        let r = &self.colidx[self.rowptr[i]..self.rowptr[i + 1]];
        match r.binary_search(&j) {
            Ok(k) => self.values[self.rowptr[i] + k],
            Err(_) => 0.0,
        }
    }

    /// y = A·x
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(MathError::DimensionMismatch {
                op: "sparse matvec",
                left: (self.rows, self.cols),
                right: (x.len(), 1),
            });
        }
        Ok((0..self.rows)
            .map(|i| (self.rowptr[i]..self.rowptr[i + 1]).map(|p| self.values[p] * x[self.colidx[p]]).sum())
            .collect())
    }

    /// y = Aᵀ·x
    pub fn t_matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.rows {
            return Err(MathError::DimensionMismatch {
                op: "sparse t_matvec",
                left: (self.rows, self.cols),
                right: (x.len(), 1),
            });
        }
        let mut y = vec![0.0; self.cols];
        for i in 0..self.rows {
            for p in self.rowptr[i]..self.rowptr[i + 1] {
                y[self.colidx[p]] += self.values[p] * x[i];
            }
        }
        Ok(y)
    }

    pub fn transpose(&self) -> Csr {
        let t: Vec<(usize, usize, f64)> = self.iter().map(|(i, j, v)| (j, i, v)).collect();
        Csr::from_triplets(self.cols, self.rows, &t)
    }

    /// (row, col, value) for every stored entry.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.rows)
            .flat_map(move |i| (self.rowptr[i]..self.rowptr[i + 1]).map(move |p| (i, self.colidx[p], self.values[p])))
    }

    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        for (i, j, v) in self.iter() {
            m[(i, j)] = v;
        }
        m
    }

    pub fn to_csc(&self) -> Csc {
        let t = self.transpose();
        Csc { rows: self.rows, cols: self.cols, colptr: t.rowptr, rowidx: t.colidx, values: t.values }
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).collect()
    }

    /// Sparse product A·B.
    pub fn matmul(&self, b: &Csr) -> Result<Csr> {
        if self.cols != b.rows {
            return Err(MathError::DimensionMismatch {
                op: "sparse matmul",
                left: (self.rows, self.cols),
                right: (b.rows, b.cols),
            });
        }
        let mut t = Vec::new();
        let mut acc = vec![0.0; b.cols];
        let mut mark = vec![usize::MAX; b.cols];
        let mut touched = Vec::new();
        for i in 0..self.rows {
            touched.clear();
            for p in self.rowptr[i]..self.rowptr[i + 1] {
                let (k, a) = (self.colidx[p], self.values[p]);
                for q in b.rowptr[k]..b.rowptr[k + 1] {
                    let j = b.colidx[q];
                    if mark[j] != i {
                        mark[j] = i;
                        acc[j] = 0.0;
                        touched.push(j);
                    }
                    acc[j] += a * b.values[q];
                }
            }
            for &j in &touched {
                t.push((i, j, acc[j]));
            }
        }
        Ok(Csr::from_triplets(self.rows, b.cols, &t))
    }
}

/// Compressed sparse column matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Csc {
    pub rows: usize,
    pub cols: usize,
    pub colptr: Vec<usize>,
    pub rowidx: Vec<usize>,
    pub values: Vec<f64>,
}

impl Csc {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn to_csr(&self) -> Csr {
        let t: Vec<(usize, usize, f64)> = (0..self.cols)
            .flat_map(|j| (self.colptr[j]..self.colptr[j + 1]).map(move |p| (self.rowidx[p], j, self.values[p])))
            .collect();
        Csr::from_triplets(self.rows, self.cols, &t)
    }

    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(MathError::DimensionMismatch {
                op: "sparse matvec",
                left: (self.rows, self.cols),
                right: (x.len(), 1),
            });
        }
        let mut y = vec![0.0; self.rows];
        for j in 0..self.cols {
            let xj = x[j];
            for p in self.colptr[j]..self.colptr[j + 1] {
                y[self.rowidx[p]] += self.values[p] * xj;
            }
        }
        Ok(y)
    }

    pub fn to_dense(&self) -> Matrix {
        self.to_csr().to_dense()
    }
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/// Minimum-degree ordering of the symmetric pattern of `a` (exact degrees on
/// the explicit elimination graph; ties broken by lowest index so the result
/// is deterministic). Returns `perm`: the k-th pivot is row/column `perm[k]`.
pub fn minimum_degree(a: &Csr) -> Vec<usize> {
    let n = a.rows;
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, j, _) in a.iter() {
        if i != j {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }
    let mut queue: BTreeSet<(usize, usize)> = (0..n).map(|i| (adj[i].len(), i)).collect();
    let mut eliminated = vec![false; n];
    let mut perm = Vec::with_capacity(n);
    while let Some((_, v)) = queue.pop_first() {
        eliminated[v] = true;
        perm.push(v);
        let nbrs: Vec<usize> = adj[v].iter().copied().filter(|&u| !eliminated[u]).collect();
        // Remove v and connect its neighbours into a clique.
        for &u in &nbrs {
            queue.remove(&(adj[u].len(), u));
            adj[u].remove(&v);
        }
        for (a_i, &x) in nbrs.iter().enumerate() {
            for &y in &nbrs[a_i + 1..] {
                adj[x].insert(y);
                adj[y].insert(x);
            }
        }
        for &u in &nbrs {
            queue.insert((adj[u].len(), u));
        }
        adj[v].clear();
    }
    perm
}

// ---------------------------------------------------------------------------
// Sparse Cholesky
// ---------------------------------------------------------------------------

/// Sparse Cholesky P·A·Pᵀ = L·Lᵀ of a symmetric positive-definite matrix,
/// with L stored by columns (diagonal first in each column).
#[derive(Debug, Clone)]
pub struct SparseCholesky {
    n: usize,
    perm: Vec<usize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<f64>,
}

impl SparseCholesky {
    /// Factorise using a minimum-degree ordering. Only the lower triangle
    /// (i ≥ j) of `a` is read.
    pub fn new(a: &Csr) -> Result<Self> {
        let perm = minimum_degree(a);
        Self::with_ordering(a, perm)
    }

    /// Factorise with a caller-supplied ordering (`perm[k]` = k-th pivot).
    pub fn with_ordering(a: &Csr, perm: Vec<usize>) -> Result<Self> {
        let n = a.rows;
        if a.cols != n {
            return Err(MathError::NotSquare { rows: a.rows, cols: a.cols });
        }
        if perm.len() != n {
            return Err(MathError::InvalidArgument("ordering length differs from matrix size".into()));
        }
        let mut pinv = vec![usize::MAX; n];
        for (k, &p) in perm.iter().enumerate() {
            if p >= n || pinv[p] != usize::MAX {
                return Err(MathError::InvalidArgument("ordering is not a permutation".into()));
            }
            pinv[p] = k;
        }
        // C = upper triangle of P·A·Pᵀ in CSC: column max(pi,pj), row min.
        let mut t = Vec::new();
        for (i, j, v) in a.iter() {
            if i >= j {
                let (pi, pj) = (pinv[i], pinv[j]);
                t.push((pi.max(pj), pi.min(pj), v)); // (column, row) as CSR of the transpose
            }
        }
        let ct = Csr::from_triplets(n, n, &t); // row k of ct = column k of C
        let (cp, ci, cx) = (&ct.rowptr, &ct.colidx, &ct.values);

        // Elimination tree.
        let mut parent = vec![usize::MAX; n];
        let mut ancestor = vec![usize::MAX; n];
        for k in 0..n {
            for &i0 in &ci[cp[k]..cp[k + 1]] {
                let mut i = i0;
                while i != usize::MAX && i < k {
                    let next = ancestor[i];
                    ancestor[i] = k;
                    if next == usize::MAX {
                        parent[i] = k;
                    }
                    i = next;
                }
            }
        }

        // Row patterns of L via ereach; first pass counts column sizes.
        let mut stack = vec![0usize; n];
        let mut mark = vec![usize::MAX; n];
        let mut counts = vec![1usize; n]; // diagonal
        let mut path = vec![0usize; n];
        for k in 0..n {
            let top = ereach_into(k, cp, ci, &parent, &mut path, &mut stack, &mut mark);
            for &i in &stack[top..n] {
                counts[i] += 1;
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + counts[k];
        }
        let nnz = lp[n];
        let mut li = vec![0usize; nnz];
        let mut lx = vec![0.0; nnz];
        let mut c: Vec<usize> = lp[..n].to_vec(); // next free slot per column
        let mut x = vec![0.0; n];
        mark.iter_mut().for_each(|m| *m = usize::MAX);

        for k in 0..n {
            let top = ereach_into(k, cp, ci, &parent, &mut path, &mut stack, &mut mark);
            // Scatter column k of C (upper part).
            for p in cp[k]..cp[k + 1] {
                if ci[p] <= k {
                    x[ci[p]] += cx[p];
                }
            }
            let mut d = x[k];
            x[k] = 0.0;
            for &i in &stack[top..n] {
                let lki = x[i] / lx[lp[i]];
                x[i] = 0.0;
                for p in lp[i] + 1..c[i] {
                    x[li[p]] -= lx[p] * lki;
                }
                d -= lki * lki;
                let p = c[i];
                c[i] += 1;
                li[p] = k;
                lx[p] = lki;
            }
            if d <= 0.0 || !d.is_finite() {
                return Err(MathError::NotPositiveDefinite { pivot: perm[k] });
            }
            let p = c[k];
            c[k] += 1;
            li[p] = k;
            lx[p] = d.sqrt();
        }
        Ok(SparseCholesky { n, perm, lp, li, lx })
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// Non-zeros in L (a measure of fill-in).
    pub fn nnz_l(&self) -> usize {
        self.lx.len()
    }

    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Solve A·x = b.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let n = self.n;
        if b.len() != n {
            return Err(MathError::DimensionMismatch {
                op: "sparse Cholesky solve",
                left: (n, n),
                right: (b.len(), 1),
            });
        }
        let mut y: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        // L·z = y (column-oriented forward substitution).
        for j in 0..n {
            y[j] /= self.lx[self.lp[j]];
            let yj = y[j];
            for p in self.lp[j] + 1..self.lp[j + 1] {
                y[self.li[p]] -= self.lx[p] * yj;
            }
        }
        // Lᵀ·w = z.
        for j in (0..n).rev() {
            let mut s = y[j];
            for p in self.lp[j] + 1..self.lp[j + 1] {
                s -= self.lx[p] * y[self.li[p]];
            }
            y[j] = s / self.lx[self.lp[j]];
        }
        let mut x = vec![0.0; n];
        for k in 0..n {
            x[self.perm[k]] = y[k];
        }
        Ok(x)
    }

    /// ln det(A) = 2 Σ ln L_kk.
    pub fn log_det(&self) -> f64 {
        2.0 * (0..self.n).map(|j| self.lx[self.lp[j]].ln()).sum::<f64>()
    }

    /// The factor L (of the permuted matrix) as a dense matrix.
    pub fn l_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.n, self.n);
        for j in 0..self.n {
            for p in self.lp[j]..self.lp[j + 1] {
                m[(self.li[p], j)] = self.lx[p];
            }
        }
        m
    }
}

/// Non-zero pattern of row k of L (Davis' `ereach`): returns `top` such that
/// `out[top..n]` lists the columns i < k with L(k, i) ≠ 0 in topological order.
fn ereach_into(
    k: usize,
    cp: &[usize],
    ci: &[usize],
    parent: &[usize],
    path: &mut [usize],
    out: &mut [usize],
    mark: &mut [usize],
) -> usize {
    let n = out.len();
    let mut top = n;
    mark[k] = k;
    for &i0 in &ci[cp[k]..cp[k + 1]] {
        if i0 > k {
            continue;
        }
        let mut i = i0;
        let mut len = 0;
        while mark[i] != k {
            path[len] = i;
            len += 1;
            mark[i] = k;
            i = parent[i];
        }
        while len > 0 {
            len -= 1;
            top -= 1;
            out[top] = path[len];
        }
    }
    top
}

// ---------------------------------------------------------------------------
// Preconditioned conjugate gradients
// ---------------------------------------------------------------------------

/// Outcome of [`pcg`].
#[derive(Debug, Clone)]
pub struct PcgResult {
    pub x: Vec<f64>,
    pub iterations: usize,
    /// ‖b − A·x‖₂ / ‖b‖₂ at exit.
    pub relative_residual: f64,
}

/// Inverse diagonal for Jacobi preconditioning (zero diagonals → 1).
pub fn jacobi_preconditioner(a: &Csr) -> Vec<f64> {
    a.diagonal().iter().map(|&d| if d != 0.0 { 1.0 / d } else { 1.0 }).collect()
}

/// Solve A·x = b for symmetric positive-definite A by preconditioned
/// conjugate gradients. `minv` is the inverse diagonal preconditioner
/// (`None` = identity). Stops when the relative residual ≤ `tol`.
pub fn pcg(
    a: &Csr,
    b: &[f64],
    x0: Option<&[f64]>,
    minv: Option<&[f64]>,
    tol: f64,
    max_iter: usize,
) -> Result<PcgResult> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(MathError::DimensionMismatch { op: "pcg", left: (a.rows, a.cols), right: (b.len(), 1) });
    }
    let bnorm = super::blas::nrm2(b);
    let mut x = x0.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
    if bnorm == 0.0 {
        return Ok(PcgResult { x: vec![0.0; n], iterations: 0, relative_residual: 0.0 });
    }
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
    let apply_m = |r: &[f64]| -> Vec<f64> {
        match minv {
            Some(m) => r.iter().zip(m).map(|(a, b)| a * b).collect(),
            None => r.to_vec(),
        }
    };
    let mut z = apply_m(&r);
    let mut p = z.clone();
    let mut rz = super::blas::dot(&r, &z);
    for it in 0..=max_iter {
        let rel = super::blas::nrm2(&r) / bnorm;
        if rel <= tol {
            return Ok(PcgResult { x, iterations: it, relative_residual: rel });
        }
        if it == max_iter {
            break;
        }
        let ap = a.matvec(&p)?;
        let pap = super::blas::dot(&p, &ap);
        if pap <= 0.0 {
            return Err(MathError::NotPositiveDefinite { pivot: it });
        }
        let alpha = rz / pap;
        super::blas::axpy(alpha, &p, &mut x);
        super::blas::axpy(-alpha, &ap, &mut r);
        z = apply_m(&r);
        let rz_new = super::blas::dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for (pi, zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
    }
    Err(MathError::NoConvergence { algorithm: "conjugate gradient", iterations: max_iter })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::testutil::*;

    /// 2-D Poisson matrix on a k×k grid (5-point stencil), SPD.
    fn poisson(k: usize) -> Csr {
        let n = k * k;
        let mut t = Triplet::new(n, n);
        for i in 0..k {
            for j in 0..k {
                let r = i * k + j;
                t.push(r, r, 4.0).unwrap();
                if i > 0 {
                    t.push(r, r - k, -1.0).unwrap();
                }
                if i + 1 < k {
                    t.push(r, r + k, -1.0).unwrap();
                }
                if j > 0 {
                    t.push(r, r - 1, -1.0).unwrap();
                }
                if j + 1 < k {
                    t.push(r, r + 1, -1.0).unwrap();
                }
            }
        }
        t.to_csr()
    }

    #[test]
    fn formats_and_products() {
        let mut t = Triplet::new(3, 4);
        t.push(0, 1, 2.0).unwrap();
        t.push(2, 3, 5.0).unwrap();
        t.push(0, 1, 1.0).unwrap(); // duplicate summed
        t.push(1, 0, -1.0).unwrap();
        assert!(t.push(3, 0, 1.0).is_err());
        let a = t.to_csr();
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.get(0, 1), 3.0);
        assert_eq!(a.get(2, 2), 0.0);
        let d = a.to_dense();
        assert_eq!(t.to_csc().to_dense(), d);
        assert_eq!(a.to_csc().to_csr(), a);
        assert_eq!(a.transpose().to_dense(), d.t());
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(a.matvec(&x).unwrap(), d.matvec(&x).unwrap());
        assert_eq!(a.to_csc().matvec(&x).unwrap(), d.matvec(&x).unwrap());
        assert_eq!(a.t_matvec(&[1.0, 1.0, 1.0]).unwrap(), d.t_matvec(&[1.0, 1.0, 1.0]).unwrap());
        let r = random(4, 5, 3);
        let b = Csr::from_dense(&r, 0.3);
        let prod = a.matmul(&b).unwrap().to_dense();
        assert!(max_abs_diff(&prod, &d.matmul(&b.to_dense()).unwrap()) < 1e-14);
    }

    #[test]
    fn sparse_cholesky_matches_dense_and_reduces_fill() {
        let a = poisson(12); // n = 144
        let n = a.rows;
        let f = SparseCholesky::new(&a).unwrap();
        // L·Lᵀ equals P·A·Pᵀ.
        let l = f.l_dense();
        let pa = a.to_dense().select(f.perm(), f.perm());
        assert!(max_abs_diff(&l.matmul(&l.t()).unwrap(), &pa) < 1e-12);
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
        let b = a.matvec(&x).unwrap();
        let sol = f.solve(&b).unwrap();
        assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-11));
        let dense = super::super::cholesky::Cholesky::new(&a.to_dense()).unwrap();
        assert!((f.log_det() - dense.log_det()).abs() < 1e-9);
        // Minimum degree beats the natural (banded) ordering on fill.
        let natural = SparseCholesky::with_ordering(&a, (0..n).collect()).unwrap();
        assert!(f.nnz_l() < natural.nnz_l(), "md {} vs natural {}", f.nnz_l(), natural.nnz_l());
        // Not positive definite.
        let mut t = Triplet::new(2, 2);
        for (i, j, v) in [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 2.0), (1, 0, 2.0)] {
            t.push(i, j, v).unwrap();
        }
        assert!(matches!(SparseCholesky::new(&t.to_csr()), Err(MathError::NotPositiveDefinite { .. })));
        assert!(SparseCholesky::with_ordering(&a, vec![0; n]).is_err());
    }

    #[test]
    fn random_spd_sparse_cholesky() {
        // Random sparse SPD: diagonally dominant with random symmetric pattern.
        let n = 80;
        let mut rng = Rng(12345);
        let mut t = Triplet::new(n, n);
        let mut diag = vec![1.0; n];
        for i in 0..n {
            for j in 0..i {
                if rng.next() > 0.9 {
                    let v = rng.next();
                    t.push(i, j, v).unwrap();
                    t.push(j, i, v).unwrap();
                    diag[i] += v.abs();
                    diag[j] += v.abs();
                }
            }
        }
        for (i, &d) in diag.iter().enumerate() {
            t.push(i, i, d).unwrap();
        }
        let a = t.to_csr();
        let f = SparseCholesky::new(&a).unwrap();
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sol = f.solve(&a.matvec(&x).unwrap()).unwrap();
        assert!(sol.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-9));
    }

    #[test]
    fn conjugate_gradient() {
        let a = poisson(20); // n = 400
        let n = a.rows;
        let x: Vec<f64> = (0..n).map(|i| ((i % 17) as f64).sqrt()).collect();
        let b = a.matvec(&x).unwrap();
        let plain = pcg(&a, &b, None, None, 1e-12, 1000).unwrap();
        assert!(plain.relative_residual <= 1e-12);
        assert!(plain.x.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-9));
        let m = jacobi_preconditioner(&a);
        let pre = pcg(&a, &b, None, Some(&m), 1e-12, 1000).unwrap();
        assert!(pre.iterations <= plain.iterations);
        // Warm start from the solution converges immediately.
        let warm = pcg(&a, &b, Some(&x), None, 1e-12, 10).unwrap();
        assert!(warm.iterations <= 1);
        assert!(matches!(pcg(&a, &b, None, None, 1e-14, 3), Err(MathError::NoConvergence { .. })));
        assert_eq!(pcg(&a, &vec![0.0; n], None, None, 1e-10, 5).unwrap().iterations, 0);
    }
}
