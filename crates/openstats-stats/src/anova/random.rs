//! Analysis of variance with random factors: adjusted sums of squares,
//! expected mean squares, F tests against (synthesized) error terms and
//! ANOVA-method variance components.
//!
//! The sums of squares are the Type III (adjusted) ones of the sum-to-zero
//! coded fixed-effects design, so they agree with the general linear model.
//! Every random term `j` contributes `σ²ⱼ` to the expected mean square of term
//! `i` with the coefficient `tr(Pᵢ Zⱼ Zⱼᵀ)/dfᵢ`, where `Pᵢ` is the projection
//! behind the sum of squares of `i` and `Zⱼ` holds the indicator columns of the
//! levels of `j`. For balanced data this is the familiar product of level
//! counts; for unbalanced data it is the coefficient of the "unrestricted"
//! mixed model that Minitab prints.

use crate::error::{invalid, Result, StatsError};
use crate::model::{build_design, AtomKind, Contrast, Data, DesignOptions, Formula, Term};
use openstats_math::dist::{ContinuousDistribution, FisherF};
use std::collections::{BTreeMap, BTreeSet};

/// Options of a random-factor analysis.
#[derive(Debug, Clone)]
pub struct RandomOptions {
    /// Design coding (sum-to-zero by default; other codings change the sums of squares).
    pub design: DesignOptions,
    /// Use the restricted form of the mixed model (variance components of
    /// interactions with fixed factors do not enter the expected mean squares
    /// of the lower-order terms). Minitab's Balanced ANOVA default.
    pub restricted: bool,
}

impl Default for RandomOptions {
    fn default() -> Self {
        RandomOptions {
            design: DesignOptions { default_contrast: Contrast::SumToZero, ..DesignOptions::default() },
            restricted: false,
        }
    }
}

/// One term of the analysis of variance with its expected mean square.
#[derive(Debug, Clone, PartialEq)]
pub struct EmsRow {
    /// Term label.
    pub source: String,
    /// Whether the term is random (contains a random factor).
    pub random: bool,
    /// Degrees of freedom.
    pub df: usize,
    /// Adjusted sum of squares.
    pub ss: f64,
    /// Adjusted mean square.
    pub ms: f64,
    /// F statistic (`None` when no error term can be formed).
    pub f: Option<f64>,
    /// Degrees of freedom of the (synthesized) denominator.
    pub den_df: Option<f64>,
    /// p-value.
    pub p_value: Option<f64>,
    /// Denominator as `(row index, weight)` pairs; the error row has index
    /// `rows.len()`.
    pub denominator: Vec<(usize, f64)>,
    /// Coefficients of the variance components in the expected mean square
    /// (same order as [`RandomAnova::components`], error last).
    pub coefficients: Vec<f64>,
    /// Whether the denominator is a synthesized combination (not one mean square).
    pub synthesized: bool,
}

/// A variance component.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceComponent {
    /// Source label (`Error` for the residual).
    pub source: String,
    /// ANOVA (method of moments) estimate, possibly negative.
    pub estimate: f64,
    /// The estimate truncated at zero.
    pub used: f64,
    /// Percent of the total of the truncated estimates.
    pub percent: f64,
    /// Standard deviation `√used`.
    pub std_dev: f64,
}

/// Mean of the response for one level combination of a term.
#[derive(Debug, Clone, PartialEq)]
pub struct TermMeans {
    /// Term label.
    pub term: String,
    /// Level labels of each combination and the number of observations and mean.
    pub rows: Vec<(Vec<String>, usize, f64)>,
}

/// Result of an analysis of variance with random factors.
#[derive(Debug, Clone)]
pub struct RandomAnova {
    /// Response column.
    pub response: String,
    /// Model formula.
    pub formula: String,
    /// Whether the restricted model form was used.
    pub restricted: bool,
    /// Whether every cell has the same number of observations (balanced design).
    pub balanced: bool,
    /// Terms in model order.
    pub rows: Vec<EmsRow>,
    /// Error degrees of freedom.
    pub df_error: usize,
    /// Error sum of squares.
    pub ss_error: f64,
    /// Error mean square.
    pub ms_error: f64,
    /// Total degrees of freedom.
    pub df_total: usize,
    /// Total sum of squares (about the mean).
    pub ss_total: f64,
    /// Variance components: random terms in model order, then the error.
    pub components: Vec<VarianceComponent>,
    /// Observations used.
    pub n: usize,
    /// Group means for every factor term.
    pub means: Vec<TermMeans>,
}

// ---------------------------------------------------------------------------
// Linear-algebra helpers (orthonormal bases of column spaces)
// ---------------------------------------------------------------------------

/// Orthonormal basis of the column space spanned by `cols` (dependent columns are dropped).
pub(crate) fn orthobasis(cols: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let n = cols.first().map_or(0, Vec::len);
    for c in cols {
        let mut v = c.clone();
        let n0 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n0 == 0.0 {
            continue;
        }
        for _ in 0..2 {
            for q in &basis {
                let d: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(q).for_each(|(a, b)| *a -= d * b);
            }
        }
        let nr = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if nr <= tol * n0 || basis.len() >= n {
            continue;
        }
        v.iter_mut().for_each(|a| *a /= nr);
        basis.push(v);
    }
    basis
}

fn proj_ss(b: &[Vec<f64>], y: &[f64]) -> f64 {
    b.iter()
        .map(|q| {
            let d: f64 = q.iter().zip(y).map(|(a, c)| a * c).sum();
            d * d
        })
        .sum()
}

/// `‖Bᵀ Z‖²_F` where `Z` has one indicator column per group.
fn proj_groups(b: &[Vec<f64>], groups: &[usize], m: usize) -> f64 {
    let mut total = 0.0;
    let mut acc = vec![0.0; m];
    for q in b {
        acc.iter_mut().for_each(|a| *a = 0.0);
        for (&g, v) in groups.iter().zip(q) {
            acc[g] += v;
        }
        total += acc.iter().map(|a| a * a).sum::<f64>();
    }
    total
}

fn group_ids(cols: &[&Vec<String>]) -> (Vec<usize>, Vec<Vec<String>>) {
    let n = cols.first().map_or(0, |c| c.len());
    let mut map: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    let mut keys: Vec<Vec<String>> = Vec::new();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let key: Vec<String> = cols.iter().map(|c| c[i].clone()).collect();
        let next = keys.len();
        let id = *map.entry(key.clone()).or_insert_with(|| {
            keys.push(key);
            next
        });
        ids.push(id);
    }
    (ids, keys)
}

/// One mean-square row used by the engine.
struct EngineRow {
    label: String,
    random: bool,
    df: usize,
    ss: f64,
    /// Which components can enter the expected mean square (restricted rule).
    allowed: Vec<bool>,
    coef: Vec<f64>,
}

/// Solve a small dense system by Gaussian elimination with partial pivoting.
fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &v)| r.iter().copied().chain([v]).collect()).collect();
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[piv][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                if f != 0.0 {
                    for k in c..=n {
                        let t = m[c][k];
                        m[r][k] -= f * t;
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// Least-squares solution of `A w = t` for a few columns (normal equations) and its residual norm.
fn small_lsq(cols: &[&Vec<f64>], t: &[f64]) -> Option<(Vec<f64>, f64)> {
    let s = cols.len();
    let mut g = vec![vec![0.0; s]; s];
    let mut r = vec![0.0; s];
    for i in 0..s {
        for j in 0..s {
            g[i][j] = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
        }
        r[i] = cols[i].iter().zip(t).map(|(a, b)| a * b).sum();
    }
    let w = solve_dense(&g, &r)?;
    let mut res = 0.0;
    for (k, tk) in t.iter().enumerate() {
        let fit: f64 = (0..s).map(|i| w[i] * cols[i][k]).sum();
        res += (fit - tk) * (fit - tk);
    }
    Some((w, res.sqrt()))
}

/// Find a combination of the other mean squares whose expectation equals `target`.
fn find_denominator(target: &[f64], cands: &[(usize, &Vec<f64>)]) -> Option<Vec<(usize, f64)>> {
    let tn = target.iter().map(|a| a * a).sum::<f64>().sqrt();
    let tol = 1e-8 * (1.0 + tn);
    let mut mixed: Option<Vec<(usize, f64)>> = None;
    for size in 1..=3usize {
        let mut idx = vec![0usize; size];
        // Enumerate combinations of `size` candidates.
        #[allow(clippy::too_many_arguments)]
        fn rec(
            pos: usize,
            start: usize,
            size: usize,
            idx: &mut Vec<usize>,
            cands: &[(usize, &Vec<f64>)],
            target: &[f64],
            tol: f64,
            mixed: &mut Option<Vec<(usize, f64)>>,
        ) -> Option<Vec<(usize, f64)>> {
            if pos == size {
                let cols: Vec<&Vec<f64>> = idx.iter().map(|&i| cands[i].1).collect();
                if let Some((w, res)) = small_lsq(&cols, target) {
                    if res <= tol && w.iter().all(|v| v.abs() > 1e-9) {
                        let sol: Vec<(usize, f64)> =
                            idx.iter().zip(&w).map(|(&i, &v)| (cands[i].0, (v * 1e9).round() / 1e9)).collect();
                        if w.iter().all(|&v| v > 0.0) {
                            return Some(sol);
                        }
                        if mixed.is_none() {
                            *mixed = Some(sol);
                        }
                    }
                }
                return None;
            }
            for i in start..cands.len() {
                idx[pos] = i;
                if let Some(s) = rec(pos + 1, i + 1, size, idx, cands, target, tol, mixed) {
                    return Some(s);
                }
            }
            None
        }
        if cands.len() >= size {
            if let Some(s) = rec(0, 0, size, &mut idx, cands, target, tol, &mut mixed) {
                return Some(s);
            }
        }
        if mixed.is_some() {
            return mixed;
        }
    }
    mixed
}

#[allow(clippy::too_many_arguments)]
fn finish(
    response: String,
    formula: String,
    restricted: bool,
    balanced: bool,
    rows: Vec<EngineRow>,
    df_error: usize,
    ss_error: f64,
    ss_total: f64,
    n: usize,
    means: Vec<TermMeans>,
) -> Result<RandomAnova> {
    if df_error == 0 {
        return invalid("the model leaves no error degrees of freedom");
    }
    let ms_error = ss_error / df_error as f64;
    let random_idx: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.random).map(|(i, _)| i).collect();
    let nc = random_idx.len() + 1;
    let ms: Vec<f64> = rows.iter().map(|r| if r.df > 0 { r.ss / r.df as f64 } else { f64::NAN }).collect();

    // Coefficient vectors over (random terms…, error); clean and apply the restricted rule.
    let mut coef_vecs: Vec<Vec<f64>> = Vec::new();
    for r in &rows {
        let mut c: Vec<f64> = (0..nc - 1)
            .map(|k| {
                let v = r.coef[k];
                if (restricted && !r.allowed[k]) || v.abs() < 1e-9 {
                    0.0
                } else {
                    v
                }
            })
            .collect();
        c.push(1.0);
        coef_vecs.push(c);
    }
    let err_vec: Vec<f64> = (0..nc).map(|k| if k == nc - 1 { 1.0 } else { 0.0 }).collect();

    // F tests.
    let mut out_rows = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let mut target = coef_vecs[i].clone();
        if r.random {
            if let Some(pos) = random_idx.iter().position(|&x| x == i) {
                target[pos] = 0.0;
            }
        }
        let mut cands: Vec<(usize, &Vec<f64>)> = Vec::new();
        for (k, c) in coef_vecs.iter().enumerate() {
            if k != i && rows[k].df > 0 && rows[k].random {
                cands.push((k, c));
            }
        }
        cands.push((rows.len(), &err_vec));
        let (mut f, mut den_df, mut p, mut den) = (None, None, None, Vec::new());
        if r.df > 0 {
            if let Some(sol) = find_denominator(&target, &cands) {
                let value: f64 = sol.iter().map(|&(k, w)| w * if k == rows.len() { ms_error } else { ms[k] }).sum();
                let denom: f64 = sol
                    .iter()
                    .map(|&(k, w)| {
                        let (m, d) = if k == rows.len() { (ms_error, df_error) } else { (ms[k], rows[k].df) };
                        (w * m) * (w * m) / d as f64
                    })
                    .sum();
                if value > 0.0 && denom > 0.0 {
                    let nu = value * value / denom;
                    let fv = ms[i] / value;
                    f = Some(fv);
                    den_df = Some(nu);
                    p = Some(FisherF::new(r.df as f64, nu)?.sf(fv));
                    den = sol;
                }
            }
        }
        let synthesized = den.len() > 1 || den.iter().any(|&(_, w)| (w - 1.0).abs() > 1e-9);
        out_rows.push(EmsRow {
            source: r.label.clone(),
            random: r.random,
            df: r.df,
            ss: r.ss,
            ms: ms[i],
            f,
            den_df,
            p_value: p,
            denominator: den,
            coefficients: coef_vecs[i].clone(),
            synthesized,
        });
    }

    // ANOVA variance components: solve MS_i = EMS_i for the random terms and the error.
    let mut a = vec![vec![0.0; nc]; nc];
    let mut b = vec![0.0; nc];
    for (row_k, &i) in random_idx.iter().enumerate() {
        if rows[i].df == 0 {
            return invalid(format!("random term {} has no degrees of freedom", rows[i].label));
        }
        a[row_k] = coef_vecs[i].clone();
        b[row_k] = ms[i];
    }
    a[nc - 1] = err_vec;
    b[nc - 1] = ms_error;
    let est = solve_dense(&a, &b).ok_or_else(|| {
        StatsError::InvalidArgument("the variance components cannot be estimated (singular system)".into())
    })?;
    let used: Vec<f64> = est.iter().map(|v| v.max(0.0)).collect();
    let total: f64 = used.iter().sum();
    let mut components = Vec::new();
    for k in 0..nc {
        let label = if k == nc - 1 { "Error".to_string() } else { rows[random_idx[k]].label.clone() };
        components.push(VarianceComponent {
            source: label,
            estimate: est[k],
            used: used[k],
            percent: if total > 0.0 { 100.0 * used[k] / total } else { f64::NAN },
            std_dev: used[k].sqrt(),
        });
    }
    let df_total = rows.iter().map(|r| r.df).sum::<usize>() + df_error;
    Ok(RandomAnova {
        response,
        formula,
        restricted,
        balanced,
        rows: out_rows,
        df_error,
        ss_error,
        ms_error,
        df_total,
        ss_total,
        components,
        n,
        means,
    })
}

fn factor_set(t: &Term, data: &Data) -> BTreeSet<String> {
    t.factors
        .iter()
        .map(|(a, _)| a)
        .chain(t.nested_in.iter())
        .filter_map(|a| match &a.kind {
            AtomKind::Var(n) if data.factor(n).is_some() => Some(n.clone()),
            AtomKind::AsFactor(n) => Some(n.clone()),
            _ => None,
        })
        .collect()
}

/// Within-parent contrasts of a nested term `A(B)`: for every cell of `B`, sum-to-zero contrasts of
/// the levels of `A` observed in that cell (zero elsewhere). `None` for other terms.
fn nested_block(t: &Term, fcols: &BTreeMap<String, Vec<String>>, n: usize) -> Option<Vec<Vec<f64>>> {
    if t.nested_in.is_empty() || t.factors.iter().any(|(_, p)| *p != 1) {
        return None;
    }
    let name = |a: &crate::model::Atom| match &a.kind {
        AtomKind::Var(v) if fcols.contains_key(v) => Some(v.clone()),
        _ => None,
    };
    let fs: Vec<String> = t.factors.iter().map(|(a, _)| name(a)).collect::<Option<_>>()?;
    let ps: Vec<String> = t.nested_in.iter().map(name).collect::<Option<_>>()?;
    let join =
        |names: &[String], i: usize| names.iter().map(|f| fcols[f][i].as_str()).collect::<Vec<_>>().join("\u{1f}");
    let fl: Vec<String> = (0..n).map(|i| join(&fs, i)).collect();
    let pl: Vec<String> = (0..n).map(|i| join(&ps, i)).collect();
    let cells: BTreeSet<&String> = pl.iter().collect();
    let mut cols = Vec::new();
    for cell in cells {
        let levels: Vec<&String> =
            (0..n).filter(|&i| &pl[i] == cell).map(|i| &fl[i]).collect::<BTreeSet<_>>().into_iter().collect();
        let m = levels.len();
        for j in 0..m.saturating_sub(1) {
            cols.push(
                (0..n)
                    .map(|i| {
                        if &pl[i] != cell {
                            0.0
                        } else if fl[i] == *levels[j] {
                            1.0
                        } else if fl[i] == *levels[m - 1] {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
        }
    }
    Some(cols)
}

/// Analysis of variance with random factors for the model `formula`.
///
/// `random` names the random factors; a term is random when it contains one.
pub fn glm_random(formula: &str, random: &[String], data: &Data, opts: &RandomOptions) -> Result<RandomAnova> {
    let parsed = Formula::parse(formula)?;
    let response = match &parsed.response {
        Some(crate::model::Expr::Var(n)) => n.clone(),
        _ => return invalid("the response must be a single worksheet column"),
    };
    for r in random {
        if data.factor(r).is_none() {
            return invalid(format!("random factor '{r}' is not a categorical column"));
        }
    }
    let (_, dm) = build_design(formula, data, &opts.design)?;
    let n = dm.x.rows();
    let y = dm.y.clone().ok_or_else(|| StatsError::InvalidArgument("the model has no response".into()))?;
    if dm.terms.len() != parsed.terms.len() {
        return invalid("internal error: term bookkeeping mismatch");
    }
    let rs: BTreeSet<String> = random.iter().cloned().collect();
    let sets: Vec<BTreeSet<String>> = parsed.terms.iter().map(|t| factor_set(t, data)).collect();
    let is_random: Vec<bool> = sets.iter().map(|s| s.iter().any(|f| rs.contains(f))).collect();
    if !is_random.iter().any(|&r| r) {
        return invalid("the model has no random term: name at least one random factor that appears in the model");
    }
    // Every random factor must appear in a term (otherwise it is silently ignored).
    for r in random {
        if !sets.iter().any(|s| s.contains(r)) {
            return invalid(format!("random factor '{r}' does not appear in the model"));
        }
    }

    // Observation groups of every random term.
    let used_factor = |name: &str| -> Vec<String> {
        let col = data.factor(name).unwrap_or(&[]);
        dm.rows_used.iter().map(|&r| col[r].clone()).collect()
    };
    let mut fcols: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for s in &sets {
        for f in s {
            fcols.entry(f.clone()).or_insert_with(|| used_factor(f));
        }
    }
    let random_terms: Vec<usize> = (0..parsed.terms.len()).filter(|&i| is_random[i]).collect();
    let mut groups: Vec<(Vec<usize>, usize)> = Vec::new();
    for &j in &random_terms {
        let cols: Vec<&Vec<String>> = sets[j].iter().map(|f| &fcols[f]).collect();
        let (ids, keys) = group_ids(&cols);
        groups.push((ids, keys.len()));
    }

    // Columns: those outside every term (the constant), then each term's block. A nested term A(B)
    // uses contrasts of the levels of A *within* each cell of B (sum-to-zero inside the cell), so that
    // it does not contain the main effect of B even when A's labels are unique across B.
    let p = dm.x.cols();
    let mut in_term = vec![false; p];
    for tc in &dm.terms {
        for j in tc.start..tc.start + tc.len {
            in_term[j] = true;
        }
    }
    let base_cols: Vec<Vec<f64>> = (0..p).filter(|&j| !in_term[j]).map(|j| dm.x.col(j)).collect();
    let mut term_cols: Vec<Vec<Vec<f64>>> = Vec::new();
    for (i, t) in parsed.terms.iter().enumerate() {
        let tc = &dm.terms[i];
        term_cols.push(
            nested_block(t, &fcols, n).unwrap_or_else(|| (tc.start..tc.start + tc.len).map(|j| dm.x.col(j)).collect()),
        );
    }
    let tol = 1e-10;
    let mut all_cols = base_cols.clone();
    all_cols.extend(term_cols.iter().flatten().cloned());
    let full = orthobasis(&all_cols, tol);
    let rank_full = full.len();
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let ss_model = proj_ss(&full, &y);
    let ss_error = (yy - ss_model).max(0.0);
    let df_error = n.saturating_sub(rank_full);
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let ss_total: f64 = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum();

    let mut rows = Vec::new();
    for (i, t) in parsed.terms.iter().enumerate() {
        let mut keep = base_cols.clone();
        for (k, blk) in term_cols.iter().enumerate() {
            if k != i {
                keep.extend(blk.iter().cloned());
            }
        }
        let red = orthobasis(&keep, tol);
        let df = rank_full - red.len();
        let ss = (ss_model - proj_ss(&red, &y)).max(0.0);
        let mut coef = Vec::new();
        let mut allowed = Vec::new();
        for (k, &j) in random_terms.iter().enumerate() {
            let (ids, m) = &groups[k];
            let v = if df > 0 { (proj_groups(&full, ids, *m) - proj_groups(&red, ids, *m)) / df as f64 } else { 0.0 };
            coef.push(v);
            let ok = sets[i].is_subset(&sets[j]) && sets[j].difference(&sets[i]).all(|f| rs.contains(f));
            allowed.push(ok);
        }
        rows.push(EngineRow { label: t.label(), random: is_random[i], df, ss, allowed, coef });
    }

    // Balance: all cells of the highest-order factor term have the same count.
    let all_factors: Vec<&Vec<String>> = fcols.values().collect();
    let balanced = if all_factors.is_empty() {
        true
    } else {
        let (ids, keys) = group_ids(&all_factors);
        let mut cnt = vec![0usize; keys.len()];
        ids.iter().for_each(|&g| cnt[g] += 1);
        let full_grid: usize = fcols.values().map(|c| c.iter().collect::<BTreeSet<_>>().len()).product();
        cnt.iter().all(|&c| c == cnt[0]) && keys.len() == full_grid
    };

    // Means of every factor term.
    let mut means = Vec::new();
    for (i, t) in parsed.terms.iter().enumerate() {
        if sets[i].is_empty() {
            continue;
        }
        let cols: Vec<&Vec<String>> = sets[i].iter().map(|f| &fcols[f]).collect();
        let (ids, keys) = group_ids(&cols);
        let mut sum = vec![0.0; keys.len()];
        let mut cnt = vec![0usize; keys.len()];
        for (&g, v) in ids.iter().zip(&y) {
            sum[g] += v;
            cnt[g] += 1;
        }
        let mut order: Vec<usize> = (0..keys.len()).collect();
        order.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
        means.push(TermMeans {
            term: t.label(),
            rows: order.into_iter().map(|g| (keys[g].clone(), cnt[g], sum[g] / cnt[g] as f64)).collect(),
        });
    }
    finish(response, formula.to_string(), opts.restricted, balanced, rows, df_error, ss_error, ss_total, n, means)
}

/// Balanced analysis of variance: like [`glm_random`] but the design must be
/// balanced (equal counts in every cell, no empty cells, no missing values).
pub fn balanced_anova(formula: &str, random: &[String], data: &Data, opts: &RandomOptions) -> Result<RandomAnova> {
    let r = glm_random(formula, random, data, opts)?;
    if !r.balanced {
        return invalid("Balanced ANOVA needs a balanced design: every combination of factor levels must have the same number of observations");
    }
    if r.n != data.n_rows() {
        return invalid("Balanced ANOVA does not allow missing values");
    }
    Ok(r)
}

/// Fully nested analysis of variance. `factors` run from the highest level of
/// the hierarchy to the lowest; every factor is nested in the ones before it
/// and all are random. Source rows are the factors and the error.
pub fn nested_anova(response: &[f64], factors: &[(&str, &[String])]) -> Result<RandomAnova> {
    if factors.is_empty() {
        return invalid("a nested ANOVA needs at least one factor");
    }
    let n_all = response.len();
    if factors.iter().any(|(_, f)| f.len() != n_all) {
        return invalid("the response and the factors differ in length");
    }
    let keep: Vec<usize> =
        (0..n_all).filter(|&i| response[i].is_finite() && factors.iter().all(|(_, f)| !f[i].is_empty())).collect();
    let n = keep.len();
    let y: Vec<f64> = keep.iter().map(|&i| response[i]).collect();
    let cols: Vec<Vec<String>> = factors.iter().map(|(_, f)| keep.iter().map(|&i| f[i].clone()).collect()).collect();
    // Nested cell identifiers: the combination of the first k factors.
    let mut ids: Vec<(Vec<usize>, usize)> = Vec::new();
    for k in 0..factors.len() {
        let refs: Vec<&Vec<String>> = cols[..=k].iter().collect();
        let (g, keys) = group_ids(&refs);
        ids.push((g, keys.len()));
    }
    let indicator_basis = |g: &[usize], m: usize| -> Vec<Vec<f64>> {
        let cols: Vec<Vec<f64>> = (0..m).map(|c| g.iter().map(|&x| if x == c { 1.0 } else { 0.0 }).collect()).collect();
        orthobasis(&cols, 1e-10)
    };
    let mut bases: Vec<Vec<Vec<f64>>> = vec![orthobasis(&[vec![1.0; n]], 1e-10)];
    for (g, m) in &ids {
        bases.push(indicator_basis(g, *m));
    }
    let top = bases.len() - 1;
    let mean_y = y.iter().sum::<f64>() / n.max(1) as f64;
    let ss_total: f64 = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum();
    let ss_full = proj_ss(&bases[top], &y);
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let ss_error = (yy - ss_full).max(0.0);
    let df_error = n.saturating_sub(bases[top].len());
    let k_f = factors.len();
    let mut rows = Vec::new();
    for i in 0..k_f {
        let (lo, hi) = (&bases[i], &bases[i + 1]);
        let df = hi.len() - lo.len();
        let ss = (proj_ss(hi, &y) - proj_ss(lo, &y)).max(0.0);
        let mut coef = Vec::new();
        for (j, (g, m)) in ids.iter().enumerate() {
            let v = if j >= i && df > 0 { (proj_groups(hi, g, *m) - proj_groups(lo, g, *m)) / df as f64 } else { 0.0 };
            coef.push(v);
        }
        rows.push(EngineRow { label: factors[i].0.to_string(), random: true, df, ss, allowed: vec![true; k_f], coef });
    }
    let cells = &ids[k_f - 1];
    let mut cnt = vec![0usize; cells.1];
    cells.0.iter().for_each(|&g| cnt[g] += 1);
    let balanced = cnt.iter().all(|&c| c == cnt[0]) && {
        // Every parent has the same number of children at every level.
        (0..k_f).all(|k| {
            let (g, m) = &ids[k];
            let mut c = vec![0usize; *m];
            g.iter().for_each(|&x| c[x] += 1);
            c.iter().all(|&v| v == c[0])
        })
    };
    let mut means = Vec::new();
    for k in 0..k_f {
        let (g, m) = &ids[k];
        let mut sum = vec![0.0; *m];
        let mut c = vec![0usize; *m];
        let mut labels: Vec<Vec<String>> = vec![Vec::new(); *m];
        for (r, &gi) in g.iter().enumerate() {
            sum[gi] += y[r];
            c[gi] += 1;
            if labels[gi].is_empty() {
                labels[gi] = (0..=k).map(|q| cols[q][r].clone()).collect();
            }
        }
        let mut order: Vec<usize> = (0..*m).collect();
        order.sort_by(|&a, &b| labels[a].cmp(&labels[b]));
        means.push(TermMeans {
            term: factors[k].0.to_string(),
            rows: order.into_iter().map(|q| (labels[q].clone(), c[q], sum[q] / c[q] as f64)).collect(),
        });
    }
    finish(
        "response".into(),
        factors.iter().map(|f| f.0).collect::<Vec<_>>().join(" > "),
        true,
        balanced,
        rows,
        df_error,
        ss_error,
        ss_total,
        n,
        means,
    )
}

impl RandomAnova {
    /// Label of the denominator of row `i` (`Error`, a term or a synthesized combination).
    pub fn denominator_label(&self, i: usize) -> String {
        let row = &self.rows[i];
        if row.denominator.is_empty() {
            return "*".into();
        }
        let name = |k: usize| if k == self.rows.len() { "Error".to_string() } else { format!("{}", k + 1) };
        if !row.synthesized {
            let k = row.denominator[0].0;
            return if k == self.rows.len() { "Error".into() } else { format!("({})", k + 1) };
        }
        let mut s = String::new();
        for (idx, &(k, w)) in row.denominator.iter().enumerate() {
            let term = if k == self.rows.len() { "Error".to_string() } else { format!("({})", name(k)) };
            let coef = if (w.abs() - 1.0).abs() < 1e-9 { String::new() } else { format!("{:.4} ", w.abs()) };
            if idx == 0 {
                if w < 0.0 {
                    s.push('-');
                }
            } else {
                s.push_str(if w < 0.0 { " - " } else { " + " });
            }
            s.push_str(&format!("{coef}{term}"));
        }
        s
    }

    /// Expected mean square of row `i` as text, e.g. `(5) + 3.0000 (3) + 12.0000 Q[1]`.
    pub fn ems_text(&self, i: usize) -> String {
        let row = &self.rows[i];
        let random_rows: Vec<usize> = self.rows.iter().enumerate().filter(|(_, r)| r.random).map(|(k, _)| k).collect();
        let mut parts: Vec<String> = Vec::new();
        let err_label = format!("({})", self.rows.len() + 1);
        parts.push(err_label);
        for (c, &k) in random_rows.iter().enumerate() {
            let v = row.coefficients[c];
            if v.abs() > 1e-9 {
                let lab = format!("({})", k + 1);
                parts.push(if (v - 1.0).abs() < 1e-9 { lab } else { format!("{v:.4} {lab}") });
            }
        }
        if !row.random {
            let q = self.rows[..i].iter().filter(|r| !r.random).count() + 1;
            parts.push(format!("Q[{q}]"));
        }
        parts.join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_way_random_balanced_components() {
        // A (3 levels) x B (2 levels) x 2 replicates, both random.
        let a: Vec<String> = (0..12).map(|i| ["a1", "a2", "a3"][(i / 4) % 3].to_string()).collect();
        let b: Vec<String> = (0..12).map(|i| ["b1", "b2"][(i / 2) % 2].to_string()).collect();
        let y = vec![10.0, 11.0, 14.0, 13.5, 12.0, 12.5, 18.0, 17.0, 9.0, 10.5, 15.0, 16.5];
        let d = Data::new().with_factor("A", &a).unwrap().with_factor("B", &b).unwrap().with_numeric("y", y).unwrap();
        let r = glm_random("y ~ A*B", &["A".into(), "B".into()], &d, &RandomOptions::default()).unwrap();
        assert!(r.balanced);
        // EMS(A) = s2 + 2 s2_AB + 4 s2_A
        let ra = &r.rows[0];
        assert!((ra.coefficients[0] - 4.0).abs() < 1e-9, "{:?}", ra.coefficients);
        assert!((ra.coefficients[2] - 2.0).abs() < 1e-9);
        assert_eq!(ra.denominator, vec![(2, 1.0)]);
    }
}
