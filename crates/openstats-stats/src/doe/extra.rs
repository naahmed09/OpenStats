//! Design-of-experiments completion: mixture designs with Scheffé models, optimal (exchange
//! algorithm) designs, and two-level split-plot designs.

use crate::error::{invalid, Result};
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};
use openstats_math::linalg::Cholesky;
use openstats_math::random::{Pcg64, Rng};
use openstats_math::stats::{ols, R2Kind};
use openstats_math::Matrix;

// ---------------------------------------------------------------------------------------------
// Mixture designs
// ---------------------------------------------------------------------------------------------

/// A mixture design: each point holds the component proportions (summing to 1).
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureDesign {
    /// Component names.
    pub components: Vec<String>,
    /// Proportions, one row per run.
    pub points: Vec<Vec<f64>>,
    /// Point type per run: 1 pure component / vertex, 2 blend on an edge or face, 0 overall
    /// centroid, −1 axial.
    pub point_types: Vec<i64>,
    /// Design family name.
    pub kind: &'static str,
}

impl MixtureDesign {
    /// Number of runs.
    pub fn runs(&self) -> usize {
        self.points.len()
    }
}

fn check_names(names: &[String], q: usize) -> Result<()> {
    if names.len() != q {
        return invalid("one name per component is needed");
    }
    if names.iter().enumerate().any(|(i, n)| n.is_empty() || names[..i].contains(n)) {
        return invalid("component names must be non-empty and distinct");
    }
    Ok(())
}

fn compositions(q: usize, m: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if cur.len() == q - 1 {
        let used: usize = cur.iter().sum();
        let mut v = cur.clone();
        v.push(m - used);
        out.push(v);
        return;
    }
    let used: usize = cur.iter().sum();
    for x in 0..=(m - used) {
        cur.push(x);
        compositions(q, m, cur, out);
        cur.pop();
    }
}

fn label(p: &[f64], centroid: bool) -> i64 {
    if centroid {
        return 0;
    }
    let nz = p.iter().filter(|&&x| x > 1e-12).count();
    if nz == 1 {
        1
    } else {
        2
    }
}

fn finish(
    components: &[String],
    mut points: Vec<Vec<f64>>,
    mut types: Vec<i64>,
    kind: &'static str,
    replicates: usize,
    randomize: Option<u64>,
) -> Result<MixtureDesign> {
    if replicates == 0 || replicates > 100 {
        return invalid("replicates must be between 1 and 100");
    }
    let (p0, t0) = (points.clone(), types.clone());
    for _ in 1..replicates {
        points.extend(p0.iter().cloned());
        types.extend(t0.iter().cloned());
    }
    if let Some(seed) = randomize {
        let mut rng = Pcg64::new(u128::from(seed), 54);
        for i in (1..points.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            points.swap(i, j);
            types.swap(i, j);
        }
    }
    Ok(MixtureDesign { components: components.to_vec(), points, point_types: types, kind })
}

fn overall_centroid(q: usize) -> Vec<f64> {
    vec![1.0 / q as f64; q]
}

fn axial_points(q: usize) -> Vec<Vec<f64>> {
    (0..q)
        .map(|i| {
            (0..q).map(|j| if i == j { (q as f64 + 1.0) / (2.0 * q as f64) } else { 1.0 / (2.0 * q as f64) }).collect()
        })
        .collect()
}

/// `{q, m}` simplex-lattice design (proportions that are multiples of 1/m), optionally with the
/// overall centroid and axial points.
pub fn simplex_lattice(
    components: &[String],
    degree: usize,
    center: bool,
    axial: bool,
    replicates: usize,
    randomize: Option<u64>,
) -> Result<MixtureDesign> {
    let q = components.len();
    if !(2..=24).contains(&q) {
        return invalid("a mixture design needs 2 to 24 components");
    }
    check_names(components, q)?;
    if !(1..=10).contains(&degree) {
        return invalid("the lattice degree must be between 1 and 10");
    }
    let mut comps = Vec::new();
    compositions(q, degree, &mut Vec::new(), &mut comps);
    if comps.len() > 20_000 {
        return invalid("this lattice has more than 20,000 points");
    }
    let mut points: Vec<Vec<f64>> =
        comps.iter().map(|c| c.iter().map(|&x| x as f64 / degree as f64).collect()).collect();
    let mut types: Vec<i64> = points.iter().map(|p| label(p, false)).collect();
    let c = overall_centroid(q);
    if center && !points.iter().any(|p| p.iter().zip(&c).all(|(a, b)| (a - b).abs() < 1e-12)) {
        points.push(c);
        types.push(0);
    }
    if axial {
        for a in axial_points(q) {
            points.push(a);
            types.push(-1);
        }
    }
    finish(components, points, types, "Simplex Lattice Design", replicates, randomize)
}

/// Simplex-centroid design: the centroid of every non-empty subset of components.
pub fn simplex_centroid(
    components: &[String],
    axial: bool,
    replicates: usize,
    randomize: Option<u64>,
) -> Result<MixtureDesign> {
    let q = components.len();
    if !(2..=12).contains(&q) {
        return invalid("a simplex-centroid design needs 2 to 12 components");
    }
    check_names(components, q)?;
    let mut points = Vec::new();
    let mut types = Vec::new();
    for mask in 1u32..(1 << q) {
        let k = mask.count_ones() as f64;
        let p: Vec<f64> = (0..q).map(|i| if mask >> i & 1 == 1 { 1.0 / k } else { 0.0 }).collect();
        types.push(label(&p, mask.count_ones() as usize == q));
        points.push(p);
    }
    if axial {
        for a in axial_points(q) {
            points.push(a);
            types.push(-1);
        }
    }
    finish(components, points, types, "Simplex Centroid Design", replicates, randomize)
}

/// Extreme-vertices design for components with lower and upper bounds: every vertex of the
/// constrained region plus the centroid of the vertices.
pub fn extreme_vertices(
    components: &[String],
    lower: &[f64],
    upper: &[f64],
    center: bool,
    replicates: usize,
    randomize: Option<u64>,
) -> Result<MixtureDesign> {
    let q = components.len();
    if !(2..=10).contains(&q) {
        return invalid("an extreme-vertices design needs 2 to 10 components");
    }
    check_names(components, q)?;
    if lower.len() != q || upper.len() != q {
        return invalid("one lower and one upper bound per component are needed");
    }
    if (0..q).any(|i| !(lower[i] >= 0.0 && upper[i] <= 1.0 && lower[i] <= upper[i])) {
        return invalid("bounds must satisfy 0 <= lower <= upper <= 1");
    }
    let (sl, su): (f64, f64) = (lower.iter().sum(), upper.iter().sum());
    if sl > 1.0 + 1e-12 || su < 1.0 - 1e-12 {
        return invalid(
            "the bounds leave no feasible mixture (sum of lower bounds must be <= 1 <= sum of upper bounds)",
        );
    }
    let mut verts: Vec<Vec<f64>> = Vec::new();
    for free in 0..q {
        for mask in 0u32..(1 << (q - 1)) {
            let mut p = vec![0.0; q];
            let mut bit = 0;
            let mut s = 0.0;
            for (j, pj) in p.iter_mut().enumerate() {
                if j == free {
                    continue;
                }
                *pj = if mask >> bit & 1 == 1 { upper[j] } else { lower[j] };
                s += *pj;
                bit += 1;
            }
            p[free] = 1.0 - s;
            if p[free] >= lower[free] - 1e-12 && p[free] <= upper[free] + 1e-12 {
                p[free] = p[free].clamp(lower[free], upper[free]);
                if !verts.iter().any(|v| v.iter().zip(&p).all(|(a, b)| (a - b).abs() < 1e-10)) {
                    verts.push(p);
                }
            }
        }
    }
    if verts.is_empty() {
        return invalid("no vertices were found");
    }
    verts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut types: Vec<i64> = vec![1; verts.len()];
    if center {
        let n = verts.len() as f64;
        let c: Vec<f64> = (0..q).map(|j| verts.iter().map(|v| v[j]).sum::<f64>() / n).collect();
        verts.push(c);
        types.push(0);
    }
    finish(components, verts, types, "Extreme Vertices Design", replicates, randomize)
}

/// Scheffé polynomial for mixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixtureModel {
    /// `Σ βᵢ xᵢ`.
    Linear,
    /// Linear plus `βᵢⱼ xᵢ xⱼ`.
    Quadratic,
    /// Quadratic plus `βᵢⱼₖ xᵢ xⱼ xₖ`.
    SpecialCubic,
    /// Quadratic plus `δᵢⱼ xᵢ xⱼ (xᵢ − xⱼ)`.
    FullCubic,
}

impl MixtureModel {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            MixtureModel::Linear => "Linear",
            MixtureModel::Quadratic => "Quadratic",
            MixtureModel::SpecialCubic => "Special Cubic",
            MixtureModel::FullCubic => "Full Cubic",
        }
    }
    fn rank(self) -> usize {
        self as usize
    }
}

type TermSet = (Vec<String>, Vec<Vec<(usize, i32)>>, Vec<usize>);

fn model_terms(q: usize, model: MixtureModel, names: &[String]) -> TermSet {
    // each term: list of (component, sign) – sign −1 marks the (xi − xj) factor's j part
    let mut name = Vec::new();
    let mut terms: Vec<Vec<(usize, i32)>> = Vec::new();
    let mut group = Vec::new(); // 0 linear, 1 quadratic, 2 cubic
    for i in 0..q {
        name.push(names[i].clone());
        terms.push(vec![(i, 1)]);
        group.push(0);
    }
    if model.rank() >= 1 {
        for i in 0..q {
            for j in i + 1..q {
                name.push(format!("{}*{}", names[i], names[j]));
                terms.push(vec![(i, 1), (j, 1)]);
                group.push(1);
            }
        }
    }
    if model == MixtureModel::SpecialCubic {
        for i in 0..q {
            for j in i + 1..q {
                for k in j + 1..q {
                    name.push(format!("{}*{}*{}", names[i], names[j], names[k]));
                    terms.push(vec![(i, 1), (j, 1), (k, 1)]);
                    group.push(2);
                }
            }
        }
    }
    if model == MixtureModel::FullCubic {
        for i in 0..q {
            for j in i + 1..q {
                name.push(format!("{}*{}*({}-{})", names[i], names[j], names[i], names[j]));
                terms.push(vec![(i, 1), (j, 1), (i, 2)]);
                group.push(2);
            }
        }
    }
    (name, terms, group)
}

fn term_value(t: &[(usize, i32)], x: &[f64]) -> f64 {
    // a `(c, 2)` entry marks the difference factor (x_i − x_j) built from the first two entries
    if t.len() == 3 && t[2].1 == 2 {
        return x[t[0].0] * x[t[1].0] * (x[t[0].0] - x[t[1].0]);
    }
    t.iter().map(|&(c, _)| x[c]).product()
}

/// One row of the mixture ANOVA table.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureAnovaRow {
    /// Source.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sequential sum of squares.
    pub ss: f64,
    /// Mean square.
    pub ms: f64,
    /// F (`NaN` for the error row).
    pub f: f64,
    /// P-value (`NaN` for the error row).
    pub p: f64,
}

/// A fitted Scheffé model (no intercept).
#[derive(Debug, Clone)]
pub struct MixtureFit {
    /// Model.
    pub model: MixtureModel,
    /// Component names.
    pub components: Vec<String>,
    /// Term names.
    pub terms: Vec<String>,
    /// Coefficients.
    pub coef: Vec<f64>,
    /// Standard errors.
    pub se: Vec<f64>,
    /// t statistics.
    pub t: Vec<f64>,
    /// Two-sided p-values.
    pub p: Vec<f64>,
    /// Sequential ANOVA: Regression, Linear, Quadratic, Cubic (as present), Error, Total.
    pub anova: Vec<MixtureAnovaRow>,
    /// Residual standard deviation.
    pub s: f64,
    /// R² about the mean.
    pub r2: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// Residual degrees of freedom.
    pub df_error: usize,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// Residuals.
    pub residuals: Vec<f64>,
    term_defs: Vec<Vec<(usize, i32)>>,
}

impl MixtureFit {
    /// Predicted response at a blend.
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.term_defs.iter().zip(&self.coef).map(|(t, b)| b * term_value(t, x)).sum()
    }
}

fn design_matrix(x: &[Vec<f64>], terms: &[Vec<(usize, i32)>]) -> Matrix {
    let mut m = Matrix::zeros(x.len(), terms.len());
    for (i, row) in x.iter().enumerate() {
        for (j, t) in terms.iter().enumerate() {
            m[(i, j)] = term_value(t, row);
        }
    }
    m
}

/// Fit a Scheffé model. `x[row][component]` are proportions that must sum to 1.
pub fn fit_mixture(x: &[Vec<f64>], y: &[f64], components: &[String], model: MixtureModel) -> Result<MixtureFit> {
    let q = components.len();
    if q < 2 || x.len() != y.len() || x.iter().any(|r| r.len() != q) {
        return invalid("the proportions and response must line up (at least two components)");
    }
    for (i, r) in x.iter().enumerate() {
        if r.iter().any(|v| !v.is_finite() || *v < -1e-9) || (r.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return invalid(format!("row {}: the proportions must be non-negative and sum to 1", i + 1));
        }
    }
    let (names, terms, group) = model_terms(q, model, components);
    let n = x.len();
    let p = terms.len();
    if n <= p {
        return invalid(format!("the {} model has {p} terms but only {n} runs", model.name().to_lowercase()));
    }
    let fit = ols(&design_matrix(x, &terms), y, R2Kind::Centered).map_err(|_| {
        crate::error::StatsError::InvalidArgument("the design is too sparse for this model (singular)".into())
    })?;
    let ybar = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();
    let sse = fit.sse;
    let df_e = n - p;
    let tdist = StudentT::new(df_e as f64)?;
    let t: Vec<f64> = fit.coef.iter().zip(&fit.std_err).map(|(b, s)| b / s).collect();
    let pv: Vec<f64> = t.iter().map(|&v| 2.0 * tdist.sf(v.abs())).collect();
    // sequential sums of squares from nested fits
    let ngroups = group.iter().copied().max().unwrap_or(0) + 1;
    let mut sse_prev = sst;
    let mut rows = vec![];
    let mse = sse / df_e as f64;
    let fdist = |d1: usize, d2: usize| FisherF::new(d1 as f64, d2 as f64);
    let gname = ["Linear", "Quadratic", "Cubic"];
    for g in 0..ngroups {
        let cols: Vec<usize> = (0..p).filter(|&j| group[j] <= g).collect();
        let sub: Vec<Vec<(usize, i32)>> = cols.iter().map(|&j| terms[j].clone()).collect();
        let sse_g = if g + 1 == ngroups {
            sse
        } else {
            ols(&design_matrix(x, &sub), y, R2Kind::Centered).map(|f| f.sse).unwrap_or(f64::NAN)
        };
        let df_g = group.iter().filter(|&&gg| gg == g).count() - usize::from(g == 0);
        let ss = sse_prev - sse_g;
        // linear group df = (#linear terms − 1) because the regression is about the mean
        let ms = ss / df_g.max(1) as f64;
        let f = ms / mse;
        let pval = if df_g > 0 { fdist(df_g, df_e)?.sf(f) } else { f64::NAN };
        let label = if g + 1 == ngroups && g == 2 && model == MixtureModel::SpecialCubic {
            "Special Cubic".to_string()
        } else if g == 2 && model == MixtureModel::FullCubic {
            "Full Cubic".to_string()
        } else {
            gname[g].to_string()
        };
        rows.push(MixtureAnovaRow { source: label, df: df_g, ss, ms, f, p: pval });
        sse_prev = sse_g;
    }
    let ss_reg = sst - sse;
    let df_reg = p - 1;
    let f_reg = ss_reg / df_reg as f64 / mse;
    let mut anova = vec![MixtureAnovaRow {
        source: "Regression".into(),
        df: df_reg,
        ss: ss_reg,
        ms: ss_reg / df_reg as f64,
        f: f_reg,
        p: fdist(df_reg, df_e)?.sf(f_reg),
    }];
    anova.extend(rows);
    anova.push(MixtureAnovaRow {
        source: "Residual Error".into(),
        df: df_e,
        ss: sse,
        ms: mse,
        f: f64::NAN,
        p: f64::NAN,
    });
    anova.push(MixtureAnovaRow { source: "Total".into(), df: n - 1, ss: sst, ms: f64::NAN, f: f64::NAN, p: f64::NAN });
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    let r2_adj = if sst > 0.0 { 1.0 - (sse / df_e as f64) / (sst / (n - 1) as f64) } else { f64::NAN };
    Ok(MixtureFit {
        model,
        components: components.to_vec(),
        terms: names,
        coef: fit.coef,
        se: fit.std_err,
        t,
        p: pv,
        anova,
        s: mse.sqrt(),
        r2,
        r2_adj,
        df_error: df_e,
        fitted: fit.fitted,
        residuals: fit.residuals,
        term_defs: terms,
    })
}

// ---------------------------------------------------------------------------------------------
// Optimal designs
// ---------------------------------------------------------------------------------------------

/// Optimality criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Maximise `|XᵀX|`.
    D,
    /// Minimise the average prediction variance over the candidate set.
    I,
}

/// Model whose terms the design must estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptModel {
    /// Intercept and main effects.
    Linear,
    /// Plus two-factor interactions.
    Interactions,
    /// Plus squares (needs at least three levels).
    Quadratic,
}

/// Options for [`optimal_design`].
#[derive(Debug, Clone)]
pub struct OptimalOptions {
    /// Runs in the design.
    pub runs: usize,
    /// Criterion.
    pub criterion: Criterion,
    /// Model.
    pub model: OptModel,
    /// Candidate levels per factor (2–5, equally spaced on [−1, 1]); one entry per factor.
    pub levels: Vec<usize>,
    /// Random restarts of the exchange.
    pub restarts: usize,
    /// Seed.
    pub seed: u64,
}

/// The chosen design and its quality measures.
#[derive(Debug, Clone)]
pub struct OptimalDesign {
    /// Coded settings, one row per run (sorted).
    pub points: Vec<Vec<f64>>,
    /// Criterion.
    pub criterion: Criterion,
    /// Model.
    pub model: OptModel,
    /// Number of model terms.
    pub n_terms: usize,
    /// `ln |XᵀX|`.
    pub log_det: f64,
    /// D-efficiency in percent: `100 |XᵀX|^(1/p) / n`.
    pub d_efficiency: f64,
    /// Average of `f(x)ᵀ (XᵀX)⁻¹ f(x)` over the candidates (variance in units of σ²).
    pub avg_prediction_variance: f64,
    /// Maximum of that quantity over the candidates.
    pub max_prediction_variance: f64,
    /// Number of candidate points searched.
    pub candidates: usize,
}

fn model_row(x: &[f64], model: OptModel) -> Vec<f64> {
    let k = x.len();
    let mut r = vec![1.0];
    r.extend_from_slice(x);
    if model != OptModel::Linear {
        for i in 0..k {
            for j in i + 1..k {
                r.push(x[i] * x[j]);
            }
        }
    }
    if model == OptModel::Quadratic {
        for v in x {
            r.push(v * v);
        }
    }
    r
}

fn quad(a: &[Vec<f64>], u: &[f64], v: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, ai) in a.iter().enumerate() {
        let mut t = 0.0;
        for (j, aij) in ai.iter().enumerate() {
            t += aij * v[j];
        }
        s += u[i] * t;
    }
    s
}

fn matvec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter().map(|r| r.iter().zip(v).map(|(x, y)| x * y).sum()).collect()
}

fn inverse(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let p = m.len();
    let mut a = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            a[(i, j)] = m[i][j];
        }
    }
    let inv = Cholesky::new(&a).ok()?.inverse().ok()?;
    Some((0..p).map(|i| (0..p).map(|j| inv[(i, j)]).collect()).collect())
}

fn info(rows: &[&Vec<f64>], p: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; p]; p];
    for r in rows {
        for i in 0..p {
            for j in 0..p {
                m[i][j] += r[i] * r[j];
            }
        }
    }
    m
}

fn log_det(m: &[Vec<f64>]) -> Option<f64> {
    let p = m.len();
    let mut a = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            a[(i, j)] = m[i][j];
        }
    }
    Cholesky::new(&a).ok().map(|c| c.log_det())
}

/// Search for an optimal design by the exchange algorithm over a full-factorial candidate set.
pub fn optimal_design(opts: &OptimalOptions) -> Result<OptimalDesign> {
    let k = opts.levels.len();
    if !(1..=8).contains(&k) {
        return invalid("an optimal design handles 1 to 8 factors");
    }
    if opts.levels.iter().any(|&l| !(2..=5).contains(&l)) {
        return invalid("each factor needs 2 to 5 candidate levels");
    }
    if opts.model == OptModel::Quadratic && opts.levels.iter().any(|&l| l < 3) {
        return invalid("a quadratic model needs at least three levels per factor");
    }
    let ncand: usize = opts.levels.iter().product();
    if ncand > 30_000 {
        return invalid("more than 30,000 candidate points; use fewer levels or factors");
    }
    let cands: Vec<Vec<f64>> = (0..ncand)
        .map(|mut idx| {
            opts.levels
                .iter()
                .map(|&l| {
                    let v = idx % l;
                    idx /= l;
                    -1.0 + 2.0 * v as f64 / (l - 1) as f64
                })
                .collect()
        })
        .collect();
    let f: Vec<Vec<f64>> = cands.iter().map(|c| model_row(c, opts.model)).collect();
    let p = f[0].len();
    let n = opts.runs;
    if n < p {
        return invalid(format!("the model has {p} terms; at least {p} runs are needed"));
    }
    if n > 2000 {
        return invalid("at most 2000 runs");
    }
    // moment matrix of the candidate set (for the I criterion)
    let mut bmat = vec![vec![0.0; p]; p];
    for r in &f {
        for i in 0..p {
            for j in 0..p {
                bmat[i][j] += r[i] * r[j] / ncand as f64;
            }
        }
    }
    let score = |a: &[Vec<f64>]| -> f64 {
        match opts.criterion {
            Criterion::D => log_det_inv(a),
            Criterion::I => (0..p).map(|i| (0..p).map(|j| a[i][j] * bmat[j][i]).sum::<f64>()).sum(),
        }
    };
    let mut rng = Pcg64::new(u128::from(opts.seed), 54);
    let mut best: Option<(Vec<usize>, f64)> = None; // lower score is better
    for _ in 0..opts.restarts.max(1) {
        // random start with a non-singular information matrix
        let mut idx: Vec<usize> = Vec::new();
        let mut a_inv = None;
        for _ in 0..200 {
            idx = (0..n).map(|_| rng.below(ncand as u64) as usize).collect();
            let rows: Vec<&Vec<f64>> = idx.iter().map(|&i| &f[i]).collect();
            if let Some(a) = inverse(&info(&rows, p)) {
                a_inv = Some(a);
                break;
            }
        }
        let Some(mut a) = a_inv else { continue };
        for _pass in 0..60 {
            let mut improved = false;
            for row in 0..n {
                let xo = &f[idx[row]];
                let ax_o = matvec(&a, xo);
                let h_o: f64 = xo.iter().zip(&ax_o).map(|(x, y)| x * y).sum();
                if 1.0 - h_o < 1e-9 {
                    continue; // removing this run would make the design singular
                }
                // A1 = inverse after removing the run
                let mut a1 = a.clone();
                let c = 1.0 / (1.0 - h_o);
                for i in 0..p {
                    for j in 0..p {
                        a1[i][j] += c * ax_o[i] * ax_o[j];
                    }
                }
                let cur = score(&a);
                let mut best_c: Option<(usize, f64)> = None;
                let a1b: Vec<Vec<f64>> = if opts.criterion == Criterion::I {
                    (0..p).map(|i| (0..p).map(|j| (0..p).map(|l| a1[i][l] * bmat[l][j]).sum()).collect()).collect()
                } else {
                    Vec::new()
                };
                let tr_a1b: f64 = if opts.criterion == Criterion::I { (0..p).map(|i| a1b[i][i]).sum() } else { 0.0 };
                for (ci, xn) in f.iter().enumerate() {
                    let a1x = matvec(&a1, xn);
                    let d: f64 = xn.iter().zip(&a1x).map(|(x, y)| x * y).sum();
                    let s = match opts.criterion {
                        Criterion::D => {
                            // det ratio of (M − xo xoᵀ + xn xnᵀ) to M:  (1 + xnᵀA1xn)(1 − h_o)
                            -(((1.0 + d) * (1.0 - h_o)).max(1e-300)).ln()
                        }
                        Criterion::I => {
                            // x'ᵀ A1 B A1 x' / (1 + x'ᵀ A1 x')
                            let ba1x = matvec(&bmat, &a1x);
                            let num: f64 = a1x.iter().zip(&ba1x).map(|(x, y)| x * y).sum();
                            tr_a1b - num / (1.0 + d)
                        }
                    };
                    if best_c.is_none_or(|b| s < b.1) {
                        best_c = Some((ci, s));
                    }
                }
                if let Some((ci, _)) = best_c {
                    // compare through the actual criterion value of the new design
                    if ci != idx[row] {
                        let old = idx[row];
                        idx[row] = ci;
                        let rows: Vec<&Vec<f64>> = idx.iter().map(|&i| &f[i]).collect();
                        match inverse(&info(&rows, p)) {
                            Some(anew) if score(&anew) < cur - 1e-10 * cur.abs().max(1.0) => {
                                a = anew;
                                improved = true;
                            }
                            _ => idx[row] = old,
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        let sc = score(&a);
        if best.as_ref().is_none_or(|b| sc < b.1) {
            best = Some((idx, sc));
        }
    }
    let (idx, _) = best.ok_or_else(|| {
        crate::error::StatsError::InvalidArgument(
            "no non-singular starting design was found; use more runs or levels".into(),
        )
    })?;
    let rows: Vec<&Vec<f64>> = idx.iter().map(|&i| &f[i]).collect();
    let m = info(&rows, p);
    let ld = log_det(&m).ok_or_else(|| crate::error::StatsError::InvalidArgument("the design is singular".into()))?;
    let a = inverse(&m).ok_or_else(|| crate::error::StatsError::InvalidArgument("the design is singular".into()))?;
    let (mut sum, mut mx) = (0.0, 0.0_f64);
    for r in &f {
        let v = quad(&a, r, r);
        sum += v;
        mx = mx.max(v);
    }
    let mut points: Vec<Vec<f64>> = idx.iter().map(|&i| cands[i].clone()).collect();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(OptimalDesign {
        points,
        criterion: opts.criterion,
        model: opts.model,
        n_terms: p,
        log_det: ld,
        d_efficiency: 100.0 * (ld / p as f64).exp() / n as f64,
        avg_prediction_variance: sum / ncand as f64,
        max_prediction_variance: mx,
        candidates: ncand,
    })
}

fn log_det_inv(a: &[Vec<f64>]) -> f64 {
    // −ln|M| = ln|A|
    log_det(a).unwrap_or(f64::INFINITY)
}

// ---------------------------------------------------------------------------------------------
// Split-plot designs
// ---------------------------------------------------------------------------------------------

/// One run of a split-plot design.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlotRun {
    /// Whole-plot number (1-based).
    pub whole_plot: usize,
    /// Coded whole-plot factor levels followed by the coded subplot factor levels.
    pub levels: Vec<f64>,
}

/// A two-level full-factorial split-plot design.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlot {
    /// Number of whole-plot factors (they come first in each run).
    pub whole_factors: usize,
    /// Number of subplot factors.
    pub sub_factors: usize,
    /// Runs in run order.
    pub runs: Vec<SplitPlotRun>,
}

/// Build a split-plot design: every combination of the whole-plot factors (repeated
/// `whole_replicates` times) forms a whole plot, and every combination of the subplot factors
/// is run inside each whole plot. Whole plots and the runs inside them are randomised when a
/// seed is given.
pub fn split_plot(
    whole_factors: usize,
    sub_factors: usize,
    whole_replicates: usize,
    randomize: Option<u64>,
) -> Result<SplitPlot> {
    if !(1..=6).contains(&whole_factors) || !(1..=6).contains(&sub_factors) {
        return invalid("a split-plot design needs 1 to 6 whole-plot and 1 to 6 subplot factors");
    }
    if !(1..=20).contains(&whole_replicates) {
        return invalid("whole-plot replicates must be between 1 and 20");
    }
    let nw = (1usize << whole_factors) * whole_replicates;
    let ns = 1usize << sub_factors;
    if nw * ns > 5000 {
        return invalid("more than 5000 runs");
    }
    let level = |bits: usize, i: usize| if bits >> i & 1 == 1 { 1.0 } else { -1.0 };
    let mut wp: Vec<usize> = (0..nw).map(|w| w % (1usize << whole_factors)).collect();
    let mut rng = randomize.map(|s| Pcg64::new(u128::from(s), 54));
    if let Some(r) = rng.as_mut() {
        for i in (1..wp.len()).rev() {
            let j = r.below(i as u64 + 1) as usize;
            wp.swap(i, j);
        }
    }
    let mut runs = Vec::with_capacity(nw * ns);
    for (w, &bits) in wp.iter().enumerate() {
        let mut sp: Vec<usize> = (0..ns).collect();
        if let Some(r) = rng.as_mut() {
            for i in (1..sp.len()).rev() {
                let j = r.below(i as u64 + 1) as usize;
                sp.swap(i, j);
            }
        }
        for s in sp {
            let mut levels: Vec<f64> = (0..whole_factors).map(|i| level(bits, i)).collect();
            levels.extend((0..sub_factors).map(|i| level(s, i)));
            runs.push(SplitPlotRun { whole_plot: w + 1, levels });
        }
    }
    Ok(SplitPlot { whole_factors, sub_factors, runs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(q: usize) -> Vec<String> {
        (0..q).map(|i| format!("X{}", i + 1)).collect()
    }

    #[test]
    fn lattice_counts() {
        // {3,2}: 6 points; {4,3}: 20 points
        assert_eq!(simplex_lattice(&names(3), 2, false, false, 1, None).unwrap().runs(), 6);
        assert_eq!(simplex_lattice(&names(4), 3, false, false, 1, None).unwrap().runs(), 20);
        let d = simplex_lattice(&names(3), 2, true, true, 2, Some(1)).unwrap();
        assert_eq!(d.runs(), 2 * (6 + 1 + 3));
        assert!(d.points.iter().all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-12));
    }

    #[test]
    fn centroid_counts() {
        assert_eq!(simplex_centroid(&names(3), false, 1, None).unwrap().runs(), 7);
        assert_eq!(simplex_centroid(&names(4), false, 1, None).unwrap().runs(), 15);
    }

    #[test]
    fn extreme_vertices_of_a_known_region() {
        // x1 in [0.1, 0.5], x2 in [0.2, 0.6], x3 in [0, 0.4]: 6 vertices
        let d = extreme_vertices(&names(3), &[0.1, 0.2, 0.0], &[0.5, 0.6, 0.4], false, 1, None).unwrap();
        assert_eq!(d.runs(), 6);
        assert!(d.points.iter().all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-12));
        assert!(extreme_vertices(&names(3), &[0.5, 0.5, 0.5], &[1.0, 1.0, 1.0], false, 1, None).is_err());
    }

    #[test]
    fn scheffe_recovers_exact_model() {
        let d = simplex_lattice(&names(3), 2, true, false, 2, None).unwrap();
        let truth = |x: &[f64]| 10.0 * x[0] + 20.0 * x[1] + 30.0 * x[2] + 8.0 * x[0] * x[1] - 6.0 * x[1] * x[2];
        let y: Vec<f64> = d.points.iter().enumerate().map(|(i, p)| truth(p) + 1e-3 * (i % 2) as f64).collect();
        let f = fit_mixture(&d.points, &y, &d.components, MixtureModel::Quadratic).unwrap();
        assert!((f.coef[0] - 10.0).abs() < 0.05 && (f.coef[3] - 8.0).abs() < 0.2);
        assert!(f.r2 > 0.9999);
        assert!((f.predict(&[0.2, 0.3, 0.5]) - truth(&[0.2, 0.3, 0.5])).abs() < 0.05);
        assert!(f.anova.iter().any(|r| r.source == "Quadratic"));
    }

    #[test]
    fn d_optimal_two_level_is_a_full_factorial() {
        let o = OptimalOptions {
            runs: 8,
            criterion: Criterion::D,
            model: OptModel::Linear,
            levels: vec![2, 2, 2],
            restarts: 5,
            seed: 3,
        };
        let d = optimal_design(&o).unwrap();
        assert!((d.d_efficiency - 100.0).abs() < 1e-6, "{}", d.d_efficiency);
        // an orthogonal design has X'X = 8 I
        assert!((d.avg_prediction_variance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn i_optimal_beats_d_on_prediction_variance() {
        let mk = |c| OptimalOptions {
            runs: 12,
            criterion: c,
            model: OptModel::Quadratic,
            levels: vec![3, 3],
            restarts: 8,
            seed: 7,
        };
        let d = optimal_design(&mk(Criterion::D)).unwrap();
        let i = optimal_design(&mk(Criterion::I)).unwrap();
        assert!(i.avg_prediction_variance <= d.avg_prediction_variance + 1e-9);
        assert!(d.log_det >= i.log_det - 1e-9);
    }

    #[test]
    fn split_plot_structure() {
        let s = split_plot(2, 2, 2, Some(5)).unwrap();
        assert_eq!(s.runs.len(), 8 * 4);
        for w in 1..=8 {
            let r: Vec<_> = s.runs.iter().filter(|r| r.whole_plot == w).collect();
            assert_eq!(r.len(), 4);
            assert!(r.iter().all(|x| x.levels[..2] == r[0].levels[..2]));
        }
        assert!(split_plot(0, 2, 1, None).is_err());
    }
}
