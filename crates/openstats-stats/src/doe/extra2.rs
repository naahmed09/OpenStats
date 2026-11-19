//! Phase 5o-b: definitive screening designs, path of steepest ascent, analysis of variability
//! (dispersion effects) and overlaid contour regions.

use super::design::FactorSpec;
use super::model::PolynomialModel;
use super::plots::contour_segments;
use crate::error::{invalid, Result};
use crate::report::{ChartKind, ChartSpec, Series};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::random::{Pcg64, Rng};
use openstats_math::stats::{ols, R2Kind};
use openstats_math::Matrix;

// ---------------------------------------------------------------------------------------------
// Conference matrices (Paley construction) and definitive screening designs
// ---------------------------------------------------------------------------------------------

/// A finite field GF(q) for q = p or p² with `p` an odd prime: element `a + b·ω` is index `a + p·b`.
struct Field {
    p: usize,
    ext: bool,
    nonres: usize,
}

impl Field {
    fn new(q: usize) -> Option<Field> {
        let is_prime = |n: usize| n >= 3 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
        if is_prime(q) {
            return Some(Field { p: q, ext: false, nonres: 0 });
        }
        let r = (3..).take_while(|d| d * d <= q).find(|d| d * d == q)?;
        if !is_prime(r) {
            return None;
        }
        let nonres = (2..r).find(|&a| (1..r).all(|b| b * b % r != a))?;
        Some(Field { p: r, ext: true, nonres })
    }
    fn size(&self) -> usize {
        if self.ext {
            self.p * self.p
        } else {
            self.p
        }
    }
    fn sub(&self, x: usize, y: usize) -> usize {
        let p = self.p;
        if !self.ext {
            return (x + p - y) % p;
        }
        let (a, b) = (x % p, x / p);
        let (c, d) = (y % p, y / p);
        (a + p - c) % p + p * ((b + p - d) % p)
    }
    fn mul(&self, x: usize, y: usize) -> usize {
        let p = self.p;
        if !self.ext {
            return x * y % p;
        }
        let (a, b) = (x % p, x / p);
        let (c, d) = (y % p, y / p);
        (a * c + b * d * self.nonres) % p + p * ((a * d + b * c) % p)
    }
}

/// Symmetric or skew conference matrix of order `q + 1` for an odd prime power `q` (3, 5, 7, 9, 11, 13,
/// 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49): `C Cᵀ = q I`, zero diagonal, ±1 elsewhere.
pub fn conference_matrix(order: usize) -> Option<Vec<Vec<i8>>> {
    let q = order.checked_sub(1)?;
    let f = Field::new(q)?;
    let sz = f.size();
    let mut square = vec![false; sz];
    for a in 1..sz {
        square[f.mul(a, a)] = true;
    }
    let chi = |a: usize| -> i8 {
        if a == 0 {
            0
        } else if square[a] {
            1
        } else {
            -1
        }
    };
    let skew = q % 4 == 3;
    let n = q + 1;
    let mut c = vec![vec![0i8; n]; n];
    for j in 1..n {
        c[0][j] = 1;
        c[j][0] = if skew { -1 } else { 1 };
    }
    for i in 0..q {
        for j in 0..q {
            c[i + 1][j + 1] = chi(f.sub(i, j));
        }
    }
    Some(c)
}

/// Orders for which [`conference_matrix`] exists here.
pub fn conference_orders() -> Vec<usize> {
    (4..=50).filter(|&n| n % 2 == 0 && conference_matrix(n).is_some()).collect()
}

/// A definitive screening design in coded units (−1, 0, +1).
#[derive(Debug, Clone, PartialEq)]
pub struct DsdDesign {
    /// Number of factors.
    pub factors: usize,
    /// Runs (rows) × factors.
    pub runs: Vec<Vec<f64>>,
    /// Order of the conference matrix used.
    pub conference_order: usize,
    /// Columns of the conference matrix dropped (when its order exceeded the factor count).
    pub dropped_columns: usize,
    /// Number of extra centre runs added.
    pub extra_center: usize,
}

impl DsdDesign {
    /// Largest absolute correlation between two main-effect columns.
    pub fn max_main_correlation(&self) -> f64 {
        let n = self.runs.len() as f64;
        let k = self.factors;
        let mean: Vec<f64> = (0..k).map(|j| self.runs.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let mut worst = 0.0f64;
        for a in 0..k {
            for b in a + 1..k {
                let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
                for r in &self.runs {
                    let (x, y) = (r[a] - mean[a], r[b] - mean[b]);
                    sab += x * y;
                    saa += x * x;
                    sbb += y * y;
                }
                if saa > 0.0 && sbb > 0.0 {
                    worst = worst.max((sab / (saa * sbb).sqrt()).abs());
                }
            }
        }
        worst
    }
}

/// Definitive screening design for `k` (2–30) three-level factors: the rows of a conference matrix `C`,
/// the rows of `−C` and a centre run (Jones & Nachtsheim, 2011), i.e. `2m + 1` runs for the smallest
/// available conference order `m ≥ k` (columns beyond `k` are dropped). `extra_center` adds centre runs.
/// With `randomize` the run order is shuffled; otherwise each `C` row is followed by its mirror row.
pub fn definitive_screening(k: usize, extra_center: usize, randomize: Option<u64>) -> Result<DsdDesign> {
    if !(3..=30).contains(&k) {
        return invalid("a definitive screening design handles 3 to 30 factors");
    }
    if extra_center > 20 {
        return invalid("at most 20 extra centre runs");
    }
    let order = (k..=50)
        .filter(|n| n % 2 == 0)
        .find(|&n| conference_matrix(n).is_some())
        .ok_or_else(|| crate::error::StatsError::InvalidArgument("no conference matrix is available".into()))?;
    let c = conference_matrix(order)
        .ok_or_else(|| crate::error::StatsError::InvalidArgument("no conference matrix".into()))?;
    let mut runs: Vec<Vec<f64>> = Vec::new();
    for row in &c {
        let r: Vec<f64> = row[..k].iter().map(|&v| f64::from(v)).collect();
        runs.push(r.clone());
        runs.push(r.iter().map(|v| -v).collect());
    }
    runs.push(vec![0.0; k]);
    for _ in 0..extra_center {
        runs.push(vec![0.0; k]);
    }
    if let Some(seed) = randomize {
        let mut rng = Pcg64::new(u128::from(seed), 54);
        for i in (1..runs.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            runs.swap(i, j);
        }
    }
    Ok(DsdDesign { factors: k, runs, conference_order: order, dropped_columns: order - k, extra_center })
}

// ---------------------------------------------------------------------------------------------
// Path of steepest ascent / descent
// ---------------------------------------------------------------------------------------------

/// One point on the path of steepest ascent.
#[derive(Debug, Clone, PartialEq)]
pub struct PathPoint {
    /// Step number (0 = centre).
    pub step: usize,
    /// Coded factor settings.
    pub coded: Vec<f64>,
    /// Natural factor settings.
    pub natural: Vec<f64>,
    /// Response predicted by the fitted model at this point.
    pub predicted: f64,
}

/// The path of steepest ascent (`ascent = true`) or descent from the design centre: the direction is the
/// vector of first-order coefficients; the factor with the largest `|coefficient|` advances `step`
/// coded units per step and the others advance in proportion. The predicted response uses the whole
/// model (so curvature, if fitted, is included).
pub fn steepest_path(
    model: &PolynomialModel,
    factors: &[FactorSpec],
    ascent: bool,
    step: f64,
    steps: usize,
) -> Result<Vec<PathPoint>> {
    let k = factors.len();
    if model.n_factors() != k {
        return invalid("the model and the factor list disagree");
    }
    if !(step.is_finite() && step > 0.0) || !(1..=100).contains(&steps) {
        return invalid("the step must be positive and the number of steps between 1 and 100");
    }
    let mut grad = vec![0.0; k];
    for (t, &b) in model.terms.iter().zip(&model.coef) {
        if let [(j, 1)] = t.powers[..] {
            grad[j] += b;
        }
    }
    let gmax = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    if gmax <= 0.0 {
        return invalid("all first-order coefficients are zero: there is no direction of steepest ascent");
    }
    let sign = if ascent { 1.0 } else { -1.0 };
    let mut out = Vec::new();
    for s in 0..=steps {
        let coded: Vec<f64> = grad.iter().map(|g| sign * g / gmax * step * s as f64).collect();
        let natural = coded.iter().zip(factors).map(|(&c, f)| f.to_natural(c)).collect();
        out.push(PathPoint { step: s, predicted: model.predict(&coded), coded, natural });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Analysis of variability
// ---------------------------------------------------------------------------------------------

/// A term of the variability model.
#[derive(Debug, Clone, PartialEq)]
pub struct VariabilityTerm {
    /// Name (`A`, `A*B`).
    pub name: String,
    /// Coefficient on ln(s) per coded unit.
    pub coef: f64,
    /// Effect (high − low) on ln(s).
    pub effect: f64,
    /// Ratio of standard deviations high / low = exp(effect).
    pub ratio: f64,
    /// Standard error of the coefficient (NaN without error df).
    pub se: f64,
    /// t statistic.
    pub t: f64,
    /// Two-sided p-value.
    pub p: f64,
}

/// Result of [`analyze_variability`].
#[derive(Debug, Clone, PartialEq)]
pub struct VariabilityAnalysis {
    /// Distinct factor combinations with ≥ 2 replicates: coded levels, standard deviation and df.
    pub groups: Vec<(Vec<f64>, f64, usize)>,
    /// Constant (average ln s) first, then the terms.
    pub terms: Vec<VariabilityTerm>,
    /// Intercept.
    pub constant: f64,
    /// Error degrees of freedom of the ln(s) regression.
    pub df_error: usize,
    /// R² of the ln(s) regression.
    pub r2: f64,
    /// Fitted standard deviation of each group.
    pub fitted_sd: Vec<f64>,
}

/// Dispersion analysis of a replicated two-level design: the standard deviation `s` of the response is
/// computed at every distinct factor combination that has at least two runs and `ln s` is regressed on the
/// coded factors (main effects, plus two-factor interactions when `interactions`). A factor with a large
/// effect on `ln s` changes the spread of the response (ratio = exp(effect)).
pub fn analyze_variability(
    coded: &[Vec<f64>],
    y: &[f64],
    names: &[String],
    interactions: bool,
) -> Result<VariabilityAnalysis> {
    let k = names.len();
    if coded.len() != y.len() || coded.is_empty() || coded.iter().any(|r| r.len() != k) {
        return invalid("the factor columns and the response must have the same length");
    }
    if y.iter().any(|v| !v.is_finite()) {
        return invalid("the response has missing values");
    }
    let key = |r: &[f64]| -> Vec<i64> { r.iter().map(|v| (v * 1e6).round() as i64).collect() };
    let mut order: Vec<Vec<i64>> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (i, r) in coded.iter().enumerate() {
        let kk = key(r);
        match order.iter().position(|o| *o == kk) {
            Some(g) => members[g].push(i),
            None => {
                order.push(kk);
                members.push(vec![i]);
            }
        }
    }
    let mut groups = Vec::new();
    let mut lns = Vec::new();
    for m in &members {
        if m.len() < 2 {
            continue;
        }
        let mean = m.iter().map(|&i| y[i]).sum::<f64>() / m.len() as f64;
        let ss: f64 = m.iter().map(|&i| (y[i] - mean).powi(2)).sum();
        let s = (ss / (m.len() - 1) as f64).sqrt();
        if s <= 0.0 {
            return invalid("a combination of factor levels has identical replicates (s = 0): ln(s) is undefined");
        }
        groups.push((coded[m[0]].clone(), s, m.len() - 1));
        lns.push(s.ln());
    }
    if groups.len() < 3 {
        return invalid("variability analysis needs at least 3 factor combinations with replicate runs");
    }
    let mut cols: Vec<(String, Vec<usize>)> = (0..k).map(|j| (names[j].clone(), vec![j])).collect();
    if interactions {
        for a in 0..k {
            for b in a + 1..k {
                cols.push((format!("{}*{}", names[a], names[b]), vec![a, b]));
            }
        }
    }
    let p = cols.len() + 1;
    if groups.len() <= p {
        return invalid(format!(
            "{} factor combinations are not enough for {} model terms; drop the interactions or add replicated combinations",
            groups.len(),
            p
        ));
    }
    let mut x = Matrix::zeros(groups.len(), p);
    for (i, g) in groups.iter().enumerate() {
        x[(i, 0)] = 1.0;
        for (j, (_, idx)) in cols.iter().enumerate() {
            x[(i, j + 1)] = idx.iter().map(|&f| g.0[f]).product();
        }
    }
    let fit = ols(&x, &lns, R2Kind::Centered).map_err(|_| {
        crate::error::StatsError::InvalidArgument("the replicated combinations are too sparse for this model".into())
    })?;
    let df = groups.len() - p;
    let tdist = StudentT::new(df as f64)?;
    let terms = cols
        .iter()
        .enumerate()
        .map(|(j, (name, _))| {
            let coef = fit.coef[j + 1];
            let se = fit.std_err[j + 1];
            let t = coef / se;
            VariabilityTerm {
                name: name.clone(),
                coef,
                effect: 2.0 * coef,
                ratio: (2.0 * coef).exp(),
                se,
                t,
                p: 2.0 * tdist.sf(t.abs()),
            }
        })
        .collect();
    Ok(VariabilityAnalysis {
        groups,
        terms,
        constant: fit.coef[0],
        df_error: df,
        r2: fit.r_squared,
        fitted_sd: fit.fitted.iter().map(|v| v.exp()).collect(),
    })
}

// ---------------------------------------------------------------------------------------------
// Overlaid contour plot
// ---------------------------------------------------------------------------------------------

/// A response model with the acceptable interval for the overlaid contour plot.
#[derive(Debug, Clone)]
pub struct OverlayResponse<'a> {
    /// Response name.
    pub name: String,
    /// Fitted model.
    pub model: &'a PolynomialModel,
    /// Lower acceptable value (−∞ for none).
    pub lower: f64,
    /// Upper acceptable value (+∞ for none).
    pub upper: f64,
}

/// Result of [`overlaid_contour`].
#[derive(Debug, Clone)]
pub struct Overlay {
    /// Contour chart with each response's boundaries.
    pub chart: ChartSpec,
    /// Chart of the grid points where every response is acceptable.
    pub feasible_chart: ChartSpec,
    /// Fraction of the plotted region where all responses are acceptable.
    pub feasible_fraction: f64,
    /// A feasible grid point closest to the middle of the feasible set (natural units), if any.
    pub best_point: Option<(f64, f64)>,
}

/// Overlaid contour plot of several responses over factors `ix` × `iy` (others held at coded `fixed`).
pub fn overlaid_contour(
    responses: &[OverlayResponse<'_>],
    factors: &[FactorSpec],
    ix: usize,
    iy: usize,
    fixed: &[f64],
    range: ([f64; 2], [f64; 2]),
    grid_points: usize,
) -> Result<Overlay> {
    let k = factors.len();
    if responses.is_empty() {
        return invalid("select at least one response");
    }
    if ix == iy || ix >= k || iy >= k || fixed.len() != k {
        return invalid("choose two different factors and a setting for each factor");
    }
    if responses.iter().any(|r| r.model.n_factors() != k) {
        return invalid("a model does not match the factor list");
    }
    if responses.iter().any(|r| r.lower.is_nan() || r.upper.is_nan() || r.lower >= r.upper) {
        return invalid("each response needs lower < upper");
    }
    if !(3..=400).contains(&grid_points) {
        return invalid("the grid needs between 3 and 400 points per axis");
    }
    let axis = |r: [f64; 2]| -> Vec<f64> {
        (0..grid_points).map(|i| r[0] + (r[1] - r[0]) * i as f64 / (grid_points - 1) as f64).collect()
    };
    let xs = axis([range.0[0], range.0[1]]);
    let ys = axis([range.1[0], range.1[1]]);
    let (fx, fy) = (&factors[ix], &factors[iy]);
    let mut series = Vec::new();
    let mut feasible = vec![vec![true; grid_points]; grid_points];
    for r in responses {
        let z: Vec<Vec<f64>> = ys
            .iter()
            .map(|&y| {
                xs.iter()
                    .map(|&x| {
                        let mut p = fixed.to_vec();
                        p[ix] = x;
                        p[iy] = y;
                        r.model.predict(&p)
                    })
                    .collect()
            })
            .collect();
        for (j, row) in z.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                if v < r.lower || v > r.upper {
                    feasible[j][i] = false;
                }
            }
        }
        for (label, level) in [("lower", r.lower), ("upper", r.upper)] {
            if !level.is_finite() {
                continue;
            }
            let (mut sx, mut sy) = (Vec::new(), Vec::new());
            for (a, b) in contour_segments(&xs, &ys, &z, level) {
                sx.extend([fx.to_natural(a[0]), fx.to_natural(b[0]), f64::NAN]);
                sy.extend([fy.to_natural(a[1]), fy.to_natural(b[1]), f64::NAN]);
            }
            if !sx.is_empty() {
                series.push(Series { name: format!("{} {label} = {level}", r.name), x: sx, y: sy });
            }
        }
    }
    let chart = ChartSpec {
        kind: ChartKind::Contour,
        title: format!(
            "Overlaid Contour Plot of {}",
            responses.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join(", ")
        ),
        x_label: fx.name.clone(),
        y_label: fy.name.clone(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    };
    let (mut fxs, mut fys) = (Vec::new(), Vec::new());
    let mut count = 0usize;
    for (j, row) in feasible.iter().enumerate() {
        for (i, &ok) in row.iter().enumerate() {
            if ok {
                count += 1;
                fxs.push(fx.to_natural(xs[i]));
                fys.push(fy.to_natural(ys[j]));
            }
        }
    }
    let best_point = (count > 0).then(|| {
        let (mx, my) = (fxs.iter().sum::<f64>() / count as f64, fys.iter().sum::<f64>() / count as f64);
        let mut best = (fxs[0], fys[0]);
        let mut bd = f64::INFINITY;
        for (&a, &b) in fxs.iter().zip(&fys) {
            let d = (a - mx).powi(2) + (b - my).powi(2);
            if d < bd {
                bd = d;
                best = (a, b);
            }
        }
        best
    });
    let feasible_chart = ChartSpec {
        kind: ChartKind::Scatter,
        title: "Feasible Region (all responses acceptable)".into(),
        x_label: fx.name.clone(),
        y_label: fy.name.clone(),
        categories: Vec::new(),
        series: vec![Series { name: "Feasible".into(), x: fxs, y: fys }],
        reference_lines: Vec::new(),
    };
    Ok(Overlay {
        chart,
        feasible_chart,
        feasible_fraction: count as f64 / (grid_points * grid_points) as f64,
        best_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conference_matrices_are_valid() {
        for n in conference_orders() {
            let c = conference_matrix(n).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let dot: i32 = (0..n).map(|l| i32::from(c[i][l]) * i32::from(c[j][l])).sum();
                    assert_eq!(dot, if i == j { n as i32 - 1 } else { 0 }, "order {n} rows {i},{j}");
                }
            }
        }
        assert!(
            conference_orders().contains(&6) && conference_orders().contains(&10) && conference_orders().contains(&26)
        );
    }

    #[test]
    fn dsd_has_orthogonal_main_effects_for_even_k() {
        for k in [6usize, 8, 10, 14] {
            let d = definitive_screening(k, 0, None).unwrap();
            assert_eq!(d.runs.len(), 2 * k + 1);
            assert!(d.max_main_correlation() < 1e-12, "k={k}");
        }
        let d = definitive_screening(5, 0, None).unwrap();
        assert_eq!((d.conference_order, d.runs.len()), (6, 13));
    }

    #[test]
    fn steepest_path_follows_the_gradient() {
        use super::super::model::{PolyTerm, PolynomialModel};
        let model = PolynomialModel {
            factor_names: vec!["A".into(), "B".into()],
            terms: vec![
                PolyTerm { powers: vec![] },
                PolyTerm { powers: vec![(0, 1)] },
                PolyTerm { powers: vec![(1, 1)] },
            ],
            coef: vec![10.0, 4.0, -2.0],
            cov: None,
            mse: None,
            df_error: 0,
        };
        let f = vec![
            FactorSpec { name: "A".into(), low: 10.0, high: 20.0 },
            FactorSpec { name: "B".into(), low: 0.0, high: 100.0 },
        ];
        let p = steepest_path(&model, &f, true, 0.5, 4).unwrap();
        assert_eq!(p.len(), 5);
        assert_eq!(p[4].coded, vec![2.0, -1.0]);
        assert_eq!(p[4].natural, vec![25.0, 0.0]);
        assert!((p[4].predicted - (10.0 + 8.0 + 2.0)).abs() < 1e-12);
        let d = steepest_path(&model, &f, false, 1.0, 2).unwrap();
        assert_eq!(d[2].coded, vec![-2.0, 1.0]);
    }

    #[test]
    fn variability_recovers_a_dispersion_effect() {
        // two replicates at each of the 8 corners of a 2^3; factor A doubles the spread
        let mut coded = Vec::new();
        let mut y = Vec::new();
        for r in 0..8usize {
            let c: Vec<f64> = (0..3).map(|j| if r >> j & 1 == 1 { 1.0 } else { -1.0 }).collect();
            let sd = if c[0] > 0.0 { 2.0 } else { 1.0 } * (1.0 + 0.01 * r as f64);
            let m = 10.0 + c[1];
            coded.push(c.clone());
            y.push(m - sd / 2f64.sqrt());
            coded.push(c);
            y.push(m + sd / 2f64.sqrt());
        }
        let names: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let a = analyze_variability(&coded, &y, &names, false).unwrap();
        assert_eq!(a.groups.len(), 8);
        assert!((a.terms[0].ratio - 2.0).abs() < 0.05, "{}", a.terms[0].ratio);
        assert!(a.terms[1].effect.abs() < 0.05);
        assert_eq!(a.df_error, 4);
    }
}
