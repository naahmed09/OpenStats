//! Response surface designs (central composite, Box–Behnken) and the full
//! quadratic model with its ANOVA, stationary point and canonical analysis.

use super::design::{Design, DesignKind, PointType};
use super::factorial::default_generators;
use super::model::{block_columns, fit_cols, term_values, Col, PolyTerm, PolynomialModel};
use crate::error::{invalid, Result};
use crate::regression::RegressionFit;
use openstats_math::dist::{ContinuousDistribution, FisherF};
use openstats_math::linalg::{Lu, Matrix, SymmetricEigen};

/// Which central composite design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcdKind {
    /// Axial points at ±α beyond the factorial cube (α > 1).
    Circumscribed,
    /// Factorial points at ±1/α inside the axial points at ±1.
    Inscribed,
    /// Axial points on the faces of the cube (α = 1).
    FaceCentered,
}

/// Rule for the axial distance α.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaRule {
    /// `α = F^{1/4}` (F = number of factorial points): rotatable.
    Rotatable,
    /// Orthogonal quadratic terms: `α² = (√(F·N) − F)/2`, `N` the total runs
    /// (with blocks: `α² = F(k + n_as/2)/(F + n_fs)`).
    Orthogonal,
    /// An explicit α.
    Custom(f64),
}

/// Options of a central composite design.
#[derive(Debug, Clone, PartialEq)]
pub struct CcdOptions {
    /// Design type.
    pub kind: CcdKind,
    /// Axial distance rule (ignored for face-centred designs).
    pub alpha: AlphaRule,
    /// Fraction exponent `p`: the cube is a 2^(k−p) fraction (0 = full factorial).
    pub fraction_p: usize,
    /// Centre points with the factorial runs.
    pub center_factorial: usize,
    /// Centre points with the axial runs.
    pub center_axial: usize,
    /// Two blocks: the factorial (with its centre points) and the axial runs.
    pub blocked: bool,
    /// Randomize the run order within blocks.
    pub randomize: Option<u64>,
}

impl CcdOptions {
    /// Defaults: rotatable, full factorial, 4 + 2 centre points, one block.
    pub fn new(kind: CcdKind) -> CcdOptions {
        CcdOptions {
            kind,
            alpha: AlphaRule::Rotatable,
            fraction_p: 0,
            center_factorial: 4,
            center_axial: 2,
            blocked: false,
            randomize: None,
        }
    }
}

/// Central composite design for 2 ≤ k ≤ 10 factors.
pub fn central_composite(k: usize, opts: &CcdOptions) -> Result<Design> {
    if !(2..=10).contains(&k) {
        return invalid(format!("a central composite design takes 2..=10 factors (got {k})"));
    }
    if opts.fraction_p >= k - 1 {
        return invalid("the fraction leaves fewer than two base factors");
    }
    let p = opts.fraction_p;
    let base = k - p;
    let (words, defining) = if p == 0 {
        (Vec::new(), None)
    } else {
        let g = default_generators(k, p)?;
        let d = super::factorial::defining_for_generators(&g)?;
        (g, Some(d))
    };
    let f_pts = 1usize << base;
    if let Some(d) = &defining {
        if d.resolution < 5 {
            return invalid(format!(
                "the 2^({k}-{p}) cube has resolution {}; a central composite design needs a resolution V cube to estimate all two-factor interactions",
                d.resolution
            ));
        }
    }
    let nf = f_pts as f64;
    let total_center = opts.center_factorial + opts.center_axial;
    let n_total = f_pts + 2 * k + total_center;
    let alpha_raw = match opts.kind {
        CcdKind::FaceCentered => 1.0,
        _ => match opts.alpha {
            AlphaRule::Rotatable => nf.powf(0.25),
            AlphaRule::Orthogonal => {
                if opts.blocked {
                    let v = nf * (k as f64 + opts.center_axial as f64 / 2.0) / (nf + opts.center_factorial as f64);
                    v.sqrt()
                } else {
                    (((nf * n_total as f64).sqrt() - nf) / 2.0).sqrt()
                }
            }
            AlphaRule::Custom(a) => {
                if !(a > 0.0 && a.is_finite()) {
                    return invalid("alpha must be positive and finite");
                }
                a
            }
        },
    };
    if alpha_raw.is_nan() || alpha_raw <= 0.0 {
        return invalid("the orthogonal alpha is not defined for these centre-point counts");
    }
    let (cube, axial) = match opts.kind {
        CcdKind::Inscribed => (1.0 / alpha_raw, 1.0),
        _ => (1.0, alpha_raw),
    };
    let mut rows: Vec<(Vec<f64>, PointType, usize)> = Vec::new();
    let blk = |b: usize| if opts.blocked { b } else { 1 };
    for r in 0..f_pts {
        let mut x: Vec<f64> = (0..base).map(|j| if r >> j & 1 == 1 { 1.0 } else { -1.0 }).collect();
        for &g in &words {
            let letters = g & ((1u32 << base) - 1);
            let v: f64 = (0..base).filter(|j| letters >> j & 1 == 1).map(|j| x[j]).product();
            x.push(v);
        }
        rows.push((x.into_iter().map(|v| v * cube).collect(), PointType::Factorial, blk(1)));
    }
    for _ in 0..opts.center_factorial {
        rows.push((vec![0.0; k], PointType::Center, blk(1)));
    }
    for j in 0..k {
        for s in [-1.0, 1.0] {
            let mut x = vec![0.0; k];
            x[j] = s * axial;
            rows.push((x, PointType::Axial, blk(2)));
        }
    }
    for _ in 0..opts.center_axial {
        rows.push((vec![0.0; k], PointType::Center, blk(2)));
    }
    let mut d =
        Design::assemble(DesignKind::CentralComposite, k, rows, if opts.blocked { 2 } else { 1 }, defining, Vec::new());
    d.std_order = (1..=d.n_runs()).collect();
    Ok(match opts.randomize {
        Some(s) => d.randomized(s),
        None => d,
    })
}

/// Box–Behnken design for k = 3, 4, 5 (every pair of factors at ±1 with the
/// others at 0) or k = 7 (the seven triples of the Fano plane at the eight
/// corners of the cube, the others at 0), plus centre points.
pub fn box_behnken(k: usize, center_points: usize, randomize: Option<u64>) -> Result<Design> {
    let mut rows: Vec<(Vec<f64>, PointType, usize)> = Vec::new();
    match k {
        3..=5 => {
            for a in 0..k {
                for b in a + 1..k {
                    for sb in [-1.0, 1.0] {
                        for sa in [-1.0, 1.0] {
                            let mut x = vec![0.0; k];
                            x[a] = sa;
                            x[b] = sb;
                            rows.push((x, PointType::Edge, 1));
                        }
                    }
                }
            }
        }
        7 => {
            // Lines of the Fano plane.
            for t in [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]] {
                for r in 0..8 {
                    let mut x = vec![0.0; k];
                    for (i, &f) in t.iter().enumerate() {
                        x[f] = if r >> i & 1 == 1 { 1.0 } else { -1.0 };
                    }
                    rows.push((x, PointType::Edge, 1));
                }
            }
        }
        _ => return invalid(format!("Box-Behnken designs are available for 3, 4, 5 or 7 factors (got {k})")),
    }
    for _ in 0..center_points {
        rows.push((vec![0.0; k], PointType::Center, 1));
    }
    let mut d = Design::assemble(DesignKind::BoxBehnken, k, rows, 1, None, Vec::new());
    d.std_order = (1..=d.n_runs()).collect();
    Ok(match randomize {
        Some(s) => d.randomized(s),
        None => d,
    })
}

/// Order of the polynomial fitted by [`analyze_surface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceModel {
    /// Linear terms only.
    Linear,
    /// Linear and two-factor interactions.
    LinearInteractions,
    /// Linear, squares and two-factor interactions (full quadratic).
    FullQuadratic,
}

/// Options of a response-surface analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceOptions {
    /// Model.
    pub model: SurfaceModel,
    /// Confidence level (0.95).
    pub level: f64,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        SurfaceOptions { model: SurfaceModel::FullQuadratic, level: 0.95 }
    }
}

/// Nature of the stationary point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationaryKind {
    /// All eigenvalues negative.
    Maximum,
    /// All eigenvalues positive.
    Minimum,
    /// Eigenvalues of both signs.
    Saddle,
    /// An eigenvalue is numerically zero (a ridge; the point is not unique).
    Ridge,
}

/// Canonical analysis of a quadratic surface.
#[derive(Debug, Clone, PartialEq)]
pub struct Canonical {
    /// Stationary point in coded units.
    pub point_coded: Vec<f64>,
    /// Stationary point in natural units.
    pub point_natural: Vec<f64>,
    /// Predicted response at the stationary point.
    pub response: f64,
    /// Eigenvalues of the matrix of quadratic coefficients (ascending).
    pub eigenvalues: Vec<f64>,
    /// Nature of the point.
    pub kind: StationaryKind,
}

/// ANOVA row of the surface model.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceAnovaRow {
    /// Source.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sequential sum of squares.
    pub seq_ss: Option<f64>,
    /// Adjusted sum of squares.
    pub adj_ss: f64,
    /// Adjusted mean square.
    pub adj_ms: Option<f64>,
    /// F statistic.
    pub f: Option<f64>,
    /// p-value.
    pub p: Option<f64>,
}

/// Result of a response-surface analysis.
#[derive(Debug, Clone)]
pub struct SurfaceAnalysis {
    /// The design.
    pub design: Design,
    /// Response.
    pub y: Vec<f64>,
    /// Regression with coefficient table, ANOVA, S, R², PRESS.
    pub fit: RegressionFit,
    /// ANOVA by term group (Blocks, Model, Linear, Square, 2-Way Interaction, Error, Lack-of-Fit, Pure Error, Total).
    pub anova: Vec<SurfaceAnovaRow>,
    /// Terms dropped as confounded.
    pub dropped: Vec<String>,
    /// Polynomial model (coded units).
    pub model: PolynomialModel,
    /// Canonical analysis (full quadratic models with a non-singular B).
    pub canonical: Option<Canonical>,
}

/// Fit a response-surface model to a design (central composite, Box–Behnken or any design in coded units).
pub fn analyze_surface(design: &Design, y: &[f64], opts: &SurfaceOptions) -> Result<SurfaceAnalysis> {
    let k = design.n_factors();
    if y.len() != design.n_runs() {
        return invalid(format!("the response has {} values for {} runs", y.len(), design.n_runs()));
    }
    let names: Vec<String> = design.factors.iter().map(|f| f.name.clone()).collect();
    let mut cols = block_columns(design);
    let add = |powers: Vec<(usize, u32)>, group: &'static str, cols: &mut Vec<Col>| {
        let t = PolyTerm { powers };
        cols.push(Col { name: t.label(&names), values: term_values(design, &t), term: Some(t), group });
    };
    for i in 0..k {
        add(vec![(i, 1)], "Linear", &mut cols);
    }
    if opts.model == SurfaceModel::FullQuadratic {
        for i in 0..k {
            add(vec![(i, 2)], "Square", &mut cols);
        }
    }
    if opts.model != SurfaceModel::Linear {
        for i in 0..k {
            for j in i + 1..k {
                add(vec![(i, 1), (j, 1)], "2-Way Interaction", &mut cols);
            }
        }
    }
    let fc = fit_cols(cols, y, opts.level)?;
    let Some(fit) = fc.fit.clone() else {
        return invalid("the design has no error degrees of freedom for this model; add runs or use a smaller model");
    };
    // ANOVA by group.
    let df_err = fc.df_error;
    let sse = fc.sse;
    let mse = sse / df_err as f64;
    let n = y.len();
    let ybar = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    let mut order: Vec<&'static str> = Vec::new();
    for g in &fc.groups[1..] {
        if !order.contains(g) {
            order.push(g);
        }
    }
    let count = |g: &str| fc.groups[1..].iter().filter(|x| **x == g).count();
    let mk = |source: &str, df: usize, seq: Option<f64>, adj: f64| -> SurfaceAnovaRow {
        let ms = adj / df as f64;
        let f = ms / mse;
        let p = FisherF::new(df as f64, df_err as f64).ok().map(|d| d.sf(f));
        SurfaceAnovaRow { source: source.into(), df, seq_ss: seq, adj_ss: adj, adj_ms: Some(ms), f: Some(f), p }
    };
    let mut anova = Vec::new();
    anova.push(mk("Model", fc.groups.len() - 1, Some(sst - sse), sst - sse));
    let mut prev = sst;
    for (pos, g) in order.iter().enumerate() {
        let without: Vec<Col> = fc.cols.iter().filter(|c| c.group != *g).cloned().collect();
        let adj = (fit_cols(without, y, opts.level)?.sse - sse).max(0.0);
        let upto: Vec<Col> = fc.cols.iter().filter(|c| order[..=pos].contains(&c.group)).cloned().collect();
        let here = fit_cols(upto, y, opts.level)?.sse;
        anova.push(mk(g, count(g), Some(prev - here), adj));
        prev = here;
    }
    anova.push(SurfaceAnovaRow {
        source: "Error".into(),
        df: df_err,
        seq_ss: Some(sse),
        adj_ss: sse,
        adj_ms: Some(mse),
        f: None,
        p: None,
    });
    if let Some(l) = &fit.lack_of_fit {
        anova.push(SurfaceAnovaRow {
            source: "Lack-of-Fit".into(),
            df: l.df_lof,
            seq_ss: Some(l.ss_lof),
            adj_ss: l.ss_lof,
            adj_ms: Some(l.ss_lof / l.df_lof as f64),
            f: Some(l.f),
            p: Some(l.p_value),
        });
        anova.push(SurfaceAnovaRow {
            source: "Pure Error".into(),
            df: l.df_pe,
            seq_ss: Some(l.ss_pe),
            adj_ss: l.ss_pe,
            adj_ms: Some(l.ss_pe / l.df_pe as f64),
            f: None,
            p: None,
        });
    }
    anova.push(SurfaceAnovaRow {
        source: "Total".into(),
        df: n - 1,
        seq_ss: Some(sst),
        adj_ss: sst,
        adj_ms: None,
        f: None,
        p: None,
    });
    let model = fc.model(&names);
    let canonical = if opts.model == SurfaceModel::FullQuadratic { canonical_analysis(&model, design) } else { None };
    Ok(SurfaceAnalysis { design: design.clone(), y: y.to_vec(), fit, anova, dropped: fc.dropped, model, canonical })
}

/// Stationary point and eigenvalues of a full quadratic model.
pub fn canonical_analysis(model: &PolynomialModel, design: &Design) -> Option<Canonical> {
    let k = model.n_factors();
    let coef_of = |powers: &[(usize, u32)]| -> Option<f64> {
        model.terms.iter().position(|t| t.powers == powers).map(|i| model.coef[i])
    };
    let b0 = coef_of(&[])?;
    let mut b = vec![0.0; k];
    let mut bm = Matrix::zeros(k, k);
    for i in 0..k {
        b[i] = coef_of(&[(i, 1)])?;
        bm[(i, i)] = coef_of(&[(i, 2)])?;
        for j in i + 1..k {
            let v = coef_of(&[(i, 1), (j, 1)])? / 2.0;
            bm[(i, j)] = v;
            bm[(j, i)] = v;
        }
    }
    let eig = SymmetricEigen::new(&bm).ok()?;
    let scale = eig.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return None;
    }
    let ridge = eig.values.iter().any(|v| v.abs() < 1e-10 * scale);
    let kind = if ridge {
        StationaryKind::Ridge
    } else if eig.values.iter().all(|&v| v < 0.0) {
        StationaryKind::Maximum
    } else if eig.values.iter().all(|&v| v > 0.0) {
        StationaryKind::Minimum
    } else {
        StationaryKind::Saddle
    };
    if ridge {
        return Some(Canonical {
            point_coded: Vec::new(),
            point_natural: Vec::new(),
            response: f64::NAN,
            eigenvalues: eig.values,
            kind,
        });
    }
    let rhs: Vec<f64> = b.iter().map(|v| -0.5 * v).collect();
    let xs = Lu::new(&bm).ok()?.solve(&rhs).ok()?;
    let response = b0 + 0.5 * b.iter().zip(&xs).map(|(a, c)| a * c).sum::<f64>();
    let natural = xs.iter().zip(&design.factors).map(|(&c, f)| f.to_natural(c)).collect();
    Some(Canonical { point_coded: xs, point_natural: natural, response, eigenvalues: eig.values, kind })
}
