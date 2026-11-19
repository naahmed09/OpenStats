//! Multiple-response optimisation with Derringer–Suich desirability functions.
//!
//! Each response has a fitted [`PolynomialModel`] in coded units and a goal
//! (maximise, minimise or hit a target). The individual desirability `dᵢ ∈ [0,1]`
//! follows Derringer & Suich (1980); the composite is the weighted geometric mean
//! `D = (∏ dᵢ^{wᵢ})^{1/Σwᵢ}` where `wᵢ` are the importances. The search maximises
//! `D` over a box in coded factor space with multi-start Nelder–Mead on a smooth
//! bounded transform `x = lo + (hi−lo)(1+sin u)/2`.

use super::design::FactorSpec;
use super::model::PolynomialModel;
use crate::error::{invalid, Result};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use openstats_math::optim::{nelder_mead, NelderMeadOptions};
use openstats_math::random::{Pcg64, Rng};

/// What a response should do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Goal {
    /// Larger is better: `d = 0` below `lower`, 1 above `target`.
    Maximize {
        /// Smallest acceptable value.
        lower: f64,
        /// Value at which `d` reaches 1.
        target: f64,
        /// Shape (weight) exponent; 1 is linear.
        shape: f64,
    },
    /// Smaller is better: `d = 1` below `target`, 0 above `upper`.
    Minimize {
        /// Value up to which `d` is 1.
        target: f64,
        /// Largest acceptable value.
        upper: f64,
        /// Shape exponent.
        shape: f64,
    },
    /// Hit a target between two limits.
    Target {
        /// Lower limit.
        lower: f64,
        /// Target.
        target: f64,
        /// Upper limit.
        upper: f64,
        /// Shape exponent of the lower side.
        shape_low: f64,
        /// Shape exponent of the upper side.
        shape_high: f64,
    },
}

impl Goal {
    fn validate(&self) -> Result<()> {
        let ok = match *self {
            Goal::Maximize { lower, target, shape } => lower < target && shape > 0.0,
            Goal::Minimize { target, upper, shape } => target < upper && shape > 0.0,
            Goal::Target { lower, target, upper, shape_low, shape_high } => {
                lower < target && target < upper && shape_low > 0.0 && shape_high > 0.0
            }
        };
        let finite = match *self {
            Goal::Maximize { lower, target, shape } => [lower, target, shape].iter().all(|v| v.is_finite()),
            Goal::Minimize { target, upper, shape } => [target, upper, shape].iter().all(|v| v.is_finite()),
            Goal::Target { lower, target, upper, shape_low, shape_high } => {
                [lower, target, upper, shape_low, shape_high].iter().all(|v| v.is_finite())
            }
        };
        if ok && finite {
            Ok(())
        } else {
            invalid("goal limits must be finite and ordered (lower < target < upper) with positive shapes")
        }
    }

    /// Individual desirability of the value `y`.
    pub fn desirability(&self, y: f64) -> f64 {
        match *self {
            Goal::Maximize { lower, target, shape } => {
                if y < lower {
                    0.0
                } else if y >= target {
                    1.0
                } else {
                    ((y - lower) / (target - lower)).powf(shape)
                }
            }
            Goal::Minimize { target, upper, shape } => {
                if y > upper {
                    0.0
                } else if y <= target {
                    1.0
                } else {
                    ((upper - y) / (upper - target)).powf(shape)
                }
            }
            Goal::Target { lower, target, upper, shape_low, shape_high } => {
                if y < lower || y > upper {
                    0.0
                } else if y <= target {
                    ((y - lower) / (target - lower)).powf(shape_low)
                } else {
                    ((upper - y) / (upper - target)).powf(shape_high)
                }
            }
        }
    }

    /// Distance outside the acceptable region, relative to its width (0 inside).
    fn violation(&self, y: f64) -> f64 {
        match *self {
            Goal::Maximize { lower, target, .. } => ((lower - y) / (target - lower)).max(0.0),
            Goal::Minimize { target, upper, .. } => ((y - upper) / (upper - target)).max(0.0),
            Goal::Target { lower, upper, .. } => ((lower - y).max(y - upper) / (upper - lower)).max(0.0),
        }
    }
}

/// One response to optimise.
#[derive(Debug, Clone)]
pub struct ResponseGoal {
    /// Name.
    pub name: String,
    /// Fitted model (coded factors).
    pub model: PolynomialModel,
    /// Goal.
    pub goal: Goal,
    /// Importance `wᵢ > 0` (exponent in the composite).
    pub importance: f64,
}

/// Search options.
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Coded range of each factor; `None` means [−1, 1] for every factor.
    pub ranges: Option<Vec<(f64, f64)>>,
    /// Factors held fixed, in natural units (`None` = free).
    pub fixed: Vec<Option<f64>>,
    /// Number of random starting points (besides the centre and the box corners when k ≤ 4).
    pub starts: usize,
    /// Number of distinct solutions to report.
    pub top: usize,
    /// Seed of the starting-point generator.
    pub seed: u64,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        OptimizeOptions { ranges: None, fixed: Vec::new(), starts: 40, top: 5, seed: 1 }
    }
}

/// One candidate setting.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Coded factor values.
    pub coded: Vec<f64>,
    /// Natural factor values.
    pub natural: Vec<f64>,
    /// Predicted responses.
    pub fit: Vec<f64>,
    /// Standard error of each fitted response.
    pub se_fit: Vec<Option<f64>>,
    /// Individual desirabilities.
    pub d: Vec<f64>,
    /// Composite desirability.
    pub composite: f64,
}

/// Result of an optimisation.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Factors.
    pub factors: Vec<FactorSpec>,
    /// Responses.
    pub responses: Vec<ResponseGoal>,
    /// Distinct local optima, best first.
    pub solutions: Vec<Solution>,
}

fn composite(goals: &[ResponseGoal], y: &[f64]) -> (f64, Vec<f64>) {
    let d: Vec<f64> = goals.iter().zip(y).map(|(g, &v)| g.goal.desirability(v)).collect();
    let sw: f64 = goals.iter().map(|g| g.importance).sum();
    if d.iter().any(|&v| v <= 0.0) {
        return (0.0, d);
    }
    let lg: f64 = goals.iter().zip(&d).map(|(g, &v)| g.importance * v.ln()).sum();
    ((lg / sw).exp(), d)
}

fn evaluate(goals: &[ResponseGoal], factors: &[FactorSpec], x: &[f64]) -> Solution {
    let fit: Vec<f64> = goals.iter().map(|g| g.model.predict(x)).collect();
    let (c, d) = composite(goals, &fit);
    Solution {
        natural: x.iter().zip(factors).map(|(&v, f)| f.to_natural(v)).collect(),
        se_fit: goals.iter().map(|g| g.model.se_fit(x)).collect(),
        coded: x.to_vec(),
        fit,
        d,
        composite: c,
    }
}

/// Composite desirability of a coded setting.
pub fn composite_desirability(goals: &[ResponseGoal], factors: &[FactorSpec], coded: &[f64]) -> Solution {
    evaluate(goals, factors, coded)
}

/// Find settings that maximise the composite desirability.
pub fn optimize_responses(
    goals: &[ResponseGoal],
    factors: &[FactorSpec],
    opts: &OptimizeOptions,
) -> Result<OptimizationResult> {
    let k = factors.len();
    if goals.is_empty() {
        return invalid("at least one response is needed");
    }
    if k == 0 {
        return invalid("at least one factor is needed");
    }
    for g in goals {
        g.goal.validate()?;
        if !(g.importance.is_finite() && g.importance > 0.0) {
            return invalid("importance must be positive");
        }
        if g.model.n_factors() != k {
            return invalid(format!("model for '{}' has {} factors, expected {k}", g.name, g.model.n_factors()));
        }
    }
    let ranges = match &opts.ranges {
        Some(r) if r.len() == k => r.clone(),
        Some(_) => return invalid("ranges must have one (low, high) pair per factor"),
        None => vec![(-1.0, 1.0); k],
    };
    if ranges.iter().any(|r| r.0.is_nan() || r.0 >= r.1 || !r.0.is_finite() || !r.1.is_finite()) {
        return invalid("each range needs low < high");
    }
    let fixed_coded: Vec<Option<f64>> =
        (0..k).map(|i| opts.fixed.get(i).copied().flatten().map(|v| factors[i].to_coded(v))).collect();
    for (i, f) in fixed_coded.iter().enumerate() {
        if let Some(v) = f {
            if *v < ranges[i].0 - 1e-12 || *v > ranges[i].1 + 1e-12 {
                return invalid(format!("fixed value of '{}' is outside its range", factors[i].name));
            }
        }
    }
    let free: Vec<usize> = (0..k).filter(|&i| fixed_coded[i].is_none()).collect();
    let to_x = |u: &[f64]| -> Vec<f64> {
        let mut x = vec![0.0; k];
        let mut j = 0;
        for i in 0..k {
            x[i] = match fixed_coded[i] {
                Some(v) => v,
                None => {
                    let v = ranges[i].0 + (ranges[i].1 - ranges[i].0) * (1.0 + u[j].sin()) / 2.0;
                    j += 1;
                    v
                }
            };
        }
        x
    };
    let objective = |u: &[f64]| -> f64 {
        let x = to_x(u);
        let y: Vec<f64> = goals.iter().map(|g| g.model.predict(&x)).collect();
        let (c, _) = composite(goals, &y);
        if c > 0.0 {
            -c
        } else {
            1.0 + goals.iter().zip(&y).map(|(g, &v)| g.goal.violation(v)).sum::<f64>()
        }
    };
    // Starting points in u-space.
    let inv = |i: usize, v: f64| -> f64 {
        let t = (2.0 * (v - ranges[i].0) / (ranges[i].1 - ranges[i].0) - 1.0).clamp(-1.0, 1.0);
        t.asin()
    };
    let mut starts: Vec<Vec<f64>> = Vec::new();
    starts.push(free.iter().map(|&i| inv(i, 0.5 * (ranges[i].0 + ranges[i].1))).collect());
    if free.len() <= 4 {
        for m in 0..(1usize << free.len()) {
            starts.push(
                free.iter()
                    .enumerate()
                    .map(|(j, &i)| {
                        let t = if m >> j & 1 == 1 { 0.9 } else { -0.9 };
                        inv(i, ranges[i].0 + (ranges[i].1 - ranges[i].0) * (1.0 + t) / 2.0)
                    })
                    .collect(),
            );
        }
    }
    let mut rng = Pcg64::new(opts.seed as u128, 77);
    for _ in 0..opts.starts {
        starts.push(
            free.iter()
                .map(|&i| {
                    let t = rng.next_f64();
                    inv(i, ranges[i].0 + (ranges[i].1 - ranges[i].0) * t)
                })
                .collect(),
        );
    }
    let nm = NelderMeadOptions { max_iter: 4000, ..NelderMeadOptions::default() };
    let mut found: Vec<Solution> = Vec::new();
    if free.is_empty() {
        found.push(evaluate(goals, factors, &to_x(&[])));
    } else {
        for s in &starts {
            let mut f = |u: &[f64]| objective(u);
            let m =
                nelder_mead(&mut f, s, &nm).map_err(|e| crate::error::StatsError::InvalidArgument(e.to_string()))?;
            found.push(evaluate(goals, factors, &to_x(&m.x)));
        }
    }
    found.sort_by(|a, b| b.composite.partial_cmp(&a.composite).unwrap_or(std::cmp::Ordering::Equal));
    // Keep distinct optima (coded distance > 1e-3 of the full range).
    let mut solutions: Vec<Solution> = Vec::new();
    for s in found {
        let dup = solutions
            .iter()
            .any(|t| s.coded.iter().zip(&t.coded).zip(&ranges).all(|((a, b), r)| (a - b).abs() <= 1e-3 * (r.1 - r.0)));
        if !dup {
            solutions.push(s);
        }
        if solutions.len() >= opts.top.max(1) {
            break;
        }
    }
    Ok(OptimizationResult { factors: factors.to_vec(), responses: goals.to_vec(), solutions })
}

impl OptimizationResult {
    /// Best solution.
    pub fn best(&self) -> &Solution {
        &self.solutions[0]
    }

    /// Optimisation plot: composite desirability against each factor, the others at the best setting.
    pub fn charts(&self, ranges: Option<&[(f64, f64)]>, points: usize) -> Vec<ChartSpec> {
        let best = self.best();
        let k = self.factors.len();
        let n = points.max(3);
        (0..k)
            .map(|i| {
                let (lo, hi) = ranges.map(|r| r[i]).unwrap_or((-1.0, 1.0));
                let mut xs = Vec::new();
                let mut ys = Vec::new();
                for p in 0..n {
                    let v = lo + (hi - lo) * p as f64 / (n - 1) as f64;
                    let mut x = best.coded.clone();
                    x[i] = v;
                    let y: Vec<f64> = self.responses.iter().map(|g| g.model.predict(&x)).collect();
                    xs.push(self.factors[i].to_natural(v));
                    ys.push(composite(&self.responses, &y).0);
                }
                ChartSpec {
                    kind: ChartKind::Line,
                    title: format!("Composite Desirability vs {}", self.factors[i].name),
                    x_label: self.factors[i].name.clone(),
                    y_label: "Composite desirability".into(),
                    categories: Vec::new(),
                    series: vec![Series { name: "D".into(), x: xs, y: ys }],
                    reference_lines: vec![ReferenceLine {
                        axis: Axis::X,
                        value: best.natural[i],
                        label: "Optimum".into(),
                    }],
                }
            })
            .collect()
    }
}
