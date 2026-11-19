//! Acceptance sampling: single and double plans by attributes, single plans by variables.
//!
//! Attribute plans are described by a sample size `n` and an acceptance number `c`; the lot is
//! accepted when the sample holds at most `c` defectives. Operating-characteristic (OC) curves
//! use the binomial, Poisson or (with a lot size) hypergeometric distribution. Average outgoing
//! quality (AOQ) and average total inspection (ATI) assume that rejected lots are screened 100 %
//! and that defectives found in samples are replaced by good items.
//!
//! Variables plans use a single specification limit and the criterion `(U − x̄)/s ≥ k`
//! (or `(x̄ − L)/s ≥ k`), with σ known (the σ-method) or unknown (the s-method). Acceptance
//! probabilities are exact: normal for a known σ and non-central t for an unknown σ.

use crate::error::{invalid, Result, StatsError};
use crate::report::{Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Section, Series, Table};
use openstats_math::dist::{
    Binomial, ContinuousDistribution, DiscreteDistribution, Hypergeometric, NonCentralT, Poisson,
};
use openstats_math::optim::brent_minimize;
use openstats_math::special::erf::{norm_cdf, norm_ppf};

const S: ColumnKind = ColumnKind::Significant(6);
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn table(title: &str, cols: &[(&str, ColumnKind)], rows: Vec<Vec<Cell>>) -> Table {
    let mut t = Table::new(title, cols.iter().map(|(h, k)| Column::new(*h, *k)).collect());
    for r in rows {
        t.row(r);
    }
    t
}

/// Distribution used for the number of defectives in a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LotModel {
    /// Binomial (infinite lot, or a lot much larger than the sample).
    Binomial,
    /// Poisson approximation, mean `n·p`.
    Poisson,
    /// Hypergeometric; needs a lot size.
    Hypergeometric,
}

impl LotModel {
    /// Name used in reports.
    pub fn name(self) -> &'static str {
        match self {
            LotModel::Binomial => "Binomial",
            LotModel::Poisson => "Poisson",
            LotModel::Hypergeometric => "Hypergeometric",
        }
    }
}

fn check_p(p: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&p) || p.is_nan() {
        return invalid(format!("the proportion defective must be in [0, 1] (got {p})"));
    }
    Ok(())
}

fn check_risk(name: &str, r: f64) -> Result<()> {
    if !(r > 0.0 && r < 1.0) {
        return invalid(format!("{name} must be strictly between 0 and 1 (got {r})"));
    }
    Ok(())
}

/// A single sampling plan by attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributePlan {
    /// Sample size.
    pub n: u64,
    /// Acceptance number: accept when the defectives found are at most `c`.
    pub c: u64,
}

fn count_cdf(model: LotModel, lot: Option<u64>, n: u64, c: u64, p: f64) -> Result<f64> {
    check_p(p)?;
    if p == 0.0 {
        return Ok(1.0);
    }
    Ok(match model {
        LotModel::Binomial => {
            if p >= 1.0 {
                return Ok(if c >= n { 1.0 } else { 0.0 });
            }
            Binomial::new(n, p)?.cdf(c as f64)
        }
        LotModel::Poisson => Poisson::new(n as f64 * p)?.cdf(c as f64),
        LotModel::Hypergeometric => {
            let Some(big_n) = lot else {
                return invalid("the hypergeometric model needs a lot size");
            };
            if n > big_n {
                return invalid(format!("the sample size {n} exceeds the lot size {big_n}"));
            }
            let d = ((big_n as f64) * p).round() as u64;
            Hypergeometric::new(big_n, d.min(big_n), n)?.cdf(c as f64)
        }
    })
}

impl AttributePlan {
    /// Probability of accepting a lot with proportion defective `p`.
    pub fn accept_probability(&self, model: LotModel, lot: Option<u64>, p: f64) -> Result<f64> {
        if self.n == 0 {
            return invalid("the sample size must be at least 1");
        }
        count_cdf(model, lot, self.n, self.c, p)
    }

    /// Average outgoing quality at `p`: `p·Pa·(N − n)/N` (just `p·Pa` without a lot size).
    pub fn aoq(&self, model: LotModel, lot: Option<u64>, p: f64) -> Result<f64> {
        let pa = self.accept_probability(model, lot, p)?;
        Ok(match lot {
            Some(big_n) if big_n > 0 => p * pa * (big_n as f64 - self.n as f64) / big_n as f64,
            _ => p * pa,
        })
    }

    /// Average total inspection at `p`: `n + (1 − Pa)(N − n)`; `None` without a lot size.
    pub fn ati(&self, model: LotModel, lot: Option<u64>, p: f64) -> Result<Option<f64>> {
        let pa = self.accept_probability(model, lot, p)?;
        Ok(lot.map(|big_n| self.n as f64 + (1.0 - pa) * (big_n as f64 - self.n as f64)))
    }
}

/// OC, AOQ and ATI evaluated on a grid of proportions defective.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeCurve {
    /// Proportions defective.
    pub p: Vec<f64>,
    /// Probability of acceptance.
    pub pa: Vec<f64>,
    /// Average outgoing quality.
    pub aoq: Vec<f64>,
    /// Average total inspection (empty without a lot size).
    pub ati: Vec<f64>,
}

/// Evaluate a plan on the grid `p`.
pub fn attribute_curve(plan: &AttributePlan, model: LotModel, lot: Option<u64>, p: &[f64]) -> Result<AttributeCurve> {
    let mut c = AttributeCurve { p: p.to_vec(), pa: Vec::new(), aoq: Vec::new(), ati: Vec::new() };
    for &x in p {
        c.pa.push(plan.accept_probability(model, lot, x)?);
        c.aoq.push(plan.aoq(model, lot, x)?);
        if let Some(a) = plan.ati(model, lot, x)? {
            c.ati.push(a);
        }
    }
    Ok(c)
}

/// The proportion at which the plan's acceptance probability falls to `target` (bisection).
fn p_at_pa(plan: &AttributePlan, model: LotModel, lot: Option<u64>, target: f64) -> Result<f64> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if plan.accept_probability(model, lot, mid)? > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// A grid of `points` proportions from 0 to the point where the plan accepts with probability 0.005
/// (at least one defective per lot is always covered).
pub fn default_grid(plan: &AttributePlan, model: LotModel, lot: Option<u64>, points: usize) -> Result<Vec<f64>> {
    let pmax = p_at_pa(plan, model, lot, 0.005)?.max(1.0 / (plan.n as f64 + 1.0)).min(1.0);
    let k = points.max(2);
    Ok((0..k).map(|i| pmax * i as f64 / (k - 1) as f64).collect())
}

/// The largest average outgoing quality (AOQL) and the proportion where it occurs.
pub fn aoql(plan: &AttributePlan, model: LotModel, lot: Option<u64>) -> Result<(f64, f64)> {
    let pmax = (3.0 * (plan.c as f64 + 4.0) / plan.n as f64).min(1.0);
    let steps = 2000;
    let mut best = (0.0, 0.0);
    let mut bi = 0;
    for i in 1..=steps {
        let p = pmax * i as f64 / steps as f64;
        let a = plan.aoq(model, lot, p)?;
        if a > best.1 {
            best = (p, a);
            bi = i;
        }
    }
    let lo = pmax * (bi.max(1) - 1) as f64 / steps as f64;
    let hi = (pmax * (bi + 1) as f64 / steps as f64).min(1.0);
    if hi > lo {
        let mut err = None;
        let m = brent_minimize(
            |p| match plan.aoq(model, lot, p) {
                Ok(a) => -a,
                Err(e) => {
                    err = Some(e);
                    0.0
                }
            },
            lo,
            hi,
            1e-12,
            200,
        )?;
        if let Some(e) = err {
            return Err(e);
        }
        if -m.f > best.1 {
            best = (m.x, -m.f);
        }
    }
    Ok((best.1, best.0))
}

/// A plan found for given quality and risk requirements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanSearch {
    /// The plan.
    pub plan: AttributePlan,
    /// Probability of accepting a lot at the acceptable quality level (≥ 1 − α).
    pub pa_aql: f64,
    /// Probability of accepting a lot at the rejectable quality level (≤ β).
    pub pa_rql: f64,
}

/// Smallest `n` (and the smallest `c` for it) with `Pa(AQL) ≥ 1 − α` and `Pa(RQL) ≤ β`.
pub fn find_attribute_plan(
    aql: f64,
    rql: f64,
    alpha: f64,
    beta: f64,
    model: LotModel,
    lot: Option<u64>,
) -> Result<PlanSearch> {
    check_risk("the acceptable quality level (AQL)", aql)?;
    check_risk("the rejectable quality level (RQL)", rql)?;
    check_risk("the producer's risk", alpha)?;
    check_risk("the consumer's risk", beta)?;
    if aql >= rql {
        return invalid("the AQL must be smaller than the RQL");
    }
    if model == LotModel::Hypergeometric && lot.is_none() {
        return invalid("the hypergeometric model needs a lot size");
    }
    let limit = lot.unwrap_or(300_000).min(300_000);
    for n in 1..=limit {
        let c = match model {
            LotModel::Binomial => Binomial::new(n, aql)?.quantile(1.0 - alpha),
            LotModel::Poisson => Poisson::new(n as f64 * aql)?.quantile(1.0 - alpha),
            LotModel::Hypergeometric => {
                let big_n = lot.unwrap_or(0);
                let d = ((big_n as f64) * aql).round() as u64;
                Hypergeometric::new(big_n, d.min(big_n), n)?.quantile(1.0 - alpha)
            }
        } as u64;
        if c >= n {
            continue;
        }
        let plan = AttributePlan { n, c };
        let pa_rql = plan.accept_probability(model, lot, rql)?;
        if pa_rql <= beta {
            let pa_aql = plan.accept_probability(model, lot, aql)?;
            if pa_aql >= 1.0 - alpha {
                return Ok(PlanSearch { plan, pa_aql, pa_rql });
            }
        }
    }
    invalid("no sampling plan meets both risks within the searched sample sizes; relax the risks or move the quality levels apart")
}

/// A double sampling plan: sample `n1`; accept when the defectives `d1 ≤ c1`, reject when
/// `d1 ≥ r1`, otherwise sample `n2` more and accept when `d1 + d2 ≤ c2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoublePlan {
    /// First sample size.
    pub n1: u64,
    /// First acceptance number.
    pub c1: u64,
    /// First rejection number.
    pub r1: u64,
    /// Second sample size.
    pub n2: u64,
    /// Cumulative acceptance number after the second sample.
    pub c2: u64,
}

/// Probabilities of the outcomes of a double plan at one quality level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleOutcome {
    /// Accepted on the first sample.
    pub accept_first: f64,
    /// Accepted after the second sample.
    pub accept_second: f64,
    /// Total probability of acceptance.
    pub pa: f64,
    /// A second sample is needed.
    pub second_sample: f64,
    /// Average sample number `n1 + n2·P(second sample)`.
    pub asn: f64,
}

impl DoublePlan {
    fn validate(&self) -> Result<()> {
        if self.n1 == 0 || self.n2 == 0 {
            return invalid("both sample sizes must be at least 1");
        }
        if self.r1 <= self.c1 {
            return invalid("the first rejection number must exceed the first acceptance number");
        }
        if self.r1 > self.c2 + 1 {
            return invalid("the first rejection number cannot exceed the second acceptance number plus 1");
        }
        Ok(())
    }

    /// Outcome probabilities at proportion defective `p` (binomial or Poisson).
    pub fn outcome(&self, model: LotModel, p: f64) -> Result<DoubleOutcome> {
        self.validate()?;
        check_p(p)?;
        if model == LotModel::Hypergeometric {
            return invalid("double sampling plans use the binomial or Poisson model");
        }
        if p == 0.0 {
            return Ok(DoubleOutcome {
                accept_first: 1.0,
                accept_second: 0.0,
                pa: 1.0,
                second_sample: 0.0,
                asn: self.n1 as f64,
            });
        }
        let pmf1 = |d: u64| -> Result<f64> {
            Ok(match model {
                LotModel::Binomial => Binomial::new(self.n1, p)?.pmf(d as f64),
                _ => Poisson::new(self.n1 as f64 * p)?.pmf(d as f64),
            })
        };
        let accept_first = count_cdf(model, None, self.n1, self.c1, p)?;
        let mut accept_second = 0.0;
        let mut second = 0.0;
        for d1 in (self.c1 + 1)..self.r1 {
            let w = pmf1(d1)?;
            second += w;
            accept_second += w * count_cdf(model, None, self.n2, self.c2 - d1, p)?;
        }
        Ok(DoubleOutcome {
            accept_first,
            accept_second,
            pa: accept_first + accept_second,
            second_sample: second,
            asn: self.n1 as f64 + self.n2 as f64 * second,
        })
    }

    /// Average outgoing quality with 100 % screening of rejected lots.
    pub fn aoq(&self, model: LotModel, lot: Option<u64>, p: f64) -> Result<f64> {
        let o = self.outcome(model, p)?;
        Ok(match lot {
            Some(big_n) if big_n > 0 => {
                let n = big_n as f64;
                p * (o.accept_first * (n - self.n1 as f64) + o.accept_second * (n - (self.n1 + self.n2) as f64)) / n
            }
            _ => p * o.pa,
        })
    }

    /// Average total inspection with 100 % screening of rejected lots; `None` without a lot size.
    pub fn ati(&self, model: LotModel, lot: Option<u64>, p: f64) -> Result<Option<f64>> {
        let o = self.outcome(model, p)?;
        Ok(lot.map(|big_n| {
            o.accept_first * self.n1 as f64 + o.accept_second * (self.n1 + self.n2) as f64 + (1.0 - o.pa) * big_n as f64
        }))
    }
}

/// Side of the specification limit for a variables plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitSide {
    /// Upper specification limit: accept when `(U − x̄)/s ≥ k`.
    Upper,
    /// Lower specification limit: accept when `(x̄ − L)/s ≥ k`.
    Lower,
}

/// A single sampling plan by variables (one specification limit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariablesPlan {
    /// Sample size.
    pub n: u64,
    /// Acceptability constant.
    pub k: f64,
    /// Whether σ is known (σ-method) or estimated by `s` (s-method).
    pub sigma_known: bool,
}

impl VariablesPlan {
    /// Probability of acceptance when the proportion nonconforming is `p`.
    pub fn accept_probability(&self, p: f64) -> Result<f64> {
        check_p(p)?;
        if self.n == 0 || (!self.sigma_known && self.n < 2) {
            return invalid("the sample size is too small for this plan");
        }
        if p == 0.0 {
            return Ok(1.0);
        }
        if p == 1.0 {
            return Ok(0.0);
        }
        let z = norm_ppf(1.0 - p);
        let rn = (self.n as f64).sqrt();
        if self.sigma_known {
            Ok(norm_cdf(rn * (z - self.k)))
        } else {
            Ok(NonCentralT::new(self.n as f64 - 1.0, rn * z)?.sf(self.k * rn))
        }
    }
}

/// A variables plan found for given requirements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariablesSearch {
    /// The plan.
    pub plan: VariablesPlan,
    /// Probability of acceptance at the AQL (equal to 1 − α).
    pub pa_aql: f64,
    /// Probability of acceptance at the RQL (at most β).
    pub pa_rql: f64,
}

/// Smallest sample size and the constant `k` for `Pa(AQL) = 1 − α` and `Pa(RQL) ≤ β`.
pub fn find_variables_plan(aql: f64, rql: f64, alpha: f64, beta: f64, sigma_known: bool) -> Result<VariablesSearch> {
    check_risk("the acceptable quality level (AQL)", aql)?;
    check_risk("the rejectable quality level (RQL)", rql)?;
    check_risk("the producer's risk", alpha)?;
    check_risk("the consumer's risk", beta)?;
    if aql >= rql {
        return invalid("the AQL must be smaller than the RQL");
    }
    let (z1, z2) = (norm_ppf(1.0 - aql), norm_ppf(1.0 - rql));
    let (za, zb) = (norm_ppf(1.0 - alpha), norm_ppf(1.0 - beta));
    let n_known = (((za + zb) / (z1 - z2)).powi(2)).ceil().max(1.0) as u64;
    if sigma_known {
        let k = z1 - za / (n_known as f64).sqrt();
        let plan = VariablesPlan { n: n_known, k, sigma_known: true };
        return Ok(VariablesSearch {
            plan,
            pa_aql: plan.accept_probability(aql)?,
            pa_rql: plan.accept_probability(rql)?,
        });
    }
    let start = n_known.max(2);
    for n in start..start + 100_000 {
        let rn = (n as f64).sqrt();
        let t = NonCentralT::new(n as f64 - 1.0, rn * z1)?.quantile(alpha);
        let pa_rql = NonCentralT::new(n as f64 - 1.0, rn * z2)?.sf(t);
        if pa_rql <= beta {
            let plan = VariablesPlan { n, k: t / rn, sigma_known: false };
            return Ok(VariablesSearch { plan, pa_aql: 1.0 - alpha, pa_rql });
        }
    }
    Err(StatsError::InvalidArgument("no variables plan found within the searched sample sizes".into()))
}

/// The accept/reject decision of a variables plan for observed data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariablesDecision {
    /// Number of observations used.
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// σ used (known value or the sample standard deviation).
    pub sigma: f64,
    /// The quality index `(U − x̄)/σ` or `(x̄ − L)/σ`.
    pub index: f64,
    /// The plan's constant.
    pub k: f64,
    /// Whether the lot is accepted.
    pub accept: bool,
}

/// Apply a variables plan to measurements (missing values ignored).
pub fn decide_variables(
    plan: &VariablesPlan,
    data: &[f64],
    limit: f64,
    side: LimitSide,
    known_sigma: Option<f64>,
) -> Result<VariablesDecision> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.len() < 2 {
        return Err(StatsError::InsufficientData { what: "acceptance decision", needed: 2, got: v.len() });
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sigma = if plan.sigma_known {
        match known_sigma {
            Some(s) if s > 0.0 => s,
            _ => return invalid("this plan assumes a known σ; give a positive standard deviation"),
        }
    } else {
        let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
        (ss / (n - 1.0)).sqrt()
    };
    if sigma == 0.0 {
        return invalid("the sample standard deviation is zero");
    }
    let index = match side {
        LimitSide::Upper => (limit - mean) / sigma,
        LimitSide::Lower => (mean - limit) / sigma,
    };
    Ok(VariablesDecision { n: v.len(), mean, sigma, index, k: plan.k, accept: index >= plan.k })
}

fn oc_chart(title: &str, xlabel: &str, ylabel: &str, series: Vec<Series>, refs: Vec<ReferenceLine>) -> ChartSpec {
    ChartSpec {
        kind: ChartKind::Line,
        title: title.into(),
        x_label: xlabel.into(),
        y_label: ylabel.into(),
        categories: Vec::new(),
        series,
        reference_lines: refs,
    }
}

fn level_refs(aql: Option<f64>, rql: Option<f64>) -> Vec<ReferenceLine> {
    let mut v = Vec::new();
    if let Some(a) = aql {
        v.push(ReferenceLine { axis: Axis::X, value: a, label: "AQL".into() });
    }
    if let Some(r) = rql {
        v.push(ReferenceLine { axis: Axis::X, value: r, label: "RQL".into() });
    }
    v
}

/// Report for one or more single attribute plans (compare plans by passing several).
pub fn attribute_report(
    plans: &[(String, AttributePlan)],
    model: LotModel,
    lot: Option<u64>,
    aql: Option<f64>,
    rql: Option<f64>,
) -> Result<Section> {
    if plans.is_empty() {
        return invalid("there are no sampling plans");
    }
    let mut sec = Section::new("Acceptance Sampling by Attributes");
    let lot_text = lot.map_or("not given".to_string(), |n| n.to_string());
    sec.text(format!("Measurement unit: proportion defective. Distribution: {}. Lot size: {lot_text}.", model.name()));
    let mut rows = Vec::new();
    for (name, p) in plans {
        rows.push(vec![Cell::text(name), Cell::Int(p.n as i64), Cell::Int(p.c as i64), Cell::Int(p.c as i64 + 1)]);
    }
    sec.table(table("Sampling Plan", &[("Plan", T), ("Sample Size", I), ("Accept", I), ("Reject", I)], rows));
    let levels: Vec<(&str, f64)> = aql.map(|a| ("AQL", a)).into_iter().chain(rql.map(|r| ("RQL", r))).collect();
    if !levels.is_empty() {
        let mut rows = Vec::new();
        for (name, p) in plans {
            for (lab, q) in &levels {
                let pa = p.accept_probability(model, lot, *q)?;
                let ati = p.ati(model, lot, *q)?;
                rows.push(vec![
                    Cell::text(name),
                    Cell::text(*lab),
                    Cell::Num(*q),
                    Cell::Num(pa),
                    Cell::Num(1.0 - pa),
                    ati.map_or(Cell::Missing, Cell::Num),
                ]);
            }
        }
        sec.table(table(
            "Accept / Reject at the Quality Levels",
            &[
                ("Plan", T),
                ("Level", T),
                ("Proportion Defective", S),
                ("Probability Accepting", S),
                ("Probability Rejecting", S),
                ("ATI", S),
            ],
            rows,
        ));
    }
    let mut oc = Vec::new();
    let mut aoq = Vec::new();
    let mut ati = Vec::new();
    let mut summary = Vec::new();
    for (name, p) in plans {
        let grid = default_grid(p, model, lot, 101)?;
        let c = attribute_curve(p, model, lot, &grid)?;
        let (aoql_v, at) = aoql(p, model, lot)?;
        summary.push(vec![Cell::text(name), Cell::Num(aoql_v), Cell::Num(at)]);
        oc.push(Series { name: name.clone(), x: c.p.clone(), y: c.pa.clone() });
        aoq.push(Series { name: name.clone(), x: c.p.clone(), y: c.aoq.clone() });
        if !c.ati.is_empty() {
            ati.push(Series { name: name.clone(), x: c.p.clone(), y: c.ati.clone() });
        }
    }
    let first = &plans[0].1;
    let grid = default_grid(first, model, lot, 11)?;
    let c = attribute_curve(first, model, lot, &grid)?;
    let mut rows = Vec::new();
    for i in 0..grid.len() {
        rows.push(vec![
            Cell::Num(c.p[i]),
            Cell::Num(c.pa[i]),
            Cell::Num(c.aoq[i]),
            c.ati.get(i).map_or(Cell::Missing, |v| Cell::Num(*v)),
        ]);
    }
    sec.table(table(
        &format!("Operating Characteristics of {}", plans[0].0),
        &[("Proportion Defective", S), ("Probability Accepting", S), ("AOQ", S), ("ATI", S)],
        rows,
    ));
    sec.table(table(
        "Average Outgoing Quality Limit",
        &[("Plan", T), ("AOQL", S), ("At Proportion Defective", S)],
        summary,
    ));
    let refs = level_refs(aql, rql);
    sec.chart(oc_chart("OC Curve", "Proportion Defective", "Probability of Acceptance", oc, refs.clone()));
    sec.chart(oc_chart("AOQ Curve", "Proportion Defective", "Average Outgoing Quality", aoq, refs.clone()));
    if !ati.is_empty() {
        sec.chart(oc_chart("ATI Curve", "Proportion Defective", "Average Total Inspection", ati, refs));
    }
    Ok(sec)
}

/// Report for a double sampling plan.
pub fn double_report(
    plan: &DoublePlan,
    model: LotModel,
    lot: Option<u64>,
    aql: Option<f64>,
    rql: Option<f64>,
) -> Result<Section> {
    plan.validate()?;
    let mut sec = Section::new("Double Acceptance Sampling by Attributes");
    let lot_text = lot.map_or("not given".to_string(), |n| n.to_string());
    sec.text(format!("Distribution: {}. Lot size: {lot_text}.", model.name()));
    sec.table(table(
        "Sampling Plan",
        &[("Sample", T), ("Sample Size", I), ("Accept", I), ("Reject", I)],
        vec![
            vec![Cell::text("First"), Cell::Int(plan.n1 as i64), Cell::Int(plan.c1 as i64), Cell::Int(plan.r1 as i64)],
            vec![
                Cell::text("Second (cumulative defectives)"),
                Cell::Int(plan.n2 as i64),
                Cell::Int(plan.c2 as i64),
                Cell::Int(plan.c2 as i64 + 1),
            ],
        ],
    ));
    let pmax = {
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if plan.outcome(model, mid)?.pa > 0.005 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let grid: Vec<f64> = (0..101).map(|i| pmax * i as f64 / 100.0).collect();
    let (mut pa, mut asn, mut aoq, mut ati, mut p1) = (vec![], vec![], vec![], vec![], vec![]);
    for &p in &grid {
        let o = plan.outcome(model, p)?;
        pa.push(o.pa);
        p1.push(o.accept_first);
        asn.push(o.asn);
        aoq.push(plan.aoq(model, lot, p)?);
        if let Some(a) = plan.ati(model, lot, p)? {
            ati.push(a);
        }
    }
    let mut rows = Vec::new();
    for i in (0..grid.len()).step_by(10) {
        rows.push(vec![
            Cell::Num(grid[i]),
            Cell::Num(p1[i]),
            Cell::Num(pa[i] - p1[i]),
            Cell::Num(pa[i]),
            Cell::Num(asn[i]),
            Cell::Num(aoq[i]),
            ati.get(i).map_or(Cell::Missing, |v| Cell::Num(*v)),
        ]);
    }
    sec.table(table(
        "Operating Characteristics",
        &[
            ("Proportion Defective", S),
            ("Accept on First", S),
            ("Accept on Second", S),
            ("Probability Accepting", S),
            ("ASN", S),
            ("AOQ", S),
            ("ATI", S),
        ],
        rows,
    ));
    let levels: Vec<(&str, f64)> = aql.map(|a| ("AQL", a)).into_iter().chain(rql.map(|r| ("RQL", r))).collect();
    if !levels.is_empty() {
        let mut rows = Vec::new();
        for (lab, q) in &levels {
            let o = plan.outcome(model, *q)?;
            rows.push(vec![Cell::text(*lab), Cell::Num(*q), Cell::Num(o.pa), Cell::Num(1.0 - o.pa), Cell::Num(o.asn)]);
        }
        sec.table(table(
            "Accept / Reject at the Quality Levels",
            &[
                ("Level", T),
                ("Proportion Defective", S),
                ("Probability Accepting", S),
                ("Probability Rejecting", S),
                ("ASN", S),
            ],
            rows,
        ));
    }
    let refs = level_refs(aql, rql);
    sec.chart(oc_chart(
        "OC Curve",
        "Proportion Defective",
        "Probability of Acceptance",
        vec![
            Series { name: "Total".into(), x: grid.clone(), y: pa },
            Series { name: "First sample".into(), x: grid.clone(), y: p1 },
        ],
        refs.clone(),
    ));
    sec.chart(oc_chart(
        "Average Sample Number",
        "Proportion Defective",
        "ASN",
        vec![Series { name: "ASN".into(), x: grid.clone(), y: asn }],
        refs.clone(),
    ));
    sec.chart(oc_chart(
        "AOQ Curve",
        "Proportion Defective",
        "Average Outgoing Quality",
        vec![Series { name: "AOQ".into(), x: grid.clone(), y: aoq }],
        refs.clone(),
    ));
    if !ati.is_empty() {
        sec.chart(oc_chart(
            "ATI Curve",
            "Proportion Defective",
            "Average Total Inspection",
            vec![Series { name: "ATI".into(), x: grid, y: ati }],
            refs,
        ));
    }
    Ok(sec)
}

/// Report for one or more variables plans.
pub fn variables_report(
    plans: &[(String, VariablesPlan)],
    side: LimitSide,
    aql: Option<f64>,
    rql: Option<f64>,
) -> Result<Section> {
    if plans.is_empty() {
        return invalid("there are no sampling plans");
    }
    let mut sec = Section::new("Acceptance Sampling by Variables");
    sec.text(format!(
        "One {} specification limit; the lot is accepted when the quality index is at least k.",
        match side {
            LimitSide::Upper => "upper",
            LimitSide::Lower => "lower",
        }
    ));
    let rows: Vec<Vec<Cell>> = plans
        .iter()
        .map(|(name, p)| {
            vec![
                Cell::text(name),
                Cell::Int(p.n as i64),
                Cell::Num(p.k),
                Cell::text(if p.sigma_known { "Known (σ-method)" } else { "Unknown (s-method)" }),
            ]
        })
        .collect();
    sec.table(table("Sampling Plan", &[("Plan", T), ("Sample Size", I), ("k", S), ("Standard Deviation", T)], rows));
    let levels: Vec<(&str, f64)> = aql.map(|a| ("AQL", a)).into_iter().chain(rql.map(|r| ("RQL", r))).collect();
    if !levels.is_empty() {
        let mut rows = Vec::new();
        for (name, p) in plans {
            for (lab, q) in &levels {
                let pa = p.accept_probability(*q)?;
                rows.push(vec![Cell::text(name), Cell::text(*lab), Cell::Num(*q), Cell::Num(pa), Cell::Num(1.0 - pa)]);
            }
        }
        sec.table(table(
            "Accept / Reject at the Quality Levels",
            &[
                ("Plan", T),
                ("Level", T),
                ("Proportion Defective", S),
                ("Probability Accepting", S),
                ("Probability Rejecting", S),
            ],
            rows,
        ));
    }
    let hi = levels.iter().map(|l| l.1).fold(0.0_f64, f64::max);
    let pmax = if hi > 0.0 { (hi * 2.5).min(1.0) } else { 0.1 };
    let grid: Vec<f64> = (0..101).map(|i| pmax * i as f64 / 100.0).collect();
    let mut series = Vec::new();
    for (name, p) in plans {
        let pa: Result<Vec<f64>> = grid.iter().map(|&x| p.accept_probability(x)).collect();
        series.push(Series { name: name.clone(), x: grid.clone(), y: pa? });
    }
    sec.chart(oc_chart(
        "OC Curve",
        "Proportion Nonconforming",
        "Probability of Acceptance",
        series,
        level_refs(aql, rql),
    ));
    Ok(sec)
}

/// Decision report for an attribute plan.
pub fn attribute_decision_section(plan: &AttributePlan, defectives: u64) -> Section {
    let mut sec = Section::new("Accept/Reject Based on Attributes");
    let accept = defectives <= plan.c;
    sec.table(table(
        "Decision",
        &[("Sample Size", I), ("Acceptance Number", I), ("Defectives Found", I), ("Decision", T)],
        vec![vec![
            Cell::Int(plan.n as i64),
            Cell::Int(plan.c as i64),
            Cell::Int(defectives as i64),
            Cell::text(if accept { "Accept lot" } else { "Reject lot" }),
        ]],
    ));
    sec
}

/// Decision report for a variables plan.
pub fn variables_decision_section(d: &VariablesDecision, side: LimitSide, limit: f64) -> Section {
    let mut sec = Section::new("Accept/Reject Based on Variables");
    sec.table(table(
        "Decision",
        &[
            ("N", I),
            ("Mean", S),
            ("Standard Deviation", S),
            ("Limit", S),
            ("Quality Index", S),
            ("k", S),
            ("Decision", T),
        ],
        vec![vec![
            Cell::Int(d.n as i64),
            Cell::Num(d.mean),
            Cell::Num(d.sigma),
            Cell::Num(limit),
            Cell::Num(d.index),
            Cell::Num(d.k),
            Cell::text(if d.accept { "Accept lot" } else { "Reject lot" }),
        ]],
    ));
    sec.text(match side {
        LimitSide::Upper => "Index = (upper limit − mean) / standard deviation.",
        LimitSide::Lower => "Index = (mean − lower limit) / standard deviation.",
    });
    sec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_oc_closed_form() {
        let plan = AttributePlan { n: 5, c: 1 };
        let p = 0.1_f64;
        let pa = plan.accept_probability(LotModel::Binomial, None, p).unwrap();
        let want = 0.9_f64.powi(5) + 5.0 * 0.1 * 0.9_f64.powi(4);
        assert!((pa - want).abs() < 1e-14);
    }

    #[test]
    fn double_plan_probabilities_add_up() {
        let plan = DoublePlan { n1: 50, c1: 1, r1: 4, n2: 100, c2: 4 };
        let o = plan.outcome(LotModel::Binomial, 0.02).unwrap();
        assert!((o.pa - (o.accept_first + o.accept_second)).abs() < 1e-15);
        assert!(o.second_sample > 0.0 && o.pa <= 1.0);
        assert!((o.asn - (50.0 + 100.0 * o.second_sample)).abs() < 1e-12);
    }

    #[test]
    fn variables_plan_meets_both_points() {
        for known in [true, false] {
            let s = find_variables_plan(0.01, 0.08, 0.05, 0.10, known).unwrap();
            assert!((s.plan.accept_probability(0.01).unwrap() - 0.95).abs() < 1e-8);
            assert!(s.plan.accept_probability(0.08).unwrap() <= 0.10 + 1e-12);
        }
    }

    #[test]
    fn rejects_bad_requirements() {
        assert!(find_attribute_plan(0.05, 0.01, 0.05, 0.1, LotModel::Binomial, None).is_err());
        assert!(find_attribute_plan(0.01, 0.05, 0.0, 0.1, LotModel::Binomial, None).is_err());
        assert!(DoublePlan { n1: 10, c1: 2, r1: 2, n2: 10, c2: 3 }.outcome(LotModel::Binomial, 0.1).is_err());
    }
}
