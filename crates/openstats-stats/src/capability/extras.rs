//! Capability by subgroup, the running-average chart and the extended
//! Capability Sixpack bundle.
//!
//! * **Capability by subgroup**: for every subgroup with its own mean `x̄ᵢ` and
//!   standard deviation `sᵢ`, `Ppkᵢ = min(USL − x̄ᵢ, x̄ᵢ − LSL)/(3sᵢ)`; it shows
//!   how the capability moves over time.
//! * **Running average**: the cumulative mean of all observations after each
//!   subgroup, with limits `x̿ ± 3σ_within/√Nᵢ` around the overall mean, `Nᵢ`
//!   being the observations accumulated so far. (For a stable process it
//!   stays inside the limits; the limits narrow as data accumulate.)

use crate::capability::ci::pooled_within_df;
use crate::error::{invalid, Result};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use crate::spc::capability::{NormalCapability, Specs};
use crate::spc::charts::subgroup_stats;
use crate::spc::sixpack::{
    capability_sixpack_individuals, capability_sixpack_subgroups, CapabilitySixpack, SixpackOptions,
};

/// Capability of one subgroup against the specifications.
#[derive(Debug, Clone, PartialEq)]
pub struct SubgroupCapability {
    /// 1-based subgroup number.
    pub index: usize,
    /// Observations in the subgroup.
    pub n: usize,
    /// Subgroup mean.
    pub mean: f64,
    /// Subgroup standard deviation (n − 1).
    pub sd: f64,
    /// `(USL − LSL)/(6 sᵢ)` (`None` for one-sided specifications or `sᵢ = 0`).
    pub pp: Option<f64>,
    /// `min(PPL, PPU)` from the subgroup's own mean and σ (`None` when `sᵢ = 0`).
    pub ppk: Option<f64>,
}

/// Per-subgroup capability, in order.
pub fn capability_by_subgroup(groups: &[Vec<f64>], specs: &Specs) -> Result<Vec<SubgroupCapability>> {
    specs.validate()?;
    let stats = subgroup_stats(groups)?;
    Ok(stats
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let positive = s.sd > 0.0;
            let l = specs.lsl.map(|l| (s.mean - l) / (3.0 * s.sd));
            let u = specs.usl.map(|u| (u - s.mean) / (3.0 * s.sd));
            let ppk = match (l, u) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            };
            let pp = match (specs.lsl, specs.usl) {
                (Some(l), Some(u)) => Some((u - l) / (6.0 * s.sd)),
                _ => None,
            };
            SubgroupCapability {
                index: i + 1,
                n: s.n,
                mean: s.mean,
                sd: s.sd,
                pp: if positive { pp } else { None },
                ppk: if positive { ppk } else { None },
            }
        })
        .collect())
}

/// Cumulative means with their control limits.
#[derive(Debug, Clone, PartialEq)]
pub struct RunningAverage {
    /// Overall mean (centre line).
    pub center: f64,
    /// Within σ used for the limits.
    pub sigma: f64,
    /// Number of σ in the limits (3).
    pub k: f64,
    /// Cumulative number of observations after each subgroup.
    pub count: Vec<usize>,
    /// Cumulative mean after each subgroup.
    pub mean: Vec<f64>,
    /// Lower limit after each subgroup.
    pub lower: Vec<f64>,
    /// Upper limit after each subgroup.
    pub upper: Vec<f64>,
}

/// Running average of subgroups (`groups` of size 1 for individual data).
pub fn running_average(groups: &[Vec<f64>], sigma_within: f64, k: f64) -> Result<RunningAverage> {
    if !(sigma_within > 0.0 && sigma_within.is_finite()) {
        return invalid("the within σ must be positive and finite");
    }
    if !(k > 0.0 && k.is_finite()) {
        return invalid("k must be positive and finite");
    }
    let mut total = 0usize;
    let mut sum = 0.0;
    let mut all = Vec::new();
    for g in groups {
        if g.iter().any(|v| v.is_infinite()) {
            return Err(crate::error::StatsError::NonFinite);
        }
        let v: Vec<f64> = g.iter().copied().filter(|x| !x.is_nan()).collect();
        total += v.len();
        sum += v.iter().sum::<f64>();
        all.push((total, sum));
    }
    if total == 0 {
        return Err(crate::error::StatsError::InsufficientData { what: "running average", needed: 1, got: 0 });
    }
    let center = sum / total as f64;
    let mut out =
        RunningAverage { center, sigma: sigma_within, k, count: vec![], mean: vec![], lower: vec![], upper: vec![] };
    for (cnt, s) in all {
        if cnt == 0 {
            continue;
        }
        let w = k * sigma_within / (cnt as f64).sqrt();
        out.count.push(cnt);
        out.mean.push(s / cnt as f64);
        out.lower.push(center - w);
        out.upper.push(center + w);
    }
    Ok(out)
}

impl RunningAverage {
    /// Chart: the cumulative means with the narrowing limits and the centre line.
    pub fn chart_spec(&self) -> ChartSpec {
        let x: Vec<f64> = self.count.iter().map(|&c| c as f64).collect();
        ChartSpec {
            kind: ChartKind::Line,
            title: "Running Average Chart".into(),
            x_label: "Observations".into(),
            y_label: "Cumulative mean".into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "Running average".into(), x: x.clone(), y: self.mean.clone() },
                Series { name: "UCL".into(), x: x.clone(), y: self.upper.clone() },
                Series { name: "LCL".into(), x, y: self.lower.clone() },
            ],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: self.center, label: "Mean".into() }],
        }
    }
}

/// A Capability Sixpack with the additional panels.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedSixpack {
    /// The six standard panels.
    pub sixpack: CapabilitySixpack,
    /// Capability of each subgroup (blocks for individual data).
    pub by_subgroup: Vec<SubgroupCapability>,
    /// Running average of the observations by subgroup.
    pub running_average: RunningAverage,
    /// Degrees of freedom of the within σ when it is the pooled standard deviation of subgroups.
    pub within_df: Option<f64>,
}

impl ExtendedSixpack {
    /// The six standard panels followed by capability by subgroup and the running average.
    pub fn charts(&self) -> Vec<ChartSpec> {
        let mut v = self.sixpack.charts();
        let x: Vec<f64> = self.by_subgroup.iter().map(|s| s.index as f64).collect();
        let pick = |f: fn(&SubgroupCapability) -> Option<f64>| -> (Vec<f64>, Vec<f64>) {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for (xi, s) in x.iter().zip(&self.by_subgroup) {
                if let Some(y) = f(s) {
                    xs.push(*xi);
                    ys.push(y);
                }
            }
            (xs, ys)
        };
        let (px, py) = pick(|s| s.ppk);
        let mut series = vec![Series { name: "Ppk".into(), x: px, y: py }];
        if self.sixpack.capability.specs.lsl.is_some() && self.sixpack.capability.specs.usl.is_some() {
            let (qx, qy) = pick(|s| s.pp);
            series.push(Series { name: "Pp".into(), x: qx, y: qy });
        }
        v.push(ChartSpec {
            kind: ChartKind::Line,
            title: "Capability by Subgroup".into(),
            x_label: "Subgroup".into(),
            y_label: "Index".into(),
            categories: Vec::new(),
            series,
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 1.33, label: "1.33".into() }],
        });
        v.push(self.running_average.chart_spec());
        v
    }

    /// The capability statistics.
    pub fn capability(&self) -> &NormalCapability {
        &self.sixpack.capability
    }
}

/// Extended sixpack for subgroup data.
pub fn extended_sixpack_subgroups(
    groups: &[Vec<f64>],
    specs: &Specs,
    opts: &SixpackOptions,
) -> Result<ExtendedSixpack> {
    let sixpack = capability_sixpack_subgroups(groups, specs, opts)?;
    let by_subgroup = capability_by_subgroup(groups, specs)?;
    let running = running_average(groups, sixpack.capability.sd_within, 3.0)?;
    let pooled = sixpack.capability.within_method.contains("pooled");
    let within_df = if pooled { Some(pooled_within_df(groups)?) } else { None };
    Ok(ExtendedSixpack { sixpack, by_subgroup, running_average: running, within_df })
}

/// Extended sixpack for individual observations in time order. `block` is the
/// number of consecutive observations that form one "subgroup" in the
/// capability-by-subgroup panel (the running average uses single observations).
pub fn extended_sixpack_individuals(
    data: &[f64],
    specs: &Specs,
    opts: &SixpackOptions,
    block: usize,
) -> Result<ExtendedSixpack> {
    if block < 2 {
        return invalid("the block size for capability by subgroup must be at least 2");
    }
    let sixpack = capability_sixpack_individuals(data, specs, opts)?;
    let valid: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let mut blocks: Vec<Vec<f64>> = valid.chunks(block).map(<[f64]>::to_vec).collect();
    if blocks.last().is_some_and(|b| b.len() < 2) {
        blocks.pop();
    }
    if blocks.is_empty() {
        return invalid("not enough observations to form a block");
    }
    let by_subgroup = capability_by_subgroup(&blocks, specs)?;
    let singles: Vec<Vec<f64>> = valid.iter().map(|&v| vec![v]).collect();
    let running = running_average(&singles, sixpack.capability.sd_within, 3.0)?;
    Ok(ExtendedSixpack { sixpack, by_subgroup, running_average: running, within_df: None })
}
