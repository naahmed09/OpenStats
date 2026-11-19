//! Process capability and performance.
//!
//! The normal indices (Cp, Cpl, Cpu, Cpk, Cpm, Pp, Ppl, Ppu, Ppk), capability
//! after a Box–Cox or Johnson transformation, capability from a fitted
//! distribution and the Capability Sixpack live in [`crate::spc::capability`] and
//! [`crate::spc::sixpack`]; they are re-exported here. This module adds
//!
//! * [`ci`]: confidence intervals for every index — χ² for Cp / Pp, exact
//!   non-central-t limits for the one-sided indices Ppl / Ppu (and for Cpl / Cpu
//!   when the within estimate is χ²-distributed), Bissell's normal
//!   approximation for Cpk / Ppk and Boyles' χ² approximation for Cpm;
//! * [`empirical`]: non-normal capability from empirical percentiles
//!   (`Pp = (USL − LSL)/(P99.865 − P0.135)`, median in place of the mean);
//! * [`identify`]: individual-distribution identification over the normal,
//!   lognormal, exponential, Weibull, gamma, logistic and loglogistic families
//!   and the Johnson system, ranked by the Anderson–Darling statistic;
//! * [`extras`]: capability by subgroup, the running-average chart and the
//!   extended sixpack bundle.

pub mod ci;
pub mod empirical;
pub mod extras;
pub mod identify;
mod report_impl;

pub use report_impl::{intervals_section, ranking_section};

pub use crate::spc::capability::*;
pub use crate::spc::sixpack::{
    capability_sixpack_individuals, capability_sixpack_subgroups, CapabilitySixpack, SixpackOptions,
};
pub use ci::{capability_intervals, pooled_within_df, CapabilityIntervals};
pub use empirical::{capability_empirical, EmpiricalCapability, EmpiricalOptions};
pub use extras::{
    capability_by_subgroup, extended_sixpack_individuals, extended_sixpack_subgroups, running_average, ExtendedSixpack,
    RunningAverage, SubgroupCapability,
};
pub use identify::{identify_distributions, DistributionRank};
