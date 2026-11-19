//! Reliability and survival analysis.
//!
//! * [`LifeData`] — exact, right-, left- and interval-censored observations with frequencies.
//! * [`fit_distribution`] — maximum-likelihood fits of the Weibull, lognormal, exponential,
//!   normal, logistic and log-logistic families, with percentiles, reliabilities and mean life
//!   and normal-approximation confidence intervals.
//! * [`kaplan_meier`] — the Kaplan–Meier estimate with Greenwood intervals, and log-rank,
//!   Wilcoxon (Breslow) and Tarone–Ware comparisons of groups.
//! * [`fit_alt`] — accelerated life testing: a location–scale regression on transformed stress.
//! * [`warranty_forecast`] — returns forecasts from shipment and return counts.

// Negated comparisons are deliberate: they reject NaN as well as out-of-range values.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod alt;
pub mod cox;
pub mod data;
pub mod dist;
pub mod extra;
pub mod fit;
pub mod km;
pub mod repairable;
pub mod report_impl;
pub mod testplan;
pub mod warranty;

pub use alt::{fit_alt, AltFit, Relationship, StressVariable};
pub use cox::{cox_formula, cox_ph, CoxCoefficient, CoxFit, CoxOptions, CoxTies};
pub use data::{LifeData, Obs, ObsKind};
pub use dist::{Family, FAMILIES};
pub use extra::{fit_eyring, fit_gamma, fit_threshold, turnbull, EyringFit, GammaFit, ThresholdFit, Turnbull};
pub use fit::{fit_distribution, obs_loglik, rank_distributions, Estimate, LifeFit};
pub use km::{compare_groups, kaplan_meier, CiMethod, GroupTest, KaplanMeier, KmStep, WeightKind};
pub use repairable::{
    crow_amsaa, mean_cumulative_function, systems_from_rows, CrowAmsaa, McfPoint, System, Termination,
};
pub use testplan::{acceptance_probability, plan_sample_size, plan_test_time, DemoPlan};
pub use warranty::{warranty_forecast, WarrantyForecast, WarrantyInput};
