//! Quality tools: control-chart constants, control charts with special-cause
//! tests, and process capability.
//!
//! * [`charts`], [`rules`] — Shewhart charts (I-MR, X̄-R, X̄-S, p, np, c, u) and the
//!   Nelson / Western Electric rule sets.
//! * [`stages`] — staged limits for the Shewhart, Laney, G/T, time-weighted and T² charts.
//! * [`variables_ext`] — I-MR-R/S (between/within), zone, Z-MR (short run) and Box-Cox charts.
//! * [`attributes`] — exact binomial/Poisson limits for p/np/c/u, Laney P′/U′ and the rare-event G and T charts.
//! * [`time_weighted`] — moving-average, EWMA and CUSUM (tabular, one- or two-sided, FIR; V-mask) charts.
//! * [`multivariate`] — Hotelling T² charts with variable decomposition, and the generalized-variance chart.
//! * [`capability`], [`sixpack`], [`transform`] — capability analysis.

pub mod attributes;
pub mod capability;
pub mod charts;
pub mod constants;
pub mod multivariate;
mod report_ext;
mod report_impl;
pub mod rules;
pub mod sixpack;
pub mod stages;
pub mod time_weighted;
pub mod transform;
pub mod variables_ext;
pub use report_impl::constants_section;
