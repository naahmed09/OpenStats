//! Measurement systems analysis: crossed Gage R&R (ANOVA and X̄-R methods),
//! nested Gage R&R for destructive testing, and the six-panel chart data.
//!
//! Variance components follow the Minitab / AIAG conventions: a *Total Gage
//! R&R* of repeatability and reproducibility, *Part-to-Part* variation, the
//! study variation `k·σ` (default `k = 6`), its share of the total study
//! variation, its share of a tolerance, and the number of distinct
//! categories `ndc = max(1, ⌊1.41·σ_part/σ_gage⌋)`.
//!
//! [`linearity`] covers gage linearity and bias and [`attribute`] attribute
//! agreement analysis (Fleiss' and Cohen's κ, effectiveness, miss and false-alarm rates).

pub mod attribute;
mod charts;
mod crossed;
pub mod linearity;
mod nested;
mod report_impl;
mod types;

pub use attribute::{AttributeAgreement, AttributeOptions, AttributeStudy};
pub use crossed::CrossedStudy;
pub use linearity::{LinearityBias, LinearityBiasOptions, LinearityBiasStudy};
pub use nested::NestedStudy;
pub use types::{
    AnovaRow, GageMethod, GageOptions, GageResult, Interaction, Source, Tolerance, VarianceRow, XbarRDetail,
};
