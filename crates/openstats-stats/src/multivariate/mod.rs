//! Multivariate analysis — Minitab's Stat > Multivariate.
//!
//! * [`pca`] — principal components of the correlation or covariance matrix, with scores.
//! * [`factor_analysis`] — principal-component and maximum-likelihood factor analysis, orthomax
//!   rotations (varimax, quartimax, equamax, general γ) and regression factor scores.
//! * [`discriminant`] — linear and quadratic discriminant analysis with priors, resubstitution and
//!   leave-one-out classification, and canonical discriminant functions.
//! * [`hierarchical`] / [`cluster_variables`] / [`kmeans`] — agglomerative clustering of observations
//!   or variables (Lance–Williams updates) and k-means with optimal-transfer refinement.
//! * [`manova`] — general MANOVA by formula with Wilks, Pillai, Lawley–Hotelling and Roy statistics.
//! * [`cronbach_alpha`] — item analysis.
//!
//! Variables are passed as columns (`&[Vec<f64>]`, one `Vec` per variable); missing values are `NaN`
//! and rows with any missing value are dropped (listwise). Every result records the worksheet rows it
//! used so callers can store scores and memberships back in the right places.
// Negated comparisons are deliberate: they reject NaN as well as out-of-range values.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod cluster;
mod common;
mod correspondence;
mod discriminant;
mod factor;
mod item;
mod manova;
mod pca;
mod report_impl;

pub use cluster::{
    cluster_variables, hierarchical, kmeans, Distance, KMeans, KMeansInit, KMeansOptions, Linkage, LinkageMethod,
    MergeStep, PartitionSummary,
};
pub use common::{complete_rows, MatrixKind};
pub use correspondence::{correspondence_analysis, multiple_correspondence, promax, CorrespondenceAnalysis, Promax};
pub use discriminant::{discriminant, CanonicalFunctions, Discriminant, DiscriminantOptions, Priors};
pub use factor::{factor_analysis, FactorAnalysis, FactorMethod, FactorOptions, Rotation};
pub use item::{cronbach_alpha, ItemAnalysis};
pub use manova::{manova, Manova, ManovaOptions, ManovaTerm, MultivariateTest};
pub use pca::{pca, Pca, PcaOptions};
pub use report_impl::partition_section;
