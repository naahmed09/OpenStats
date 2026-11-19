//! Linear regression: OLS with a full coefficient / ANOVA table and
//! diagnostics, stepwise selection and best-subsets regression.

mod diagnostics;
mod fit;

pub use diagnostics::{Diagnostics, FourInOne, ResidualKind, UnusualFlag, UnusualObservation};
pub(crate) use fit::independent_columns;
pub use fit::{
    fit_design, fit_weighted, fit_xy, regress, regress_weighted, AnovaRow, Coefficient, LackOfFit, Prediction,
    RegressionFit, RegressionOptions,
};
mod report_impl;
mod selection;
mod subsets;

pub use selection::{
    stepwise, stepwise_design, Step, StepAction, StepCriterion, StepMethod, StepwiseOptions, StepwiseResult,
};
pub use subsets::{best_subsets, BestSubsets, SubsetModel, SubsetOptions};
