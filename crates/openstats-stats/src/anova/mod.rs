//! Analysis of variance: one-way ANOVA with Welch's test, multiple
//! comparisons (Tukey, Fisher, Dunnett, Hsu's MCB) and the general linear
//! model with Type I / Type III sums of squares, least-squares means and
//! main-effects / interaction plot data.

pub mod comparisons;
pub mod glm;
pub mod oneway;
pub mod random;
mod report_impl;

pub use comparisons::{
    dunnett, fisher, hsu_mcb, tukey, Alternative, Best, Comparisons, Difference, GroupingRow, McbResult, McbRow, Means,
    Method,
};
pub use glm::{glm, GlmAnova, GlmOptions, LsMean, LsMeansTable, SequentialRow};
pub use oneway::{one_way, one_way_stacked, LevelSummary, OneWayAnova, OneWayOptions, Welch};
pub use random::{
    balanced_anova, glm_random, nested_anova, EmsRow, RandomAnova, RandomOptions, TermMeans, VarianceComponent,
};
