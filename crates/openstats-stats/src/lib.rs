//! OpenStats statistical procedures (Phases 2a–2k).
//!
//! * [`descriptive`] — Minitab "Display Descriptive Statistics".
//! * [`quantile`] — the nine Hyndman–Fan quantile definitions (Minitab uses
//!   type 6).
//! * [`graphical`] — data behind Minitab's "Graphical Summary": binning,
//!   histograms, ECDF, box-plot statistics, probability-plot positions,
//!   confidence intervals and the Anderson–Darling normality test.
//! * [`crosstab`] — cross-tabulation with chi-square, likelihood-ratio,
//!   Mantel–Haenszel, Fisher's exact and McNemar tests.
//! * [`correlation`] — covariance, Pearson, Spearman, Kendall τ-b/τ-c and
//!   partial correlation.
//! * [`ttest`] — one-sample, paired and two-sample (pooled / Welch) t-tests
//!   and the one-sample Z-test.
//! * [`proportion`], [`poisson`] — one- and two-sample proportion tests
//!   (exact and normal) and Poisson rate tests.
//! * [`variance`] — F-test, Levene, Brown–Forsythe and Bonett.
//! * [`normality`] — Shapiro–Wilk, Ryan–Joiner, Kolmogorov–Smirnov, Lilliefors.
//! * [`outlier`] — Grubbs, generalized ESD (multiple outliers), Dixon's Q.
//! * [`equivalence`] — TOST for one-sample, paired and two-sample means.
//! * [`report`], [`reportable`] — the unified, serialisable [`report::SessionReport`]
//!   (sections, aligned tables, footnotes, warnings, chart specs; text and JSON
//!   renderers) and `Reportable` implementations for every result type above.
//! * [`spc`] — quality and statistical process control: unbiasing constants
//!   (d2, d3, c4, A2, A3, B3–B6, D1–D4, n = 2–25), I-MR / X̄-R / X̄-S and
//!   p / np / c / u control charts with Minitab tests 1–8, normal and
//!   non-normal (Box-Cox, Johnson, Weibull/lognormal/gamma) capability analysis
//!   and the capability sixpack; Western Electric / Nelson rule sets, staged
//!   limits, I-MR-R/S, zone, Z-MR and Box-Cox charts, Laney P′/U′, G and T
//!   charts, moving-average / EWMA / CUSUM (tabular and V-mask) charts and
//!   Hotelling T² charts; exact binomial/Poisson limits, one-sided CUSUM, T² variable
//!   decomposition and the generalized-variance chart.
//! * [`anova`] — one-way ANOVA (Welch, R², PRESS), Tukey / Fisher / Dunnett / Hsu MCB
//!   comparisons with grouping letters, and the general linear model with
//!   Type I and Type III sums of squares, least-squares means and plot data.
//! * [`capability`] — process capability and performance: the normal indices,
//!   non-normal capability (Box-Cox, Johnson, fitted Weibull / lognormal / gamma /
//!   exponential / normal / logistic / loglogistic, empirical percentiles),
//!   confidence intervals (χ², exact non-central t), distribution identification
//!   and the extended Capability Sixpack.
//! * [`msa`] — Gage R&R: crossed (ANOVA and X̄-R methods) and nested studies,
//!   variance components, %Study Var, %Tolerance, ndc and the six-panel charts;
//!   gage linearity and bias; attribute agreement (Fleiss' and Cohen's κ,
//!   effectiveness, miss and false-alarm rates).
//! * [`power`] — power and sample-size solvers (any one of n, power, effect)
//!   for Z/t/paired/2-sample t, proportions, Poisson rates, variances, ANOVA,
//!   2-level factorial and Plackett–Burman designs, and TOST equivalence.
//! * [`tolerance`] — normal (exact and Howe) and nonparametric tolerance intervals.
//! * [`model`] — Minitab/R-style model formulas (`A*B`, `A(B)`, `I(x^2)`,
//!   `poly`, `offset`, intercept control) and the design-matrix builder
//!   (treatment, sum-to-zero, Helmert and orthogonal-polynomial coding).
//! * [`regression`] — OLS by Householder QR with iterative refinement:
//!   coefficient and ANOVA tables, R²/adjusted/predicted R², residual
//!   diagnostics (leverage, Cook's D, DFFITS, VIF, Durbin–Watson), lack of
//!   fit, prediction intervals, stepwise selection and best subsets
//!   (Furnival–Wilson leaps and bounds); weighted least squares.
//! * [`glm`] — generalized linear models by IRLS (binomial/logistic with logit, probit and
//!   complementary log-log links, Poisson, negative binomial, gamma, Gaussian): deviance table,
//!   Hosmer–Lemeshow, classification table, ROC, case diagnostics, term-level stepwise; ordinal
//!   (proportional odds) and nominal logistic regression.
//! * [`nonlinear`], [`pls`], [`orthogonal`] — nonlinear least squares of user equations
//!   (bounds, standard errors, confidence and prediction limits), partial least squares with
//!   cross-validation and VIP, and orthogonal (Deming) regression.
//! * [`timeseries`] — autocorrelation (ACF with Bartlett limits, PACF, Ljung–Box), single /
//!   double / Winters exponential smoothing, seasonal ARIMA by conditional least squares or
//!   exact Kalman-filter maximum likelihood, and trend–seasonal–irregular decomposition, each
//!   with forecasts and prediction intervals.
//! * [`doe`] — design of experiments: 2-level full and fractional factorial
//!   designs (generator selection, defining relation, resolution, alias
//!   structure, blocking, centre points, replicates), Plackett–Burman designs
//!   from Hadamard matrices, central composite (circumscribed, inscribed,
//!   face-centred) and Box–Behnken designs, Taguchi orthogonal arrays with
//!   S/N ratios and ANOVA, factorial and response-surface analysis (effects,
//!   ANOVA, Lenth's method, lack of fit, canonical analysis), Derringer–Suich
//!   desirability optimisation and the chart data (main effects, interaction,
//!   cube, Pareto, normal, contour, surface).
//!
//! * [`ml`] — tree-based machine learning: CART with cost-complexity pruning, cross-validation and
//!   surrogate splits, random forests (out-of-bag error, permutation importance, quantile-forest
//!   intervals) and gradient boosted trees (squared, Huber, binomial and multinomial deviance).
//!
//! * [`nonparametric`] — sign, Wilcoxon signed-rank (with Hodges–Lehmann estimate and interval),
//!   Mann–Whitney, Kruskal–Wallis, Mood's median, Friedman and runs tests, with exact p-values where
//!   the null distribution is tractable.
//!
//! Numeric missing values are `NaN` (displayed `*` by the worksheet), as in
//! `openstats-core`. Every function either drops missing values as documented
//! or reports an error; none panics on user data.

pub mod acceptance;
pub mod anova;
pub mod attrcap;
pub mod bootstrap;
pub mod capability;
pub mod correlation;
pub mod crosstab;
pub mod descriptive;
pub mod doe;
pub mod eda;
pub mod equivalence;
pub mod error;
pub mod glm;
pub mod graphical;
pub mod hypothesis;
pub mod mixed;
pub mod ml;
pub mod model;
pub mod msa;
pub mod multivari;
pub mod multivariate;
pub mod nonlinear;
pub mod nonparametric;
pub mod normality;
pub mod orthogonal;
pub mod outlier;
pub mod parallel;
pub mod pls;
pub mod poisson;
pub mod power;
pub mod proportion;
pub mod quantile;
pub mod regression;
pub mod reliability;
pub mod report;
pub mod reportable;
pub mod runchart;
pub mod spc;
pub mod stemleaf;
pub mod timeseries;
pub mod tolerance;
pub mod ttest;
pub mod variance;

pub use error::{Result, StatsError};

/// Non-missing values of `x`; errors on ±∞ (which is not a missing value).
pub(crate) fn clean(x: &[f64]) -> Result<Vec<f64>> {
    let mut v = Vec::with_capacity(x.len());
    for &a in x {
        if a.is_nan() {
            continue;
        }
        if a.is_infinite() {
            return Err(StatsError::NonFinite);
        }
        v.push(a);
    }
    Ok(v)
}

/// Sorted non-missing values.
pub(crate) fn clean_sorted(x: &[f64]) -> Result<Vec<f64>> {
    let mut v = clean(x)?;
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(v)
}
