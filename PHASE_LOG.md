# OpenStats — Phase Log

## Decisions (approved 2026-11-16)
- Stop after **every sub-phase** (1a, 1b, …) for review.
- License: `Apache-2.0 OR MIT`.
- NIST StRD: embed the small datasets in tests. A `.dat` parser harness reads the official larger files from `tests/fixtures/nist/` (harness lands in 1g).
- Delivery: a workspace archive per sub-phase. The user runs the Wasm and Tauri builds locally or in CI.

## 1a — Workspace + `openstats-core` (done 2026-11-16)
- Delivered as `openstats-phase1a.zip` (git commit `cfb3d2b`). About 7.2k lines of Rust, std only.
- 72 tests pass (69 unit + 3 integration, including a 600-step randomized undo/redo test and a 200k-row round trip). Clippy `-D warnings` and rustfmt clean.
- Design: edits return their own inverse and `Edit::Batch` is atomic; equality is NaN-aware; missing values sort last in both directions; two-digit years below 30 map to the 2000s; time-only values sit on 1970-01-01; date/time → numeric uses the Excel serial number.
- Not verified here: the `wasm32` compile (target not installed). Minimum Rust version 1.82.

## 1b — Calculator, compression and containers (done 2026-11-16)
- Delivered as `openstats-phase1b.zip` (git commit `b4dd569`). About 14.2k lines total; 118 tests; clippy, rustdoc clean; no stubs.
- Formula engine: precedence-climbing parser, vectorised evaluator, 175 function names in an extensible `FunctionRegistry`, column formulas recalculated in dependency order with cycle detection through undoable edits.
- Semantics: unequal-length columns are an error (scalars broadcast); division by zero and non-finite results give `*`; `AND`/`OR` are three-valued; `ROUND` is decimal half-away-from-zero; `NOW()`/`TODAY()` read the clock from `EvalCtx`.
- Verified against reference tools: DEFLATE/zlib/gzip both directions, ZIP (incl. ZIP64) against Python `zipfile`, XLSX import (openpyxl fixture plus hand-built package).
- `.otb` format v1.1 adds column formulas; v1.0 still loads.
- Minitab `.mpx`/`.mwx` (ZIP) and `.mpj`/`.mtw` (OLE2) containers are inspected only. Data import is not implemented because no spec is published; needs real sample files.

## 1c — `openstats-math` linear algebra (done 2026-11-16)
- Delivered as `openstats-phase1c.zip` (git commit `489ea1e`). About 4.2k new lines, std only; 152 tests across the workspace.
- `Matrix`, cache-blocked GEMM, LU (+rcond), Householder QR (with pivoting), Cholesky/LDLᵀ/modified Cholesky, symmetric eigen (QL and Jacobi), general eigen (Hessenberg + Francis QR), Golub–Kahan–Reinsch SVD, sparse CSR/CSC with minimum-degree Cholesky and Jacobi-PCG.
- Verified by property tests, analytic spectra and NumPy/LAPACK references (about 1e-13). Factorisations never panic on singular input; iterative methods cap iterations and return `NoConvergence`.

## 1d — Special functions + continuous distributions (done 2026-11-16)
- Delivered as `openstats-phase1d.zip` (git commit `6746a04`). 181 tests; clippy, rustfmt, rustdoc, `--no-default-features` clean.
- Γ family, erf family and accurate normal tails, incomplete gamma/beta with inverses, Bessel J/Y/I/K, Studentized range, Dunnett; Gauss–Legendre, adaptive quadrature, Brent.
- 22 continuous distributions behind `ContinuousDistribution`; calculator NSCORES, PDF, CDF, SFCDF, INVCDF.
- 2,268 SciPy reference values, worst 8.7e-13; 10 cases where SciPy was wrong were settled with 40-digit decimals or round trips.
- Known limitation: incomplete-beta continued fraction reaches about 1.3e-13 at a = b = 1e6.

## 1e — Additional special functions, discrete + non-central distributions (done 2026-11-17)
- Delivered as `openstats-phase1e.zip`. About 3.0k new lines of source plus a 3.9k-line reference table; std only.
- 208 tests pass across the workspace. Clippy (`-D warnings`), rustfmt, rustdoc and a `--no-default-features` build are clean; no stubs.
- **Special functions** (`special::`):
  - Owen's T (Gauss–Legendre panels with the Gaussian factor pulled out, plus the a>1 reflection identity).
  - E₁ and Ei (series, Lentz continued fraction, asymptotics; Ei zero exposed as `EI_ZERO`).
  - ζ(s) for all real s (Euler–Maclaurin; functional equation for s<0; trivial zeros exact).
  - Lambert W₀ and W₋₁ (branch-point series from an accurate p=√(2(ex+1)) using a double-double −1/e, then Halley).
- **Discrete distributions** (`DiscreteDistribution`: pmf, ln_pmf, cdf, sf, quantile, isf, mean, variance, std_dev, support):
  - Binomial, Poisson, Hypergeometric, Negative binomial and Geometric (both count trials, so Geometric support starts at 1), Discrete uniform (alias `Integer`), and a user-specified `Discrete`.
  - pmfs use Loader's saddle-point terms (stirlerr, bd0) for full accuracy in the tails. cdf/sf go through the incomplete beta or gamma. Hypergeometric sums a pmf recurrence toward the nearer tail.
  - Quantile and isf are exact smallest-k searches, evaluated on the survival side when p>½. A 64-ulp fuzz absorbs the rounding of p. Round trips of quantile(cdf(k)) are therefore only exact for cdf ≤ ½, which is a property of double-precision p rather than the algorithm.
- **Non-central** (`ContinuousDistribution`): χ², Beta, F, t.
  - χ², Beta and F are Poisson-weighted mixtures summed outward from the mode, with complementary-tail series so both tails keep relative accuracy.
  - t: series for the same-sign tail; log-domain quadrature where a series would cancel (the opposite tail, and the pdf when δt<0).
  - Quantiles come from a doubling bracket plus Brent on whichever tail is smaller.
- **Calculator:** `PDF`, `CDF`, `SFCDF`, `INVCDF` accept binomial, poisson, hypergeometric, negativebinomial, geometric, integer and the nc names (`nct`, `ncchisquare`, `ncf`, `ncbeta`). Discrete `PDF` is the pmf, and count parameters must be whole numbers.
- **Verification:**
  - SciPy 1.18 reference tables (`tests/reference/scipy2.rs`) for the new special functions and distributions.
  - Discrete pmf/cdf/sf compared with exact 60-digit decimal sums; ppf/isf checked against the quantile definition.
  - Worst errors: special ≤ 1.1e-13; non-central ≤ 1e-11 (nct tails, 36 quantiles accepted by round trip); discrete ≤ 1e-11.
  - SciPy was wrong in many far-tail cases (hypergeometric, Poisson pmf and sf, randint sf, geometric p=1, discrete isf for tiny q, nct tails, underflowing ncx2/ncf, ncf isf at 1e-100). In those cases the exact or independent-quadrature value replaced SciPy's.
  - The Lambert W reference at x = −0.36787944117144 was likewise recomputed from the exact double.
  - Fixed along the way: `beta_prefix` lost a·ε accuracy for huge a with tiny y; it now uses ln_1p.
- **Known limitations:**
  - Binomial cdf/sf at n=1e9 have about 3e-11 relative error, from the incomplete beta at huge parameters; the error stays ≤ 1e-11 for n ≲ 1e6.
  - Negative binomial sf is about 7e-12 at r=1000, p=0.001 deep in the tail.
  - The nct pdf series is capped at |δ| ≲ 1e3.

## 1f — Optimizers, random numbers, calculator RANDOM (done 2026-11-17)
- Delivered as `openstats-phase1f.zip`. About 2.8k new lines of source and 0.8k of tests; std only.
- 239 tests pass across the workspace. Clippy (`-D warnings`), rustfmt, rustdoc and a `--no-default-features` build are clean; no stubs.
- **Optimizers** (`optim::`):
  - `brent_minimize` and `bracket_minimum` (1-D).
  - `nelder_mead`: Gao–Han adaptive coefficients, automatic restarts.
  - `bfgs` (dense) and `lbfgs`, both with a strong-Wolfe line search (Nocedal & Wright 3.5/3.6, cubic zoom).
  - `lbfgs_bounded`: box constraints by a projected active-set L-BFGS with projected Armijo backtracking. This is **not** the Cauchy-point L-BFGS-B algorithm; it is documented as such.
  - `newton` (modified-Cholesky shifted Hessian, Armijo) and `newton_dogleg` (trust region, Algorithm 4.1).
  - `levenberg_marquardt` (Marquardt scaling, QR on the augmented system, Nielsen damping) and `gauss_newton` (pivoted QR, Armijo), with `covariance()`, `std_errors()`, `mse()` and `rank()` from the SVD of J.
  - Gradients, Hessians and Jacobians are analytic or differenced (central for ∇ and ∇², forward or central for J). Non-convergence is reported in `converged` and `termination`.
- **Random numbers** (`random::`):
  - Generators: MT19937, MT19937-64, PCG64 (XSL-RR 128/64 with selectable streams) and xoshiro256**, seeded by SplitMix64 where a single u64 is given.
  - `Rng` is object safe: uniform [0,1), open (0,1), unbiased `below(n)` (Lemire).
  - `sample`/`sample_n` are methods on both distribution traits. Dedicated exact samplers cover Normal, Lognormal, Half/Folded normal, Gamma, χ², Beta, t, F, the four non-central laws, Poisson, Binomial, Hypergeometric, Negative binomial, Geometric, Discrete uniform and Discrete. All other distributions (Cauchy, Laplace, Logistic, Loglogistic, Weibull, Exponential, Uniform, Triangular, SEV/LEV, Johnson) use quantile inversion on the open interval.
  - Algorithms:
    - normal: Marsaglia polar.
    - gamma: Marsaglia–Tsang, with a log-scale boost for shape < 1.
    - beta and F: ratios of log-scale gammas, with no 0/0 for tiny parameters.
    - Poisson: inversion below 10, PTRS above.
    - binomial: inversion for n·min(p,1−p) < 30, otherwise Knuth's exact beta splitting.
    - hypergeometric: inversion walking outward from the mode.
    - negative binomial: gamma–Poisson mixture.
    - non-central laws: their defining mixtures.
  - Multivariate: `MultivariateNormal` (Cholesky, or an eigen factor for singular covariance), `wishart` (Bartlett), `dirichlet` (log-scale gammas), `multinomial` (conditional binomials).
- **Calculator:** `RANDOM(n, "dist", …)` uses a per-thread PCG64 stream seeded from the clock on first use. `BASE(seed)` makes it reproducible. `RANDOMSEED(seed, n, "dist", …)` is stateless and reproducible. Every distribution name that `PDF` accepts works. `n` must be a whole number ≤ 1e8.
- **Verification:**
  - Generators match published vectors: MT19937 (default seed, 10000th output, `init_by_array`), MT19937-64 (default and `init_by_array64`), PCG64 against NumPy and the reference seeding, xoshiro256** and SplitMix64 against reference implementations.
  - Generator quality: uniformity χ², lag-1 correlation, bit balance, unbiased `below`.
  - Optimizers: Rosenbrock (2-D and 10-D), Himmelblau (all four basins), Beale, Powell singular, Wood, Goldstein–Price, Booth, an indefinite-Hessian problem (x⁴ − x² + y²), a 20-D quadratic, and bounded problems with KKT checks.
  - Least squares: SciPy `curve_fit` parameters (rel. ≤ 1e-7) and standard errors (rel. ≤ 1e-6) for three models, covariance against σ²(XᵀX)⁻¹ for a linear model, and a rank-deficient model (rank 1).
  - Samplers: one-sample Kolmogorov–Smirnov tests for all 31 central and 10 non-central parameter sets, Pearson χ² for 22 discrete parameter sets (Poisson up to 1e4, Binomial up to 1e6, Hypergeometric up to 1e5), and a mean z-test for each. Samplers are also exercised with all four generators.
  - Multivariate: mean/covariance of the multivariate normal, Wishart mean and variance, Dirichlet moments and multinomial mean and covariance.
  - Mutation spot checks showed the tests fail on a wrong gamma constant and a wrong binomial inversion pmf.
- **Known limits:**
  - Quasi-Newton and Newton converge only linearly on Powell's singular function (Hessian singular at the solution), so tests assert a tiny objective rather than a tiny x-error.
  - A covariance from a differenced Jacobian carries differencing noise (≈1e-11 relative for central differences), so exact rank deficiency is only detected with an analytic Jacobian.
  - Newton started exactly at a saddle stays there (zero gradient).
  - Marsaglia polar normals discard the second variate (stateless samplers).

## 1g — NIST StRD harness and Phase 1 integration (done 2026-11-17)

**Library additions** (`openstats_math::stats`): Neumaier compensated sums, corrected two-pass mean/variance/SD, NIST-definition lag-k autocorrelation, `ols` (Householder QR + iterative refinement with double-double residuals, standard errors, centred/uncentred R²), `anova_oneway`.

**Harness** (`crates/openstats-math/tests/nist_strd.rs`, `tests/nist/{parser,expr,mod}.rs`): tolerant parser for all four StRD procedures (handles `Infinity`, zero SDs, multi-line models, "Start 1/Start 2" columns); model-formula parser (`+ - * / ** ^`, exp/log/sin/cos/arctan/sqrt, `[]`/`()`, pi) with forward-mode dual numbers so nonlinear fits use exact Jacobians; LRE reporting. It discovers official files in `tests/fixtures/nist/*.dat` (git-ignored, take precedence) and committed `embedded/*.dat`.

**Embedded datasets (reconstructed, see `tests/fixtures/nist/README.md`)** — official files could not be downloaded here, so data were reconstructed and certified values recomputed independently (exact rational / 80-digit decimal) and cross-checked with published digits: NumAcc1–4; Longley, Wampler1, Wampler2, NoInt1, NoInt2; Misra1a, MGH09, MGH10, MGH17, Rat42, Rat43. `OpenStatsAnova1` is a **synthetic** ANOVA set (not NIST). PiDigits was dropped: recalled certified values were not reproducible from π.

**Results**: linear/univariate 9–15 LRE; nonlinear LM (both starts, except MGH10 Start 1) and GN from Start 2 reach worst-case LRE ≥ 7 on every embedded set (typ. 8–10). Gauss–Newton does not converge from several Start-1 points and LM does not converge on MGH10 Start 1 within 5000 iterations; these are reported, not asserted. Spread statistics on large-offset data are limited by double representation of the decimal inputs (NumAcc3 ≈ 9.5, NumAcc4 ≈ 8.3 digits); thresholds account for this.

**Bugs found by the harness and fixed**: Gauss–Newton reported `converged` when the step was not a descent direction (now `LineSearchFailed` unless the gradient is negligible), and any fit with an all-zero Jacobian column (model underflow) was reported converged (now degenerate ⇒ not converged).

**Integration**: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`, and `--no-default-features` builds of `openstats-core` and `openstats-math` are clean; no TODO/unimplemented markers.

## Phase 1 complete — carry-forward gaps vs the Phase 0 blueprint
Delivered per the approved 1a–1g scope. Not delivered (listed in the blueprint, absent from the approved sub-phase specs): numerical integration is present but **interpolation/splines, FFT, polynomial roots**; **multivariate normal/t distributions (Genz MVN CDF)**; **QP, LP, IRLS**; NIST datasets beyond those embedded (need official files). Schedule these when first needed by a Phase 2+ procedure.

## 2a — `openstats-stats`: descriptive, graphical-summary data, cross-tabulation, correlation (done 2026-11-17)
New crate `openstats-stats` (std only; depends on `openstats-math` without its `formula` feature). Missing numeric values are `NaN`, as in `openstats-core`; text categories use the empty string. 38 new tests (30 unit, 8 reference); workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings` and `--no-default-features` builds are clean; no stubs.

- **Quantiles** (`quantile`): all nine Hyndman–Fan definitions; Minitab quartiles are type 6, (n+1)p with end clamping. Checked against NumPy for every type over 11 probabilities.
- **Descriptives** (`descriptive::describe`, `describe_by`): N, N*, mean, SE mean, 5 % trimmed mean (k = round(0.05n) per end), SD, variance, CV, sum, sum of squares, min, Q1, median, Q3, max, range, IQR, modes with count, bias-adjusted skewness and kurtosis, MSSD. Quantities that need more data (or are undefined) are `None`. Missing values are dropped and counted; ±∞ is an error. Uses the compensated two-pass sums from 1g.
- **Graphical summary data** (`graphical`): histogram with Sturges / Scott / Freedman–Diaconis / fixed count / fixed width / explicit edges (left-closed bins, last closed, densities), ECDF, box-plot statistics (1.5·IQR fences, outliers), normal-probability-plot positions (Benard default, Blom, Hazen, Weibull), t interval for the mean, χ² interval for the SD, distribution-free median interval (reports achieved confidence), Anderson–Darling A² with the D'Agostino–Stephens p-value, and a `graphical_summary` bundle.
- **Cross-tabulation** (`crosstab`): `Crosstab` from labels, numeric codes or counts; totals, row/column/total percentages, expected counts, Pearson and adjusted residuals; Pearson χ², likelihood-ratio χ² (with expected-count warnings), Mantel–Haenszel linear-by-linear (default or custom scores), Fisher's exact test (2×2 by hypergeometric with one-sided p-values and odds ratio; r×c by full enumeration, limit 2·10⁸ tables, then `TooLarge`), McNemar (asymptotic, continuity-corrected, exact binomial), phi / Cramér's V / contingency coefficient. Empty rows or columns are ignored.
- **Correlation** (`correlation`): covariance and Pearson matrices with pairwise or listwise deletion, p-values and Fisher-z intervals; Spearman (ranks with ties, t-approximation p-value); Kendall τ-a/τ-b/τ-c by Knight's O(n log n) algorithm with the S-test (exact null distribution when there are no ties and n ≤ 60, tie-corrected normal otherwise); partial correlation from the inverse covariance matrix with df = n−2−k; `pairwise_matrix` for matrices of any pairwise statistic.
- **Verification**: SciPy 1.18 / NumPy values (`tests/common/data.rs`, generated by `gen/stats_ref.py`) for descriptives (5 datasets incl. ties, missing values and a 1e6 offset), quantiles, A–D, intervals, χ²/G²/Mantel–Haenszel on 5 tables, Fisher 2×2 (5 tables, three alternatives), McNemar (4), Pearson/Spearman/Kendall (with heavy ties and exact), partial correlation and covariance. Fisher r×c is checked against **exact rational enumeration** done independently in Python, because SciPy has no r×c test.
- **Known limits and judgement calls**:
  - Minitab parity of the definitions above is from Minitab's published method descriptions; no Minitab output was available here to compare against. The trimmed-mean rounding rule and "no mode when all values are distinct" follow those descriptions.
  - The median interval is the classical order-statistic interval with achieved confidence, not an interpolated one.
  - Spearman p-values use the t approximation; Kendall's exact p is used only up to n = 60 (SciPy switches at 33, so n = 34–60 differ slightly in method, not in validity).
  - Fisher r×c is enumeration, not the network algorithm: large tables with big counts return `TooLarge` instead of an answer.
  - The Pearson matrix leaves undefined entries (constant variable, n < 3) as `NaN`.
- **Blueprint items for 2a that were not in the approved scope and are not delivered**: report model, design matrix/formula, t/Z/proportion/Poisson/variance tests, outlier tests, equivalence tests, general normality tests beyond Anderson–Darling. Carried forward with the Phase 1 gaps above.

## 2b — Hypothesis tests, normality, outliers, equivalence (done 2026-11-17)
Seven new modules in `openstats-stats` (std only). 39 new tests (28 unit + 11 SciPy / handbook / Monte Carlo reference tests; the crate now has 77). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds clean; no stubs.

- **t / Z** (`ttest`): one-sample, paired (complete pairs), two-sample pooled and Welch (Satterthwaite df); two- and one-sided p-values and confidence intervals (one-sided intervals are open-ended); `z_test` with known σ.
- **Proportions** (`proportion`): one-proportion exact binomial (two-sided by the "small p-values" rule, found by binary search so it is fast for huge n), Clopper–Pearson and Wald intervals, normal z with optional continuity correction; two-proportion pooled z (unpooled when the null difference ≠ 0), continuity correction, Fisher's exact p-value, Wald interval.
- **Poisson** (`poisson`): one-sample exact test and Garwood interval plus the normal test; two-sample difference z-test, exact conditional test of equal rates (binomial) and exact rate-ratio interval; `poisson_totals` from per-unit counts.
- **Variances** (`variance`): F-test with ratio and SD-ratio intervals; Levene (mean, median, trimmed centre), Brown–Forsythe, Bonett (k groups).
- **Normality** (`normality`): Shapiro–Wilk (Royston AS R94, 3 ≤ n ≤ 5000), Ryan–Joiner, Kolmogorov–Smirnov for a specified normal (exact Marsaglia–Tsang–Wang for n ≤ 1000), Lilliefors. Anderson–Darling remains in `graphical` (2a).
- **Outliers** (`outlier`): Grubbs (either end / larger / smaller, p-value and critical value), generalized ESD (Rosner) as the multi-outlier test, Dixon's Q (3 ≤ n ≤ 30) with p-values and critical values from numerical integration of the exact normal null.
- **Equivalence** (`equivalence`): TOST for one-sample (vs a target), paired and two-sample (pooled/Welch); both one-sided p-values, the (1 − 2α) interval, and the equivalence decision (identical to the interval lying inside the limits).
- **Verification**:
  - SciPy 1.18 for t (all alternatives, intervals), proportions (binomial exact + CP bounds, Fisher), conditional Poisson, Levene (all three centres), F-test, Shapiro–Wilk (10 datasets, n = 3 … 1000), exact KS, and TOST (recomputed from t quantiles).
  - Exact Poisson two-sided p-values and the Garwood interval against brute-force summation and χ² quantiles.
  - NIST/SEMATECH e-Handbook generalized-ESD example (Rosner's 54 values): R₁–R₃ and λ₁–λ₃ match the published 3.118/3.158, 2.942/3.151, 3.179/3.143 and three outliers are found.
  - Dixon critical values match the classic Q tables for n = 3–10 (to the tables' own rounding; the printed n = 4, 99 % value .926 is off — the computed .9207 gives tail probability .0050 by Monte Carlo) and four (n, α) cases are confirmed by a 400,000-sample Monte Carlo.
  - Calibration (600–800 simulated normal samples each): Shapiro–Wilk, Ryan–Joiner and Lilliefors reject at ≈5 % at α = 0.05.
  - SciPy's exact KS p-value for n = 200 is off by ≈5·10⁻⁶ relative; OpenStats's matches a 60-digit evaluation of the same matrix-power algorithm to 1e-12.
- **Judgement calls and limits (please review)**:
  - **Bonett's test** is implemented from the published formulas as I know them (trimmed-mean kurtosis estimate, c = n/(n − z), inverse-variance pooling for k > 2). It is checked only against an independent re-implementation of the same formulas, *not* against Minitab or published numbers, and the trimming rule (⌊n/(2√(n−4))⌋ per end) and the dependence of c on the confidence level are assumptions. Treat it as unverified until compared with Minitab output.
  - **Ryan–Joiner** statistic is the Blom-score correlation; its p-value uses Royston's Shapiro–Francia normalising transformation (checked for calibration), whereas Minitab interpolates tables.
  - **Lilliefors** uses the Dallal–Wilkinson approximation as in R's `nortest`; verified at three published 5 % critical points (n = 10, 20, 30) and by calibration, not against software output.
  - "Multi-outlier Grubbs" is delivered as the generalized ESD test (iterating single Grubbs tests suffers masking and has no valid error rate).
  - Dixon's Q covers r₁₀ only, for n ≤ 30; for "either end" the p-value is twice the one-end value (exact for Q > ½, an upper bound otherwise).
  - TOST limits are absolute differences from the target / between means; ratio-form or percent-of-target limits and equivalence tests for proportions or standard deviations are not implemented.
  - The two-proportion exact (Fisher) and the two-sample Poisson exact tests apply to a null difference of 0 only.
  - Wald intervals are not clamped to [0, 1].
- **Still outstanding from the blueprint's Phase 2**: report model and design-matrix/formula layer.

## 2c — Session report model, power & sample size, tolerance intervals (done 2026-11-17)
Three new modules in `openstats-stats` (std only) plus `reportable`. 52 new tests (40 unit, 12 SciPy-reference integration; the crate now has 98 unit + 31 integration tests). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds and `cargo test --workspace` are clean; no stubs.

- **`report`**: `SessionReport` → `Section`s of `Block::{Text, Table, Chart}`, plus section/report warnings. `Table` has typed columns (`Text`, `Integer`, `Fixed(d)`, `Significant(s)`, `PValue`), `Cell::{Text, Int, Num, Missing}` and footnotes. Numbers are stored raw and formatted per column, so decimal points align (a `Significant` column uses the largest decimals any cell needs) and p-values print Minitab-style (`0.000` below 0.0005, `*` for missing/NaN, `∞`). `ChartSpec` (histogram, boxplot, probability plot, scatter, line, bar, interval plot + reference lines) is plain data for the UI. `render_text()` gives the Session-window text; `to_json()` / `from_json()` are a hand-written, lossless round trip (finite f64 exact, NaN/±∞ encoded as strings; full string escaping incl. surrogate pairs).
- **`reportable`**: `Reportable` implemented for every 2a–2c result type (descriptives, graphical summary with histogram + boxplot charts, Anderson–Darling, crosstab, χ², Fisher, McNemar, Pearson/Spearman/Kendall/partial, t/Z, proportions, Poisson, F, Levene/Brown–Forsythe, Bonett, Shapiro–Wilk, Ryan–Joiner, KS/Lilliefors, Grubbs, ESD, Dixon, TOST, tolerance intervals) and `power_section()` (parameters, results table, power-curve chart). A test builds one report out of 25 sections, validates it and round-trips it through JSON. Not yet covered: the correlation/covariance *matrices* and `describe_by` (they need variable labels, which belong to the caller).
- **`power`**: trait `PowerModel` with `power(n, effect)`, `sample_size(effect, target)`, `effect(n, target)` and `solve(n, power, effect)` (exactly one `None`). Designs: `OneSampleZ`, `OneSampleT`, `PairedT`, `TwoSampleT` (non-central t); `OneProportion` / `TwoProportions` (normal approximation or exact binomial / Fisher's exact); `OnePoisson` / `TwoPoisson` (normal or exact / conditional exact); `OneVariance` (χ²), `TwoVariances` (F); `OneWayAnova` (non-central F, Minitab's max-difference parametrisation, plus `power_for_means`); `Factorial` over `FactorialDesign::{full, fractional, plackett_burman}` (t test of an effect, error df from replicates / centre points / blocks / omitted terms, solves for replicates); `Tost` (one-sample, paired, two-sample). Sample size = smallest integer reaching the target: doubling + bisection for continuous powers, a full scan for the saw-toothed exact tests. Exact-test power uses the true rejection region of the same p-values the tests in 2b use (`actual_alpha()` reports the achieved size). TOST power is exact: `E_W[max(0, Φ(δ_U − t_c W) − Φ(δ_L + t_c W))]` by adaptive quadrature over the χ² variable.
- **`tolerance`**: one-sided factor exact via non-central t; two-sided exact by solving the defining coverage integral (inner coverage condition solved exactly, outer integral over χ², substitution removing the endpoint singularity) or Howe's approximation; `normal_*_confidence` give the confidence of any k. Nonparametric: order-statistic intervals with exact binomial confidence, tightest symmetric ranks, `nonparametric_min_sample_size` (93 for 95/95 two-sided, 59 one-sided), `nonparametric_confidence` for arbitrary ranks.
- **Verification**:
  - SciPy 1.18 (`nct`, `ncf`, `chi2`, `f`, `binom(test)`, `poisson`, `fisher_exact`) for t, ANOVA and variance power, exact 1-proportion power and sample size (region from `binomtest`), Fisher power (all-table enumeration with `scipy.stats.fisher_exact`), 1- and 2-sample Poisson exact power, factorial power and df, one-sided tolerance factors; sample sizes agree with G*Power values (34 / 64 / 26 for the t tests).
  - TOST power against an independent integral that integrates over the mean first (different order from the library), and against a 60,000-run simulation.
  - Two-sided tolerance factors against an independent SciPy quad/brentq of the defining integral, and against a 100,000-run simulation (confidence 0.95 ± 0.003).
  - Library cross-checks: brute-force enumeration through the crate's own `two_proportions` / `two_sample_poisson` p-values for the exact-power regions; z-test and 2-group-ANOVA ↔ t identities; α at zero effect; symmetry of variance powers.
- **Judgement calls and limits (please review)**:
  - **Two-sided normal tolerance factors**: I first recalled published table values (n = 10, 95/95: 3.379) that did *not* agree with the integral; the independent integral and a 400,000-run simulation both give 3.3934 (3.379 yields only 94.85 % confidence), so the tests pin the computed values, not the recalled ones. The published-table comparison was *not* done (no tables available offline); Minitab's own factors are unverified. Howe's factor (3.382 at n = 10) is within 1 % for n ≥ 10. Odeh–Owen is not implemented separately — the exact integral supersedes it.
  - Normal-approximation power for proportions and Poisson rates uses the null-variance / alternative-variance form (pooled-at-the-mean null for two samples, no continuity correction); Minitab's documented formulas were not available, so numbers can differ slightly from Minitab for the approximate methods. Exact methods have no such ambiguity apart from the two-sided "small p-values" rule.
  - Factorial / Plackett–Burman power: the model-term and error-df accounting (terms = 1 + (base runs − 1 − omitted) + (blocks − 1) + curvature) is my reading of Minitab's inputs, not verified against Minitab output. Designs are described by run counts only (no generator checking; fractional requires k − p ≥ 2 and k ≤ 2^(k−p) − 1).
  - ANOVA effect = maximum difference with all other means at the centre; other configurations through `power_for_means`.
  - Unequal sample sizes / allocation ratios are not supported (equal n per group everywhere); the 2-sample t power assumes pooled variances; TOST is for means only (no ratio limits, proportions or SDs), and its α is that of each one-sided test (< 0.5).
  - Exact-method search caps: 1-proportion 20,000, 1-sample Poisson 20,000, 2-sample Poisson 5,000, Fisher 1,000 per group (`TooLarge` beyond). Solving a two-sided effect searches one side (`direction` field, default increase).
  - Power for exact tests is saw-toothed in n; the reported n is the *first* n reaching the target, so a slightly larger n can dip below it again.
- **Still outstanding from the blueprint**: the design-matrix/formula layer (moves to the modelling phase), and the Phase 1 gaps listed above.

## 2d — Formula/design-matrix layer and linear regression (done 2026-11-17)
Two new modules in `openstats-stats` (std only): `model/` and `regression/`. 38 new tests (18 unit, 16 NumPy/SciPy-reference integration, 4 NIST StRD; the crate now has 116 unit + 51 integration tests). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds and `cargo test --workspace` are clean; no stubs.

- **`model` (formula parser)**: tokenizer, expression parser and term algebra. `+`/`-`/whitespace separate terms; `*` and `|` cross; `:` and `.` interact only; `/` nests; `A(B)` is A nested in B; `(a+b+c)^k` expands to interactions up to order k; `I(expr)`, functions (`log`, `sqrt`, `exp`, …), `poly(x, k)`, `C(x)`/`factor(x)`, `offset(expr)`; intercept control with `0`, `1`, `-1`; backtick-quoted names. Terms are ordered by interaction order, then by appearance.
- **`model` (design matrix)**: `Data` (numeric columns, factor columns), `build_design` → `DesignMatrix` (columns, term map, response, offset, rows used). Codings: treatment (first/last/named reference), sum-to-zero, Helmert and orthogonal polynomial (R-compatible `contr.poly` / `contr.helmert`, checked), plus custom matrices; per-factor overrides and explicit level order. Missing values: listwise deletion (NaN for numbers, empty string for text). `ModelSpec` remembers levels and codings so new data encode identically for prediction; unseen levels are an error.
- **`regression::fit`**: Householder QR from `openstats-math` plus up to four rounds of iterative refinement with FMA-compensated residuals; `(XᵀX)⁻¹ = R⁻¹R⁻ᵀ` for standard errors, leverage from the thin Q. Output: coefficient table (estimate, SE, t, p, CI, VIF), ANOVA (Regression, each term with sequential SS, adjusted SS/MS, F, p; Error; Lack-of-Fit and Pure Error when the design has replicates; Total), S, R², adjusted R², predicted R² (1 − PRESS/SST), PRESS, overall F, AIC/AICc/BIC, `predict` with confidence and prediction intervals, Minitab's unusual-observation flags (R, X, D). VIF is computed from the inverse correlation matrix by Cholesky. Rank-deficient columns are aliased (Gram–Schmidt, tolerance 1e-10) and reported, as Minitab drops them.
- **`regression::diagnostics`**: ordinary, standardized and studentized-deleted residuals, leverage, Cook's D, DFFITS (= DFITS), PRESS residuals, Durbin–Watson, and the residual four-in-one chart data (normal probability plot, residuals vs fits, histogram, residuals vs order).
- **`regression::selection`**: stepwise / forward / backward over *terms* with p-value (F test) or AIC / AICc / BIC rules, forced-in terms, optional hierarchy (a term needs its lower-order margins; a margin cannot leave while a term containing it stays). Minitab's default α (0.15/0.15, 0.25 forward, 0.10 backward); α-remove < α-enter is rejected (it can cycle) and revisited models are never re-entered.
- **`regression::subsets`**: Furnival–Wilson leaps and bounds on the SWEEP operator (centred, unit-scaled cross-products) with the bound RSS(all remaining), `n_best` models per size, forced-in predictors, size limits, intercept or not, automatic exclusion of aliased predictors. Final statistics (R², adjusted, predicted, Cp with the full-model MSE, S, PRESS, AICc, BIC) are recomputed by QR, so SWEEP rounding never reaches the output.
- **Reports**: `Reportable` for `RegressionFit` (ANOVA, model summary, coefficients, Durbin–Watson, lack of fit, unusual observations, four charts), `StepwiseResult` and `BestSubsets`.
- **Verification**:
  - NIST StRD Longley (≥ 9 correct digits on estimates and SEs, ≥ 10 on S, R², SS), Wampler1 (exact polynomial, degree 5), Wampler2, NoInt1/2 — **the fixture files are OpenStats reconstructions** (their headers say so; certified values recomputed in exact arithmetic), not the official downloads.
  - NumPy/SciPy references (`gen/stats_ref4.py` → `tests/common/data4.rs`): coefficients, SE, t, p, CI, VIF, all scalar statistics, 25 rows of diagnostics, sequential/adjusted SS, prediction/confidence intervals, lack of fit, a factor + covariate model with treatment coding (and invariance of the fit under sum-to-zero coding).
  - Best subsets equals exhaustive search over all 2⁷ subsets (top two per size: variable set, SSE, R², adjusted R², Cp), with forced variables, size limits, no intercept and an aliased predictor. Stepwise/forward/backward steps and final models equal an independent Python implementation of the documented rules for 8 configurations (p-value and AIC/AICc/BIC).
- **Judgement calls and limits (please review)**:
  - AIC/AICc/BIC use the forms `n ln(SSE/n) + 2k`, `+ n(n+k)/(n−k−2)`, `+ k ln n` (k = coefficients) — recalled, **not verified against Minitab**, which may count the error variance as a parameter (a constant offset in k that does not change rankings). Unusual-observation thresholds (|std resid| > 2, h > 3p/n, Cook's D > median of F(p, n−p)) are likewise from memory.
  - No Minitab output was available for comparison. Where Minitab defines something I could not check (nesting coding, stepwise tie-breaks, order of terms in the ANOVA table), I chose the documented/R-like behaviour.
  - Nesting `A(B)`: coded(A) ⊗ full indicators of the observed B cells. This is my interpretation.
  - Without an intercept only the first pure-factor main effect gets full dummy coding (R's rule).
  - Hierarchy detection works on `*` terms; polynomial terms (`I(x^2)`) and `poly()` are treated as independent terms, so stepwise may keep `I(x^2)` without `x`.
  - Best subsets works on predictor *columns* (a factor must be expanded by the caller); stepwise works on terms.
  - Weights, robust and orthogonal regression, transformations of the response (Box–Cox), and categorical predictors in best subsets are not part of this phase.
  - The exact-fit Wampler1 case yields S ≈ 1e-9 relative (not exactly 0), as any floating-point solver does; t and p there are not meaningful.

## 2e — Quality & SPC (done 2026-11-17)
New module `openstats_stats::spc` (std only): `constants`, `rules`, `charts`, `transform`, `capability`, `sixpack`, plus `Reportable` implementations. 21 new tests (3 + 5 + 8 + 5 + 8 + 2 unit tests across the submodules and 18 integration tests in `tests/spc.rs`). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds and `cargo test --workspace` are clean; no stubs.

- **Constants** (`spc::constants`): d2, d3, d4 (median of the range), c4, A2, A3, B3–B6, D1–D4 for n = 2–25, *computed* (not tabulated): d2/d3 by Gauss–Legendre integration of the exact distribution of the range of n standard normals (`openstats_math` Tukey range CDF), c4 from Gamma functions; `c4_real` accepts non-integer n. The table is built lazily once (first call about 1 s in debug builds).
- **Control charts** (`spc::charts`): I-MR, X̄-R, X̄-S (also from precomputed subgroup statistics) and p, np, c, u. σ estimators: average/median moving range or MSSD for individuals; R̄, S̄ or pooled for subgroups; unequal subgroup sizes with per-point limits; known centre/σ; k-sigma width. Limits are truncated to the feasible range (≥ 0, ≤ 1, ≤ n).
- **Special-cause tests** (`spc::rules`): Minitab tests 1–8 with configurable K and a flag on the completing point and every later point that keeps the pattern. Attribute charts allow tests 1–4 only.
- **Capability** (`spc::capability`): Cp, CPL, CPU, Cpk, Pp, PPL, PPU, Ppk, Cpm; Z.Bench, Z.LSL, Z.USL; expected (within/overall) and observed PPM; Pp (χ²) and Ppk (Bissell) confidence intervals; one- or two-sided specifications. Non-normal: Box-Cox (λ optimal, rounded to 0.5, or fixed), Johnson (SB/SL/SU), and Weibull / lognormal / gamma maximum likelihood with a fixed threshold; `rank_distributions` orders fits by Anderson-Darling A². Non-normal Pp/Ppk use the 0.135 % / 99.865 % points and the median.
- **Capability sixpack** (`spc::sixpack`): I/X̄ chart, MR/R/S chart, last-N observations, capability histogram with spec lines, normal probability plot (A²/p in the title) and capability plot, as six `ChartSpec`s plus the summary tables. Control charts emit UCL/Center/LCL/Flagged series.
- **Reports**: `Reportable` for variable and attribute charts, the three capability results and the sixpack; `constants_section(from, to)` gives the unbiasing-constants table.
- **Verification**:
  - Constants against an independent NumPy Gauss–Legendre integration (d2 1e-11, d3 1e-9, d4 1e-10, c4 1e-13, n = 2–25) and against the published AIAG/ASTM table values to their printed precision, including A2, A3, B3, B4, D3, D4 for n = 5, 10, 25.
  - Test rules 1–8, I-MR (three σ estimators), X̄-R, X̄-S, pooled σ, unequal sizes, normal capability (every index, CIs, PPM) and the capability from subgroups against independent Python implementations / SciPy (`gen/stats_ref5.py` → `tests/common/data5.rs`).
  - Box-Cox λ and log-likelihood against SciPy; Weibull, gamma and lognormal MLE and the Weibull capability against SciPy-grade references (Weibull shape and gamma shape solved with `brentq` at 1e-15; `scipy.stats.fit` was only good to about 1e-5).
  - Montgomery data: orange-juice p chart (p̄ = 0.2313, UCL 0.4102, LCL 0.0524, samples 15 and 23 out; revised p̄ = 0.2150, UCL 0.3893, and sample 21 then exceeds the revised limit, as in the book), circuit-board c chart (c̄ = 19.85, UCL 33.22, LCL 6.48, samples 6 and 20 out; revised 19.67 / 32.97 / 6.36), and the piston-ring X̄-S chart (x̿ = 74.0012, s̄ = 0.0094, S-chart UCL 0.0196, X̄ limits 74.0147 / 73.9877 to the printed digits, in control).
  - Johnson and Box-Cox capability on a lognormal process of known Z.Bench (within 0.15 and 0.1 respectively; the naive normal result is further off); Johnson SU and SB populations are fitted with p > 0.1.
- **Judgement calls and limits (please review)**:
  - A recalled d3 (n = 19: 0.734) disagreed with the computed 0.7335; the test now uses the published 0.733. As in 2c, the computed value and the independent reference win over memory.
  - **No Minitab output was available.** Where Minitab's definitions are not fully known to me I chose the documented-in-literature form: unequal-size σ weights (R̄: (d2/d3)²; S̄: c4²/(1−c4²)), the default subgroup σ (R̄ when max n ≤ 8, otherwise S̄), the Johnson selection rule (z grid 0.25–1.25, maximum Anderson-Darling p of the transformed data; needs n ≥ 10), SL fitted for positive skew only, and fixed (default 0) thresholds for the non-normal fits (no three-parameter Weibull / lognormal).
  - Cp and Cpk confidence intervals are not provided (Pp and Ppk are).
  - The raw Montgomery capability data were not available, so capability tests use seeded SciPy data plus the first twelve piston-ring samples; for piston rings 13–25 only means and variances are known, so the full R-chart golden values are not tested (the X̄-S chart is).
  - Box-Cox needs positive data; λ is searched on [−5, 5] (grid 0.05, then Brent).
  - Charts are vector metadata only (`ChartSpec`); flagged points are a `Flagged` series that the renderer draws as markers.

## 2f — ANOVA, multiple comparisons, GLM and Gage R&R (done 2026-11-17, commit 38cd100, approved)
Two new modules in `openstats-stats` (std only): `anova` (`oneway`, `comparisons`, `glm`) and `msa` (crossed / nested Gage R&R, chart data), plus `Reportable` implementations and `spc::constants::d2_star`. 22 new tests (6 unit, 16 integration in `tests/anova.rs`; the crate now has 153 unit + 83 integration tests). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds and `cargo test --workspace` are clean; no stubs. The ANOVA integration tests take about 33 s in a debug build (Dunnett integrals).

- **One-way ANOVA** (`anova::one_way`, `one_way_stacked`): Factor/Error/Total table, S, R², R²(adj), R²(pred) (PRESS in closed form from the leverages 1/nᵢ), per-level N, mean, SD, confidence interval from the pooled SD and from the level's own SD, Welch's test (F, df₁, df₂, p) when every level has ≥ 2 values and a positive SD, interval-plot data (`IntervalPlot` with estimate/lower/upper) and individual-value-plot data. Missing values are dropped; unequal sizes are fine.
- **Multiple comparisons** (`anova::comparisons`, also on `OneWayAnova`): Tukey (Tukey–Kramer, adjusted p from the studentized range), Fisher LSD, Dunnett (two-sided, greater, less; exact multivariate-t with the correlation structure of the actual group sizes), Hsu MCB (largest or smallest is best; constrained simultaneous intervals from one-sided Dunnett constants). Each gives differences, SE, t, adjusted p, intervals, the individual confidence level implied by the simultaneous critical value, and grouping letters (A, B, C, … by connected runs of non-significantly different levels sorted by mean). The functions take plain `Means` (labels, n, means, MSE, df), so they apply to any model's means.
- **GLM** (`anova::glm`): fixed factors and covariates from a formula, sum-to-zero coding by default. Type I (sequential, with F and p against MSE) and Type III (adjusted) sums of squares, coefficient table and model summary from the regression layer, least-squares means with standard errors and intervals (equal-weight average over the other factors, covariates at their means) for every factor-only term, and data for main-effects and interaction plots.
- **Gage R&R, crossed** (`msa::CrossedStudy`): from stacked columns, balanced with ≥ 2 replicates, no missing values.
  - ANOVA method: Part, Operator, Part × Operator tables with F tests; the interaction is dropped (pooled with repeatability) when p ≥ 0.25, with `Interaction::{Auto, Keep, Drop}` and a configurable level; both ANOVA tables are kept when it is dropped. Variance components (negative estimates set to 0 and reported), Total Gage R&R, Repeatability, Reproducibility (Operator, Operator*Part), Part-To-Part, Total Variation, %Contribution, StdDev, Study Var (k × SD, default k = 6), %Study Var, %Tolerance (from a width or LSL/USL) and ndc = max(1, ⌊1.41·σ_part/σ_gage⌋).
  - X̄-R method: repeatability R̄/d2, reproducibility from the range of operator means with `d2*(operators, 1)`, part variation from the range of part means with `d2*(parts, 1)`.
- **Gage R&R, nested** (`msa::NestedStudy`): Operator, Part(Operator), Repeatability ANOVA with F tests (operator against part(operator)), the same components table.
- **Six-panel chart data**: components of variation (bars with %Contribution, %Study Var, %Tolerance), R chart by operator, X̄ chart by operator (per-operator centre and R̄-based limits), measurements by part (points plus means), by operator (box plots with outliers) and the Operator × Part interaction. Nested designs have five panels (no interaction).
- **Reports**: `Reportable` for `OneWayAnova`, `Comparisons`, `McbResult`, `GlmAnova` and `GageResult`, with Minitab-style table captions and column headers (ANOVA tables, Gage R&R variance components, Gage Evaluation, ndc line).
- **Verification**:
  - Montgomery Example 3.1 (plasma etching): SS treatments 66,870.55, SSE 5,339.20, MS 22,290.18, MSE 333.70, F 66.80, LSD 24.49, Tukey half-width 33.05 (the book's 33.09 uses the rounded table value q = 4.05).
  - Montgomery Example 5.1 (battery life, 3 × 3 with 4 replicates): SS 10,683.72 / 39,118.72 / 9,613.78 / 18,230.75 / 77,646.97, F 7.91 / 28.97 / 3.56, p 0.0020 / 0.0186, cell mean 134.75.
  - AIAG measurement-systems data (10 parts, 3 operators, 3 trials): ANOVA sums of squares and F values reproduce the Minitab documentation (88.3619, 3.1673, 0.3590, 2.7589; F 492.291, 79.406, 0.434; interaction p = 0.974; without interaction F 245.614, 39.617; repeatability 0.03997, reproducibility 0.05146, part-to-part 1.08645, %Contribution 7.76, %Study Var 27.86, ndc 4). The X̄-R method reproduces the AIAG manual (EV 0.2019, AV 0.2297, GRR 0.3058, %GRR 26.68 %, ndc 5). `d2*(m, 1)` reproduces the AIAG K2/K3 constants (0.7071, 0.5231, 0.3146).
  - SciPy 1.18 / NumPy references (`gen/stats_ref6.py` → `tests/common/data6.rs`): F, p, S, R², R²(pred), Welch, pooled and individual intervals on an unbalanced, unequal-variance data set; `tukey_hsd` (p, intervals); Fisher LSD; Dunnett p-values, intervals and critical values against an independent double integral (the SciPy Dunnett is randomized, so it was not used); Hsu MCB against an independent computation (balanced and unbalanced, both directions); Type I and Type III sums of squares and least-squares means for an unbalanced 3 × 2 design with a covariate and for an unbalanced 2 × 3 × 2 full factorial (7 terms); the Gage R&R ANOVA tables and variance components for a data set with a strong interaction (kept), the forced-interaction model, the X̄-R method and a nested study.
- **Judgement calls and limits (please review)**:
  - No Minitab output was available beyond the published AIAG/Minitab-documentation numbers above. Column headers and table captions are reproduced from memory of Minitab's layout.
  - NIST StRD ANOVA files (SiRstv, SmLs, AtmWtAg) were not available offline, so they are not part of the tests; the textbook data are Montgomery and AIAG.
  - **GLM is fixed-effects only**: no random factors or expected-mean-squares F tests (Gage R&R has its own random-effects ANOVA), no nested factors in the least-squares means (they are skipped for nested or numeric-as-factor terms), no Tukey/Dunnett comparisons of *least-squares* means inside a model (the comparison functions accept any `Means`, but a convenience wrapper is not provided), and a factor combination with no data gives `NaN` least-squares means. Empty cells in other terms are not detected.
  - Hsu MCB with unequal group sizes uses, for each level, a Dunnett constant built from the other levels' sizes and takes the minimum over the other levels of the lower and upper limits (equal to Hsu's closed form when sizes are equal); this generalisation is mine and is checked only against my independent implementation of the same definition.
  - Dunnett's grouping letters are simplified: the control and every level not significantly different from it get "A", the others "B".
  - X̄-R method: repeatability uses `d2` with g = ∞ (the AIAG K1), not Duncan's `d2*` for the finite number of ranges; operator and part ranges use `d2*(m, 1)` (the AIAG K2/K3). Minitab may use finite-g `d2*` for repeatability; differences are in the third digit.
  - Gage R&R needs balanced complete data; a part × operator count mismatch or a missing value is an error rather than a partial analysis. Operators and parts are treated as random effects; fixed-operator variants are not offered. For X̄-R, parts or operators beyond 25 work (via `d2*`), but replicates beyond 25 do not (tabulated `d2`).
  - The R and X̄ charts by operator draw the operator's own centre line and R̄-based limits; the control-limit series are dropped for more than 25 replicates.

## 2g — SPC: further control charts, rule sets and staging (done 2026-11-17, commit fe586d1, approved)
Six new modules in `openstats_stats::spc` (std only): `stages`, `variables_ext`, `attributes_ext` (renamed `attributes` in 2h), `timeweighted` (renamed `time_weighted` in 2h), `multivariate`, plus `report_ext` (`Reportable` / `ChartSpec` for every new result), and rule-set presets in `rules`. 26 new unit tests and 26 integration tests (`tests/spc2.rs`; the crate now has 179 unit + 109 integration tests). Workspace clippy `-D warnings`, rustfmt, rustdoc `-D warnings`, `--no-default-features` builds and `cargo test --workspace` (537 tests) are clean; no stubs. Unbiasing constants stay computed (d2, d3, c4, `d2_star`), never tabulated.

- **Rule sets** (`SpecialCauseTests`): `nelson()` (= the eight tests, Minitab defaults), `western_electric()` (3σ; 2 of 3 beyond 2σ; 4 of 5 beyond 1σ; 8 in a row on one side) and `western_electric_supplementary()` (adds trend 6, alternation 14, stratification 15). Every parameter stays configurable (K per test, σ multiple of test 1).
- **Staged limits** (`stages`): `staged_i_mr`, `staged_xbar` (R or S), `staged_p`, `_np`, `_c`, `_u`. A stage is a contiguous block of equal labels; each stage gets its own centre, σ and limits, tests run per stage (patterns never span a boundary; the first moving range of a stage is missing) and the joined panel reports violations with global point numbers.
- **Variables** (`variables_ext`): I-MR-R/S between/within (σ_within from R̄/d2 or s̄/c4, σ_between² = max(0, (MR̄/d2)² − σ_w²·mean(1/nᵢ)), means limits ±k√(σ_b² + σ_w²/nᵢ), MR chart of the means, R or S chart); zone chart (weights 0/2/4/8, signal at a cumulative 8, restart on crossing the centre line and after a signal; weights and level configurable; individuals or subgroups); Z-MR short-run (σ pooled from within-part moving ranges, by part, or known; targets from the part means or given; then I-MR with centre 0 and σ 1); Box-Cox I-MR and X̄-R/S (λ optimal, rounded or fixed; limits in transformed units and back-transformed, `NaN` where 1 + λz ≤ 0).
- **Attributes** (`attributes_ext`, now `attributes`): Laney P′ and U′ (z-scores, σ_z = MR̄(z)/d2(2), limits `centre ± k·σ_z·σᵢ`, truncated); G chart (geometric on {0, 1, …}, p̂ = 1/(1 + mean)) and T chart (Weibull MLE, threshold 0) with the median as centre line and probability limits Φ(−k) / 1 − Φ(−k).
- **Time-weighted** (`timeweighted`, now `time_weighted`): `TimeWeightedInput` from individuals (σ = MR̄/d2) or subgroups (pooled σ by default); moving average (variance σ²/m²·Σ1/nⱼ, shorter windows at the start); EWMA (exact variance recursion or asymptotic limits; λ is always supplied by the caller); tabular CUSUM in standard-error units with K, H, head start and the counts since the sum was last zero, `in_data_units()` for constant sizes; V-mask CUSUM (cumulative sums, lead distance d = h/k, half-angle atan(k), mask arms at any point) computed independently of the tabular form.
- **Multivariate** (`multivariate`): Hotelling T² for individuals and equal-size subgroups; Phase I (Beta for individuals, F for subgroups), Phase II with historical parameters (F) and known parameters (χ²); sample or successive-difference covariance; α default 0.00135.
- **Verification**:
  - Montgomery Example 9.1 (30 observations, target 10, σ = 1): tabular CUSUM K = 0.5, H = 5 gives C⁺₂₉ = 5.28 (first signal at observation 29, as published) with all C⁺, C⁻ and N⁺, N⁻ equal to an independent NumPy recursion; EWMA (λ = 0.1, L = 2.7) first signals at observation 29 and matches the exact and asymptotic NumPy limits. The data transcription was checked numerically in `gen/stats_ref7.py` (it reproduces 5.28) before being used.
  - NumPy/SciPy references (`gen/stats_ref7.py` → `tests/common/data7.rs`): d2, d3 by independent integration; unequal-size CUSUM with head start, EWMA and MA; Laney P′ and U′ (z, σ_z, limits); G chart against the closed form and `scipy.stats.geom.ppf`; T chart against a `brentq`-solved Weibull MLE and `weibull_min.ppf`; I-MR-R and I-MR-S (σ components, all three panels); zone scores and signals by an independent loop; Z-MR pooled and by part; Box-Cox λ against `boxcox_normmax` and the limits in both units; Hotelling T² (Beta, F, χ² limits; sample and successive-difference covariance; subgroups; Σ T² = (m − 1)p identity as a unit test).
  - V-mask signals equal tabular CUSUM signals (K = 0.5, H = 5) on the Montgomery data and on a seeded random series with a shift.
  - Rules: Western Electric patterns, configurable thresholds, staged limits equal to separate charts, JSON round trips and text rendering for every new report, and error cases (bad λ, K, H, head start, sizes, covariance, ragged input).
- **Judgement calls and limits (please review)**:
  - **No Minitab output was available.** The following follow my reading of the literature or of Minitab's documentation and are not confirmed against Minitab: the zone-chart weights and reset convention; the G and T chart centre (median) and probability limits (Minitab's T chart may transform the data instead); the unequal-size handling in I-MR-R/S (an average-variance moment estimate); the V-mask scale (one observation = one σ unit, half-angle atan(k)); the default α = 0.00135 for T²; CUSUM/EWMA on subgroups use the pooled standard deviation by default.
  - Successive-difference T² uses the Sullivan–Woodall beta approximation (ν = 2(m − 1)²/(3m − 4)), which is published but approximate; it is checked against SciPy's Beta quantile, not against a simulation.
  - The G chart is for counts of non-events between events; Minitab's variants with the time-series of event dates are not offered. The T chart fits a Weibull with threshold 0.
  - Time-weighted charts and T² need complete data (no missing values) and offer test 1 only; Laney charts and attribute-style charts offer tests 1–4 (as in 2e).
  - (2g as delivered; the decomposition was added in 2h) T² had no variable-contribution decomposition, and there is no Phase II with unequal subgroup sizes. CUSUM limits with unequal sizes are in standard-error units (data units only for a constant size).
  - Staging applies to the Shewhart charts (I-MR, X̄-R/S, p, np, c, u), not to the time-weighted, Laney, G/T or T² charts; Box-Cox charts with a known σ are not offered (σ would be in transformed units), only a known centre.
  - Charts are vector metadata only (`ChartSpec`).

## 2h — SPC completion: exact limits, one-sided CUSUM, decomposition, generalized variance, staging (done 2026-11-17)
**Scope note.** Most of the 2h request list (P/NP/C/U with stair-step limits, Laney P′/U′, MA, EWMA with exact and asymptotic limits, tabular CUSUM with FIR, G and T charts, Hotelling T² with Phase I Beta and Phase II F limits, Nelson/Western Electric integration, `Reportable`/`ChartSpec`) had already been delivered in 2g. 2h renames the modules to the requested layout (`spc/attributes.rs`, `spc/time_weighted.rs`, `spc/multivariate.rs`) and adds what was genuinely missing. 14 new integration tests (`tests/spc3.rs`; the crate now has 179 unit + 123 integration tests).

- **Exact limits** (`attributes`): `LimitMethod::{Normal, Exact}` with `p_chart_with`, `np_chart_with`, `c_chart_with`, `u_chart_with`. Exact limits are binomial (p, np) or Poisson (c, u) quantiles with tail probability Φ(−k) on each side: the upper limit is the smallest x with P(X > x) ≤ tail, the lower the smallest x with P(X ≤ x) ≥ tail, and a count fails test 1 when it is above the upper or below the lower limit. Sizes may vary per sample (stair-step limits). Tests 2–4 keep the normal σ zones.
- **T chart**: `t_chart_with(…, TimeModel::{Weibull, Exponential})` (exponential: scale = mean, shape 1).
- **CUSUM** (`time_weighted`): `cusum_tabular_sided` with `CusumSides::{Both, UpperOnly, LowerOnly}`; FIR through `head_start` (applies only to the computed sides).
- **Rules on smoothing charts**: `moving_average_with_tests`, `ewma_with_tests` accept any `SpecialCauseTests` (Nelson, Western Electric, custom); the default stays test 1 at k.
- **Staging** (`stages`): `staged_smoothing_individuals` / `_subgroups` (MA, EWMA), `staged_cusum_individuals`, `staged_laney_p` / `_u`, `staged_g`, `staged_t`, `staged_hotelling_individuals`, each with a `Reportable`. Every stage has its own target, σ, limits and model; patterns do not span stage boundaries.
- **T² decomposition** (`multivariate`): `HotellingChart::contributions` (n·dⱼ(S⁻¹d)ⱼ, rows add up to T², can be negative) and `hotelling_decomposition_individuals` / `_subgroups`, which also give the Mason–Young–Tracy unconditional terms as one-variable T² charts with their own limits at the same α (Phase I, Phase II and known-parameter references are reduced consistently).
- **Generalized variance** (`multivariate`): `generalized_variance` chart of |Sᵢ| with limits b₁|Σ| ± k√b₂|Σ| (Montgomery §11.3), |Σ| known or |S̄|/b₁; `generalized_variance_constants(n, p)`.
- **Verification** (`gen/stats_ref8.py` → `tests/common/data8.rs`): exact limits and flagged points for variable-size p and u charts, np and c, against a brute-force search on SciPy `binom`/`poisson` cdf and sf (an exact lower limit flags a zero count that the normal chart cannot); one-sided and FIR CUSUM against NumPy; exponential T against the closed form; T² contributions, unconditional terms and limits (individuals and subgroups) against NumPy/SciPy; b₁, b₂ against an independent product formula and a 150,000-sample Monte Carlo of |S| inside the generator (mean within 2 %, variance within 10 %); |S| chart limits with estimated and known Σ; p = 1 identity (|S| = s²: b₁ = 1, b₂ = 2/(n−1)); staged results equal separate charts; row sums of the contributions equal T²; JSON round trips of every new report; error cases.
- **Judgement calls and limits (please review)**:
  - **No Minitab output was available.** The exact-limit convention (tail Φ(−k) each side, discrete quantiles, strict inequalities) is mine; Minitab's treatment of the discrete boundary may differ by one count.
  - Run and zone rules on MA/EWMA points are allowed on request, but smoothed points are autocorrelated, so those rules signal more often than their nominal rates; the default remains test 1.
  - The generalized-variance limits are the usual 3σ moment limits, not exact probability limits (|S| is a product of independent χ² variables; its exact quantiles were not implemented). They are asymmetric in probability and can be loose in the lower tail for small n.
  - No separate Montgomery data set was available for Hotelling T² or |S|; they are verified against independent NumPy/SciPy computations and identities, not against the book's printed tables. The Montgomery CUSUM/EWMA example (2g) remains the textbook check.
  - Staged charts: T² and G/T/Laney stages are estimated independently, so each stage needs enough points for its own estimate (T²: more than p + 1).
  - The decomposition's additive contributions can be negative; the MYT *conditional* terms are not provided.

## 2i + 2j — Process capability & measurement systems analysis (done 2026-11-17)
**Scope note.** Much of the requested list had been delivered earlier: normal Cp/Cpl/Cpu/Cpk/Cpm/Pp/Ppl/Ppu/Ppk, Pp (χ²) and Ppk intervals, Box-Cox (profile likelihood) and Johnson SB/SU/SL capability, Weibull/lognormal/gamma capability by the 0.135 %/99.865 % points and the sixpack (2e); crossed ANOVA and X̄-R Gage R&R with EV/AV/PV/%SV/%Tolerance/ndc and nested Gage R&R (2f). 2i + 2j adds what was missing, as a new top-level `openstats_stats::capability` module (re-exporting the 2e items) and new `msa` modules. Crate tests: 181 unit + 16 new integration tests (`tests/capmsa.rs`); the workspace runs 569 tests.

- **Capability confidence intervals** (`capability::ci`, `capability_intervals`): Cp and Pp by χ² (`Ĉ√(χ²/ν)`); the one-sided indices CPL, CPU, PPL, PPU by inverting the non-central t (`T = 3√n·Ĉ ~ t_ν(δ)`, δ solved by Brent so that `P(T ≤ t; δ) = 1 − α/2` and `α/2`); Cpk and Ppk by Bissell's normal approximation; Cpm by Boyles' χ² approximation with Patnaik degrees of freedom. The overall (Pp family) intervals need no extra input; the within-σ intervals take the σ estimate's degrees of freedom (`pooled_within_df` gives N − k for a pooled standard deviation). `intervals_section` renders the table.
- **Distributions** (`Family` extended): exponential (scale = mean − threshold), normal (MLE, divisor n), logistic (nested root-finding on the two score equations), loglogistic (logistic MLE on ln(x − threshold)); all usable in `fit_distribution`, `capability_distribution` and the new `rank_families`. `identify_distributions` fits normal, lognormal, exponential, Weibull, gamma, logistic, loglogistic and the Johnson system and ranks them by Anderson–Darling statistic (`ranking_section` for the report).
- **Empirical percentiles** (`capability_empirical`): `Pp = (USL − LSL)/(P99.865 − P0.135)`, Ppl/Ppu with the sample median, type 6 quantiles (configurable). With fewer than 740 observations the 0.135 % point is not an interior order statistic: this is an error unless `allow_extremes` is set, which uses the sample minimum/maximum and records a note.
- **Sixpack extras** (`capability::extras`): `capability_by_subgroup` (Ppkᵢ and Ppᵢ from each subgroup's own mean and s), `running_average` (cumulative mean with `x̿ ± 3σ_within/√Nᵢ` limits) and `extended_sixpack_subgroups` / `_individuals` (the six standard panels plus the two new ones, 8 `ChartSpec`s, a `Reportable`).
- **Gage linearity and bias** (`msa::linearity`): bias = measurement − reference; OLS of bias on reference (N − 2 df) with coefficient tests and intervals, `Linearity = |slope|·process variation`, `%Linearity = 100|slope|`; per-reference bias with one-sample t tests (nⱼ − 1 df), `%Bias = 100|bias|/process variation`, and the average bias over all measurements (N − 1 df); chart data with the fitted line and its confidence band. The process variation is supplied by the caller.
- **Attribute agreement** (`msa::attribute`): from long-format columns (sample, appraiser, rating, optional standard; balanced design). Within appraisers, versus the standard, between appraisers and all appraisers versus the standard: matched counts with Clopper–Pearson intervals; Fleiss' κ (overall and per category, SE and Z under H₀, one-sided P) within appraisers and between appraisers; Cohen's κ (pooled trial pairs) of each appraiser against the standard, all appraisers against the standard, every pair of appraisers, and per category against the rest, with the Fleiss–Cohen–Everitt null SE and the Fleiss (1981) non-null SE for the interval; with a designated defect category: miss rate, false-alarm rate and the number of mixed samples. Interval-plot `ChartSpec`s for the agreement percentages.
- **Reports/charts**: `Reportable` for `EmpiricalCapability`, `ExtendedSixpack`, `RunningAverage`, `LinearityBias`, `AttributeAgreement`; `intervals_section`, `ranking_section`; every report round-trips through JSON in the tests.
- **Verification** (`gen/stats_ref9.py` → `tests/common/data9.rs`; independent NumPy/SciPy code): exact PPL/PPU limits against `scipy.stats.nct` and `brentq` (and the generator checks the method's 95 % coverage by simulation), Pp, Ppk and Cpm intervals, within-σ intervals with df 31; exponential, logistic, loglogistic and normal MLEs against `scipy.stats.logistic.fit` and closed forms (parameters and log-likelihoods); empirical percentiles against `numpy.percentile(method="weibull")`; linearity and bias against `scipy.stats.linregress` and `ttest_1samp`; attribute agreement (24 samples, 3 appraisers, 3 trials) against hand-written NumPy code including Clopper–Pearson from the beta quantile, Fleiss κ/SE, Cohen κ, miss/false-alarm rates, and the non-null Cohen SE by the multinomial delta method (numerical gradient); the Cohen null SE is checked by a 40,000-run Monte Carlo inside the generator. Unit tests reproduce the textbook Fleiss example (10 subjects, 14 raters, 5 categories: κ ≈ 0.210) and Cohen's 2×2 example (po 0.7, pe 0.5, κ 0.4).
- **Judgement calls and limits (please review)**:
  - **No Minitab output was available**, and I did not reproduce the AIAG MSA manual's linearity/bias or attribute data sets (I could not recall them reliably, so I did not guess). The conventions below follow the literature and my reading of Minitab's documentation and are not confirmed against Minitab.
  - There is **no exact interval for two-sided Cpk/Ppk**; Bissell's approximation is used, and Cpm's interval is also an approximation. The within-σ one-sided intervals are exact only when the within estimate is a scaled χ² (a pooled standard deviation); for R̄/s̄ or moving-range estimates they treat the supplied degrees of freedom as if it were, and no default degrees of freedom is assumed for moving ranges.
  - Extra distributions are fitted with the threshold **fixed** (default 0), as in 2e; no threshold estimation. The Johnson system enters identification through its transformed-data A² and p-value, so its A² is not strictly comparable with the other candidates' fitted-distribution A².
  - Empirical-percentile capability needs a large sample; see above. The quantile type is configurable but only type 6 was checked against NumPy.
  - Attribute agreement: the Cohen κ "versus standard" pools all trials; Minitab's own pooling and standard errors for its Fleiss-versus-standard statistic may differ. Ordinal statistics (Kendall's coefficients) are not implemented. κ is `NaN` when raters use a single category throughout.
  - Linearity & bias uses the average of the individual biases for the "Average" row (N − 1 df); the regression uses all N measurements.
  - The extended sixpack's capability-by-subgroup uses each subgroup's own mean and standard deviation (a within-subgroup Ppk), not the pooled σ; for individuals it needs a block size.
  - Charts are vector metadata only (`ChartSpec`).

## 2k — Design of experiments and response optimisation (done 2026-11-17)
New module `openstats_stats::doe` (std only). Crate tests: 17 new integration tests (`tests/doe.rs`); the workspace runs 586 tests, clippy `-D warnings`, rustdoc `-D warnings` and `--no-default-features` builds are clean.

- **Designs** (`doe::design`, `factorial`, `hadamard`): `Design` (coded matrix, natural units via `FactorSpec`, point types, blocks, standard order, seeded randomisation within blocks). `full_factorial` (k ≤ 15, replicates, centre points, blocking by generator words); `fractional_factorial(k, p, generators)` with generators like `"D=ABC"` or an automatic choice (maximum resolution, then minimum aberration; exhaustive search up to 2·10⁶ combinations, greedy beyond); `Design::alias_structure(max_order)` gives the defining relation, resolution and confounded classes. `plackett_burman(runs, factors)` for 4–48 runs from Hadamard matrices (Paley I/II and Sylvester doubling); every matrix is checked orthogonal.
- **Response surface** (`doe::rsm`): `central_composite` (circumscribed, inscribed, face-centred; rotatable, orthogonal or custom α; half/quarter cubes at resolution V or higher; separate centre-point counts; two blocks) for k = 2–10; `box_behnken` for k = 3, 4, 5, 7. `analyze_surface` fits linear, linear + interactions or full quadratic models (coefficient table, ANOVA by Linear/Square/2-Way, lack of fit and pure error, S, R², adjusted, predicted R²) and `canonical_analysis` (stationary point, eigenvalues, maximum/minimum/saddle/ridge).
- **Factorial analysis** (`doe::analysis`): effects (2 × coefficient), coefficients with SE/T/P, sequential and adjusted ANOVA by order, curvature from centre points, lack of fit, aliases per term, Lenth's PSE with individual and simultaneous margins when there are no error degrees of freedom.
- **Taguchi** (`doe::taguchi`): arrays L4, L8, L9, L12, L16, L18, L25, L27, L32, L81; factor-to-column assignment; S/N (smaller, larger, nominal `10 log ȳ²/s²`, nominal `−10 log s²`); response tables with Delta and Rank for S/N and means; ANOVA with % of SS; best levels and additive predictions.
- **Optimiser** (`doe::optimize`): Derringer–Suich `Goal::{Maximize, Minimize, Target}` with shape exponents, importance-weighted composite `D = (∏ dᵢ^wᵢ)^(1/Σw)`, multi-start Nelder–Mead (Phase 1f) on a bounded sine transform, fixed factors, restricted ranges, distinct local optima, predictions with SE fit.
- **Reports and charts**: `Reportable` for `Design`, `AliasStructure`, `FactorialAnalysis`, `SurfaceAnalysis`, `TaguchiAnalysis`, `OptimizationResult`. Chart builders: main effects, interaction, cube, Pareto of effects, normal plot of effects, Taguchi main effects, optimisation plots, contour and surface (`SurfacePlot`). `ChartKind::{Contour, Surface}` added to the report model (JSON names `contour`, `surface`).
- **Verification** (`gen/stats_ref10.py` → `tests/common/data10.rs`, independent NumPy/SciPy): the Box–Hunter–Hunter pilot-plant 2³ (T 23, C −5, TK 10, …) and Lenth margins; Montgomery's 2⁴ filtration example (A 21.625, AC −18.125, AD 16.625, …) with the full order-≤2 regression (SE, t, p, S, R², adjusted, PRESS-based R²) and ANOVA F; the 2⁴⁻¹ (D = ABC) contrasts, aliases and Lenth; a 13-run rotatable CCD quadratic fit with lack of fit, pure error, canonical analysis, prediction and SE; a Taguchi L9 experiment (all four S/N types, response tables, both ANOVAs, best levels); structure tests (orthogonality of every array and Hadamard order 4–48, CCD α, Box–Behnken zero pattern, blocking); contour points lie on their levels; the optimiser against a 401 × 401 brute-force grid, a closed-form target line, fixed factors and ranges; JSON round trips; error cases.
- **Judgement calls and limits (please review)**:
  - **No Minitab output was available.** Conventions (term labels, ANOVA layout, generator choice by minimum aberration) follow Minitab's documentation from memory and are not confirmed against it.
  - Plackett–Burman designs come from Hadamard constructions and may differ in run order or column signs from the classical cyclic tables; they are orthogonal and balanced.
  - There is **no L11 array** (the 12-run, 11-column array is L12). Four-level and mixed 2/4-level arrays (L16 4⁵, L32 2¹4⁹, L36, L50, L64) are not provided. L18 is a tabulated array verified as orthogonal.
  - Box–Behnken is available for k = 3, 4, 5, 7 only; CCD blocking supports two blocks.
  - Default generators are the output of my minimum-aberration search; Minitab's catalogue may choose a different, equal-resolution design.
  - The Montgomery data used are the published 2⁴ responses; the CCD and Taguchi data are synthetic and verified only against NumPy/SciPy.
  - Prediction models (`PolynomialModel`) ignore block and curvature terms.
  - Cube and contour data are `ChartSpec` conventions (categories label the cube corners with their means; contour polylines are NaN-separated); drawing is left to the front end.
  - The optimiser is a local search from many starts, not a global guarantee; the response models must be supplied in coded units.

## Roadmap reordering (approved 2026-11-17, after 2k)
Phase 2 (`openstats-stats`) is complete. The remaining roadmap is reordered so OpenStats becomes usable end-to-end first:
- **Phase 3 (immediate priority, formerly Phase 4):** `openstats-charts` (3a 2D vector engine, 3b statistical/quality plots), then `openstats-ui`, the Tauri v2 desktop wrapper and the WebAssembly browser target.
- **Phase 4 (deferred, formerly Phase 3):** advanced analytics and machine learning (CART, Random Forests, GBDT, MARS), reliability/survival, multivariate (PCA, clustering), time series (ARIMA), nonparametrics, and the `.mac` macro / session-command engine (`openstats-script`). Nothing is abandoned.
- **Extension points.** UI menus, the Navigator tree and `SessionReport` hooks are designed so Phase 4 modules plug in later: every procedure is a `Reportable` that yields `Section`s of tables and `ChartSpec`s, and `openstats-charts` renders any `ChartSpec` without per-procedure code, so a new procedure needs no change to the chart or UI layers.

## 3a + 3b — `openstats-charts` (done 2026-11-17)

Pure std Rust: no charting, font or graphics crates. Depends only on `openstats-core`, `openstats-math`, `openstats-stats`.

**Core (`src/core/`)** — flat display-list `Scene` (lines, polylines, polygons, rects, circles, wedges, text, markers, clip push/pop) with hit targets and axes metadata; `Scene::embed` for multi-panel layouts. Scales: linear, log, probability (normal / Weibull / lognormal), band, time (Excel serial days). Ticks: Talbot–Lin–Hanrahan extended algorithm, decade log ticks, probability ticks, calendar ticks. `Frame` lays out margins from measured text (title, legend, axis labels, rotated category labels, optional secondary axis, extra right margin) and draws the chrome. Fonts: Helvetica-compatible advance widths, 5×8 bitmap glyphs for ASCII plus hand-drawn Greek/math symbols (X̄ σ μ χ² λ …). Palette is Okabe–Ito based with colour, marker and dash cycling; light and dark themes.

**Renderers (`src/render/`)** — SVG emitter (clip paths, overline text); Canvas 2D command stream as JSON and as a binary stream (`OTC1`) with a decoder, plus a JavaScript player; software PNG rasteriser (exact-coverage polygon fill, Wu hairlines, dashes, round joins, bitmap text) encoded through the Phase 1b zlib/DEFLATE encoder.

**Interaction** — `HitIndex` (uniform grid; nearest-within-tolerance and rectangle brushing returning series/row indices), `Viewport` zoom and pan in each scale's transformed space, `AxesMeta` pixel↔data mapping, reference-line overlays.

**Plot builders (`src/plots/`)** — `render_spec` draws every `ChartKind` emitted by openstats-stats; `render_panels` lays several specs out in a grid.
- Basic: histogram with normal/fitted/density overlays, grouped and notched box plots with outlier markers and means, individual value plot, interval plot (vertical/horizontal), clustered bar, pie, ECDF, probability plot (normal, lognormal, Weibull) with a pointwise confidence band.
- Relationships: scatter (grouped), polynomial fit with confidence and prediction bands, robust LOWESS, generic line chart (main effects, interaction), cube plot, contour, software-3D response surface.
- Quality: control-chart panels (stepped staged limits, zone shading, red special-cause markers labelled with failed test numbers, end-of-line limit labels, stage dividers), Pareto with cumulative-% axis, capability histogram with Within/Overall curves and spec lines, sixpack / four-in-one / Gage R&R panel layouts.

**Tests** — unit tests per core module plus `tests/render_all.rs`: every chart type from real openstats-stats output (I-MR, X̄-R, P chart, capability sixpack) and hand-built specs; each scene is checked for finite geometry, balanced clips, XML-valid SVG (openstats-core reader), PNG decode with size and non-blank check, canvas JSON and binary round trip, determinism, hit-testing and zoom, degenerate input giving a notice scene.

**Honest limits** — Text uses 5×8 bitmaps in PNG output and estimated widths for Greek letters, so PNG text is approximate and SVG text depends on the viewer's font. The JavaScript canvas player is shipped but not executed here (no JS runtime in the sandbox). No pixel comparison against Minitab was done. PNG goldens are determinism/difference checks on this renderer's own output, not stored reference images. The probability-plot band comes from regression on transformed plotting positions, not Minitab's exact likelihood-based band. (The surface plot was a simple projection here; 3c replaced it with a full software 3-D pipeline.) Four-in-one residual and Gage R&R layouts use whatever specs the stats crate emits.

## 3c + 3d — 3-D surface/contour engine and virtualized worksheet grid (done 2026-11-17)

**3c — `openstats-charts/src/render_3d.rs`.** Right-handed, z-up world; row-major `Mat4` on column vectors with look-at, perspective and orthographic matrices. `Camera` is an orbit camera (azimuth, elevation clamped to ±89°, distance, zoom 0.2–20, target) with hooks `orbit(dx, dy)` (0.5°/pixel), `zoom_by`, `pan_by` and `reset`; `ViewTransform` fits the reference box into the pixel area and is shared by rendering and picking. Meshes are depth-sorted far-to-near by mean view depth (ties by cell order) with Sutherland–Hodgman near-plane clipping. Lighting: directional `Light` with ambient, Lambert diffuse and Blinn–Phong specular, two-sided, flat or smooth (averaged vertex normals) shading, colour from the height ramp; modes `Surface`, `SurfaceWireframe`, `Wireframe`. `contour_lines` is marching squares (corner ≥ level is high, saddles resolved by the cell-centre mean, segments stitched into open or closed polylines); `equal_levels` and `nice_levels` choose levels; `contour_plot` adds filled bands, labels and DOE design-point overlays (factorial, axial, centre). `pick` maps a pixel back to the nearest unclipped grid vertex. `ChartKind::Surface` specs from openstats-stats now render through this engine (the old projection was removed). Tests: 13 unit tests plus `tests/surface3d.rs` (CCD and Box–Behnken specs end to end, contour points lie on their level within 0.05, design-point overlays, all modes × both projections, camera hooks, picking).

**3d — `openstats-ui/src/grid/`.** New std-only crate depending on openstats-core and openstats-charts. `GridController` owns a `Workbook` and its `History`; `render()` returns a `Scene` of only the visible window (frozen header and row gutter, clipped body, selection shading, active-cell box, in-cell editor, scrollbar thumbs), so frame cost depends on the window, not the row count. Rows up to 10M and 4000 columns; sparse per-column widths with binary-search lookup; `ScrollProxy` maps a capped scroll element to the logical range for hosts with element-size limits.
- Headers `C1`, `C2-T`, `C3-D` with the column name below; row numbers in the gutter; missing values show `*`; numbers follow the column's `NumericFormat` (automatic, fixed, scientific, percent) and date/time patterns; overlong numbers fall back to scientific, then `#`. The formula bar and editor show full precision, or `=formula` for calculated columns.
- Editing: type to replace, F2 / double-click to edit, Enter/Tab/arrows commit and move, Esc cancels, a rejected entry keeps the editor open with the message in `status()`. Typing into a column past the data creates columns, and an empty column takes the kind of what is typed first (number, date/time, else text). Entering `=expr` gives the column a formula and recalculates dependants; calculated cells reject direct edits. Every operation is one undo step and atomic; large edits replace whole columns instead of logging a million cell edits.
- Selection: multiple ranges, shift-extend, whole columns/rows from headers, select all, Ctrl+arrow jumps (block edge / next filled cell / sheet edge), Ctrl+Home/End, PageUp/Down.
- Clipboard: tab-separated copy, cut, paste (RFC-style quoting, CRLF tolerant, a single value fills the selection, text into a numeric column fails atomically); the host may use system clipboard text through `copy()` / `paste_text()`.
- Formatting commands: number format, decimals ±1, date/time pattern, insert/delete rows, column rename and autosize, drag-resize.
- Pointer: `hit_test` (corner, header, resize edge, row header, cell, scrollbars), drag-select, header drag, thumb drag, track paging, wheel.

**Tests** — 12 unit tests (metrics, selection, TSV, formatting) and 13 integration tests in `tests/grid.rs`: headers and `*`, typing/undo/redo/escape/invalid input, formula columns, kind adoption, clipboard round trip, navigation and jumps, pointer selection/resize/rename, rows and formats, date/time, scrollbars, a 5000-row paste as one undo step, and a 1,000,000-row sheet where 240 scroll positions each draw under 4000 primitives and the worst frame stays under 100 ms in a debug build.

**Honest limits** — The grid is host-agnostic: it consumes abstract key, pointer and wheel events and emits a `Scene`. There are no DOM or `web-sys` bindings in this milestone (std only, no third-party crates), so no browser has run it and the "60 fps" claim rests on frame-cost measurements, not on an on-screen run. Text is measured with the charts crate's Helvetica-like metrics, so column fitting is approximate against real fonts. The in-cell editor draws a caret but has no text selection, IME or system-clipboard access (hosts pass clipboard text in and out). Formula columns follow the Phase 1 formula engine. Cell formats are per column, not per cell. The 3-D engine is a software painter's-algorithm renderer: intersecting triangles can sort incorrectly and there are no cast shadows or textures.

## 3e — Session window, Project Navigator, menu bar and dialogs (done 2026-11-17, commit 88f16c1)

**`openstats-ui/src/dialogs/`.** `DialogSpec` (fields on pages Main / Options / Graphs / Storage: variable listboxes with numeric/any kind and min/max counts, numbers with optional bounds, integers, optional numbers, text, choices, checkboxes) → `Values` → `validate` (against column kinds and bounds, messages per field) → `Registry::run` → `Output { section, stored columns }`. A runner is either a report function over the worksheet or a mutating function over the grid; stored columns go in through `GridController::store_columns` as one undoable step. 21 analyses are registered, each with its menu path: descriptive statistics (by variable, histogram/boxplot), 1-sample, 2-sample and paired t, normality (5 tests), correlation (Pearson/Spearman/Kendall, scatterplots), graphical summary, regression (fits/residuals/standardized residuals stored, residual plots), one-way ANOVA (Tukey, Fisher, interval and individual value plots), I-MR, Xbar-R/S, normal capability, full factorial design, five Graph commands, Calculator, Standardize and Sort. `MenuBar` (File, Edit, Data, Calc, Stat, Graph, Assistant) is generated from the registry paths plus fixed commands, with pointer and keyboard (arrows, Enter, Esc, Left/Right) navigation and enabled-state callbacks (Undo/Redo). `DialogView` draws and drives a dialog (column list, Select, page tabs, OK/Cancel, errors from validation or from the run). The Assistant maps a topic and the selected columns to a recommended analysis with the reason.

**`openstats-ui/src/workspace/`.** `Navigator` tree (project → Worksheets / Graphs / Session windows) with selection, keyboard, double-click open, right-click context menu (open, rename, duplicate, export, export PNG, delete, new worksheet/session) and in-place rename. `SessionView` lays out a `SessionReport` (headings, text, tables with the report model's aligned decimals, chart thumbnails flowing in rows, warnings, footnotes), scrolls, hit-tests thumbnails, selects, copies as Minitab-style text and paginates for printing. `Workspace` composes menu bar, Navigator, Session/graph pane, formula bar, grid and a modal dialog; each analysis chart becomes a named graph (`Histogram 1`, `Scatterplot 2`, …) in the Navigator. `execute(Command)` returns an `Outcome` (file to save, clipboard text, print pages, open-file request, error) so the host stays thin. Save/open serialize worksheets plus Session windows and graphs (as report JSON in workbook metadata under `ui.*`) into one `.otb`.

**Fixed along the way** — the factorial-design "Store design in worksheet" checkbox was ignored (always stored); now honoured.

**Tests** — 21 new unit tests (spec 2, menu 4, dialog 4, assistant 2, session 5 incl. print/aligned decimals, navigator 4) in addition to 13 `tests/workspace.rs` (menu → dialog → run → Session and Navigator; regression storage with undo/redo; remembered values; error stays in dialog; rename/duplicate refusal/export SVG/PNG/CSV/text/delete rules; multiple sheets and Session windows; copy and print; save/open round trip incl. failed open; input routing by focus; menu clicks; Assistant; resize; every analysis dialog opens and draws) and 9 `tests/analyses.rs` (every registered analysis on real data, stored columns, calculator, standardize (mean 0, variance 1), sort, bad-input paths).

**Honest limits** — Still no browser or window system has run this: everything is abstract events in, `Scene` out. Dialogs cover 21 of Minitab's analyses; GLM, DOE analysis, MSA, multivariate, nonparametrics and the rest of Phase 2 are not yet exposed. The capability dialog shows tables only (no histogram panel). Dialog text fields have no selection or IME; the multi-column list does not support Ctrl/Shift multi-select (use Select repeatedly). Sheet operations (add, rename, duplicate, delete) clear the undo history. Projects keep Session output as report data, so graphs reopen from their specs, not pixels. Printing returns page scenes; the host does the actual print job. Menu accelerators and mnemonics are limited to Ctrl+N/O/S/P/C/A.

## 3f — Host bindings, Tauri shell, remaining dialogs and editor polish (done 2026-11-17, commit e97f214)

**Dialogs.** `openstats-ui/src/dialogs/analyses_adv.rs` registers four more analyses (25 in total): General Linear Model (factors, covariates, interaction order, Type I/III SS, least-squares means), Analyze Factorial Design (effects, ANOVA, curvature, Lenth PSE from a worksheet design), Gage R&R (crossed study, optional tolerance width, interaction keep / drop / automatic) and the Hotelling T² chart (individuals or subgroups). The variable listbox now supports Ctrl-click toggle and Shift-click range selection, and Ctrl+A. Dialog text fields gained selection, word moves and clipboard (`insert_text`, `pointer_down_with`).

**Grid editor.** New `Editor` with caret, selection (shift+arrows, word moves, double-click word, drag), Ctrl+C/X/V into and out of the editor, and input-method composition (`ime_update` / `ime_commit` / `ime_cancel`, preedit drawn underlined, `ime_anchor` for the candidate window). The formula bar shows and edits selections as well. `Mods` implements `BitOr`.

**Data import and graph tear-off.** `Command::ImportData` ("Open Data File…" in File) raises `Outcome::NeedImportFile`; `Workspace::import_data` loads CSV/TSV/text, XLSX (each sheet becomes a worksheet) and gzip. Graphs offer "Open in New Window" (`NavAction::TearOff`), which raises `Outcome::TearOffGraph` with the graph's report JSON. `MenuBar::to_json` and `Command::id/from_id` export the menu tree for native menu bars, with Undo/Redo enabled state.

**`crates/openstats-host`.** `cdylib` + `rlib`, std only, no third-party crates. `Host` wraps `Workspace` with numeric events (pointer, wheel, key, paste, IME, command, open/save/import), `render()` returning the OTC1 canvas stream, `menu_json`, `title`, an outcome queue and print pages; `GraphHost` renders a graph spec at any size. `abi.rs` exports a C ABI (`ot_*`) over a thread-local host. There is no `unsafe` beyond `#[no_mangle]` on the exports (a local `[lints]` override; the workspace otherwise forbids unsafe): input goes through a host-owned buffer the shell fills after `ot_input_reserve`, output comes back at `ot_out_ptr()`. Key wire codes and modifier bits are in `keys.rs`.

**Browser glue — `openstats-desktop/web`.** `openstats.js` (`decodeFrame`, `paint`, `Abi`, `OpenStats`, `GraphWindow`, `browserHandlers`) maps mouse, wheel, keyboard, paste, resize, device-pixel-ratio and IME composition to the ABI and paints the command stream onto a `<canvas>`; a hidden textarea carries focus, paste and composition, and follows the editor caret. `index.html` and `graph.html` are the pages (tear-off through `window.open` + `postMessage`).

**Tauri v2 shell — `openstats-desktop/tauri/src-tauri`.** `tauri.conf.json`, capabilities, own `Cargo.toml` (own `[workspace]`, outside the repo workspace because Tauri needs the registry), `main.rs` with `set_menu` (native menu bar built from the menu tree; items emit `menu-command`) and `open_graph_window` (new webview on `graph.html`, spec delivered after a `graph-ready` event). `web/tauri-shell.js` supplies native open/save dialogs for `.otb` and CSV/XLSX, file read/write, system clipboard, window title, menu publishing and the tear-off window entry point.

**Tests** — `openstats-host` 4 ABI tests (key-code round trip, a full session through the ABI, outcomes for files / clipboard / print / errors / dialog requests, graph windows from tear-off JSON); `openstats-ui` new unit tests for the multi-select listbox and editor, plus grid (17), analyses (13) and workspace (18) integration tests covering the new dialogs, IME, import and tear-off. 24 Node tests (`node --test openstats-desktop/web/test/*.test.mjs`): the JS decoder against `fixtures/frame.bin` written by the Rust encoder (counts, texts, dash and colour fields, malformed input), the painter against a recording context, key and modifier mapping, event wiring with a fake wasm module (mouse, wheel, keys, paste, IME, caret-following candidate window, outcomes, commands, files, graph windows), and the Tauri glue against a mock `__TAURI__`. `cargo test --workspace`: 713 passed, 0 failed.

**Deviation from the brief** — `wasm-bindgen`, `web-sys` and `js-sys` are third-party crates, which the project forbids, so the module exposes a plain C ABI and the DOM glue is hand-written JavaScript.

**Honest limits**
- No `wasm32-unknown-unknown` target was installable here: the host crate is verified natively through the same ABI functions, but its wasm build, and the JS running against the real module in a browser, have not been run.
- The Tauri project is written against the v2 API but was not compiled or launched (no registry, no network). Expect small API drift on first build. Menu items carry no native accelerators because the workspace handles its own shortcuts and a native one would fire twice.
- The Node tests use mock DOM and Tauri objects, so real IME behaviour (Windows/macOS/Linux input methods), clipboard permissions and print dialogs are untested.
- Not yet exposed as dialogs: nonparametrics, multivariate analyses other than Hotelling T², time series, reliability (Phase 4). GLM supports one response and no random factors. Sheet operations still clear the undo history. The factorial-analysis dialog rejects columns with missing values.

## 3g + 3h — End-to-end tests, shortcuts, themes and release packaging (done 2026-11-17, commit 8ab1518, version 1.0.0)

**End-to-end suite — `crates/openstats-host/tests/e2e/`.** A `Driver` speaks only the C ABI (`ot_*` calls, input buffer, outcome slots, decoded `OTC1` frames): it finds UI elements by the text that was drawn and clicks them at that position, so menus, dialog tabs, Navigator context menus and buttons are used the way a person uses them; keyboard-only paths cover dialog fields. Eight workflows:
- *CSV → regression → charts → save/reopen*: import `line.csv`, Stat ▸ Regression ▸ Regression…, response/predictor by keyboard, residual-plot and storage options by clicking the tabs, OK. The Session report text (exported through File ▸ Export Session) is parsed and its coefficients, S, R² and error df are compared with an independent least-squares done in the test; stored FITS/RESI are checked row by row (FITS = a + b·x, RESI = y − FITS, residuals sum to 0); Undo/Redo of the storage step; the four residual plots appear in the Navigator, double-clicking opens one, "Open in New Window" yields a tear-off whose JSON a graph window renders at two sizes, "Export as PNG…" yields a real PNG; Ctrl+S then reopening in a fresh window reproduces the Session text, worksheet and graphs exactly.
- *XLSX import → capability analysis*: a hand-built XLSX (shared-string header, numeric rows) is imported; Stat ▸ Quality Tools ▸ Capability Analysis ▸ Normal… with typed spec limits; sample mean, overall and within standard deviation (MR̄/d2), Pp, Ppk, Cp and Cpk agree with independent formulas to 1e-5.
- *Every menu command and dialog*: all 51 command ids published in the menu JSON run through `ot_command`; each of the ≥ 25 dialogs opens, draws OK/Cancel and closes on Escape; unknown ids are refused.
- *Shortcuts, themes, panes*: Ctrl+Shift+D flips the window background between a light and a dark colour; Alt+S, F10, Escape; Ctrl+O/I/P outcomes; Ctrl+F and F6; typing into the formula bar, Escape discards, Enter commits; Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z; Ctrl+Shift+N, Ctrl+N.
- *Editing*: paste of a TSV block, Ctrl+Home, typing then an input-method composition (`ime_update`/`ime_commit`) into A1, Ctrl+A/Ctrl+C returning the worksheet as TSV on the clipboard.
- *Bad input*: a corrupt XLSX and a corrupt project give `Error` outcomes naming the file; a project handed to Import is refused with advice; OK with nothing chosen keeps the dialog open and names the problem.
- *Session window*: select all and copy, print, wheel/PageDown/Home/End scrolling, Edit ▸ Clear Session Window.
- *Scale*: a 200,000-row CSV scrolls 40 frames, survives three window resizes, and descriptive statistics report N = 200,000.

**Problems the end-to-end tests found, now fixed.** (1) The first data import left the unused empty "Worksheet 1" behind, so a reopened project showed the empty sheet; a project that is still a single empty worksheet now has it replaced by the imported sheets. (2) Ctrl+C / Ctrl+X in the worksheet never produced a clipboard outcome (only Edit ▸ Copy did); the workspace now returns the cells as `Outcome::Clipboard`, and Ctrl+C works in the Session window and the grid. (3) Three older tests still described the menu before the View menu was added.

**Shortcuts and accessibility.** Ctrl+N, Ctrl+Shift+N, Ctrl+O, Ctrl+I (import), Ctrl+S, Ctrl+P, Ctrl+Z, Ctrl+Y and Ctrl+Shift+Z, Ctrl+F (formula bar), F6 (next pane), F10 (File menu), Alt+F/E/D/C/S/G/V/A (menu mnemonics; every top-level initial is distinct), Ctrl+Shift+D (theme), Ctrl+A / Ctrl+C / Ctrl+X in the focused pane; Cmd counts as Ctrl on macOS. Menus display their shortcuts, an open menu or dialog has the keyboard first, and the JS shell keeps these keys away from the browser (Ctrl+F, Ctrl+I and Ctrl+Shift+D included) while leaving Ctrl+R, Ctrl+T and the like alone. New `View` menu: Go to Formula Bar, Next Pane, Dark Theme. `Key::F6` and `Key::F10` are wire codes 14 and 15.

**Themes — `openstats-ui/src/style.rs`.** `ThemeKind::{Light, Dark}` and `Chrome::of(theme)` define every non-chart colour (desk, bars, headers, buttons, accent, selection, Navigator markers, muted text) once; the grid, menus, formula bar, Navigator, Session window, dialogs and focus ring all read them, and the old hexadecimal literals in `openstats-ui` are gone. `Workspace::set_theme_kind` re-themes every component, open dialogs and graph thumbnails, and `Command::ToggleTheme` (`view.dark_theme`) is in the native menu. Tests apply the WCAG 2.x contrast formula in both themes (text ≥ 4.5:1 on background, bars and buttons; accent, alert and Navigator markers ≥ 3:1; text on the accent ≥ 4.5:1) and check that a dark render (with a dialog and a menu open) contains almost no light surfaces.

**Release tooling — `scripts/`, `.github/workflows/`.** `release.sh` (version check → fmt, clippy, tests, JS tests → web bundle → the installer for the current OS → `SHA256SUMS`; `--skip-checks`, `--web-only`, `OPENSTATS_DRY_RUN`), `build-web.sh` (compiles `openstats-host` with the `web` profile, validates the WebAssembly magic, optional `wasm-opt -Oz`, copies pages and JS, writes `_headers` and `version.json` with the module's SHA-256, gzip copies, checks that every referenced file is shipped, archives to `.tar.gz`/`.zip`), `package-macos.sh` (universal `.app` and `.dmg` through `cargo tauri build --target universal-apple-darwin`, `hdiutil verify`, `lipo` check, optional signing/notarisation from Tauri's `APPLE_*` variables), `package-windows.ps1` (`.msi` and NSIS setup `.exe` for x64 or arm64, optional Authenticode signing by certificate thumbprint with a signature check, checksums), `check-versions.sh`. CI: `ci.yml` (fmt, clippy, tests on three OSes, docs, JS tests, bundle) and `release.yml` (tag-triggered web, macOS, Windows jobs and a GitHub release). The Tauri bundle config gained icons, the `.dmg`/`.msi`/NSIS targets, platform settings and a CSP; it embeds the same `dist/web` the web build produces. App icons are drawn by `examples/make_icons.rs` with the project's own rasteriser and written as PNG, ICO (PNG entries) and ICNS (PNG payloads). The workspace and every manifest are now **1.0.0**.

**Tests** — 7 `tests/ui_polish.rs` (shortcut outcomes, both Redo spellings, Ctrl+F and F6, Alt mnemonics and F10, every shortcut printed in a menu is bound, theme recolouring, graph thumbnails follow the theme), 8 `tests/e2e`, 7 `tests/release.rs` (PNG sizes of every icon, ICO directory entries, ICNS chunks tiling the file, version agreement and bundle configuration, capabilities vs the plugins and commands the shell uses, web pages only reference shipped files, the script tests), `scripts/test/test-scripts.sh` (syntax of every script, version mismatch detected, bundle assembled around a minimal valid module and checked, non-wasm and missing modules refused, macOS script refuses elsewhere, checksums verify), 24 Node tests with regenerated fixtures. `cargo test --workspace`: 738 passed, 0 failed; clippy `-D warnings`, rustfmt, `rustdoc -D warnings`, `--no-default-features`, no TODO/stub markers.

**Honest limits**
- **Nothing was built for a real platform.** The wasm module, the `.dmg`, `.msi` and `.exe` installers, signing and notarisation, and both GitHub workflows have never run: this environment has no `wasm32` target, no network, no macOS or Windows. The scripts are tested for their own logic (syntax, validation, bundle assembly, checksums) around a minimal module, not against a real build; expect fixes on the first real run.
- **The Tauri shell has never compiled** (no registry); the end-to-end tests use the Rust host through the same ABI the browser uses, with no browser or webview in the loop.
- **Accessibility is partial.** The whole interface is drawn on a canvas, so there is no accessibility tree: screen readers see nothing. What exists is full keyboard operation, tested contrast in both themes, and IME support. A DOM mirror (ARIA live region/roles) or native menu accessibility would be needed for assistive technology; there is no high-contrast mode or reduced-motion handling, and only two built-in themes.
- Shortcuts are fixed (no customisation); Ctrl+Z/Y act on the workbook, not inside dialog text fields; native Tauri menu items carry no accelerators by design (the workspace handles keys itself).
- Projects do not store undo history or the active sheet.
- The 200,000-row test checks responsiveness with a generous time bound in a debug build; it is not a benchmark.
- Carry-forward gaps from earlier phases remain (Phase 1 and the per-phase "limits" lists above): notably no Minitab output was ever available to compare against, so Minitab parity of conventions is by documentation and literature, not by side-by-side runs.

## Phase 3 complete

3a–3h deliver the end-to-end product: data engine, statistics library, charts, 3-D, worksheet grid, Session window, Project Navigator, menus, 25 dialogs, host ABI, browser glue, Tauri shell, themes, shortcuts, tests through the real ABI, and release tooling. Version 1.0.0 is the code-complete milestone; it becomes a *released* 1.0 only after the first real wasm, macOS and Windows builds pass the checks listed under "Honest limits".

## Next — Phase 4 planning (deferred scope, proposed order)

First, **build verification** (small, blocks the release): run `scripts/build-web.sh` on a machine with the wasm target, load the page in Chrome, Firefox and Safari, run `cargo tauri build` on macOS and Windows, fix API drift, enable the CI workflows, and add a smoke test that loads the real `.wasm` in Node (`WebAssembly.instantiate`) and drives `ot_*` calls.

Then Phase 4, each module as a `Reportable` plus dialogs and tests in the same pattern as Phase 2/3 (no UI or chart changes should be needed):
1. **Nonparametrics** (Mann–Whitney, Wilcoxon, Kruskal–Wallis, Mood, Friedman, runs, sign; exact and asymptotic p-values) — smallest, reuses ranks and distributions.
2. **Multivariate** (PCA, factor analysis, discriminant analysis, hierarchical and k-means clustering, MANOVA) — builds on `openstats-math` linear algebra (SVD, eigen); add Genz MVN CDF from the Phase 1 gap list.
3. **Time series** (ARIMA/SARIMA, exponential smoothing, decomposition, ACF/PACF, forecasting intervals) — builds on the optimisers; needs a time-axis chart kind (calendar ticks exist).
4. **Reliability and survival** (Weibull/lognormal/exponential fits with censoring, Kaplan–Meier, accelerated life, warranty) — needs censored-data columns in the dialogs.
5. **Machine learning** (CART, random forests, gradient boosting, MARS) — largest; define a model-object and prediction-column protocol first.
6. **`.mac` macro and session-command engine** (`openstats-script`) — last, because it must drive every analysis by command name; the stable command ids (`file.*`, `dialog:*`) are the starting point.
Cross-cutting, each phase: dialogs registered in `analyses*.rs`, menu paths, E2E workflow tests in `tests/e2e`, SciPy/R reference tables for every new statistic, an "Honest limits" entry. Also scheduled: interpolation/splines, FFT, polynomial roots, QP/LP/IRLS (Phase 1 carry-forward), nonparametric dialogs for the remaining Phase 2 procedures, an accessibility layer (DOM mirror), and data import for Minitab files if sample files become available.

Phase 4 was proposed here; Phase 4a (tree-based ML) is logged at the end of this file.

## Post-1.0 — native desktop shell (no web view), icon, focus and selection fixes (2026-11-17)

**Why.** The Tauri shell built in 3f hosts the page in the OS web view, so the desktop app was browser-based. The requirement is that only the *web* build is browser-based. Decision (user, 2026-11-17): a native shell on Rust windowing crates; the workspace itself stays std-only.

- **`openstats-charts`**: `raster::TextPainter` / `TextRun` and `rasterize_with` let a shell draw text with system fonts; with no painter (or when the painter returns `None`) the built-in bitmap face is used as before. The rasteriser now keeps alpha (transparent backgrounds stay transparent; opaque output is unchanged).
- **`openstats-host::native`** (std only, tested): `NativeApp` (events in, `Action`s out: save file, clipboard, open-file pickers, graph tear-off, print pages as PNG), frame rasterisation at any device scale, `to_xrgb` for softbuffer-style surfaces, menu JSON → `MenuNode` tree with change detection for the native menu bar and window title, `NativeGraph` for graph windows. 7 new tests (`tests/native.rs`) plus a painter test in `tests/render_all.rs`.
- **`openstats-desktop/native`** (NOT in the workspace, NOT compiled here): winit 0.30 + softbuffer 0.4 window, fontdue system fonts, muda menu bar, rfd dialogs, arboard clipboard, IME events, multi-window graphs. `scripts/package-native-macos.sh` builds a universal `OpenStats.app` + `.dmg`.
- The Tauri shell remains in the tree for reference but is no longer the intended desktop app; the web build (`openstats-desktop/web`) stays the browser target.
- **Fixes found on first run (macOS, by the user):** wasm32 overflow in `nonparametric_min_sample_size` (`1 << 40` on a 32-bit `usize`); keyboard focus was lost because the canvas mousedown default blurred the hidden textarea; a right-click selected the canvas (selection tint over the window); the icon had an opaque white background (rasteriser forced alpha 255) and the executable was named `openstats-tauri`.

**Honest limits.** The native shell has never been compiled. Expect API fixes for the five third-party crates. Whether winit also reports typed characters as `Ime::Commit` on macOS (which could double-insert text) is unverified. Printing writes PNG files; Linux has no native menu bar; the Dock icon needs the app bundle; screen readers cannot see a drawn UI.

### Post-1.0 follow-up — themes, sheet tabs, status bar, Help, File > Open (2026-11-17)
Prompted by the user's first run and a Minitab screenshot.
- **High Contrast theme** (white on black, yellow accent, bright series colours; text ≥ 7:1 in tests) next to Light and Dark. View > Theme > Light / Dark / High Contrast; Ctrl+Shift+D cycles. The native shell remembers the choice in `~/.openstats/theme`.
- **Worksheet tabs** under the grid (click to switch, "+" adds a sheet) and a **status bar** (project name, status message, active sheet, rows × columns, theme).
- **Help menu**: Keyboard Shortcuts and About OpenStats (written to the Session window).
- **File > Open** in the native shell now accepts data files as well as `.otb` (routed by extension); before, CSV/XLSX were greyed out unless the user chose "Open Data File...". Sample data lives in `samples/`.
- **Not built (vs Minitab's window):** Command Line and History panes (need the `.mac` macro engine, Phase 4), the Predictive Analytics Module and Additional Tools menus, report tabs. Column headers show `C1`, `C2-T`… with the name row beneath.

### Post-1.0 follow-up — CPU, Dock icon, theme label (2026-11-17)
- **Redraw loop (idle CPU 100%)**: `perform()` requested a redraw while answering `RedrawRequested`, so the native shell redrew forever. Fixed (commit `4ccd856`). Confirmed by the user (2026-11-17): idle CPU about 0.6%.
- **Theme label**: the status bar's right-hand label is now the only place that names the theme; the status message is cleared on a theme change (test in `tests/ui_polish.rs`).
- **Dock icon** under `cargo run`: `set_dock_icon()` (macOS only, `objc2` crates in the native shell's own `Cargo.toml`, not the workspace) sets the application icon from the bundled PNG at startup. Confirmed by the user: the icon appears.

## Phase 4a — Tree-based machine learning (done 2026-11-17)

Scope: `openstats-stats/src/ml/` (CART, random forests, gradient boosting), `Reportable` for all three, three dialogs under **Stat > Predictive Analytics**. Std only; no new dependencies in any workspace crate; no stubs or TODO markers.

**Library (`openstats-stats/src/ml/`, about 3.5k lines)**
- `data.rs` — `Dataset`/`Feature`/`Response` (numeric and categorical predictors, missing values as NaN / absent level; regression or classification response), pre-sorted index lists so a node never re-sorts.
- `tree.rs` — recursive binary partitioning with variance (regression), Gini or Shannon entropy (classification) criteria; missing-value penalty on the gain; categorical splits (level ordering for regression and two classes, exhaustive subsets for more classes with up to 10 levels); surrogate splits ranked by agreement and adjusted agreement; importance = primary gain plus surrogate-adjusted gain; prediction routes missing values through surrogates, then the majority branch.
- `cart.rs` — minimum leaf/split/depth/complexity limits; cost-complexity pruning R_α(T) = R(T) + α|T| by the weakest-link sequence (nested, α non-decreasing); K-fold cross-validated tree size with the minimum-CV and 1-SE rules, or a fixed α or leaf count; reproducible seeded folds.
- `random_forest.rs` — bootstrap by multiplicity counts, `m_try` (default √p / p/3), out-of-bag predictions, OOB error and its curve against tree count, OOB R², OOB confusion matrix, permutation importance per tree (OOB error increase, mean and SE), impurity importance, class-probability averaging, and Meinshausen quantile-forest prediction intervals (out-of-bag, and for new rows).
- `gbdt.rs` — Friedman 2001 gradient boosting: squared error, Huber (data-driven δ), binary and multiclass deviance; shrinkage, stochastic subsampling, depth and leaf-size regularisation, validation split with best-iteration choice and optional early stopping.
- `report_impl.rs` — `Reportable` for all three: model summary, R² / confusion matrix / OOB error, relative importance table and chart, pruning sequence, terminal-node rules (first 200), surrogate tables.

**UI (`openstats-ui/src/dialogs/analyses_ml.rs`)** — CART, Random Forests, Gradient Boosting (registry 25 → 28 commands). Options pages for every parameter; Graphs page; Storage page with fits, residuals, class probabilities (CART also terminal-node numbers; forests also OOB interval limits and a choice of out-of-bag or in-sample predictions). Rows with a missing response are dropped; rows with missing predictors are kept.

**Verification**
- `tests/ml_reference.rs` — 7 tests against scikit-learn 1.9.1 (data and references generated by `tests/gen/ml_ref.py` into `tests/common/ml_data.rs`): regression and 2/3-class trees match node for node (structure, thresholds, importances); the cost-complexity path matches; squared-error and deviance boosting match stage by stage; one-stage Huber matches exactly.
- `tests/ml_models.rs` — 26 behavioural tests: hand-worked impurities and a step function, XOR, categorical splits against brute force, surrogates routing missing values, pruning-sequence nesting/monotonicity/optimality, CV selection rules, structural limits, categorical level remapping, reports and JSON round trips, input errors; forests (reproducibility, OOB size ≈ 37%, importance ranking, probabilities, interval calibration, option errors); boosting (loss falls every stage, proper probabilities, validation and early stopping, Huber against outliers, option errors).
- `openstats-ui/tests/ml_dialogs.rs` — 9 tests running every dialog through the registry path and checking the stored columns.
- Gates (all clean): `cargo test --workspace` **792 passed, 0 failed** (42 new in this phase); `cargo clippy --workspace --all-targets -- -D warnings`; `cargo fmt --check`; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo build --workspace --no-default-features`; no `TODO`/`unimplemented!`/`todo!` in the new code.
- Existing tests updated for the new menu: registry count 28; Stat submenu list gains "Predictive Analytics".

**Honest limits**
- Single-threaded; forests and cross-validation on large data are slow and memory-hungry (leaf row lists are kept when intervals are requested, with a size guard).
- Huber boosting matches scikit-learn exactly for one stage but drifts over many (< 0.3 after 30 stages): with an even count scikit-learn takes the lower middle value as the "median" in leaves, this code takes the mean of the two middle values.
- Multiclass categorical splits with more than 10 levels use an ordering heuristic, not exhaustive search. No MARS yet (listed in the Phase 4 plan).
- Quantile-forest intervals under-cover when leaves are tiny (leaves chosen to fit noise): with minimum leaf 5 the 90% OOB coverage was about 77% in testing; with 20 it is near nominal. Use a larger minimum leaf when intervals matter.
- Surrogate importance credit can give a pure-noise predictor a small positive score on small samples (best-threshold selection effect), as in other CART implementations.
- The menu is named "Stat > Predictive Analytics" (CART, Random Forests, Gradient Boosting), not "Predictive Analytics / Machine Learning". No dialog yet for scoring new data from a saved model (the library's `predict` exists).
- Not run in the native shell or the browser; verified through the registry/ABI path only.

## Phase 4b — Minitab-compatible macro scripting engine, `openstats-script` (done 2026-11-17)

Scope: the new crate `openstats-script` (std only; depends on openstats-core, openstats-math, openstats-stats), the Macro Editor panel in `openstats-ui`, File > Macros and Tools menus, macro scripts stored inside `.otb` projects. No third-party crates; no stubs or TODO markers.

**Language and engine (`crates/openstats-script`, about 4k lines)**
- `lexer.rs`, `parser.rs`, `ast.rs` — recursive-descent parser. `#` comments and `NOTE` lines, `&` continuation, `"text"`, `'column name'`, `*` missing, `C1`–`C4000`, `K1`–`K1000`, `M1`–`M1000`; `GMACRO` / `MACRO … ENDMACRO` with header parameters, `ARGS`, `MCOLUMN` / `MCONSTANT` / `MMATRIX`; `DO v = a:b [BY|/ step]`, `WHILE`, `IF / ELSEIF / ELSE`, `GOTO` / `MLABEL`, `RETURN`, `STOP`; `LET` with full expression grammar (precedence, `**`, comparisons, `AND` / `OR` / `NOT`, function calls, `C1(3)` and `M1(2,3)` elements); commands abbreviated to four letters, `;` subcommands ended by `.`, inline data for `SET` / `READ` until `END`; `%name args` macro calls. Syntax errors carry line and column.
- `compile.rs` — flattens bodies to jump-based instructions, so `GOTO`, loops, stepping and breakpoints are exact.
- `machine.rs` — the interpreter. Variables are the worksheet's own columns, constants and matrices, so scripts and the worksheet always agree. Expressions are lowered to formula-engine source and evaluated by `openstats-core` (the same engine as the Calculator). Macro arguments are bound by reference; non-argument `MCOLUMN` / `MCONSTANT` / `MMATRIX` names are temporaries erased on return, stop or failure. Calls resolve to macros in the same script, then to scripts stored in the project (`%name`). Limits: instruction budget, nesting depth, output sections. Constants may hold text (string handling). Every worksheet change goes through the undo history. Run / Step / Stop / breakpoints are machine operations; Step executes one source line.
- `commands.rs` — data (`SET`, `READ`, `NAME`, `ERASE`, `COPY` with `USE`, `SORT` with `BY` / `DESCENDING`, `PRINT`), column statistics into constants (`MEAN`, `SUM`, `STDEV`, `MEDIAN`, `MINIMUM`, `MAXIMUM`, `RANGE`, `N`, `NMISS`, `COUNT`, `SSQ`), `DESCRIBE`, `CORRELATION` (Pearson, Spearman, Kendall), `REGRESS` (Minitab form `REGRESS C1 k C2…`, with `FITS`, `RESIDUALS`, `SRESIDUALS`, `COEFFICIENTS`, `MSE`, `RSQUARED`, `NOCONSTANT`, `CONFIDENCE`), `TTEST`, `ONEWAY`, `CART` / `FOREST` / `BOOST` (text response = classification; `DEPTH`, `LEAF`, `TREES`, `RATE`, `SEED`, `FOLDS`, `FITS`, `RESIDUALS`, `NODE`), matrices (`COPY` between columns and matrices, `TRANSPOSE`, `MULTIPLY`, `INVERT`), `ECHO` / `NOECHO`, `TEXT`. Results become Session-window sections; the commands call the same `openstats-stats` paths as the dialogs.
- `highlight.rs` — token classes for syntax colouring.

**UI (`openstats-ui`)**
- `workspace/macro_editor.rs` — editor panel: text editing with undo/redo, selection (keyboard, mouse, word and line), auto-indent, Tab / Shift+Tab block indent, line numbers, syntax colours in all three themes, breakpoint gutter, current-line marker, error line and error strip, Run / Continue, Step and Stop toolbar.
- The macro editor occupies the upper pane with the Session window directly below it, so output appears as the macro runs. `Top::Macro`, `MacroItem`, a **Macros** folder in the Navigator (open, rename, export `.mac`, delete, New Macro), `Command::Macro*`.
- Menus: **File > Macros** (New Macro, Open Macro File…, Save Macro As…, Macro Editor) and a new **Tools** menu (Run Macro F5, Step Macro F8, Stop Macro Shift+F5, Toggle Breakpoint F9). Keys F5 / F8 / F9 added to `Key` and to the host, native-shell and web key tables.
- Each Run or Step is one undo step ("Macro NAME"), formula columns are recalculated inside it. A failed run keeps the work done before the error (undo reverses it). Editing a paused macro abandons the run.
- Persistence: macros are written to `ui.macro.NNNN` metadata keys of the `.otb` (name line, then the source), removed from the live workbook after saving and restored on open. `.mac` files opened through the data-file picker (File > Open / Open Data File; the native picker filters now include `mac`) import as macros.

**Verification**
- `openstats-script`: 17 unit tests (lexer, parser, compiler) + `tests/machine.rs` 45 tests: arithmetic and element assignment, DO (steps, negative step, empty range), WHILE, IF chain, GOTO, text constants, macro arguments by reference, optional `ARGS` / `NARGS` / `ISSET`, temporaries erased, recursion and depth limit, error lines and call traces, endless loop stopped by the budget, step budget pause and resume, stepping, breakpoints, Stop, undo as one group, every command group (including regression, CART / forest / boosting fits and stored columns, matrices, singular matrix), macro library calls, highlighter classes.
- `openstats-ui`: 10 editor unit tests + `tests/macros.rs` 24 tests (run, one-step undo and redo, formula recalculation, step markers, breakpoints, Stop erasing temporaries, runtime and syntax errors, edit-while-paused, `%name` calls, worksheet-change guard, `.otb` round trip, `.mac` import/export, Navigator, menus and shortcuts, toolbar clicks, clipboard focus, resize, all themes).
- Existing tests updated: menu labels gain "Tools"; Navigator tree tests count the Macros folder; web key-code test covers F5/F8/F9 (17 web tests pass).
- Gates (all clean): `cargo test --workspace` **889 passed, 0 failed** (97 new this phase); `cargo clippy --workspace --all-targets -- -D warnings`; `cargo fmt --check`; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo build --workspace --no-default-features`; no `TODO` / `unimplemented!` / `todo!` in the new code.

**Honest limits**
- This is a Minitab-*style* language, not a byte-compatible one: I could not test against real Minitab macros here. Commands are the subset listed above (not the full Minitab command set); unknown commands fail with a clear error and line. `ARGS` is read as extra optional arguments; calling by `%name` uses OpenStats macro names (project macro names, spaces allowed but awkward to call).
- Expressions have no matrix arithmetic (matrix elements are read as scalars; whole-matrix work uses `COPY`, `TRANSPOSE`, `MULTIPLY`, `INVERT`) and no element indexing inside the formula engine itself (indexing is resolved by the interpreter before evaluation).
- Breakpoints live in the editor and are not saved with the project. While a run is paused, switching worksheets is refused. Editing the text abandons a paused run.
- A stored script called with `%name` that has an error reports the error line of the calling line in the open macro (plus the call trace in the Session window).
- Single-threaded; a long macro blocks the window until it finishes or reaches the step budget (200 000 instructions per Run, then it pauses; Run continues). There is no output redirection (`OUTFILE`), `EXECUTE`, `STACK` / `UNSTACK` or `PROMPT`.
- Not run in the native macOS shell or the browser: the native picker filter, F5/F8/F9 key tables and `.mac` import are covered by Rust and web unit tests only; `openstats-desktop/native` depends on crates.io crates and was not rebuilt in this sandbox.

## Phase 4c — Time series, forecasting and response optimization (done 2026-11-17)

Scope: `openstats-stats/src/timeseries/` (new), `openstats-stats/src/doe/response_optimizer.rs` (new), `openstats-ui/src/dialogs/analyses_ts.rs` (new). Std only; no new dependencies in any workspace crate; no stubs or TODO markers.

**Scope note.** The Derringer–Suich desirability engine (target / maximum / minimum goals, shape weights, importances, multi-start search, fixed factors and ranges, `Reportable`, optimization plots) already existed from Phase 2k. 4c adds the part that was missing, a worksheet-level Response Optimizer (design from columns, one model per response, goal parsing, dialog), and the whole time-series suite.

**Library — `openstats_stats::timeseries` (about 2.2k lines)**
- `acf` — autocorrelations with Bartlett standard errors, t statistics, Ljung–Box Q and confidence limits; PACF by Durbin–Levinson with ±z/√n limits; `ljung_box` with degrees-of-freedom correction; `difference` (ordinary and seasonal).
- `smoothing` — single, double (Holt) and Holt–Winters (additive and multiplicative seasonality, with or without trend). Weights left blank are chosen by least squares (multi-start Nelder–Mead on a logistic transform); initial states from the first one to three seasons. Forecasts with prediction intervals from the exact variance weights of the additive-error model; `select_smoothing` ranks every sensible model by approximate AICc.
- `arima` — ARIMA(p,d,q)×(P,D,Q)ₛ with optional constant. Methods: conditional least squares, exact Gaussian maximum likelihood (Kalman filter on the stationary state-space form; the initial covariance is found by doubling, the update step uses the companion structure so a step costs O(r²)), or CSS then ML (default). Search is unconstrained: each AR and MA block is mapped through partial autocorrelations `tanh(u)`, so every iterate is stationary and invertible. Coefficient table (AR, SAR, MA, SMA, Constant) with SE from the numerical Hessian, t and p; σ², log-likelihood, AIC, AICc, BIC; Ljung–Box on residuals; residual ACF; one-step fits and residuals; forecasts through the expanded operator `φ(B)Φ(Bˢ)(1−B)ᵈ(1−Bˢ)ᴰ` with ψ-weight prediction intervals; `rank_arima` orders candidate models by AICc.
- `decompose` — trend by regression (linear, quadratic, exponential growth, constant) or centred moving average; additive or multiplicative; seasonal indices, detrended, deseasonalised, fitted and irregular series, accuracy measures; forecasts with leverage-adjusted intervals.
- `report_impl` — `Reportable` for ACF, PACF, smoothing, ARIMA and decomposition; `report_with_forecast` adds a forecast table and a plot with the prediction limits; ACF/PACF bar charts with limits, series-with-fits plots, component charts.

**Response Optimizer — `openstats_stats::doe::response_optimizer`**: `design_from_columns` (factors coded by observed range, point types inferred), `fit_responses` (linear, interactions or full quadratic per response through `analyze_surface`; aliased terms are reported), `model_summary_section`, and `parse_goal` (`max LOWER TARGET [SHAPE]`, `min TARGET UPPER [SHAPE]`, `target LOWER TARGET UPPER [SHAPE_LOW [SHAPE_HIGH]]`). The optimization report now shows each goal's weight next to its importance.

**UI (`analyses_ts.rs`, registry 28 → 36)** — **Stat > Time Series**: Autocorrelation, Partial Autocorrelation, Single Exp Smoothing, Double Exp Smoothing, Winters' Method, ARIMA, Decomposition. **Stat > DOE > Response Optimizer**. Every dialog has Options, Graphs and Storage pages. Storage: fits, residuals, forecasts with lower and upper limits, ACF/T/LBQ/PACF, and for decomposition trend, detrended, seasonal, deseasonalised data; for the optimizer, fitted values per response and the composite desirability of each run. Series: trailing blanks are ignored, any other missing value is an error with its row number. Optimizer inputs: goals as `;`-separated text (one per response), importances, optional factor ranges and fixed settings in natural units (`*` = free).

**Verification** (new tests: 19 + 6 + 9 = 34; `cargo test --workspace` **923 passed, 0 failed**, up from 889)
- `tests/timeseries.rs`, references from NumPy/SciPy in `tests/common/ts_data.rs` (generator `tests/gen/ts_ref.py`, no time-series package):
  - ACF, Bartlett SE, LBQ, PACF (Yule–Walker solves) and the Ljung–Box p-value (`scipy.stats.chi2`) to 1e-10 or better.
  - ARIMA ML for AR(2) with mean, ARMA(1,1), MA(1) with mean, ARIMA(1,1,0) with drift and the seasonal (0,1,1)(0,1,1)₁₂ model against an **independent exact likelihood** (full autocovariance matrix, Cholesky, scipy Nelder–Mead + BFGS): estimates within 2–3·10⁻³, log-likelihood and AIC within 10⁻³.
  - Forecasts and their standard errors against the exact best linear predictor and the integrated ψ-weights (≤ 4·10⁻³ relative; the library truncates pre-sample innovations).
  - CSS for a pure AR(2) equals the regression of wₜ on its lags (coefficients, intercept and every residual).
  - Holt–Winters additive and multiplicative fits, SSE and 12-step forecasts against an independent recursion to 1e-10; optimised SSE at or below a brute-force grid minimum; single smoothing (fits, final level, optimal α against `minimize_scalar`); a perfect line is extrapolated exactly by Holt's method.
  - Decomposition (regression trend, multiplicative; moving-average trend, additive) against direct NumPy computation to 1e-10; report validation, text and JSON round trip.
  - Mutation checks: flipping the MA sign in the Kalman filter fails four tests; perturbing the smoothing recursion fails three.
- `tests/response_optimizer.rs` (6): design reconstruction, models recover exact quadratics, goal parsing, composite-desirability optimum at or above a 401 × 401 brute-force grid and within 1e-3 of it, geometric-mean identity, importance moves the optimum in the expected direction.
- `openstats-ui/tests/ts_dialogs.rs` (9): menu paths, stored columns against the library (ACF, fits + residuals = data, forecast lengths and limits), seasonal and non-seasonal ARIMA, decomposition components recombine, gap and trailing-blank handling, every optimizer validation message.
- Gates (all clean): `cargo test --workspace`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo fmt --check`; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo build --workspace --no-default-features`; no `TODO`/`unimplemented!`/`todo!` in the new code. Existing tests updated: Stat submenu list gains "Time Series", registry count 36.

**Honest limits**
- **No Minitab output was available.** Table layouts, default lag counts, the Winters initial-state rule and the Minitab MAPE/MAD/MSD definitions follow the documentation from memory; the initial values here (single: mean of the first six observations; double: a line through the first ten; Winters: first one to three seasons) are my choice and may differ from Minitab's, in which case fits and optimal weights differ too.
- Smoothing prediction intervals are exact for additive-error forms; for multiplicative seasonality they apply the same variance weights on a relative scale, an approximation.
- ARIMA: SEs come from a numerical Hessian (ML for ML fits; CSS fits use 2σ²H⁻¹). The Constant row's SE is `|1 − Σφ|·SE(mean)`, ignoring covariance with the AR terms. Forecasts use the Kalman one-step errors as innovations rather than the exact finite-sample predictor, so near-boundary models can differ from the exact predictor beyond the tolerances tested. Near a unit root or for weakly identified near-cancelling ARMA terms the optimiser can stop at a flat likelihood (the ARMA(1,1) test data were chosen to avoid that).
- ARIMA has no missing-value handling, no automatic order selection beyond `rank_arima`, no outlier or intervention terms and no regression (ARIMAX) terms; unit-root tests are not provided, so `d` is the caller's choice. Seasonal orders are limited to P, Q ≤ 4, D ≤ 2; p, q ≤ 10.
- Prediction intervals everywhere use the normal quantile, not Student's t.
- Decomposition: no trend-plus-seasonal joint fit (the trend is fitted first, as in Minitab's procedure); the moving-average trend leaves the first and last half-period missing and forecasts from a line through the trend values.
- Response Optimizer: coding is by the observed range, so a CCD's cube levels appear at ±1/α in the coded columns (natural-unit answers are unaffected). Response models are fitted per response with all columns present; rows with any missing value are dropped. Goals are typed as text rather than a grid; the dialog framework has no per-row table widget. The search is a local multi-start optimizer, not a global guarantee; block and curvature terms are not used in predictions.
- No time-series chart kind with calendar axes was added: plots use period numbers on the x axis.
- Not run in the native macOS shell or the browser; verified through the registry/ABI path only. `openstats-desktop/native` was not rebuilt.

Next: final release packaging (not started).

## Phase 4b+ — Full Minitab macro parity (`openstats-script`)

Goal: run authentic Minitab `.mac` files. Reproduced failures (bare `=` assignment, `'…'` text, `MEND`/`END`, `MTEXT`/`MSTORE`, string variables, one-line subcommands, `//` and `NOTE` lines containing quotes) and fixed them in the lexer, parser, compiler and machine.

**Added**
- Structure: `GMACRO`/`MACRO` closed by `ENDMACRO`, `MEND` or `END`; name on the header line or the next one; `MACRO name a b` arguments; named arguments on calls (`%m C1; Factor 3.`).
- Declarations: `MCOLUMN`, `MCONSTANT`, `MMATRIX`, `MTEXT` (string), `MSTORE` (any kind; a local takes its kind from its first assignment); `ARGS` optional arguments.
- Variables: `K1`–`K1000`, `C1`–`C4000`, `M1`–`M1000`, `S1`–`S1000` strings (`'text'` or `"text"`; `+` joins text); implicit declaration by assignment (`total = 5`, `words = 'x'`); `LET` is optional (`K1 = 5`, `C1(3) = 9`, `'Name' = C1 * 2`).
- Subcommands: `;`-separated, on one line or many, ended by `.`; modern `REGRESS; RESPONSE …; CONTINUOUS …` form.
- Comments: `#`, `//`, `NOTE` (the rest of a `NOTE` line is not tokenised, so apostrophes are safe).
- Control flow: `DO v = a:b[/step]`, `WHILE`, `IF/ELSEIF/ELSE/ENDIF`, `GOTO` with `MLABEL x`, `LABEL x`, `x:` and numeric labels.
- Errors: line, column, message, the source text of the failing line and the macro call trace; shown in the Session window and the Macro Editor error strip. Highlighter knows the new keywords, `S1` and `//`.

**Tests:** 939 in the workspace (923 + 16): 14 end-to-end in `openstats-script/tests/minitab_macros.rs` over eight `.mac` fixtures in `tests/fixtures/` (plus syntax-error line numbers, comments, call traces, MSTORE), and 2 parser unit tests.

**Gates:** `cargo test --workspace`, `clippy -D warnings`, `rustfmt --check`, `rustdoc -D warnings` and `cargo build --no-default-features` all clean; no TODO/`unimplemented!`.

**Limits (honest):**
- Only the Minitab statements and commands OpenStats already implements run; others report "not a command OpenStats knows". No real Minitab installation was available, so fixtures follow documented syntax rather than being validated against Minitab output.
- `MSTORE` semantics are my interpretation (any-kind argument/local).
- String variables live in the running macro, not the worksheet; they are not undoable and vanish when the run ends (a text constant such as `K20 = S5` does persist).
- Implicit variables created at the top level of a script or in a `GMACRO` entry persist (constants take high numbers such as K1000); in called macros they are erased on return.
- Unquoted `'text'` that matches no column is treated as a string.

## Phase 4d — Nonparametrics

Pure-Rust, std-only. New `openstats_stats::nonparametric` module and `Stat ▸ Nonparametrics` dialogs.

**Library:** 1-sample Sign test (order-statistic CI with achieved confidence); 1-sample / paired Wilcoxon signed-rank (exact subset-sum DP for n ≤ 50 without ties, otherwise tie-adjusted normal with optional continuity correction; Walsh-average Hodges–Lehmann estimate and CI); Mann–Whitney (exact U by DP for ≤ 50 per sample, otherwise tie-adjusted normal; HL difference and CI); Kruskal–Wallis (unadjusted and tie-adjusted H, z-values per group); Mood's median test; Friedman (unadjusted and tie-adjusted, block-adjusted estimated medians); Runs test (about mean, median or K; normal and exact p).

**Dialogs (7):** sign, wilcoxon (one-sample or paired), mannwhitney, kruskal, mood, friedman (stacked response/treatment/block, rejects duplicate and missing cells), runs. Registry now holds 43 dialogs.

**Tests:** 12 library tests against SciPy / exhaustive enumeration (`tests/gen/np_ref.py`), 7 dialog tests. All gates clean (test, clippy -D warnings, fmt, rustdoc -D warnings, no-default-features).

**Honest limits**
- No real Minitab output was available; formats follow documented layouts.
- Exact paths: signed-rank n ≤ 50, Mann–Whitney ≤ 50 per sample; HL intervals limited to n ≤ 3000 / 5M pairs and use a normal approximation for larger samples.
- Two-sided exact p-value is 2 × the smaller tail (capped at 1).
- Sign CI is an achieved-confidence order-statistic interval (no interpolated intervals).
- Mood's median has no per-group CIs and no continuity correction; values equal to the median count as "at or below".
- Runs test: exact p only when each side ≤ 100; values equal to K are omitted.
- Not implemented: two-sample Kolmogorov–Smirnov, Mood scale, Ansari–Bradley, normal scores, other rank tests. Dialogs do not store results.

Next: Multivariate (PCA, factor analysis, discriminant, clustering, MANOVA), then Reliability & survival, MARS, final release packaging.

## Phase 4e — Multivariate

Added `Stat > Multivariate` (openstats-stats `multivariate` module, 8 dialogs `stat.mv.*`):
principal components, factor analysis (PC and ML, orthomax rotations, regression scores),
discriminant analysis (linear/quadratic, priors, leave-one-out, canonical functions),
hierarchical clustering of observations and variables (7 linkages, dendrogram),
k-means, MANOVA (Wilks, Lawley–Hotelling, Pillai, Roy) and item analysis (Cronbach alpha).

Verification: 21 library tests against scipy/sklearn/numpy/R references (generator
`tests/gen/mv_ref.py`), 8 dialog tests. Workspace: 987 tests pass; clippy `-D warnings`,
fmt, rustdoc `-D warnings` and `--no-default-features` build all clean.

### Honest limits
- No Minitab output was available; references are scipy/sklearn/numpy/R.
- Hierarchical clustering is limited to 3000 observations.
- K-means is Lloyd plus optimal-transfer refinement, not exact Hartigan–Wong.
- Factor analysis has no promax/oblique rotations; ML uses bounded-ψ L-BFGS and may sit on Heywood boundaries.
- MANOVA is fixed-effects only; Type III-style tests via sum-to-zero coding; Roy's F is an upper bound.
- No correspondence analysis, stepwise discriminant or shrinkage.
- Genz multivariate-normal CDF (Phase 1 gap) is still not implemented.

## Phase 4f — Reliability & survival

New `openstats_stats::reliability` module (std only) and five dialogs under `Stat ▸ Reliability/Survival`
(`stat.rel.dist`, `.dist_id`, `.km`, `.alt`, `.warranty`; registry now 56).

**Library**
- `LifeData`: exact, right-, left- and interval-censored records with frequencies. A censoring column marks a row censored when it equals the chosen value; a start/end column pair gives arbitrary censoring (blank start = left, blank end = right).
- Maximum-likelihood fits of Weibull, lognormal, exponential, normal, logistic and log-logistic life distributions (location–scale form, `(μ, ln σ)` optimised by BFGS then Nelder–Mead from several starts, Hessian by central differences). Parameter, percentile, survival-probability and mean-life estimates with normal-approximation intervals (delta method); hazard, density, AIC/BIC; `rank_distributions` for identification.
- Kaplan–Meier with Greenwood standard errors, plain or log-log intervals, Nelson–Aalen hazard, median and quartiles with limits, restricted mean with standard error; log-rank, Wilcoxon (Breslow) and Tarone–Ware comparisons of two or more groups with observed and expected failures.
- Accelerated life testing: location–scale regression on one or more transformed stresses (Arrhenius in °C, linear, log, reciprocal), common scale, regression table with Wald tests, life distribution and percentiles at use stress with intervals, acceleration factors. The design is centred and scaled internally so strongly collinear Arrhenius designs stay well conditioned.
- Warranty prediction: shipment counts and returns by service period are fitted as interval- and right-censored data; expected returns are forecast per period, including future shipments.
- `Reportable` for every result, with survival, hazard and density charts, Kaplan–Meier step plots and forecast charts.

**Verification**
- 12 reference tests (`tests/reliability.rs`, data from `tests/gen/rel_ref.py`): all six families on a mixed exact/right/left/interval/weighted data set against an independent SciPy log-likelihood (log-likelihood 1e-8, parameters 1e-4, covariance 2e-3, percentile and reliability intervals 2e-3); exponential against total-time-on-test; Kaplan–Meier against `scipy.stats.ecdf` (survival, plain and log-log intervals) and explicit loops (Greenwood, restricted mean and its SE, Nelson–Aalen); log-rank, Wilcoxon and Tarone–Ware against numpy loops and `scipy.stats.logrank`; Arrhenius–Weibull regression and use-stress percentiles; warranty forecast; frequencies equal repeated rows; input errors; reports and JSON round trip.
- 8 dialog tests (`openstats-ui/tests/rel_dialogs.rs`).
- Gates: `cargo test --workspace` 1007 passed, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings` and `--no-default-features` clean; no TODO/`unimplemented!`.

### Honest limits
- No Minitab output was available; layouts and defaults follow the documentation from memory. References are SciPy/NumPy.
- Only two-parameter families: no three-parameter Weibull/lognormal, gamma, or extreme-value families, no fixed-shape fits, no least-squares (rank regression) estimation.
- No Anderson–Darling goodness-of-fit with censoring and no probability plots (survival, hazard and density plots only); percentile and survival intervals are normal approximations, not likelihood-ratio or bootstrap.
- Kaplan–Meier needs exact or right-censored data (arbitrary censoring is parametric only); no Turnbull estimator, no Cox regression, no competing risks.
- Accelerated life: common scale across stress levels, no interactions or per-level shapes, no accelerating-variable plots or relationship plots; the covariance for strongly collinear stress levels is sensitive to the finite-difference Hessian.
- Warranty: time is counted in whole periods, returns by service period must be filled without gaps, forecast has no prediction intervals and treats shipments within a period as shipped at its start.
- Dialogs do not store results in the worksheet.
- Not run in the native macOS shell or the browser.

## Phase 4g — MARS

New `openstats_stats::ml::mars` (std only) and the dialog `stat.ml.mars` under `Stat ▸ Predictive Analytics ▸ MARS Regression...` (registry now 57).

**Library**
- Forward pass (Friedman 1991): mirrored hinge pairs `max(0,x−t)` / `max(0,t−x)` multiplied into earlier terms up to a chosen degree (1–5), optional linear terms, candidate knots limited by minimum span and end span (automatic values from Friedman's formulas), gain evaluated by orthogonalisation against the current basis, stop on the R² threshold or the term limit.
- Backward pass: terms deleted one at a time by smallest RSS increase; the subset with the smallest GCV `(RSS/n)/(1−C/n)²`, `C = M + penalty·(M−1)/2`, is kept (penalty 2, or 3 with interactions, unless set).
- `MarsModel`: terms, coefficients, GCV/GRSq/R²/RMSE, the whole pruning path, variable importance (GCV and RSS increase, scaled to 100), `predict`.
- `Reportable`: summary, basis functions, equation, backward-elimination table and GCV chart, importance table and chart, fitted-versus-observed chart.
- Dialog: numeric response, continuous predictors, categorical predictors as 0/1 indicators (first level sorted is the reference), listwise deletion of missing rows, options for terms, degree, penalty, spans, threshold, linear terms and pruning; storage of fits and residuals.

**Verification**
- 8 library tests (`tests/mars.rs`, reference from `tests/gen/mars_ref.py`): additive and interaction models against a naive NumPy implementation of the same rules (forward term count, kept terms and knots, coefficients, RSS, GCV, the whole pruning path, predictions); recovery of a noise-free piecewise function; importance ordering; option behaviour; errors; report and JSON round trip.
- 3 dialog tests (`openstats-ui/tests/mars_dialogs.rs`).
- Gates: `cargo test --workspace` 1018 passed, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings` and `--no-default-features` clean; no TODO/`unimplemented!`.

### Honest limits
- No Minitab or `earth` output was available. The reference is my own naive NumPy implementation of the same algorithm, so it checks the Rust code against the written rules, not against an independent published implementation. Knot and span conventions follow Friedman's paper and `earth` from memory.
- Hinge pairs on one variable are exactly collinear with the intercept. Deletion ties between such redundant terms are broken by taking the earliest term (increases within 1e-7 of the total sum of squares of the smallest are tied); other implementations may break these ties differently and so may pick a different but equivalent model path.
- GCV pruning only: no cross-validation, no `Fast MARS` updating, no interaction penalty or variable-reuse limits; the forward pass is a direct search, so very large data sets are slow.
- Single numeric response, no weights; categorical predictors only as 0/1 indicators; rows with missing values are dropped; no GLM (binary) MARS.
- Dialogs store fits and residuals only; no partial-dependence plots.
- Not run in the native macOS shell or the browser.


## Phase 5a — Basic-statistics dialogs

Thirteen new dialogs under `Stat ▸ Basic Statistics` (`openstats-ui/src/dialogs/analyses_basic.rs`; registry now 70) wired to library routines that already existed from Phase 2, plus one new library function.

**Dialogs:** 1 Proportion; 2 Proportions; 1-Sample Poisson Rate; 2-Sample Poisson Rate (each takes summarized data or worksheet columns; proportions also accept a stacked outcome/sample layout and an event level); 2 Variances (F test plus Levene); Test for Equal Variances (Bonett plus Levene, k groups); Outlier Test (Grubbs, Dixon, generalized ESD); Equivalence Test (TOST, 1-sample / paired / 2-sample); Tolerance Interval (normal exact, Howe, nonparametric); Cross Tabulation and Chi-Square (raw categories or a table of counts; Pearson, likelihood-ratio, linear-by-linear, Fisher, McNemar, association measures, row/column/total percentages, expected counts, residuals); Chi-Square Goodness-of-Fit; Covariance; Partial Correlation.

**Library:** `crosstab::chi_square_gof` (equal, proportional or historical-count expectations, optional estimated-parameter df adjustment) with a `Reportable`.

**Verification:** 4 library tests against SciPy `chisquare` (`tests/basic.rs`, generator `tests/gen/basic_ref.py`); 10 dialog tests (`openstats-ui/tests/basic_dialogs.rs`) that compare each result block-for-block with the library routine it wraps, for every data layout and option, plus error paths. Gates: `cargo test --workspace` 1032 passed, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings`, `--no-default-features` clean; no TODO/`unimplemented!`.

### Honest limits
- These dialogs add no new statistics except goodness of fit; their numbers are those of the Phase 2 library (see the 2a/2b limits). No Minitab output was available for layouts.
- The 2-variances and equal-variances dialogs append Levene's test as a second section; Minitab prints the multiple-comparison table in its own layout.
- Event levels are taken as text (numbers are matched by their displayed form); blank means the last level in sorted order.
- Cross-tabulation has no layer (third) variable, no chi-square contributions table and no per-cell percentage mixing; McNemar needs a 2 x 2 table.
- No graphs are drawn by these dialogs, and none store results in the worksheet.
- Not run in the native macOS shell or the browser.

## Phase 5b — Power and sample size, regression extras, GLM comparisons

Eighteen new dialogs (registry 70 → 88) and one renderer change; no new statistics, all dialogs wrap Phase 2/3 library code.

**Power and Sample Size** (`analyses_power.rs`, menu `Stat ▸ Power and Sample Size`): 1-Sample Z, 1-Sample t, Paired t, 2-Sample t, 1 Proportion, 2 Proportions, 1-Sample and 2-Sample Poisson Rate, 1 Variance, 2 Variances, One-Way ANOVA, 2-Level Factorial (full, fractional, Plackett-Burman; centre points, blocks, omitted terms) and Equivalence (TOST; 1-sample, paired, 2-sample). Each has three boxes (sample sizes, power values, effects); exactly one is blank and is solved for, the other two may hold several values (all combinations, at most 400, are solved into one table with one power curve per sample size, at most 8). A single combination reuses the library's `power_section` unchanged.

**Regression extras** (`analyses_reg.rs`): Stepwise (stepwise / forward / backward; p-value, AIC, AICc or BIC; forced terms; stores fits and residuals of the final model), Best Subsets (n best per size, size limits, forced predictors), Predict (fit at new values read from worksheet columns, confidence and prediction intervals, optional storage), Fitted Line Plot (linear, quadratic, cubic; confidence and prediction bands), and General Linear Model: Comparisons (Tukey, Fisher, Dunnett, Hsu MCB on the least-squares means of one factor). The Regression dialog gained storage of deleted residuals, leverages, Cook's distance, DFITS, PRESS residuals and coefficients.

**Renderer:** `ChartKind::Scatter` specs now draw series named `Fit`, `Lower CI`, `Upper CI`, `Lower PI`, `Upper PI` as curves (solid / dashed / dotted) instead of markers.

**Verification:** 16 dialog tests (`openstats-ui/tests/reg_dialogs.rs`): every power dialog is run and compared with the library (including 2-sample t, delta = 1, power 0.9 giving n = 23); multi-value solving, input checking, stepwise / best subsets / prediction against the library, prediction without intercept against a closed form, the fitted line against the least-squares line with band ordering, a quadratic recovered from noiseless data, GLM Tukey comparisons against SciPy `studentized_range` (MSE 3.31532 on 14 df: adjusted p 0.347, 0.013, 0.184; critical value 2.617), and the new storage columns against the regression diagnostics (leverages sum to the number of coefficients). One renderer test.

### Honest limits
- GLM comparisons build the differences' standard errors from the least-squares-mean standard errors (n_eff = MSE / SE²). That is exact for balanced designs without covariates, which the test checks against SciPy; for unbalanced designs or models with covariates it is an approximation and the report says so. Only a single factor's levels are compared (not interaction cells), and Dunnett/Hsu use that same approximation.
- Power dialogs take no group-mean pattern for the one-way ANOVA (only the maximum difference), no unequal group sizes, and no unequal-variance or non-inferiority designs.
- Stepwise terms are the numeric predictors themselves (no interactions, squares or categorical predictors from the dialog); names are made formula-safe, so a column called `my col` appears as `my_col`.
- Predict needs one column of new values per predictor, in order; it does not accept typed constants.
- The fitted line plot fits one predictor with polynomial terms; no log transforms; the bands are drawn on 101 grid points.
- No Minitab output was available for layouts or last-digit comparisons. Not run in the native macOS shell or the browser.


## Phase 5c — Quality-tool dialogs

Twenty-three new dialogs (registry 88 → 111) in `analyses_qual.rs`, plus extra options on the existing I-MR, Xbar-R/S and capability dialogs. No new statistics: every dialog wraps the Phase 3/4 `spc`, `capability` and `msa` library.

**Variables charts:** I-MR-R/S (between/within), Zone, Z-MR (short run, pooled or per-part sigma), Box-Cox transformation chart (individuals or subgroups, optimal / rounded / fixed lambda). I-MR and Xbar gained a tests choice (Test 1 only, Nelson 1-8, Western Electric, WE supplementary), sigma estimator (average/median moving range, MSSD; R-bar, S-bar, pooled), historical mean and sigma, moving-range span, a stage column (separate limits per stage) and storage of CENTER/UCL/LCL/FLAGGED columns; Xbar accepts a subgroup-id column.

**Attribute and rare-event charts:** P, NP, C, U (normal or exact limits, stages, Tests 1 or 1-4), Laney P' and U' (also stores ZSCORE), G, T (Weibull or exponential).

**Time-weighted and multivariate:** Moving Average, EWMA (exact or asymptotic limits), CUSUM (tabular with sides and head start, or V-mask), Generalized Variance.

**Capability and MSA:** Nonnormal capability (Box-Cox, Johnson, Weibull, lognormal, gamma, exponential, logistic, loglogistic, normal), Empirical capability, Capability Sixpack (individuals or subgroups, extended version), Individual Distribution Identification, Nested Gage R&R, Gage Linearity and Bias, Attribute Agreement Analysis. The Normal capability dialog gained a confidence-intervals option.

**Verification:** 15 dialog tests (`openstats-ui/tests/qual_dialogs.rs`): I-MR, Xbar and P results compared with the library, flagged/limit columns stored and checked, stages give per-stage limits, every other dialog is run on seeded data, bad input is rejected. Workspace total 1064 passed, 0 failed; clippy, fmt, rustdoc, no-default-features clean.

### Honest limits
- Pareto, run chart, Gage run chart and Gage R&R crossed expanded are not dialogs yet (no library code for the first three); planned for 5m.
- Capability intervals for the within indices need a pooled-sigma estimate; other estimators print a note instead.
- Stages are supported on I-MR, Xbar, P, NP, C and U only. Tests for attribute charts are limited to Test 1 or Tests 1-4.
- Generalized variance uses the estimated covariance (no known-covariance field). Subgroups are consecutive blocks or an identifier column; unequal subgroup sizes work only where the library supports them.
- No Minitab output was available for layouts or last-digit comparisons. Not run in the native macOS shell or the browser.


## Phase 5d — DOE dialogs

Seven new dialogs (registry 111 → 118) in `analyses_doe.rs`, and extra options on two existing dialogs. No new statistics: all wrap the Phase 2k `doe` library.

**Create designs** (stored in the worksheet as StdOrder, RunOrder, Blocks, PtType and one column per factor, in coded or natural units): Fractional Factorial (automatic or typed generators, block generators, centre points, replicates, alias structure up to a chosen order), Plackett-Burman (4-48 runs), Central Composite (circumscribed, inscribed, face-centred; rotatable, orthogonal or custom alpha; cube fraction; centre points in cube and axial block; two blocks), Box-Behnken, Taguchi (L4-L81, factor level counts). All take factor names and natural low/high levels. The original Create Factorial Design dialog gained names, natural levels, block generators and natural-unit storage.

**Analyze:** Response Surface (linear, interactions or full quadratic from worksheet columns; ANOVA by term group, lack of fit, canonical analysis; optional contour and surface plots for any two factors; stores fits and residuals), Taguchi (factor level columns plus one or more response columns; four signal-to-noise types; main-effects plots; stores SN ratio, mean, SD). Analyze Factorial Design gained main-effects, interaction, cube, Pareto and normal-effects plots.

**Verification:** 8 dialog tests (`openstats-ui/tests/doe_dialogs.rs`): fractional dialog equals the library and stores the same columns; typed generator D=ABC holds in the stored natural-unit data; names, levels and centre points; block column equals the library's; Plackett-Burman columns are orthogonal; CCD axial distance equals 8^(1/4) in natural units, blocked and face-centred variants, Box-Behnken run count; Taguchi design created and then analyzed for all four S/N types against `analyze_taguchi`; an exactly quadratic surface is recovered (fits equal responses, residuals about 0) with both plots; factorial plots add five charts; input errors. Workspace total 1072 passed, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

### Honest limits
- The library gaps in the roadmap (mixture designs, D-/I-optimal custom designs, 4-level and mixed Taguchi arrays, more Box-Behnken sizes, 3+ block CCD, split-plot) are NOT done; they move to 5o with the other DOE completion items.
- The response-surface dialog codes each factor by its observed range, so a CCD's axial points sit at +-1 and the cube at +-1/alpha in the coded model; fit quality is unaffected but coefficient values differ from a model coded at the cube levels. Contour and surface plots hold other factors at their centre.
- The Taguchi analysis builds its design from level-number columns (1, 2, 3...); it does not know which array produced them, so it takes no interaction columns and no noise-factor arrays.
- Natural-unit levels are typed as text boxes (one low and one high per factor); the dialog framework has no per-row table.
- Analyze Factorial still needs two-level data with no missing values; the cube plot uses the first three factors.
- No Minitab output was available for layouts. Not run in the native macOS shell or the browser.


## Phase 5e — Data and Calc menus

Twenty new dialogs (registry 118 → 138) in `analyses_data.rs`, wired to routines that already existed in `openstats-core::ops` and the calculator; one new controller method (`add_worksheets`, undoable) and one fix.

**Data:** Stack Columns (optional subscript column), Unstack Columns, Transpose Columns, Copy Columns to Columns, Subset Worksheet, Split Worksheet, Merge Worksheets (side by side, or by keys: inner / left / outer), Code (rules such as `1:5=low; 2=mid; *=0`, to numeric or text), Rank (five tie rules), Concatenate, Change Data Type, Extract from Date/Time (year to weekday name), Text Functions (left, right, middle, trim, case, substitute), Display Data. Copy and Subset take a row condition: all rows, a row range, values in a column, a numeric interval, missing / not missing, each with an exclude option.

**Calc:** Random Data (24 distributions, parameters typed in the order shown, reproducible with a base), Probability Distributions (density, cumulative, upper tail or inverse for a column or one constant), Column Statistics, Row Statistics (sum, mean, SD, min, max, median, N, N missing), Make Patterned Data (first, last, step, repeats), Make Indicator Variables.

**Fix:** the worksheet calculator (and so the Calculator dialog) did not know the distribution functions PDF, CDF, SFCDF, INVCDF, RANDOM, RANDOMSEED and NSCORES; `GridController` now registers them (`openstats-ui` depends on `openstats-math` with the `formula` feature).

**Verification:** 7 dialog tests (`openstats-ui/tests/data_dialogs.rs`) run each dialog and check the resulting columns and worksheets: stack and unstack round trip, transpose labels, row-range / interval / exclude copies, subset creating a worksheet that Undo removes, split into two worksheets, merges (side by side 3 rows, inner 2, outer 4, bad worksheet name and key number rejected), recoding with ranges and missing, ranks with ties (average 3.5, lowest, descending), concatenation, type conversion (strict mode fails), date parts (2024 / 3 / Friday), text functions, display, 400 random normals (mean near 10, same base gives the same values, Undo), normal cdf 0.975, inverse 1.95996, Poisson pmf 0.2240418, column and row statistics, patterned data, indicators. Workspace total 1079 passed, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

### Honest limits
- Not done from the roadmap: matrices dialogs, Replace Missing Values, Delete/Insert columns as dialogs (the grid already has them as commands), Make Mesh Data, Calc column-wise operations on several columns at once, Make Patterned Data for dates and text, Unpivot, and Code for date/time. Row and column statistics are reported or stored in new columns; they are not stored into constants.
- These dialogs create new columns or worksheets; none edit columns in place (Minitab's "store in the same column" is not offered). Sheet operations still clear nothing but are one undo step each; Subset, Split and Merge are undoable.
- Merge takes the other worksheet by name and its key columns as numbers; the dialog framework has no worksheet picker. Random Data and Probability Distributions take parameters as text (no per-parameter boxes).
- Random numbers use the calculator's generator, so a base of 0 gives a clock-dependent sequence.
- No Minitab output was available for layouts. Not run in the native macOS shell or the browser.

## Phase 5f — Regression family

New library modules (openstats-stats) and eight new dialogs (registry 138 -> 146) in `analyses_regfam.rs`, plus weights and Box-Cox on the existing Regression dialog.

**Library:**
- `glm`: generalized linear models by IRLS with step-halving. Families binomial (0/1 or events/trials), Poisson, negative binomial (theta by ML or fixed), gamma, Gaussian; links logit, probit, cloglog, log, identity, inverse (checked against the family). Coefficient table with z (fixed dispersion) or t (estimated dispersion), odds / rate ratios with intervals, Type III likelihood-ratio deviance table by term, deviance R-sq and adjusted, log-likelihood, AIC, AICc, BIC, deviance and Pearson goodness of fit, Hosmer-Lemeshow, classification table at any cutoff, ROC curve and AUC, Somers' D / Goodman-Kruskal gamma / Kendall tau-a, deviance and Pearson residuals (raw and standardized), leverage, Cook's distance, delta chi-square / deviance / beta, prediction with intervals, and term-level forward / backward / bidirectional stepwise with likelihood-ratio p-values.
- `glm::ordinal`: ordinal (cumulative logit / probit / cloglog, proportional odds) and nominal (baseline-category) logistic regression by Newton-Raphson with analytic gradient and Hessian, odds ratios, G test, Pearson and deviance fit over covariate patterns, category probabilities, prediction.
- `nonlinear`: user equations (`a*exp(-b*x)+c`) by Levenberg-Marquardt, optional parameter bounds (smooth reparameterization), asymptotic standard errors and intervals, parameter correlations, confidence and prediction limits by the delta method.
- `pls`: NIPALS partial least squares (PLS1 and PLS2), optional standardization, leave-one-out or k-fold cross-validation, PRESS and predicted R-sq per component, VIP, scores, loadings, coefficients in original units.
- `orthogonal`: Deming / orthogonal regression with a given error-variance ratio, jackknife standard errors.
- `regression::fit_weighted` / `regress_weighted`: weighted least squares with weighted sums of squares, R-sq and F.
- `model::Expr::parse` / `evaluate` made public.

**Dialogs (Stat > Regression):** Binary Logistic Regression (binary or events/trials response, event choice, link, interactions, stepwise, classification table, association, Hosmer-Lemeshow table, ROC, ten storage options), Ordinal Logistic Regression, Nominal Logistic Regression (category order and reference), Poisson Regression (Poisson or negative binomial), Generalized Linear Model (gamma, Gaussian, Poisson, negative binomial; link choice), Nonlinear Regression (equation, starting values, bounds, confidence and prediction band), Partial Least Squares, Orthogonal Regression. The Regression dialog gained a Weights column and a Box-Cox response transformation (optimal, rounded or given lambda).

**Verification:** 16 reference tests (`openstats-stats/tests/glm_reference.rs`, data from `tests/gen/glm_ref.py`) against independent implementations: logistic and Poisson coefficients from scikit-learn (no penalty) with NumPy standard errors, deviance, AIC, BIC, leverage and residuals; probit, cloglog, negative binomial, gamma and ordinal by SciPy optimization of the log-likelihood (ordinal standard errors from a numerical Hessian); nominal from scikit-learn multinomial regression re-expressed against the baseline; nonlinear from `scipy.optimize.curve_fit` (parameters, standard errors, SSE, bounded fit, delta-method prediction interval); PLS from `sklearn.cross_decomposition.PLSRegression` (fits, coefficients, intercepts, score sums of squares) with a leave-one-out PRESS loop; orthogonal regression from `scipy.odr`; weighted regression from the normal equations (coefficients, SE, SSE, SST, R-sq, F); AUC from scikit-learn; stepwise consistency. 11 dialog tests (`openstats-ui/tests/regfam_dialogs.rs`) compare each dialog's stored columns with the library. Workspace total 1106 passed, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

### Honest limits
- Sign and threshold conventions of the ordinal model are P(Y <= j) = F(theta_j - x'beta) (positive coefficient = higher categories); Minitab's printed signs have not been checked against Minitab output.
- Hosmer-Lemeshow groups are equal-sized by predicted probability with ties kept together; Minitab's grouping rule may differ slightly. The reference test replicates this algorithm in Python, so it checks the implementation, not Minitab's rule. Standard errors for non-canonical links (probit, cloglog, identity) use the expected information.
- Not done: Johnson response transformation (only Box-Cox), offset variable in the dialogs, overdispersion (quasi) options, stepwise and Type III term tests for ordinal and nominal models, measures of association for ordinal and nominal models, Predict dialogs for the new models (the library predicts; Stat > Regression > Predict is still linear only), response-by-group nonlinear fits, user-supplied derivatives, weights in the GLM dialogs.
- Nonlinear: the optimizer is local; poor starting values can fail or reach a local minimum (the report warns when it does not converge). Bounds use smooth reparameterization, so a parameter on a bound has an unreliable standard error. Equation columns are referenced by column name.
- PLS: cross-validation groups are assigned cyclically (not randomly); component signs are arbitrary; rows with any missing value are dropped.
- Orthogonal regression standard errors are delete-one jackknife values, not Minitab's formula.
- Complete or quasi-complete separation is flagged only when the iteration does not converge.
- Library comparisons use scikit-learn / SciPy (L-BFGS reference coefficients are good to about 1e-8). No Minitab output was available for layouts. Not run in the native macOS shell or the browser.


## Phase 5g — ANOVA and mixed models

New library modules (openstats-stats) and seven new dialogs (registry 146 -> 153) in `analyses_anova.rs`, plus random factors on the GLM dialog and fixed operators on Gage R&R.

**Library:**
- `anova::random`: general random-factor ANOVA. Type III sums of squares from orthonormal projections of sum-to-zero designs (valid for unbalanced data); expected-mean-square coefficients computed exactly from the same projections; unrestricted or restricted mixed-model rule; F denominators found by an exact subset search over the other random terms and error (synthesized, with Satterthwaite df, when no single term works); ANOVA-method variance components (negatives reported raw, truncated value used for percentages); marginal means. Entry points `balanced_anova` (rejects unbalanced or missing data), `glm_random`, `nested_anova` (fully nested, sequential projections). Nested terms `A(B)` use within-parent contrasts so subject IDs unique across groups still leave the between-group df.
- `mixed`: mixed-effects model by REML or ML with any number of random-intercept terms (crossed or nested). Profiled criterion optimised by Nelder-Mead then BFGS from three starts. Fixed effects with Satterthwaite df, multi-df term tests (lmerTest style), variance components with standard errors from a finite-difference Hessian, BLUPs, conditional and marginal fits and residuals, AIC/AICc/BIC.
- Gage R&R: `operator_fixed` option scales the operator component by (o-1)/o.

**Dialogs (Stat > ANOVA):** Balanced ANOVA, Fully Nested ANOVA, Mixed Effects Model, Repeated Measures (between and within factors, subject nested in group), ANCOVA (slope-homogeneity test, adjusted means, plot), Main Effects Plot, Interactions Plot. GLM gained Random factors and a Restricted-model check; Gage R&R gained a Fixed operators check. Plots use `Fit: <group>` curves.

**Verification:** 10 reference tests (`openstats-stats/tests/anova_random.rs`, data from `tests/gen/anova_ref.py`) against dense NumPy/SciPy implementations: unbalanced fixed x random SS, df and EMS coefficients; unbalanced nested design; crossed-random REML and ML variances, coefficients, SEs, Satterthwaite df and variance SEs; balanced coefficients equal level counts; restricted and unrestricted two-way mixed; synthesized three-way denominator; closed-form split-plot SS; fixed-operator Gage. 9 dialog tests (`openstats-ui/tests/anova_dialogs.rs`). Workspace total 1127 passed, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

### Honest limits
- Random intercepts only: no random slopes, no covariance structures (AR1, unstructured), no heterogeneous residual variances. At most 2500 random-effect levels.
- Variance components in ANOVA reports are ANOVA-method estimators; the mixed model reports REML/ML. Satterthwaite df for the variances comes from a numerical Hessian, so SEs are approximate. Kenward-Roger is not implemented.
- Default is the unrestricted mixed-model convention; the restricted form is an option. Minitab's defaults and printed layouts were not checked against Minitab output; references are my own dense implementations and closed forms (statsmodels was not available).
- Repeated measures assumes sphericity (no Greenhouse-Geisser or Huynh-Feldt corrections). ANCOVA plot holds other terms at reference levels.
- Fixed-operator Gage convention ((o-1)/o scaling of the operator component) is my derivation, not verified against Minitab.
- Not done: GLM-style multiple comparisons for random-factor models, split-plot design dialogs, MANOVA (5h), and ordinal/nominal stepwise carried from 5f.

Next: 5h (nonparametrics, survival, multivariate and time-series gaps).

## Phase 5h — Nonparametrics, survival, multivariate, time series (DONE)

Added (all pure Rust, std only, SciPy/NumPy-referenced tests; workspace 1150 tests, 0 failed; clippy/fmt/doc/no-default-features clean):
- Nonparametrics: 2-sample Kolmogorov-Smirnov, Ansari-Bradley, Mood scale, normal scores, Mann-Kendall trend (5 dialogs).
- Survival: Cox proportional hazards (Efron/Breslow ties, strata, Harrell C) with dialog.
- Time series: ADF, KPSS, cross-correlation (CCF), auto-ARIMA (d chosen by KPSS) (4 dialogs).
- Multivariate: simple and multiple correspondence analysis (with symmetric plot), promax rotation in Factor Analysis (2 dialogs + factor option).
- Dialog registry is now 165.

Honest limits:
- ADF critical values come from the MacKinnon response surface (checked by Monte Carlo); no exact p-values. KPSS critical values are the standard tabulated ones.
- Cox: no time-dependent covariates, case weights or baseline survival output.
- Mood scale uses the tie-free variance. Ansari-Bradley is two-sided only; exact for both samples < 55, normal approximation otherwise (matches SciPy).
- MCA has no inertia adjustment. Promax takes varimax as input; factor scores stay unrotated.
- No real Minitab output was available; references are independent NumPy/SciPy code (statsmodels, lifelines, sksurv not installed).
- Not done, carried forward: Turnbull, 3-parameter Weibull/lognormal, gamma/extreme-value families, probability plots and Anderson-Darling with censoring (-> 5n); stepwise discriminant, Genz MVN CDF (-> 5o); ARIMAX, trend analysis, MA/lag/difference dialogs, calendar axes (-> 5k).

Next: 5i (predictive analytics completion).

## Phase 5i — Predictive analytics completion (DONE)

- Binary-response MARS (`earth(glm=binomial)` recipe): basis chosen by least-squares MARS on the 0/1 response, coefficients re-estimated by logistic IRLS; report has coefficients with SE/Z/P, deviance, McFadden R², AIC, confusion matrix, AUC, log-loss, optional stored class and event probability. Event level selectable.
- MARS cross-validation over degree and penalty (every fold refits the whole model); best setting is refitted on all data.
- Model objects for CART, Random Forests, Gradient Boosting and MARS (regression and binary), serialized as JSON inside the `.otb` project (`model.<name>` metadata), validated on load; truncated or damaged models are refused.
- New dialogs: Score New Data (by column name or by selection order; missing values handled per model), Partial Dependence and ICE (1-D, 2-D pairs, class choice), Compare Models (same folds for all candidates, RMSE/R² or accuracy/AUC/log-loss, baseline stump), Saved Models (list / rename / delete). `save_as` added to the four fit dialogs.
- Charts: ICE curves drawn as faint lines and kept out of the legend.
- Dialog registry is now 169. Workspace: 1179 tests pass, 0 fail; clippy, rustdoc, no-default-features clean.

Honest limits:
- Binary MARS re-estimates only the coefficients; the basis itself comes from least-squares GCV. CV picks degree and penalty, not the number of terms.
- Saved models keep the prediction structure only (no pruning tables, OOB data or importance tables). Saving is not undoable. There are no saved objects for the regression/logistic families.
- Partial dependence uses a row subsample (default 1000) and a 5th–95th percentile grid.
- Comparison uses identical folds across models; the shuffle is Rust's own generator, so fold membership is not reproducible in SciPy/scikit-learn (the CV arithmetic is checked against a manual loop).
- Not done: TreeNet-style regularized boosting options (subsampling variants, Huber tuning grid) and variable-selection helpers (-> 5p/5q).
- No scoring macro command yet.
- No real Minitab output was available; references are independent NumPy/SciPy/scikit-learn code.

Next: 5j (Graph menu and chart coverage).

## Phase 5j — Graph menu and chart coverage (DONE)

Added: 16 Graph dialogs (bar simple/cluster/stack, pie, dotplot, stem-and-leaf, individual value, interval, ECDF, area, bubble, matrix, marginal, heatmap, 3-D scatter, surface, contour), Edit Graph (title and axis labels), probability plots with a distribution choice, PDF export (library `to_pdf` + Navigator "Export as PDF..."), `stemleaf` library module. Registry 185. Workspace: 1200 tests, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

Honest limits:
- Layout tool (nested panels) is not done; carried to 5k.
- Calendar axes for time series are not done; carried to 5k.
- Marginal plot is a 2x2 grid with histograms (not rotated on the side).
- Heatmap uses numeric column positions rather than variable-name labels.
- Surface/contour from scattered points use inverse-distance weighting (25x25 and 40x40 grids); a complete regular grid is used as is.
- Dotplot uses 50 bins.
- PDF is single page, Helvetica, Latin-1 only (Greek becomes `?`).
- Stem-and-leaf leaf units are limited to 2/5/10 line widths, not Minitab's full increment rules.
- Probability plots cover normal, lognormal and Weibull only.
- Graph editing covers title and axis labels only, not per-element formatting.
- Appearance is not verified against Minitab output (see 5p).

## Phase 5k — Workflow, import/export and platform (DONE, with carried items)

Added:
- Macro commands EXECUTE (repeat an exec file n times), STACK, UNSTACK, OUTFILE and NOOUTFILE (session text goes to a file; files are returned by `Machine::take_files`).
- Command Line pane (View > Command Line, Ctrl+L) that runs session commands, and a History pane (View > History) that lists every command line, dialog and macro run; history is saved in the `.otb`.
- Session export to HTML and Word (.docx); the .docx was validated with python-docx and LibreOffice.
- Data/Calc dialogs: Replace Missing Values (value, mean, median, previous, next, linear interpolation), Unpivot Columns, Matrix Operations (transpose, invert, determinant, trace, multiply, add, subtract, Cholesky, symmetric eigen, columns to matrix and back), Project Properties (title, author, description in workbook metadata).
- Time-series dialogs: Lag, Differences, Moving Average (trailing, centred, 2 × m for even lengths), Trend Analysis (linear, quadratic, exponential growth, accuracy measures, forecasts, stored fits/residuals/forecasts).
- Time Series Plot takes an optional date/time column and draws a calendar axis (new chart kind `TimeLine`, Excel-serial x values, existing calendar tick generator).
- JSON import was already present (checked with a round-trip test).
- Registry is now 193 dialogs. Workspace: 1225 tests, 0 failed; clippy, fmt, rustdoc and no-default-features clean.

Honest limits:
- History records command-line text, dialog titles and macro names; it does not generate the underlying command text for dialog runs.
- Project Properties does not pre-fill the dialog with the current values; an empty title leaves the title unchanged.
- Matrix dialog works on one operation at a time; eigen results use M(result) for the values and M(result+1) for the vectors. No SVD, QR or general (non-symmetric) eigen in the dialog.
- Trend Analysis has no S-curve model, no prediction intervals and no seasonal terms (use Decomposition or Winters for those).
- Calendar axis is for the single Time Series Plot; other chart kinds still use linear axes. Date ticks are only as good as the existing tick generator.
- Replace Missing Values and the lag/difference/MA dialogs store new columns instead of overwriting.
- Not done, carried forward: layout tool with nested panels; accessibility DOM mirror and native accessibility tree (needs screen-reader testing, -> 5p); Assistant decision trees and report cards; ARIMAX; SAS/SPSS/Stata import; Minitab-file import (needs sample files); key customization; PROMPT and other macro commands.
- No real Minitab output was available; matrix and trend values are checked against closed-form results.

Next: 5l.

## Phase 5l — Tables, EDA, bootstrap and simulation (DONE, with carried items)

Added (11 dialogs; registry 204):
- Stat > Tables: Tally Individual Variables (counts, percents, cumulative), Descriptive Statistics table (up to 3 row variables, optional column variable, margins, N/mean/median/SD/min/max/sum of summary variables).
- Stat > Quality Tools > Pareto Chart (new chart kind `Pareto`: sorted bars and cumulative-percent line; raw rows or a counts column).
- Stat > EDA: Letter Values, Median Polish, Resistant Line, Rootogram (observed, expected, hanging and double-root residuals), Symmetry Plot. Library `openstats_stats::eda`.
- Stat > Resampling > Bootstrap: mean, median, SD, variance, quartiles, 5% trimmed mean, correlation, difference of means or medians, ratio of SDs; percentile, basic, normal and BCa intervals; bias and SE; replicate histogram and storage. Library `openstats_stats::bootstrap`.
- Calc > Monte Carlo Simulation (inputs like `X = normal(10, 2); Y = uniform(0, 1)`, equation in the input names, summary, percentiles, tail probabilities, histogram, optional storage) and Calc > Random Data > Sample From Columns (rows drawn jointly, with or without replacement).
- Item analysis (Cronbach's alpha) and cluster variables already existed from 4e.

Verification: 9 library tests (`tests/eda_boot.rs`): letter values and median polish against independent NumPy code (R's `medpolish` recipe), BCa/percentile/basic/normal intervals against NumPy/SciPy formulas on fixed replicate and jackknife arrays, bootstrap SE against the plug-in SE, reproducibility by seed, a 120-trial coverage check, exact recovery by the resistant line and its resistance to gross outliers. 8 dialog tests (`tests/tables_dialogs.rs`). Workspace: 1242 tests, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings`, no-default-features clean.

Honest limits:
- Not done: chi-square variants with stored expected counts and contributions (the existing cross-tabulation dialog still shows them in the report only); Cell Statistics is the Descriptive Statistics table (long layout, not Minitab's wide layout); Pareto has no "combine small defects" option, no by-variable panels.
- Median polish needs a complete table with one value per cell. The resistant line splits into thirds by sorted x with extra points going to the outer groups; R's `line()` and Minitab may use different tercile rules, so slopes can differ slightly on small samples.
- Rootogram uses equal-width bins and a normal fit only. The double-root residual is Tukey's; hanging bars are tabulated, not drawn.
- Bootstrap: resampling is by rows (paired) or within each sample (two-sample); no residual or block bootstrap; BCa jackknife is leave-one-row-out (cost grows with n); percentile intervals use the type-6 quantile. No Minitab output was available to compare layouts; the random stream is OpenStats's PCG64, so replicates differ from other software even with the same seed.
- Simulation inputs are independent (no correlation matrix) and use the calculator's distribution names; the equation is evaluated by the worksheet calculator on whole columns. Name substitution is by whole word and case-insensitive.
- Not run in the native macOS shell or the browser.

Next: 5m.

## Phase 5m — Quality completion (DONE)

New library modules in `openstats-stats` (std only) and 15 dialogs (registry 204 -> 219) in `analyses_qual2.rs`.

**Library**
- `acceptance`: single attribute plans (accept when defectives <= c) with OC, AOQ, ATI and AOQL under the binomial, Poisson or hypergeometric (lot size) model; plan search for AQL / RQL / alpha / beta (smallest n, then the smallest c with Pa(AQL) >= 1 - alpha, valid when Pa(RQL) <= beta); double sampling plans (n1, c1, r1, n2, c2) with first/second-stage acceptance, ASN, AOQ and ATI; single-limit variables plans (sigma known: normal; sigma unknown: exact non-central t), plan search, and the accept/reject decision for data; reports with OC, AOQ, ATI and ASN charts.
- `runchart`: runs about the centre (median, mean or a value) with clustering / mixtures p-values, runs up or down with trends / oscillation p-values, longest runs; subgroup mean or median points.
- `multivari`: up to four factors, cell and level means, range of level means, observations-with-means chart.
- `attrcap`: binomial capability (pooled %defective with Clopper-Pearson interval, PPM, Process Z with interval, chi-square test of a common proportion, %defective and cumulative charts) and Poisson capability (DPU with Garwood interval, min / max DPU, Process Z from exp(-DPU), chi-square test of a common DPU, DPU charts).
- `spc::transform`: `box_cox_profile` (profile log-likelihood curve) and `box_cox_interval` (likelihood-ratio interval for lambda).

**Dialogs (Stat > Quality Tools unless noted)**: Acceptance Sampling by Attributes (Create Sampling Plan, Compare Sampling Plans, Double Sampling Plan, Accept/Reject), Acceptance Sampling by Variables (Create, Compare, Accept/Reject), Multi-Vari Chart, Run Chart, Gage Study > Gage Run Chart, Capability Analysis > Binomial / Poisson / Multiple Variables (Normal), Johnson Transformation, and Stat > Control Charts > Box-Cox Transformation. The two transformation dialogs store the transformed column and can transform specification limits (Johnson).

**Verification**
- `openstats-stats/tests/quality2.rs` (9 tests, `tests/gen/qual_ref.py` -> `common/qual_data.rs`): plan search against a brute-force SciPy scan for binomial, Poisson and hypergeometric (lot 1000); OC, AOQ, ATI against `binom` / `poisson` / `hypergeom`; AOQL against a 300,001-point grid; double plans against nested SciPy sums (Pa by stage, ASN, AOQ, ATI); variables plans against `norm` and `nct` (n, k, Pa at both points, OC values); run chart against explicit loops; attribute capability against `beta.ppf`, `chi2.ppf` and `chi2.sf`; Box-Cox lambda, interval (`scipy.stats.boxcox(alpha=)`) and profile (`boxcox_llf`).
- 7 library unit tests; `openstats-ui/tests/quality_dialogs.rs` (9 tests): every dialog is run and compared with the library routine it wraps, including stored columns, validation errors and the hypergeometric / double / lower-limit / subgroup variants.
- Gates: `cargo test --workspace` 1272 passed, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings`, `--no-default-features` clean; web tests 25 pass; no TODO / `unimplemented!`.

### Honest limits
- Acceptance sampling has no ANSI/ASQ Z1.4 or Z1.9 (MIL-STD) tables, no switching rules, no multiple or sequential plans, no skip-lot, and no curtailed double sampling. AOQ and ATI assume rejected lots are screened 100% and defectives in samples are replaced.
- The attribute plan search takes the smallest n whose smallest admissible c meets both risks; Minitab's search rule and tie-breaking were not checked against Minitab. The hypergeometric model rounds the defective count to N*p.
- Variables plans cover one specification limit. Two-sided limits, plans with separate AQL / RQL for each side, and the exact two-sided OC are not provided. The plan constant k is chosen to give exactly Pa(AQL) = 1 - alpha.
- Run chart p-values use the normal approximation of the number of runs (no exact distribution for small n); runs up or down skip equal neighbours; points on the centre are skipped for runs about it. Minitab's exact conventions for ties were not checked.
- Multi-vari draws one chart with the cells in table order (not Minitab's panelled layout by factor).
- Binomial / Poisson capability: the Process Z definitions follow my reading of the documentation (Z = inverse normal of 1 - p-bar; for Poisson, of exp(-DPU)); the common-proportion and common-DPU tests are Pearson chi-square. No Minitab output was available for layouts. Capability for several variables reports one table row per variable, with no per-variable histograms.
- Gage run chart is a stacked layout (measurement, part, operator); the multiple-column layout is not offered.
- Not done: cause-and-effect diagram, Pareto "by variable" panels, symmetry-based capability extras, Gage R&R (crossed, expanded). Carried to later phases with the earlier lists.
- Not run in the native macOS shell or the browser.

Next: 5o.

## Phase 5n — Reliability completion (DONE)

New library modules in `openstats-stats::reliability` (std only) and 7 dialogs (registry 219 -> 226) in `analyses_rel2.rs`.

**Library**
- `repairable`: Nelson mean cumulative function with Lawless-Nadeau variances and log-scale normal limits (systems with different end times); Crow-AMSAA power-law process by pooled maximum likelihood (time- or failure-truncated), bias-corrected shape, chi-square limits for the shape, the beta = 1 chi-square test, the Laplace trend test, intensity, expected events and instantaneous / cumulative MTBF.
- `testplan`: Weibull (known shape; 1 = exponential) demonstration plans: sample size for a test time and allowed failures, test time for a sample size, achieved confidence, and the operating characteristic.
- `extra`: threshold (3-parameter) Weibull, lognormal and log-logistic by profile likelihood with profile limits for the threshold; two-parameter gamma with censoring; Turnbull self-consistency (EM) estimate for exact, right, left and interval data; Eyring ALT (ln t = a - ln T + b/T) by time scaling with the Jacobian correction, giving percentiles and acceleration factors on the original time scale. (Temperature-humidity and other multi-stress models already exist via `fit_alt`.)

**Dialogs (Stat > Reliability/Survival)**: Repairable Systems > Mean Cumulative Function and Crow-AMSAA Growth Model; Test Plans > Demonstration Test Plan; Turnbull Estimate; Distribution Analysis (3-Parameter and Gamma); Accelerated Life Testing (Eyring); Probability Plot (Censored Data) (Turnbull plotting positions with the ML line).

**Verification**
- `openstats-stats/tests/reliability2.rs` (7 tests, `tests/gen/rel2_ref.py` -> `common/rel2_data.rs`): MCF and its standard errors against explicit loops; Crow-AMSAA (beta, lambda, chi-square limits, both trend tests) against NumPy / `chi2`; demonstration plans against a SciPy `binom` brute force and `brentq`; censored gamma against a `scipy.optimize` MLE; threshold Weibull against a SciPy profile over the threshold; Turnbull log-likelihood against an independent NumPy EM on interval data; Eyring lognormal against a direct SciPy MLE.
- 15 library unit tests (including Turnbull = Kaplan-Meier for right-censored data) and `openstats-ui/tests/reliability_dialogs.rs` (7 tests).
- Gates: `cargo test --workspace` 1302 passed, 0 failed; clippy `-D warnings`, fmt, rustdoc `-D warnings`, `--no-default-features` clean.

### Honest limits
- No extreme-value (SEV / LEV) families, and no Anderson-Darling with censoring. The censored probability plot uses a plain scatter in transformed coordinates (axis labelled by transform, no percent ticks or confidence band).
- Crow-AMSAA: the shape limits use the exact chi-square for a single system and the same formula as an approximation for pooled systems; no Cramer-von Mises goodness of fit, no grouped (interval) data, no Duane or piecewise models.
- MCF: no comparison of two MCFs, no cost / non-unit event weights.
- Threshold fits: the profile can be unbounded for shape < 1 (flagged when no interior maximum); no standard error for the threshold, only profile limits; no probability plot with the threshold.
- Eyring covers log-time families only; no warranty or Bayesian extensions, no Weibull / exponential demonstration plans with unknown shape, no multi-sample or sequential tests.
- Layouts and conventions were not checked against real Minitab output (5p). Not run in the native macOS shell or the browser.

## Phase 5o — DOE completion (partial scope)
Added (library `openstats_stats::doe::extra`, dialogs `analyses_doe2.rs`, 6 new dialogs → 232 registered):
- Mixture designs: simplex lattice, simplex centroid, extreme vertices (checked against SciPy HalfspaceIntersection); point types 1/2/0/-1.
- Analyze Mixture Design: Scheffé linear / quadratic / special cubic / full cubic, no constant, sequential ANOVA, coefficient CIs, prediction, fits/residuals (checked against NumPy lstsq).
- Create Optimal Design: D- and I-optimal exchange algorithm over a full-factorial candidate grid (checked against exhaustive search for small cases).
- Create Split-Plot Design: two-level full factorial, WholePlot id; analysis via Mixed Effects Model with WholePlot random.
- Taguchi arrays L16(4^5) and L64(4^21) (GF(4); orthogonality tested).
Tests: 6 library reference tests (`tests/doe2.rs`), 9 dialog tests (`tests/doe2_dialogs.rs`); workspace 1324 pass, clippy/fmt/rustdoc/no-default-features clean.

Honest limits / NOT done from the 5o roadmap:
- Response optimizer (desirability), overlaid contour plots, definitive screening designs, analyze variability, steepest ascent: NOT implemented.
- No Box-Behnken sizes beyond k=3,4,5,7; no 3+ block CCD; no mixed Taguchi arrays (L32 2^1 4^9, L36, L50).
- Mixtures: no extreme-vertices edge/face centroids, no trace/contour plots, no mixture-process designs.
- Optimal: candidate-grid exchange only (no mixture/constrained candidates, blocks, covariates); heuristic with restarts, not guaranteed global.
- Split-plot: 2-level full factorial only; no dedicated analysis dialog.
- Not checked against real Minitab output (5p).

## Phase 5o-b — DOE completion, remaining items
Added (library `openstats_stats::doe::extra2`, dialogs in `analyses_doe2.rs`; 4 new dialogs → 236 registered):
- Definitive Screening Design: Paley conference matrices (orders 4–50 where q = order−1 is a prime or the square of an odd prime; verified C·Cᵀ = (n−1)I), runs = C, −C, centre; k = 3–30 factors, extra centre runs, natural-unit levels, randomize.
- Steepest Ascent/Descent: first-order direction from the fitted linear model, step in coded units, natural-unit path table with predicted response.
- Analyze Variability: ln(s) of each replicated factor combination regressed on coded effects (+ optional two-factor interactions); effect, SE, t, p and StDev ratio (checked against numpy/scipy).
- Overlaid Contour Plot: bounds per response, boundary contours and feasible-region chart, feasible fraction.
- (Response Optimizer already existed from an earlier phase.)
Honest limits: DSD orders without a Paley matrix (16, 22, 28, 34, 36, 40, 46...) use the next available order and drop columns (extra runs); no DSD blocking or categorical factors; no DSD-specific analysis dialog (use Fit Regression Model / Response Surface); Analyze Variability is the ln(s) regression and may not reproduce every Minitab convention (weighting, p-value method) — unverified; steepest ascent has no confidence cone; overlaid contour shows boundaries and a feasible-point chart, not shaded regions, and only two varying factors.

## Phase 5p — Verification against Minitab output, summarized-data tests, performance (partial scope)
**What could and could not be done.** No Minitab installation and no macOS/Windows/browser toolchains were available (no registry or `wasm32` std in the sandbox; the user's machine reaches only a Linux shell without Rust). So "gather Minitab output for every dialog" and "first real builds" were NOT done. What was done instead:

**1. Verification against Minitab's own printed output** (`crates/openstats-stats/tests/minitab_published.rs`, 13 tests; sources: support.minitab.com examples and Penn State STAT 200/415/500/503 pages that show Minitab session output). Inputs are the summary statistics or counts printed with each example; tolerances are half a unit of the last printed digit (more where the printed inputs were themselves rounded). Covered and agreeing: 1-sample t (7 examples, including a one-sided bound with T and P), paired t, 1-sample Z, 2-sample t Welch (3) and pooled (4), 1 proportion normal-approximation CI/Z/P (6), chi-square for association (Pearson, likelihood ratio, expected counts and per-cell contributions, three tables), one-way ANOVA table (Montgomery cotton data), and Power and Sample Size for 2-Sample t (n = 86, actual power 0.903230).
**Discrepancies found and fixed**
- *1 Proportion exact CI.* Minitab's printed exact interval (0.070617, 0.106130) is Blaker's, not Clopper–Pearson (0.070268, 0.106208). Added `proportion::blaker_interval` (two-sided; n ≤ 20,000; one-sided alternatives keep Clopper–Pearson). The report now lists "Exact (Blaker CI)" and "Exact (Clopper-Pearson)"; the exact p-value (small-p method, 0.0069 → 0.007) already matched.
- *Missing features.* The t dialogs had no "Summarized data" mode and there was no 1-Sample Z dialog. Added `one_sample_t_summary`, `two_sample_t_summary`, `paired_t_summary`, `z_test_summary`; 1-Sample t, 2-Sample t and Paired t gained a "Data are" choice (columns / summarized data); new dialog Stat > Basic Statistics > 1-Sample Z (columns or summarized, known σ). `DialogSpec::vars_optional_when` keeps the column boxes required in column mode. Registry 236 → 237.
**Conventions confirmed against Minitab's documentation (no change needed):** X flag when leverage > min(3p/n, 0.99); R flag |standardized residual| > 2; AICc and BIC use p = coefficients including the constant, as ours do.
**Still unverified against Minitab:** everything not listed above — in particular regression coefficient tables, GLM/ANOVA multiple comparisons, nonparametric tests, control charts, capability, Gage R&R, reliability, time series (Winters initial values, ARIMA estimation details), multivariate, DOE (effects tables, optimizer, new 5o designs), Johnson selection and stepwise tie-breaking. The published examples that carry raw data inline are scarce; every one found that could be encoded was.

**2. Performance.** New `openstats_stats::parallel::{par_map, worker_threads}` (std threads, order-preserving, sequential on wasm32, `OPENSTATS_THREADS` override). Random-forest trees (including permutation importance) grow in parallel and are merged in tree order; model-comparison and MARS cross-validation folds run in parallel. A test in its own process (`tests/parallel_forest.rs`) shows 1-thread and 4-thread fits are bit-identical (OOB error, curve, importance, predictions); on the 2-core sandbox a 120-tree forest on 1,500 × 6 took 0.36 s → 0.24 s (debug build). Not parallelised: CART cross-validation (shared random stream), GBDT (sequential by construction), large-table handling.

**3. Platform builds.** Not possible here. Checked instead: the browser shell's 25 Node tests and the script tests pass; `cargo build --workspace --no-default-features` passes on Linux; the native (winit/softbuffer) and Tauri crates still need a first real build on macOS and Windows, and `wasm32-unknown-unknown` has not been built (target not installed offline). CI and release workflows remain unrun.
Workspace: 1352 tests pass; clippy, fmt, rustdoc and no-default-features clean.

## Phase 6a — graph behavior, worksheet and UX details

**Graphs**
- A graph drawn by a plain graph dialog (`graph.*`, no stored columns or models) remembers its dialog, values and worksheet (`GraphRecipe`). It is redrawn automatically when the columns it reads change (FNV fingerprint checked after every input event); the copy in the Session window follows. `Graph > Update or Freeze Graph` updates now, freezes, or resumes. User titles and axis labels are kept across redraws.
- Brushing: drag a rectangle (Shift adds, a click picks one point, a click on empty space clears) in Scatterplot, Time Series Plot, Marginal Plot or Bubble Plot. The worksheet highlights the rows, other graphs of the same worksheet ring the same points, `Graph > Brushing` clears or stores an indicator column.
- Formatting: `Graph > Format Graph` (font scale, grid, axis ranges, series colours), `Graph > Add Note`. Recipes, styles and notes are saved in the project (`ui.graphmeta.*`).

**Worksheet and UX**
- Help: `Help > Search Commands`, F1 (opens the search; inside a dialog it writes that dialog's fields to the Session window). New key code 19 (F1) in host, JS and native key maps.
- `Tools > Customize Shortcuts`: Ctrl/Alt+letter or digit and F-keys bound to a built-in command or any dialog id; saved with the project.
- `Tools > Related Documents` and a Navigator folder (shown only when non-empty).
- `Tools > Options`: undo steps (History::set_limits) and regional settings (decimal mark and date order: US, UK, continental Europe, ISO) used for display, typing, pasting, CSV import and export.
- `Data > Value Labels`: display-only labels for numeric and text columns; `.otb` minor version 2 (older builds cannot read files with the new column field), JSON `value_labels`.
- Date/time entry edge cases pinned by tests (formats, AM/PM, leap days, two-digit years, 1900 and 9999).

**Honest limits**
- Linked graphs: only plain graph dialogs; analyses that draw charts and store results stay static. Worksheets are matched by index, so deleting or reordering sheets can detach a graph. Macro-made graphs are static.
- Brushing needs a point-to-row map; other chart kinds are not brushable. Brushing is graph to worksheet and graph to graph, not worksheet to graph.
- Formatting has no marker size or line style. Value labels do not appear in chart categories or analysis output. Shortcuts and options live in the project, not an app-wide settings file.
- No UI string translation, only the number and date conventions. Real pointer feel (drag rubber band) is tested through the API, not in a real window.
- Registry count is now 246. 1380 tests pass; clippy, fmt, rustdoc and no-default-features are clean.

## Rename: OpenTab → OpenStats (2026-11-18)

The project is now **OpenStats**, described as an open-source statistical software suite (not a Minitab clone). Crates are `openstats-*`, the desktop folder is `openstats-desktop`, the web module is `openstats.js`, the app/bundle id is `org.openstats.desktop`, and the per-user theme file is `~/.openstats/theme` (the web build uses the `openstats.theme` storage key, so a theme saved under the old name is not carried over). The `.otb` extension and the `OTAB` file magic are unchanged, so existing project files still open. One deflate test vector keeps the old product name in its source text, because its compressed bytes were produced from it. README gained an alpha / no-warranty notice; `*.zip` is now git-ignored.
