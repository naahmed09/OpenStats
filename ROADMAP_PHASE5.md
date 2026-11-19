# OpenStats — Phase 5 roadmap: closing the Minitab feature gap

Written 2026-11-18, after Phase 4g (MARS). Final release packaging is now the last phase (Phase 7); Phase 6a comes before it.

Conventions for every phase (unchanged): 100% pure Rust, std only; no stubs; SciPy/NumPy-referenced tests;
dialogs registered in `analyses*.rs` with menu paths and Storage pages; registry-count and menu tests updated;
gates = test, clippy `-D warnings`, fmt, rustdoc `-D warnings`, `--no-default-features`; honest-limits entry in
`PHASE_LOG.md`; zip, deliver, sync to the Mac.

Order is by value per effort: wire what already exists, then fill the Data/Calc basics, then new statistics,
then graphs, then platform.

## 5a — Basic statistics dialogs (library exists) — DONE
- 1-proportion, 2-proportions, 1-sample and 2-sample Poisson rate.
- 2-variances (F, Levene, Bonett, Brown–Forsythe), test for equal variances (k groups).
- Outlier tests (Grubbs, Dixon, generalized ESD), equivalence tests (TOST), tolerance intervals.
- Cross-tabulation and chi-square (Pearson, LR, Mantel–Haenszel, Fisher, McNemar), chi-square goodness of fit.
- Covariance, partial correlation, correlation matrix with p-values.
- Anderson–Darling / Shapiro–Wilk already present; add probability-plot dialog with distribution choice.
- Tests: dialog tests through the registry; E2E workflow for crosstab.

## 5b — Power & sample size, and ANOVA/regression dialogs already backed by the library — DONE
- Power and Sample Size for every `PowerModel` (z/t, paired, 2-sample, proportions, Poisson, variances, one-way
  ANOVA, factorial, TOST) with power curve; solve for any one of n / power / effect.
- Tukey, Fisher, Dunnett, Hsu MCB as standalone comparisons for GLM least-squares means.
- Stepwise / forward / backward regression and best subsets dialogs.
- Regression storage extras (leverage, Cook's D, DFFITS, PRESS residuals), prediction dialog with new data.
- Fitted line plot with confidence/prediction bands.

## 5c — Quality tools dialogs (library exists) — DONE
- Attribute charts: P, NP, C, U (normal and exact limits), Laney P′/U′, G, T.
- Time-weighted: MA, EWMA, CUSUM (tabular, V-mask), Hotelling T² with decomposition, generalized variance.
- Variables extras: between/within, zone, Z-MR short-run, Box-Cox charts, staged charts, Western Electric rule sets.
- Capability: non-normal (Box-Cox, Johnson, Weibull, etc.), empirical, extended sixpack, Pareto, run chart,
  cause-and-effect data entry.
- MSA: nested Gage R&R, Gage linearity & bias, attribute agreement analysis, Gage run chart.
- Tests: each chart end to end with stored flagged points.

## 5d — DOE dialogs and gaps — DIALOGS DONE (library gaps moved to 5o)
- Create designs: fractional factorial, Plackett–Burman, central composite, Box–Behnken, Taguchi arrays; custom
  generators; randomization and blocking options.
- Analyze: response surface, Taguchi, factorial plots (main effects, interaction, cube, Pareto, normal), contour
  and surface plots from the dialog.
- New library work: mixture designs, D-/I-optimal custom designs, 4-level and mixed Taguchi arrays (L16 4^5, L36),
  more Box–Behnken sizes, 3+ block CCD, split-plot designs.

## 5e — Data and Calc menus — DONE (matrices, replace-missing, unpivot, date patterns carried to 5k)
- Data: Stack/Unstack, Transpose, Merge worksheets, Subset worksheet, Split worksheet, Copy, Code (numeric, text,
  date), Recode, Rank, Concatenate, Convert text↔numeric↔date, Replace missing, Change data type, Delete/Insert
  columns, Sort multi-key (extend), Display data, Unpivot.
- Calc: Random Data dialogs for all distributions (library has samplers), Probability Distributions dialogs
  (pdf/cdf/inverse for all), Column Statistics, Row Statistics, Row/Column operations, Make Patterned Data,
  Make Indicator Variables, Matrices (inverse, transpose, eigen, etc.), Standardize (extend).
- Every Data/Calc action undoable as one step; macros gain matching commands.
- Tests: golden worksheets before/after for every operation; undo round trips.

## 5f — Regression family — DONE (Johnson response transform, predict dialogs for the new models, ordinal/nominal stepwise carried to 5g/5k)
- Binary, ordinal and nominal logistic regression (IRLS/Newton, odds ratios, deviance, Hosmer–Lemeshow, ROC data,
  classification table, stepwise).
- Poisson and general GLM with link choices (including negative binomial).
- Nonlinear regression (Levenberg–Marquardt, existing optimizer; user equations, confidence/prediction bands).
- Partial least squares, orthogonal regression, weighted regression, Box-Cox and Johnson response transformations,
  response-transformation dialog.
- Validation: SciPy/statsmodels references for each.

## 5g — ANOVA and mixed models — DONE
- Balanced ANOVA, fully nested ANOVA, general linear model with random factors and expected mean squares,
  mixed-effects models (REML), ANCOVA with plots, repeated measures, test for equal variances, main-effects and
  interaction plots, residual plots for each.
- Variance components (REML), expanded Gage R&R with fixed operators.
- Validation: lme4/statsmodels references.

## 5h — Nonparametrics, survival, multivariate and time-series gaps — DONE (carried to 5n/5o/5k: Turnbull, 3-param families, censored probability plots/AD, stepwise discriminant, Genz MVN, ARIMAX, trend analysis, MA/lag/difference)
- Nonparametrics: 2-sample Kolmogorov–Smirnov, Mood scale, Ansari–Bradley, normal scores, Spearman/Kendall
  dialogs, Mann–Kendall trend, Kaplan–Meier-style distribution-free options.
- Survival: Cox proportional hazards (Breslow/Efron ties, time-dependent off), Turnbull for arbitrary censoring,
  three-parameter Weibull/lognormal, gamma and extreme-value families, probability plots with censoring,
  likelihood-ratio intervals, Anderson–Darling with censoring.
- Multivariate: correspondence analysis (simple and multiple), oblique rotations (promax), stepwise discriminant,
  Genz multivariate-normal CDF.
- Time series: unit-root tests (ADF, KPSS), ARIMAX/regression with ARIMA errors, cross-correlation, automatic ARIMA
  order search, calendar axes, trend analysis, moving-average and lag/difference dialogs.

## 5i — Predictive analytics completion — DONE (TreeNet-style boosting options and variable-selection helpers carried forward)
- Binary-response MARS and logistic-link models; regularized boosting (TreeNet-style options); model comparison;
  score new data from saved model objects (save model into the project, score dialog, stored predictions);
  partial dependence and ICE plots; cross-validation for MARS; variable selection helpers.
- Model objects serialized in `.otb`.

## 5j — Graph menu and chart coverage — DONE (layout tool, calendar axes carried to 5k)
- Dialogs for: bar charts (simple, cluster, stack), pie, dotplot, stem-and-leaf, matrix plot, marginal plot,
  bubble plot, 3D scatter/surface/contour dialogs, heatmap, area, time-series with calendar axes, empirical CDF,
  interval and individual value plots as stand-alone graphs, probability plots for any distribution, layout tool
  with panel grids, graph editing (titles, axes, themes), export to SVG/PNG/PDF.
- Library work: stem-and-leaf and dotplot builders, bubble and heatmap renderers, PDF emitter.

## 5k — Workflow, import/export and platform — DONE (layout tool, accessibility mirror, Assistant trees, ARIMAX, SAS/SPSS/Stata, Minitab-file import, key customization carried)
- Command Line and History panes (macro engine already exists), session command echo, `EXECUTE`/`OUTFILE`/`STACK`
  macro commands, report printing/export to HTML/DOCX-style output, project properties.
- Import: Minitab `.mtw`/`.mpj`/`.mwx` if sample files become available; SAS/SPSS/Stata text formats, JSON, Parquet-free
  columnar text; database-free copy/paste improvements; ODBC is out of scope.
- Assistant: full decision trees and report cards for the main analyses.
- Accessibility: DOM mirror / native accessibility tree, high-contrast refinement, keyboard customization.

## Revision of 2026-11-18 — target is 90–100% of Minitab
The goal stated by the project owner is 90–100% of Minitab's functionality, so the roadmap is widened. Five areas the
first draft left out or under-sized become phases of their own, and the later phases are renamed (verification is now
5p, release 7). After every phase the coverage table (complete / library-only / missing, per menu area) is re-issued.

## 5l — Tables, EDA, bootstrap and simulation — DONE (chi-square stored contributions carried)
- Stat ▸ Tables: Tally Individual Variables, Descriptive Statistics table, Cell Statistics, chi-square variants with
  stored expected counts and contributions.
- Stat ▸ EDA: resistant line, median polish, rootogram, letter values, symmetry plot, Pareto of any column.
- Bootstrap and Monte-Carlo: bootstrap intervals (percentile, BCa) for any column statistic, simulation dialog with
  user equations, random sampling.
- Item analysis (Cronbach's alpha), cluster variables.

## 5m — Quality completion — DONE (cause-and-effect, Pareto by-variable, two-sided variables plans, ANSI Z1.4/Z1.9 tables carried)
- Acceptance sampling by attributes and by variables, OC / AOQ / ATI curves, sampling plan comparison.
- Multi-vari chart, symmetry and individual-distribution identification, Johnson and Box-Cox transformation dialogs,
  attribute capability (binomial, Poisson), capability for several variables, Pareto and run chart extras.

## 5n — Reliability completion — DONE (repairable systems, test plans, Turnbull, 3-parameter and gamma, Eyring; SEV/LEV and censored Anderson-Darling carried)
- Accelerated life testing (Arrhenius, Eyring, inverse power, temperature-humidity), repairable systems (mean cumulative
  function, Poisson process, Crow-AMSAA growth), test plans and demonstration tests, warranty analysis, Cox regression.

## 5o — DOE completion (DONE across 5o and 5o-b; see PHASE_LOG for the remaining size limits)
- Carried over from 5d: mixture designs, D-/I-optimal custom designs, 4-level and mixed Taguchi arrays (L16 4^5, L36), more Box-Behnken sizes, 3+ block CCD, split-plot designs.
- Response optimizer (desirability), overlaid contour plots, definitive screening designs, analyze variability,
  mixture and split-plot analysis, D-/I-optimal custom designs, steepest ascent.

## 5p — Verification against real Minitab and platform builds (PARTIAL: verified against Minitab's published examples and parallelised; real-Minitab checklist and first platform builds still open)
- Gather Minitab output for a defined checklist (each dialog gets one reference run); reconcile conventions
  recorded in the honest-limits sections (AIC counts, Winters initial values, unusual-observation rules, Johnson
  selection, tie-breaking).
- First real builds: wasm, macOS `.app/.dmg`, Windows `.msi/.exe`; fix native shell and CI API drift; browser
  smoke tests; performance pass (multi-threading for forests, CV, large tables).

## 7 — Final release packaging (formerly 5q; moved last on 2026-11-18)
- Version bump, changelog, licenses audit, signed installers, web deployment bundle, documentation site, sample
  data and tutorials, release workflow dry run.

## 6a — Graph behavior and worksheet/UX details — DONE (added 2026-11-18; see PHASE_LOG for limits: marker/line styles, UI translation, worksheet-to-graph brushing carried)
Graph behavior:
- Brushing and linked graphs: select points in one graph and see them highlighted in the worksheet and in other graphs.
- Graphs that update automatically when the worksheet data change (and a way to freeze a graph).
- Graph editing: attribute changes (colors, symbols, lines, fonts), annotations, axis and scale options, and a layout tool.
Worksheet and UX:
- Value labels and column formats; date/time edge cases (see 5k's carried date patterns).
- Undo/redo depth across data edits and analyses.
- Project Manager and Related Documents panes.
- Localization (decimal and date conventions, translated UI strings) and in-app help.
- Key customization (carried from 5k).

## Suggested ordering and rough size
| Phase | Theme | Mostly | Size |
|---|---|---|---|
| 5a–5d | Wire existing library into dialogs | UI + tests | small–medium each |
| 5e | Data and Calc menus | new UI logic | medium |
| 5f–5g | Regression and ANOVA/mixed models | new statistics | large |
| 5h | Nonparametric, survival, multivariate, time-series gaps | new statistics | large |
| 5i | Predictive analytics completion | ML + model objects | medium |
| 5j | Graph catalog | charts + UI | medium–large |
| 5k | Workflow, import, accessibility | platform | large |
| 5l | Tables, EDA, bootstrap, simulation | new UI + statistics | medium |
| 5m | Quality completion | acceptance sampling, transformations | medium |
| 5n | Reliability completion | ALT, repairable systems | large |
| 5o | DOE completion | optimizer, DSD, optimal designs | medium–large |
| 5p | Minitab verification and real builds | needs outside inputs | blocked on data/hardware |
| 6a | Graph behavior, worksheet and UX details | charts + UI | medium–large |
| 7 | Release (formerly 5q) | packaging | small |

Dependencies: 5p needs real Minitab output and machines with macOS/Windows/wasm toolchains; 5k's Minitab-file import needs
sample files; everything else can proceed offline.
