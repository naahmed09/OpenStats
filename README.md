# OpenStats

OpenStats is an open-source statistical software suite written from scratch in Rust: a worksheet, a session window, dialog-driven analyses (descriptive statistics, hypothesis tests, regression, ANOVA, quality and reliability tools, designed experiments, time series, multivariate methods) and an interactive graph engine. It runs as a native desktop app (a real window, no web view; macOS first) and in the browser (WebAssembly) from a single codebase.

> **Alpha software.** OpenStats is young and has been tested by very few people. Its results have **not** been validated against other statistical packages or published reference values across the board, so check anything important against a trusted tool before relying on it. It is provided without warranty (see the licences). Bug reports that include a small data file and the result you expected are the most useful.

Every crate in the workspace uses the Rust standard library only. The browser front end is a plain C ABI plus hand-written JavaScript (no `wasm-bindgen`); the only places that depend on third-party crates are the desktop shells, which live outside the workspace: the native shell in `openstats-desktop/native` (winit, softbuffer, fontdue, muda, rfd, arboard) and the older Tauri shell in `openstats-desktop/tauri/src-tauri`, which is web-view based and no longer the intended desktop app.

## Status

| Phase | Scope | State |
|---|---|---|
| 1a | Workspace, `openstats-core` data engine | **done** |
| 1b | Formula calculator (~150 functions, column formulas); DEFLATE/zlib/gzip, ZIP, XML, XLSX import, OLE2, Minitab detection | **done** |
| 1c | `openstats-math` linear algebra (dense, factorizations, eigen, SVD, sparse) | **done** |
| 1d | special functions, Tukey/Dunnett, 22 continuous distributions, calculator NSCORES/PDF/CDF/INVCDF | **done** |
| 1e | Owen's T, E₁/Ei, ζ, Lambert W; discrete and non-central distributions; calculator wiring | **done** |
| 1f | optimizers, nonlinear least squares, PRNGs, variate samplers, calculator RANDOM | **done** |
| 1g | NIST StRD harness, `stats` (OLS, ANOVA, summary), Phase 1 integration | **done** |
| 2a | `openstats-stats`: descriptive stats, graphical-summary data, cross-tabulation and chi-square/Fisher/McNemar, correlations | **done** |
| 2b | t/Z, proportion, Poisson and variance tests; normality, outlier and TOST equivalence tests | **done** |
| 2c | Session report model, power & sample size, tolerance intervals | **done** |
| 2d | Formula & design-matrix layer, linear regression (OLS, diagnostics, stepwise, best subsets) | **done** |
| 2e | Quality & SPC: unbiasing constants, variable/attribute control charts, special-cause tests, normal & non-normal capability, capability sixpack | **done** |
| 2f | ANOVA (one-way, Welch, Tukey/Fisher/Dunnett/Hsu MCB, GLM Type I/III) and Gage R&R (crossed ANOVA / X̄-R, nested, six-panel) | **done** |
| 2g | SPC charts: Western Electric/Nelson sets, staged limits, I-MR-R/S, zone, Z-MR, Box-Cox, Laney P′/U′, G/T, MA/EWMA/CUSUM (tabular + V-mask), Hotelling T² | **done** |
| 2h | SPC completion: exact binomial/Poisson limits, one-sided/FIR CUSUM, exponential T, rules and staging on every new chart, T² decomposition, generalized variance | **done** |
| 2i + 2j | Capability confidence intervals (χ², exact non-central t), more distributions and empirical percentiles, extended sixpack; gage linearity & bias, attribute agreement (Fleiss/Cohen κ) | **done** |
| 2k | Design of experiments: factorial / fractional / Plackett–Burman, CCD and Box–Behnken response surfaces, Taguchi arrays and S/N, desirability optimizer, DOE charts | **done** |
| 3a + 3b | `openstats-charts`: scene graph and layout, scales and ticks, font metrics, SVG / Canvas command stream / PNG renderers, hit-testing and viewports, plot builders for every `ChartSpec` | **done** |
| 3c | `openstats-charts::render_3d`: software 3-D pipeline (perspective/orthographic camera, painter's algorithm, Lambert + Blinn–Phong lighting), marching-squares contours, shaded surface/wireframe for CCD and Box–Behnken response surfaces, rotate/zoom/pan/pick hooks | **done** |
| 3d | `openstats-ui::grid`: virtualized worksheet grid (1M+ rows), `C1`/`C2-T`/`C3-D` headers, formula bar, undo/redo, multi-range selection, TSV clipboard, number/date formatting, `*` for missing | **done** |
| 3e | `openstats-ui::workspace` + `dialogs`: Minitab-style split workspace (Project Navigator tree with context menus, Session window with aligned tables and chart thumbnails, copy/print, worksheet grid with formula bar), menu bar (File … Assistant) and declarative dialogs bound to 21 `openstats-stats` analyses; project save/open | **done** |
| 3f | `openstats-host` (std-only wasm C ABI), browser glue (`openstats-desktop/web`), Tauri v2 shell with native menus/dialogs/clipboard/graph tear-off, GLM / factorial analysis / Gage R&R / Hotelling T² dialogs, Ctrl/Shift multi-select listboxes, grid text selection and IME, CSV/XLSX import | **done** (wasm and Tauri builds not run here) |
| 3g + 3h | End-to-end tests through the C ABI (`openstats-host/tests/e2e`: CSV/XLSX import → regression / capability via menus and dialogs → Session report checked against independent computations → charts → save/reopen), keyboard shortcuts and accessibility, light/dark chrome (`openstats-ui::style`, WCAG contrast tested), release tooling (`scripts/`: web bundle, universal macOS `.dmg`, Windows `.msi`/`.exe`, CI workflows), app icons, version 1.0.0 | **done** (installers and wasm build not run here) |
| 4 | Advanced analytics and ML, reliability/survival, multivariate, time series, nonparametrics, `.mac` macro engine (`openstats-script`) | deferred |

## Layout

```
openstats/
├── Cargo.toml              workspace manifest (shared metadata, lints, profiles)
├── rust-toolchain.toml     stable + wasm32-unknown-unknown
├── crates/
│   ├── openstats-core/       columnar data engine (std only)
│       ├── src/
│       │   ├── value.rs missing.rs datetime.rs refs.rs
│       │   ├── column.rs worksheet.rs workbook.rs
│       │   ├── edit.rs history.rs          reversible edits + undo/redo
│       │   ├── ops/                        stack, unstack, split, subset, transpose, sort,
│       │   │                               rank, recode, concatenate, merge, change type,
│       │   │                               extract, patterned data
│       │   ├── formula/                    lexer, parser, AST, vectorised evaluator,
│       │   │   ├── functions/              math, stats, row, logic, text, date (175 names)
│       │   │   └── recalc.rs               formula columns, dependency order, cycles
│       │   └── io/                         csv, json, otb, deflate (+zlib/gzip), zip, xml,
│       │                                   xlsx, cfb (OLE2), minitab (inspection), crc32
│       └── tests/                          workflow.rs (end-to-end, randomized undo/redo),
│                                           fixtures/openpyxl_sample.xlsx
│   ├── openstats-math/       numerical kernel (std only)
│       ├── src/linalg/                     matrix, blas (blocked GEMM), triangular, householder,
│       │                                   lu, qr, cholesky, eigen_sym, eigen_gen, svd, sparse, kron
│       ├── src/special/                    gamma, erf, incgamma, incbeta, bessel, tukey, dunnett
│       ├── src/dist/                       22 continuous distributions (ContinuousDistribution trait)
│       ├── src/integrate.rs roots.rs calc.rs  quadrature, Brent, calculator functions
│       └── tests/                          numpy_reference.rs, scipy_reference.rs (≈2,270 reference values)
│   └── openstats-stats/      statistical procedures (std only)
│       ├── src/                            quantile, descriptive, graphical, crosstab, correlation
│       └── tests/                          reference.rs (SciPy/NumPy/exact-rational values)
│   └── openstats-charts/     2-D charts (std only)
│       ├── src/core/                       scene graph, scales, ticks, fonts, layout, hit-testing, styles
│       ├── src/render/                     svg, canvas (JSON + binary), png + software rasteriser
│       ├── src/plots/                      basic, relation, quality builders and `render_spec`
│       ├── src/render_3d.rs                3-D camera, painter's algorithm, lighting, contours, surface plot
│       └── tests/                          render_all.rs, surface3d.rs (CCD / Box–Behnken surfaces)
│   └── openstats-ui/         worksheet grid, workspace and dialogs (std only)
│       ├── src/grid/                       controller, input, render (Scene), selection, clipboard, format, metrics
│       ├── src/dialogs/                    dialog specs, analysis registry (25 analyses), menu bar, dialog view, Assistant
│       ├── src/workspace/                  Navigator, Session window, project items, Workspace shell (input routing, save/open)
│       └── tests/                          grid.rs, analyses.rs (every analysis), workspace.rs (menu → dialog → output, files)
│   └── openstats-host/       wasm C-ABI host for the workspace (std only; native tests drive the same ABI)
│       ├── src/                            abi (exports), keys (wire codes), lib (Host, GraphHost)
│       ├── examples/                       dump_frame.rs (JS test fixtures), make_icons.rs (app icons)
│       └── tests/                          abi.rs, release.rs (icons, versions, bundle config), e2e/ (user workflows)
├── openstats-desktop/        browser and Tauri v2 front ends (see its README)
│   ├── web/                openstats.js, tauri-shell.js, index.html, graph.html, node tests + fixtures
│   └── tauri/src-tauri/    Tauri v2 shell (own workspace; written, not compiled here)
├── scripts/                release tooling: release.sh, build-web.sh, package-macos.sh, package-windows.ps1, tests
├── .github/workflows/      ci.yml and release.yml (written, not run here)
└── tests/fixtures/nist/    drop official NIST StRD .dat files here (Phase 1g)
```

## Building and testing

```sh
cargo test --workspace        # all std-only crates; no network needed after first fetch
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --no-deps --open
```

JavaScript tests (decoder, painter, events, IME, Tauri glue): `node --test openstats-desktop/web/test/*.test.mjs`.

### Releasing

```sh
scripts/release.sh                 # checks, web bundle, and the native installer for this OS
scripts/release.sh --web-only      # just dist/web and dist/openstats-web-VERSION.tar.gz
scripts/package-macos.sh           # universal OpenStats_VERSION_universal.dmg (macOS only)
pwsh scripts/package-windows.ps1   # OpenStats_VERSION_x64.msi and -setup.exe (Windows only)
```

Everything lands in `dist/` with a `SHA256SUMS` file. Code signing and notarisation are optional and read from the environment (see the header of each script). See `openstats-desktop/README.md` for prerequisites.

### Keyboard

| Keys | Action | Keys | Action |
|---|---|---|---|
| Ctrl+N | New project | Ctrl+Shift+N | New worksheet |
| Ctrl+O | Open project | Ctrl+I | Open data file (CSV, TSV, XLSX, JSON) |
| Ctrl+S | Save project | Ctrl+P | Print Session window |
| Ctrl+Z | Undo | Ctrl+Y or Ctrl+Shift+Z | Redo |
| Ctrl+F | Go to the formula bar | F6 | Next pane |
| F10 | Open the File menu | Alt+F/E/D/C/S/G/V/A | Open File, Edit, Data, Calc, Stat, Graph, View, Assistant |
| Ctrl+Shift+D | Toggle the dark theme | Ctrl+A / Ctrl+C | Select all / copy in the focused pane |

(Cmd replaces Ctrl on macOS.) Menus show their shortcuts. A menu or dialog that is open has the keyboard first.

The toolchain file asks rustup for the `wasm32-unknown-unknown` target. On an offline machine that already has a stable toolchain, run `RUSTUP_TOOLCHAIN=stable cargo test` to skip the target download.

## Core concepts (Phase 1a)

- **Missing values** follow Minitab. Numeric and date/time missing values display as `*` (stored as `NaN` / `i64::MIN`); text missing is the empty string. When reading, `*`, empty, `NA`, `NaN`, `N/A`, `.` and `NULL` count as missing. A *quoted* `"*"` in a CSV file stays text.
- **Date/time** values are `i64` microseconds since 1970-01-01 UTC on the proleptic Gregorian calendar. They support ISO-8601, US/European numeric dates, textual months, AM/PM, offsets, and Excel serial conversion.
- **Undo/redo.** Every mutation is an `Edit`, and applying an edit returns its exact inverse. `History` provides transactions (`begin_group`/`end_group`/`cancel_group`), depth and memory limits, and save-point tracking (`is_dirty`). `Edit::Batch` is atomic: it rolls back if any step fails.
- **Operations** are pure functions that return new columns or worksheets. `ops::store_columns` and `ops::add_worksheets` turn their results into undoable edits.
- **File formats:**
  - CSV/TSV: RFC 4180, delimiter auto-detection, type inference, decimal-comma locales, Windows-1252 fallback.
  - JSON: documented schema, lossless.
  - `.otb`: versioned little-endian binary with a CRC-32 checksum and allocation-safe decoding.

## Phase 1b additions

- **Calculator.** Minitab-style syntax: `C1`, names, `'quoted names'`, `K1`, `*` for missing, `**`/`^`, comparisons, `AND`/`OR`/`NOT`/`&`/`|`/`~`. Evaluation is vectorised with broadcasting; missing values propagate, and `AND`/`OR` use three-valued logic. Date arithmetic is supported, and comparing a date with text parses the text as a date. Functions live in an extensible `FunctionRegistry`; distribution functions will register there from `openstats-math`.
- **Column formulas.** `ColumnProps::formula`, with dependency-ordered recalculation and cycle detection. Setting a formula and recalculating are both undoable edits. Formulas are stored in JSON and in `.otb` v1.1, and v1.0 files still load.
- **DEFLATE.** Inflate handles stored, fixed and dynamic blocks. It is checked against streams produced by the reference zlib and has a decompression-bomb guard. Deflate does LZ77 with lazy matching and picks the smallest of stored, fixed or dynamic per block; real zlib decodes its output at every level. zlib and gzip wrappers are included.
- **ZIP** reader (ZIP64, UTF-8 and CP437 names, CRC checks) and writer; Python's `zipfile` reads the writer's output. **XML** pull parser.
- **XLSX import**: shared and inline strings, rich text, booleans, errors, cached formula results, date detection from number formats, the 1904 date system, and sheet selection.
- **OLE2 compound files**: reader for v3 and v4 plus a writer.
- **Minitab files** (`.mpx`/`.mwx`/`.mpj`/`.mtw`) are detected and their containers inspected. Data import waits on real sample files, because Minitab doesn't publish the internal layout (see `io::minitab`).

## Phase 1c additions (`openstats-math::linalg`)

- **Dense matrix type.** `Matrix` (row-major) with slicing, stacking, norms, and a cache-blocked GEMM that handles all four transpose combinations (about 6.7 GFLOP/s single-threaded). Also an overflow-safe `nrm2`, `AᵀA` via a symmetric rank-k update, and triangular solves.
- **LU** with partial pivoting: solve, transpose-solve, determinant, log-determinant, inverse, and a Hager/Higham `rcond` estimate.
- **Householder QR**, unpivoted or with column pivoting (LAPACK-style norm downdating): thin and full Q, numerical rank, least squares including the rank-deficient basic solution, and the residual sum of squares.
- **Cholesky, LDLᵀ and modified Cholesky.** LDLᵀ handles indefinite matrices and reports inertia; modified Cholesky follows Nocedal–Wright.
- **Symmetric eigensolvers**: tridiagonal QL (default) and cyclic Jacobi (for high relative accuracy).
- **General eigensolver**: Hessenberg reduction plus Francis double-shift QR, returning complex eigenpairs.
- **SVD** (Golub–Kahan–Reinsch) for any shape, with pseudo-inverse, minimum-norm least squares, rank, null space, 2-norm and condition number.
- **Sparse**: triplet, CSR and CSC formats, sparse matrix–matrix product, a sparse Cholesky with minimum-degree ordering and elimination tree, and Jacobi-preconditioned conjugate gradients.

## Phase 1d additions

- **Special functions** (`openstats_math::special`):
  - Γ, lnΓ (full relative accuracy near its zeros at 1 and 2), ψ, ψ′, B and ln B (no cancellation at large arguments).
  - erf, erfc, erfcx and their inverses; standard normal pdf, cdf, sf and quantile accurate deep into the tails.
  - Regularised incomplete gamma P and Q and incomplete beta Iₓ, each with inverses and both tails computed directly.
  - Bessel Jₙ, Yₙ, Iₙ, Kₙ.
  - Studentized range cdf and quantile; Dunnett one- and two-sided cdf and quantile for arbitrary group sizes.
- **Distributions** (`openstats_math::dist`): Normal, Lognormal, Half-normal, Folded normal, Johnson SU/SB/SL, Gamma, Chi-square, Exponential, Weibull, Beta, t, F, Logistic, Loglogistic, Cauchy, Laplace, Uniform, Triangular, and the smallest and largest extreme value (Gumbel). Each provides pdf, ln_pdf, cdf, sf, quantile, isf, mean, variance and median, with Minitab parameterisations.
- **Calculator** (default `formula` feature): `NSCORES`, `PDF`, `CDF`, `SFCDF`, `INVCDF`.
- **Verification**: about 2,270 SciPy reference values agree to ≤ 8.7e-13, most to about 1e-15. Where SciPy itself is inaccurate (10 cases, documented in the test), values are checked against 40-digit decimal results or strict round trips.

## Phase 1e additions

- **Special functions**: Owen's T, E₁ and Ei, Riemann ζ (all real s), Lambert W₀ and W₋₁ (accurate to the double-double branch point).
- **Discrete distributions** (`DiscreteDistribution`: pmf, ln_pmf, cdf, sf, quantile, isf, mean, variance, std_dev, support): Binomial, Poisson, Hypergeometric, Negative binomial and Geometric (counting trials, Minitab convention), Discrete uniform / Integer, and a user-specified Discrete. pmfs use Loader's saddle-point terms; quantiles are exact smallest-k searches.
- **Non-central** (`ContinuousDistribution`): χ², Beta, F, t, via Poisson-mixture series summed from the mode, and log-domain quadrature where a series would cancel (nct opposite tail, nct pdf for δt<0).
- **Calculator**: `PDF`, `CDF`, `SFCDF`, `INVCDF` accept all of the above (e.g. `CDF(x, "binomial", 20, 0.3)`, `INVCDF(p, "nct", 10, 1.5)`).
- **Verification**: SciPy reference tables (discrete values against exact 60-digit decimal sums). Where SciPy is wrong, exact or independent quadrature values are used instead.
- **Known limit**: Binomial/NB cdf/sf go through the incomplete beta; relative error is about 3e-11 at n = 1e9 and ≤ 1e-11 otherwise.

## Phase 1f additions

- **Optimizers** (`openstats_math::optim`): Brent 1-D minimisation with bracketing; Nelder–Mead (adaptive coefficients, restarts); BFGS and L-BFGS with strong-Wolfe line search; bound-constrained projected L-BFGS; damped Newton and trust-region dogleg on a modified-Cholesky shifted Hessian; Levenberg–Marquardt and Gauss–Newton for nonlinear least squares with analytic or differenced Jacobians and SVD-based parameter covariance and standard errors. Analytic or finite-difference derivatives throughout; non-convergence is reported, never hidden.
- **Random numbers** (`openstats_math::random`): `Rng`/`SeedableRng`, MT19937, MT19937-64, PCG64 (XSL-RR 128/64, selectable streams) and xoshiro256**, all checked against published reference outputs. `sample`/`sample_n` on every distribution: polar normal, Marsaglia–Tsang gamma, log-scale beta/F, PTRS Poisson, inversion/beta-splitting binomial, mode-walk hypergeometric, gamma–Poisson negative binomial, mixture constructions for the non-central laws, quantile inversion for the rest. Multivariate normal (also singular covariance), Wishart, Dirichlet, multinomial.
- **Calculator**: `RANDOM(n, "dist", …)`, `RANDOMSEED(seed, n, "dist", …)`, `BASE(seed)`.
- **Verification**: Rosenbrock (2-D and 10-D), Himmelblau, Beale, Powell singular, Wood, Goldstein–Price, Booth; SciPy `curve_fit` parameters and standard errors; Kolmogorov–Smirnov tests for all 35 continuous samplers and χ² tests for all discrete ones; moment checks for the multivariate samplers.

## License

Dual-licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
