# NIST StRD fixtures

The harness in `crates/openstats-math/tests/nist_strd.rs` (parser: `tests/nist/parser.rs`,
model evaluator with forward-mode autodiff: `tests/nist/expr.rs`) discovers and verifies:

* `tests/fixtures/nist/*.dat` — **official** NIST files. Git-ignored; download them from
  <https://www.itl.nist.gov/div898/strd/> and drop them here. They take precedence over
  embedded files of the same name.
* `tests/fixtures/nist/embedded/*.dat` — committed files in the official layout.

## Provenance of the embedded files (read this)

The official files could not be downloaded in the build environment. The embedded files were
**reconstructed**: data from the published datasets, certified values **recomputed independently**
in exact rational arithmetic (linear, univariate) or 80-digit decimal Gauss–Newton (nonlinear),
then compared with the published certified digits (they agree to ~1e-11 or better). Each file
carries a `Note:` line saying so.

| Procedure | Embedded |
|---|---|
| Univariate | NumAcc1, NumAcc2, NumAcc3, NumAcc4 |
| Linear | Longley, Wampler1, Wampler2, NoInt1, NoInt2 |
| ANOVA | OpenStatsAnova1 — **constructed by OpenStats, not a NIST dataset** (large-offset data) |
| Nonlinear | Misra1a, MGH09, MGH10, MGH17, Rat42, Rat43 |

Not embedded (reconstruction not possible offline): Norris, Pontius, Filip, Wampler3–5, PiDigits,
Lew, Lottery, Mavro, Michelso, SiRstv, SmLs01–09, AtmWtAg, Thurber and the other nonlinear sets.
PiDigits was attempted and dropped: the recalled certified values were not reproducible from π.
These are verified automatically as soon as the official files are present; until then they are
simply absent from the report, not claimed as passing.

Run `cargo test -p openstats-math --test nist_strd -- --nocapture` for the LRE table.
