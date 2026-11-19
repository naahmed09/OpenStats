//! Cross-tabulation and tests of association for categorical data.
//!
//! * [`Crosstab`] holds a table of counts with row/column labels and gives
//!   marginal totals, row/column/total percentages, expected counts and
//!   residuals.
//! * Tests: Pearson χ², likelihood-ratio χ² (G²), Mantel–Haenszel
//!   linear-by-linear association, Fisher's exact test (2×2 and r×c) and
//!   McNemar's test for paired binary data.
//!
//! Rows or columns whose total is zero are ignored by every test (they carry
//! no information and would make expected counts zero). Cases with a missing
//! category (empty label / `NaN`) are dropped when a table is built from raw
//! data and counted in [`Crosstab::n_excluded`].

use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{Binomial, ChiSquared, ContinuousDistribution, DiscreteDistribution, Hypergeometric};
use openstats_math::special::gamma::ln_factorial;
use openstats_math::stats::summary;

/// Table of counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Crosstab {
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    /// `counts[i][j]`: observations in row i, column j.
    pub counts: Vec<Vec<u64>>,
    /// Raw cases dropped because a category was missing.
    pub n_excluded: usize,
}

/// Row, column and total percentages (each 0–100). A row (column) with no
/// observations has `NaN` percentages in the row (column) table.
#[derive(Debug, Clone, PartialEq)]
pub struct Percentages {
    pub row: Vec<Vec<f64>>,
    pub col: Vec<Vec<f64>>,
    pub total: Vec<Vec<f64>>,
}

impl Crosstab {
    /// Table from counts; labels are `"1"`, `"2"`, …. Rows must be equal length.
    pub fn from_counts(counts: Vec<Vec<u64>>) -> Result<Self> {
        let c = counts.first().map_or(0, Vec::len);
        if counts.is_empty() || c == 0 {
            return invalid("a table needs at least one row and one column");
        }
        if counts.iter().any(|r| r.len() != c) {
            return invalid("table rows have different lengths");
        }
        Ok(Crosstab {
            row_labels: (1..=counts.len()).map(|i| i.to_string()).collect(),
            col_labels: (1..=c).map(|i| i.to_string()).collect(),
            counts,
            n_excluded: 0,
        })
    }

    /// Table from two label vectors of equal length. Empty labels are missing.
    /// Levels are sorted in text order.
    pub fn from_labels(rows: &[String], cols: &[String]) -> Result<Self> {
        if rows.len() != cols.len() {
            return invalid(format!("{} row labels but {} column labels", rows.len(), cols.len()));
        }
        let level = |v: &[String]| -> Vec<String> {
            let mut l: Vec<String> = v.iter().filter(|s| !s.is_empty()).cloned().collect();
            l.sort();
            l.dedup();
            l
        };
        let keep: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].is_empty() && !cols[i].is_empty()).collect();
        let (rl, cl) = (
            level(&keep.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>()),
            level(&keep.iter().map(|&i| cols[i].clone()).collect::<Vec<_>>()),
        );
        if rl.is_empty() || cl.is_empty() {
            return Err(StatsError::InsufficientData { what: "cross-tabulation", needed: 1, got: 0 });
        }
        let mut counts = vec![vec![0u64; cl.len()]; rl.len()];
        for &i in &keep {
            let r = rl.binary_search(&rows[i]).expect("level present");
            let c = cl.binary_search(&cols[i]).expect("level present");
            counts[r][c] += 1;
        }
        Ok(Crosstab { row_labels: rl, col_labels: cl, counts, n_excluded: rows.len() - keep.len() })
    }

    /// Table from two numeric variables; `NaN` is missing; levels are sorted
    /// numerically and labelled with the shortest round-trip text.
    pub fn from_numeric(rows: &[f64], cols: &[f64]) -> Result<Self> {
        if rows.len() != cols.len() {
            return invalid(format!("{} row values but {} column values", rows.len(), cols.len()));
        }
        if rows.iter().chain(cols).any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        let keep: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].is_nan() && !cols[i].is_nan()).collect();
        let levels = |v: &[f64]| -> Vec<f64> {
            let mut l: Vec<f64> = keep.iter().map(|&i| v[i]).collect();
            l.sort_by(|a, b| a.total_cmp(b));
            l.dedup();
            l
        };
        let (rl, cl) = (levels(rows), levels(cols));
        if rl.is_empty() || cl.is_empty() {
            return Err(StatsError::InsufficientData { what: "cross-tabulation", needed: 1, got: 0 });
        }
        let pos = |l: &[f64], v: f64| l.binary_search_by(|p| p.total_cmp(&v)).expect("level present");
        let mut counts = vec![vec![0u64; cl.len()]; rl.len()];
        for &i in &keep {
            counts[pos(&rl, rows[i])][pos(&cl, cols[i])] += 1;
        }
        Ok(Crosstab {
            row_labels: rl.iter().map(|v| v.to_string()).collect(),
            col_labels: cl.iter().map(|v| v.to_string()).collect(),
            counts,
            n_excluded: rows.len() - keep.len(),
        })
    }

    pub fn n_rows(&self) -> usize {
        self.counts.len()
    }
    pub fn n_cols(&self) -> usize {
        self.counts[0].len()
    }
    pub fn row_totals(&self) -> Vec<u64> {
        self.counts.iter().map(|r| r.iter().sum()).collect()
    }
    pub fn col_totals(&self) -> Vec<u64> {
        (0..self.n_cols()).map(|j| self.counts.iter().map(|r| r[j]).sum()).collect()
    }
    /// Total count.
    pub fn total(&self) -> u64 {
        self.counts.iter().flatten().sum()
    }

    /// Row, column and total percentages.
    pub fn percentages(&self) -> Percentages {
        let (rt, ct, n) = (self.row_totals(), self.col_totals(), self.total() as f64);
        let pct = |x: u64, d: f64| if d == 0.0 { f64::NAN } else { 100.0 * x as f64 / d };
        let mut p = Percentages { row: Vec::new(), col: Vec::new(), total: Vec::new() };
        for (i, r) in self.counts.iter().enumerate() {
            p.row.push(r.iter().map(|&x| pct(x, rt[i] as f64)).collect());
            p.col.push(r.iter().enumerate().map(|(j, &x)| pct(x, ct[j] as f64)).collect());
            p.total.push(r.iter().map(|&x| pct(x, n)).collect());
        }
        p
    }

    /// Expected counts under independence, `row_total·col_total/n`.
    pub fn expected(&self) -> Vec<Vec<f64>> {
        let (rt, ct, n) = (self.row_totals(), self.col_totals(), self.total() as f64);
        rt.iter().map(|&r| ct.iter().map(|&c| r as f64 * c as f64 / n).collect()).collect()
    }

    /// Standardised (Pearson) residuals (O−E)/√E.
    pub fn pearson_residuals(&self) -> Vec<Vec<f64>> {
        let e = self.expected();
        self.counts
            .iter()
            .zip(&e)
            .map(|(r, er)| r.iter().zip(er).map(|(&o, &e)| (o as f64 - e) / e.sqrt()).collect())
            .collect()
    }

    /// Adjusted standardised residuals (O−E)/√(E(1−r/n)(1−c/n)).
    pub fn adjusted_residuals(&self) -> Vec<Vec<f64>> {
        let (rt, ct, n) = (self.row_totals(), self.col_totals(), self.total() as f64);
        let e = self.expected();
        (0..self.n_rows())
            .map(|i| {
                (0..self.n_cols())
                    .map(|j| {
                        (self.counts[i][j] as f64 - e[i][j])
                            / (e[i][j] * (1.0 - rt[i] as f64 / n) * (1.0 - ct[j] as f64 / n)).sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    /// Indices of rows and columns with a non-zero total.
    fn active(&self) -> (Vec<usize>, Vec<usize>) {
        let (rt, ct) = (self.row_totals(), self.col_totals());
        ((0..rt.len()).filter(|&i| rt[i] > 0).collect(), (0..ct.len()).filter(|&j| ct[j] > 0).collect())
    }

    /// The table without empty rows/columns; errors if fewer than 2×2 remain.
    fn trimmed(&self, what: &'static str) -> Result<Crosstab> {
        let (ri, ci) = self.active();
        if ri.len() < 2 || ci.len() < 2 {
            return invalid(format!("{what} needs at least two non-empty rows and two non-empty columns"));
        }
        Ok(Crosstab {
            row_labels: ri.iter().map(|&i| self.row_labels[i].clone()).collect(),
            col_labels: ci.iter().map(|&j| self.col_labels[j].clone()).collect(),
            counts: ri.iter().map(|&i| ci.iter().map(|&j| self.counts[i][j]).collect()).collect(),
            n_excluded: self.n_excluded,
        })
    }
}

/// Result of a χ² test of independence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquareTest {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    /// Smallest expected count.
    pub min_expected: f64,
    /// Cells with expected count < 5 (the usual validity warning).
    pub cells_expected_below_5: usize,
    /// Total number of cells.
    pub n_cells: usize,
}

fn chi_p(stat: f64, df: usize) -> Result<f64> {
    Ok(ChiSquared::new(df as f64)?.sf(stat))
}

fn chi_square(t: &Crosstab, likelihood_ratio: bool, what: &'static str) -> Result<ChiSquareTest> {
    let t = t.trimmed(what)?;
    let e = t.expected();
    let mut terms = Vec::new();
    let (mut min_e, mut below) = (f64::INFINITY, 0);
    for (r, er) in t.counts.iter().zip(&e) {
        for (&o, &ev) in r.iter().zip(er) {
            min_e = min_e.min(ev);
            if ev < 5.0 {
                below += 1;
            }
            let o = o as f64;
            terms.push(if likelihood_ratio {
                if o == 0.0 {
                    0.0
                } else {
                    2.0 * o * (o / ev).ln()
                }
            } else {
                (o - ev) * (o - ev) / ev
            });
        }
    }
    let stat = summary::sum(terms).max(0.0);
    let df = (t.n_rows() - 1) * (t.n_cols() - 1);
    Ok(ChiSquareTest {
        statistic: stat,
        df,
        p_value: chi_p(stat, df)?,
        min_expected: min_e,
        cells_expected_below_5: below,
        n_cells: t.n_rows() * t.n_cols(),
    })
}

/// Pearson χ² = Σ(O−E)²/E, df = (r−1)(c−1).
pub fn pearson_chi_square(t: &Crosstab) -> Result<ChiSquareTest> {
    chi_square(t, false, "the Pearson chi-square test")
}

/// Likelihood-ratio χ² G² = 2ΣO·ln(O/E), df = (r−1)(c−1).
pub fn likelihood_ratio_chi_square(t: &Crosstab) -> Result<ChiSquareTest> {
    chi_square(t, true, "the likelihood-ratio chi-square test")
}

/// Mantel–Haenszel linear-by-linear association test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearByLinear {
    /// M² = (n−1)·r², χ² with 1 df.
    pub statistic: f64,
    /// Correlation of the row and column scores over the n observations.
    pub correlation: f64,
    pub df: usize,
    pub p_value: f64,
}

/// Mantel–Haenszel χ² with equally spaced scores 1, 2, … (use
/// [`linear_by_linear_scored`] for other scores).
pub fn linear_by_linear(t: &Crosstab) -> Result<LinearByLinear> {
    let rs: Vec<f64> = (1..=t.n_rows()).map(|i| i as f64).collect();
    let cs: Vec<f64> = (1..=t.n_cols()).map(|i| i as f64).collect();
    linear_by_linear_scored(t, &rs, &cs)
}

/// Mantel–Haenszel χ² with the given row and column scores.
pub fn linear_by_linear_scored(t: &Crosstab, row_scores: &[f64], col_scores: &[f64]) -> Result<LinearByLinear> {
    if row_scores.len() != t.n_rows() || col_scores.len() != t.n_cols() {
        return invalid("one score per row and per column is required");
    }
    if row_scores.iter().chain(col_scores).any(|s| !s.is_finite()) {
        return invalid("scores must be finite");
    }
    let n = t.total() as f64;
    if n < 2.0 {
        return Err(StatsError::InsufficientData { what: "linear-by-linear test", needed: 2, got: n as usize });
    }
    let (rt, ct) = (t.row_totals(), t.col_totals());
    let mr = summary::sum((0..rt.len()).map(|i| rt[i] as f64 * row_scores[i])) / n;
    let mc = summary::sum((0..ct.len()).map(|j| ct[j] as f64 * col_scores[j])) / n;
    let sxx = summary::sum((0..rt.len()).map(|i| rt[i] as f64 * (row_scores[i] - mr).powi(2)));
    let syy = summary::sum((0..ct.len()).map(|j| ct[j] as f64 * (col_scores[j] - mc).powi(2)));
    if sxx == 0.0 || syy == 0.0 {
        return invalid("the linear-by-linear test needs variation in both row and column scores among observed cases");
    }
    let sxy = summary::sum(
        (0..rt.len())
            .flat_map(|i| (0..ct.len()).map(move |j| (i, j)))
            .map(|(i, j)| t.counts[i][j] as f64 * (row_scores[i] - mr) * (col_scores[j] - mc)),
    );
    let r = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let m2 = (n - 1.0) * r * r;
    Ok(LinearByLinear { statistic: m2, correlation: r, df: 1, p_value: chi_p(m2, 1)? })
}

/// Fisher's exact test result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherExact {
    /// Two-sided p-value: total probability of tables (same margins) no more
    /// likely than the observed one (relative tolerance 1e-7, as R).
    pub p_value: f64,
    /// 2×2 only: P(X ≤ n₁₁) and P(X ≥ n₁₁).
    pub p_left: Option<f64>,
    pub p_right: Option<f64>,
    /// 2×2 only: sample odds ratio n₁₁n₂₂/(n₁₂n₂₁) (∞ or `NaN` when degenerate).
    pub odds_ratio: Option<f64>,
    /// Number of tables enumerated (r×c); 0 for 2×2.
    pub tables_enumerated: u64,
}

/// Largest number of tables the r×c enumeration will visit.
pub const FISHER_MAX_TABLES: u64 = 200_000_000;

/// Fisher's exact test. A 2×2 table (after dropping empty rows/columns) uses
/// the hypergeometric distribution; larger tables enumerate every table with
/// the observed margins and error with [`StatsError::TooLarge`] beyond
/// [`FISHER_MAX_TABLES`] tables.
pub fn fisher_exact(t: &Crosstab) -> Result<FisherExact> {
    let t = t.trimmed("Fisher's exact test")?;
    let (r, c) = (t.n_rows(), t.n_cols());
    let (rt, ct) = (t.row_totals(), t.col_totals());
    let n = t.total();
    if r == 2 && c == 2 {
        let a = t.counts[0][0];
        let h = Hypergeometric::new(n, rt[0], ct[0])?;
        let obs = h.pmf(a as f64);
        let lo = rt[0].saturating_sub(n - ct[0]);
        let hi = rt[0].min(ct[0]);
        let thr = obs * (1.0 + 1e-7);
        let p = summary::sum((lo..=hi).map(|k| h.pmf(k as f64)).filter(|&p| p <= thr)).min(1.0);
        let (b, cc, d) = (t.counts[0][1], t.counts[1][0], t.counts[1][1]);
        let odds = if b * cc == 0 {
            if a * d == 0 {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else {
            (a as f64 * d as f64) / (b as f64 * cc as f64)
        };
        return Ok(FisherExact {
            p_value: p,
            p_left: Some(h.cdf(a as f64)),
            p_right: Some(h.sf(a as f64 - 1.0).min(1.0)),
            odds_ratio: Some(odds),
            tables_enumerated: 0,
        });
    }

    let lnf: Vec<f64> = (0..=n).map(ln_factorial).collect();
    let konst = rt.iter().chain(&ct).map(|&m| lnf[m as usize]).sum::<f64>() - lnf[n as usize];
    let lp_obs = konst - t.counts.iter().flatten().map(|&x| lnf[x as usize]).sum::<f64>();
    let thr = lp_obs + 1e-7f64.ln_1p();

    // Enumerate column by column; the last column is determined by the rest.
    struct State<'a> {
        r: usize,
        c: usize,
        ct: &'a [u64],
        lnf: &'a [f64],
        konst: f64,
        thr: f64,
        rem: Vec<u64>,
        total: f64,
        comp: f64,
        tables: u64,
        overflow: bool,
    }
    impl State<'_> {
        fn add(&mut self, x: f64) {
            let t = self.total + x;
            self.comp += if self.total.abs() >= x.abs() { (self.total - t) + x } else { (x - t) + self.total };
            self.total = t;
        }
        fn column(&mut self, j: usize, acc: f64) {
            if self.overflow {
                return;
            }
            if j == self.c - 1 {
                let mut lp = self.konst - acc;
                for i in 0..self.r {
                    lp -= self.lnf[self.rem[i] as usize];
                }
                self.tables += 1;
                if self.tables > FISHER_MAX_TABLES {
                    self.overflow = true;
                    return;
                }
                if lp <= self.thr {
                    self.add(lp.exp());
                }
                return;
            }
            let need = self.ct[j];
            self.row(j, 0, need, acc);
        }
        fn row(&mut self, j: usize, i: usize, need: u64, acc: f64) {
            if self.overflow {
                return;
            }
            if i == self.r - 1 {
                if need <= self.rem[i] {
                    self.rem[i] -= need;
                    self.column(j + 1, acc + self.lnf[need as usize]);
                    self.rem[i] += need;
                }
                return;
            }
            let later: u64 = self.rem[i + 1..].iter().sum();
            let hi = need.min(self.rem[i]);
            let lo = need.saturating_sub(later);
            for x in lo..=hi {
                self.rem[i] -= x;
                self.row(j, i + 1, need - x, acc + self.lnf[x as usize]);
                self.rem[i] += x;
            }
        }
    }
    let mut st = State {
        r,
        c,
        ct: &ct,
        lnf: &lnf,
        konst,
        thr,
        rem: rt.clone(),
        total: 0.0,
        comp: 0.0,
        tables: 0,
        overflow: false,
    };
    st.column(0, 0.0);
    if st.overflow {
        return Err(StatsError::TooLarge(format!("more than {FISHER_MAX_TABLES} tables with these margins")));
    }
    let p = (st.total + st.comp).min(1.0);
    Ok(FisherExact { p_value: p, p_left: None, p_right: None, odds_ratio: None, tables_enumerated: st.tables })
}

/// McNemar's test for paired binary data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McNemar {
    /// Pairs that changed from the first to the second category (n₁₂).
    pub b: u64,
    /// Pairs that changed from the second to the first category (n₂₁).
    pub c: u64,
    /// (b−c)²/(b+c), `None` when b + c = 0.
    pub statistic: Option<f64>,
    pub p_value: Option<f64>,
    /// (|b−c|−1)²/(b+c) with continuity correction.
    pub statistic_corrected: Option<f64>,
    pub p_value_corrected: Option<f64>,
    /// Exact two-sided binomial p-value, min(1, 2·P(Bin(b+c, ½) ≤ min(b, c))).
    pub p_value_exact: f64,
}

/// McNemar's test on a 2×2 table of paired outcomes (rows: first
/// measurement, columns: second; the off-diagonal cells are the discordant
/// pairs).
pub fn mcnemar(t: &Crosstab) -> Result<McNemar> {
    if t.n_rows() != 2 || t.n_cols() != 2 {
        return invalid("McNemar's test needs a 2x2 table");
    }
    let (b, c) = (t.counts[0][1], t.counts[1][0]);
    let m = b + c;
    if m == 0 {
        return Ok(McNemar {
            b,
            c,
            statistic: None,
            p_value: None,
            statistic_corrected: None,
            p_value_corrected: None,
            p_value_exact: 1.0,
        });
    }
    let mf = m as f64;
    let d = (b as f64 - c as f64).abs();
    let stat = d * d / mf;
    let corr = ((d - 1.0).max(0.0)).powi(2) / mf;
    let exact = (2.0 * Binomial::new(m, 0.5)?.cdf(b.min(c) as f64)).min(1.0);
    Ok(McNemar {
        b,
        c,
        statistic: Some(stat),
        p_value: Some(chi_p(stat, 1)?),
        statistic_corrected: Some(corr),
        p_value_corrected: Some(chi_p(corr, 1)?),
        p_value_exact: exact,
    })
}

/// McNemar's test from two paired label vectors. Pairs with a missing label
/// are dropped; exactly two distinct labels must occur overall, and both
/// vectors use the same ordering (text order), so the first label is the
/// "row/column 1" category.
pub fn mcnemar_paired(first: &[String], second: &[String]) -> Result<McNemar> {
    if first.len() != second.len() {
        return invalid(format!("{} first measurements but {} second measurements", first.len(), second.len()));
    }
    let mut levels: Vec<&str> = first
        .iter()
        .zip(second)
        .filter(|(a, b)| !a.is_empty() && !b.is_empty())
        .flat_map(|(a, b)| [a.as_str(), b.as_str()])
        .collect();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() != 2 {
        return invalid(format!("McNemar's test needs exactly 2 distinct categories, found {}", levels.len()));
    }
    let mut counts = vec![vec![0u64; 2]; 2];
    for (a, b) in first.iter().zip(second) {
        if a.is_empty() || b.is_empty() {
            continue;
        }
        counts[usize::from(a != levels[0])][usize::from(b != levels[0])] += 1;
    }
    let mut t = Crosstab::from_counts(counts)?;
    t.row_labels = levels.iter().map(|s| s.to_string()).collect();
    t.col_labels = t.row_labels.clone();
    mcnemar(&t)
}

/// Measures of association for a table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Association {
    /// √(χ²/n) (for 2×2 this is |φ|).
    pub phi: f64,
    /// Cramér's V = √(χ²/(n·min(r−1, c−1))).
    pub cramers_v: f64,
    /// √(χ²/(χ²+n)).
    pub contingency_coefficient: f64,
}

/// Phi, Cramér's V and the contingency coefficient from the Pearson χ².
pub fn association(t: &Crosstab) -> Result<Association> {
    let tr = t.trimmed("association measures")?;
    let chi = pearson_chi_square(&tr)?.statistic;
    let n = tr.total() as f64;
    let k = (tr.n_rows().min(tr.n_cols()) - 1) as f64;
    Ok(Association {
        phi: (chi / n).sqrt(),
        cramers_v: (chi / (n * k)).sqrt(),
        contingency_coefficient: (chi / (chi + n)).sqrt(),
    })
}

/// How the expected counts of a goodness-of-fit test are specified.
#[derive(Debug, Clone, PartialEq)]
pub enum GofExpected {
    /// Equal proportions for every category.
    Equal,
    /// Hypothesised proportions (positive, rescaled to sum to 1).
    Proportions(Vec<f64>),
    /// Historical counts (positive); only their relative sizes matter.
    Counts(Vec<f64>),
}

/// Result of a χ² goodness-of-fit test for one categorical variable.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquareGof {
    pub labels: Vec<String>,
    pub observed: Vec<f64>,
    pub proportion: Vec<f64>,
    pub expected: Vec<f64>,
    /// `(observed − expected)² / expected` per category.
    pub contribution: Vec<f64>,
    pub n: f64,
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    pub min_expected: f64,
    pub cells_expected_below_5: usize,
}

/// χ² goodness-of-fit test of observed category counts against `expected`.
/// `estimated_params` is subtracted from the degrees of freedom (k − 1 − m).
pub fn chi_square_gof(
    labels: &[String],
    observed: &[f64],
    expected: &GofExpected,
    estimated_params: usize,
) -> Result<ChiSquareGof> {
    let k = observed.len();
    if k < 2 {
        return Err(StatsError::InsufficientData { what: "goodness-of-fit test", needed: 2, got: k });
    }
    if labels.len() != k {
        return invalid("one label per category is required");
    }
    if observed.iter().any(|&o| !o.is_finite() || o < 0.0) {
        return invalid("observed counts must be finite and non-negative");
    }
    let n: f64 = observed.iter().sum();
    if n <= 0.0 {
        return invalid("the observed counts are all zero");
    }
    let weights: Vec<f64> = match expected {
        GofExpected::Equal => vec![1.0; k],
        GofExpected::Proportions(w) | GofExpected::Counts(w) => {
            if w.len() != k {
                return invalid(format!("{} expected values for {k} categories", w.len()));
            }
            w.clone()
        }
    };
    if weights.iter().any(|&w| !w.is_finite() || w <= 0.0) {
        return invalid("expected proportions or counts must be positive");
    }
    let wsum: f64 = weights.iter().sum();
    let proportion: Vec<f64> = weights.iter().map(|w| w / wsum).collect();
    let exp: Vec<f64> = proportion.iter().map(|p| p * n).collect();
    let contribution: Vec<f64> = observed.iter().zip(&exp).map(|(o, e)| (o - e) * (o - e) / e).collect();
    let statistic: f64 = contribution.iter().sum();
    if k < 2 + estimated_params {
        return invalid("too many estimated parameters for the number of categories");
    }
    let df = k - 1 - estimated_params;
    Ok(ChiSquareGof {
        labels: labels.to_vec(),
        observed: observed.to_vec(),
        proportion,
        min_expected: exp.iter().cloned().fold(f64::INFINITY, f64::min),
        cells_expected_below_5: exp.iter().filter(|&&e| e < 5.0).count(),
        expected: exp,
        contribution,
        n,
        statistic,
        df,
        p_value: chi_p(statistic, df)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(rows: &[&[u64]]) -> Crosstab {
        Crosstab::from_counts(rows.iter().map(|r| r.to_vec()).collect()).unwrap()
    }
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn building_tables() {
        let t = Crosstab::from_labels(&s(&["a", "a", "b", "", "b"]), &s(&["x", "y", "x", "x", "x"])).unwrap();
        assert_eq!(t.counts, vec![vec![1, 1], vec![2, 0]]);
        assert_eq!(t.n_excluded, 1);
        assert_eq!(t.row_labels, ["a", "b"]);
        let t = Crosstab::from_numeric(&[2.0, 10.0, 2.0, f64::NAN], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(t.row_labels, ["2", "10"]); // numeric, not text, order
        assert_eq!(t.counts, vec![vec![2], vec![1]]);
        assert!(Crosstab::from_counts(vec![]).is_err());
        assert!(Crosstab::from_counts(vec![vec![1, 2], vec![3]]).is_err());
        assert!(Crosstab::from_labels(&s(&["a"]), &s(&["x", "y"])).is_err());
        assert!(Crosstab::from_labels(&s(&[""]), &s(&["x"])).is_err());
    }

    #[test]
    fn percentages_and_expected() {
        let t = tab(&[&[10, 20], &[30, 40]]);
        let p = t.percentages();
        assert!((p.row[0][0] - 100.0 / 3.0).abs() < 1e-12);
        assert!((p.col[1][1] - 100.0 * 40.0 / 60.0).abs() < 1e-12);
        assert!((p.total[1][0] - 30.0).abs() < 1e-12);
        let e = t.expected();
        assert!((e[0][0] - 12.0).abs() < 1e-12);
        let z = tab(&[&[0, 0], &[3, 4]]);
        assert!(z.percentages().row[0][0].is_nan());
        let r = t.adjusted_residuals();
        assert!((r[0][0] + r[0][1]).abs() < 1e-12);
    }

    #[test]
    fn chi_square_2x2() {
        let t = tab(&[&[10, 20], &[30, 40]]);
        let p = pearson_chi_square(&t).unwrap();
        // by hand: 100*(10*40-20*30)^2/(30*70*40*60)
        let want = 100.0 * 200.0 * 200.0 / (30.0 * 70.0 * 40.0 * 60.0);
        assert!((p.statistic - want).abs() < 1e-12);
        assert_eq!(p.df, 1);
        assert_eq!(p.cells_expected_below_5, 0);
        let g = likelihood_ratio_chi_square(&t).unwrap();
        assert!(g.statistic > 0.0 && (g.statistic - p.statistic).abs() < 0.05);
        // an empty row is ignored, a single remaining row is an error
        let t2 = tab(&[&[10, 20], &[0, 0], &[30, 40]]);
        assert!((pearson_chi_square(&t2).unwrap().statistic - want).abs() < 1e-12);
        assert!(pearson_chi_square(&tab(&[&[1, 2], &[0, 0]])).is_err());
        let a = association(&t).unwrap();
        assert!((a.phi - (want / 100.0).sqrt()).abs() < 1e-12 && (a.phi - a.cramers_v).abs() < 1e-12);
    }

    #[test]
    fn linear_by_linear_matches_correlation() {
        let t = tab(&[&[10, 5, 2], &[4, 8, 6], &[1, 5, 12]]);
        let m = linear_by_linear(&t).unwrap();
        // expand to observations and correlate
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..3 {
            for j in 0..3 {
                for _ in 0..t.counts[i][j] {
                    x.push((i + 1) as f64);
                    y.push((j + 1) as f64);
                }
            }
        }
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let r = sxy
            / (x.iter().map(|a| (a - mx).powi(2)).sum::<f64>() * y.iter().map(|b| (b - my).powi(2)).sum::<f64>())
                .sqrt();
        assert!((m.correlation - r).abs() < 1e-13);
        assert!((m.statistic - (n - 1.0) * r * r).abs() < 1e-11);
        assert!(m.p_value < 0.001);
        assert!(linear_by_linear_scored(&t, &[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(linear_by_linear_scored(&t, &[1.0], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn fisher_tea_tasting() {
        // Fisher's lady tasting tea: p(two-sided) = 34/70, one-sided right 17/70
        let f = fisher_exact(&tab(&[&[3, 1], &[1, 3]])).unwrap();
        assert!((f.p_value - 34.0 / 70.0).abs() < 1e-12);
        assert!((f.p_right.unwrap() - 17.0 / 70.0).abs() < 1e-12);
        assert!((f.odds_ratio.unwrap() - 9.0).abs() < 1e-12);
        let f = fisher_exact(&tab(&[&[5, 0], &[0, 5]])).unwrap();
        assert!(f.odds_ratio.unwrap().is_infinite());
    }

    #[test]
    fn fisher_rxc_agrees_with_2x2_path_and_enumeration() {
        // a 2xc table can be checked against an independent brute force
        let t = tab(&[&[3, 1, 2], &[1, 3, 0]]);
        let f = fisher_exact(&t).unwrap();
        assert!(f.tables_enumerated > 0);
        // brute force over all 2x3 tables with margins (6, 4 | 4, 4, 2)
        let lnf = |k: u64| ln_factorial(k);
        let lp = |a: u64, b: u64, c: u64| -> f64 {
            let (d, e, g) = (4 - a, 4 - b, 2 - c);
            lnf(6) + lnf(4) + lnf(4) + lnf(4) + lnf(2)
                - lnf(10)
                - [a, b, c, d, e, g].iter().map(|&x| lnf(x)).sum::<f64>()
        };
        let obs = lp(3, 1, 2);
        let mut p = 0.0;
        for a in 0..=4u64 {
            for b in 0..=4u64 {
                for c in 0..=2u64 {
                    if a + b + c == 6 && lp(a, b, c) <= obs + 1e-9 {
                        p += lp(a, b, c).exp();
                    }
                }
            }
        }
        assert!((f.p_value - p).abs() < 1e-12, "{} vs {p}", f.p_value);
        // independence of orientation
        let tt = tab(&[&[3, 1], &[1, 3], &[2, 0]]);
        assert!((fisher_exact(&tt).unwrap().p_value - f.p_value).abs() < 1e-12);
        assert!(fisher_exact(&tab(&[&[1, 2], &[0, 0]])).is_err());
    }

    #[test]
    fn fisher_size_limit() {
        let big = tab(&[&[30, 30, 30, 30], &[30, 30, 30, 30], &[30, 30, 30, 30], &[30, 30, 30, 30]]);
        assert!(matches!(fisher_exact(&big), Err(StatsError::TooLarge(_))));
    }

    #[test]
    fn mcnemar_known() {
        // classic: b = 5, c = 15 → chi2 = 100/20 = 5
        let m = mcnemar(&tab(&[&[30, 5], &[15, 50]])).unwrap();
        assert!((m.statistic.unwrap() - 5.0).abs() < 1e-12);
        assert!((m.statistic_corrected.unwrap() - 81.0 / 20.0).abs() < 1e-12);
        assert!((m.p_value.unwrap() - 0.025347318677468).abs() < 1e-12);
        // exact: 2*P(Bin(20, .5) <= 5) = 2*0.0206947...
        assert!((m.p_value_exact - 0.04138946533203125).abs() < 1e-12);
        let z = mcnemar(&tab(&[&[3, 0], &[0, 4]])).unwrap();
        assert!(z.statistic.is_none() && z.p_value_exact == 1.0);
        assert!(mcnemar(&tab(&[&[1, 2, 3], &[4, 5, 6]])).is_err());
        let a = s(&["y", "y", "n", "n", "y", ""]);
        let b = s(&["y", "n", "n", "y", "n", "y"]);
        let m = mcnemar_paired(&a, &b).unwrap();
        // levels sorted: n, y → row/col 1 = n; (y,n)x2 → counts[1][0]; (n,y)x1 → counts[0][1]
        assert_eq!((m.b, m.c), (1, 2));
        assert!(mcnemar_paired(&s(&["a", "b", "c"]), &s(&["a", "b", "c"])).is_err());
    }
}
