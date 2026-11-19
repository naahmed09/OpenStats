//! Exploratory data analysis: letter values, median polish, Tukey's resistant line, the
//! (hanging) rootogram and the symmetry plot.

use crate::error::{invalid, Result, StatsError};
use openstats_math::special::erf::norm_cdf;

fn clean(x: &[f64]) -> Result<Vec<f64>> {
    if x.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    Ok(x.iter().copied().filter(|v| !v.is_nan()).collect())
}

fn med(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn median_of(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    med(&s)
}

/// One row of a letter-value display.
#[derive(Debug, Clone, PartialEq)]
pub struct LetterValue {
    /// M, H, E, D, C, B, A, Z, Y, X, W, V, U, T, S, R, Q, P, O, N, … and finally `1` for the extremes.
    pub label: String,
    /// Depth (position from the nearest end; half-integers average two order statistics).
    pub depth: f64,
    /// Lower letter value (the median for M).
    pub lower: f64,
    /// Upper letter value.
    pub upper: f64,
    /// Midpoint of the pair.
    pub mid: f64,
    /// Spread (upper − lower).
    pub spread: f64,
}

/// Tukey's letter values: median, hinges, eighths, … down to the extremes.
pub fn letter_values(x: &[f64]) -> Result<Vec<LetterValue>> {
    let mut s = clean(x)?;
    let n = s.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "letter values", needed: 2, got: n });
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let at = |d: f64| -> (f64, f64) {
        // Lower value at depth d and the matching upper value.
        let lo_i = d.floor() as usize;
        let lo = if (d - d.floor()).abs() < 1e-12 { s[lo_i - 1] } else { (s[lo_i - 1] + s[lo_i]) / 2.0 };
        let hi = if (d - d.floor()).abs() < 1e-12 { s[n - lo_i] } else { (s[n - lo_i] + s[n - lo_i - 1]) / 2.0 };
        (lo, hi)
    };
    const LABELS: &str = "MHEDCBAZYXWVUTSRQPON";
    let mut out = Vec::new();
    let mut depth = (n as f64 + 1.0) / 2.0;
    let mut k = 0usize;
    loop {
        let (lo, hi) = at(depth);
        let label = LABELS.chars().nth(k).map_or_else(|| format!("L{k}"), |c| c.to_string());
        out.push(LetterValue { label, depth, lower: lo, upper: hi, mid: (lo + hi) / 2.0, spread: hi - lo });
        if depth <= 1.0 {
            break;
        }
        let next = (depth.floor() + 1.0) / 2.0;
        depth = if next <= 1.0 { 1.0 } else { next };
        k += 1;
        if depth == 1.0 {
            let (lo, hi) = at(1.0);
            out.push(LetterValue {
                label: "1".into(),
                depth: 1.0,
                lower: lo,
                upper: hi,
                mid: (lo + hi) / 2.0,
                spread: hi - lo,
            });
            break;
        }
    }
    // The median row has one value.
    if let Some(m) = out.first_mut() {
        m.upper = m.lower;
        m.spread = 0.0;
    }
    Ok(out)
}

/// Result of Tukey's median polish.
#[derive(Debug, Clone, PartialEq)]
pub struct MedianPolish {
    /// Common value.
    pub overall: f64,
    /// Row effects.
    pub row: Vec<f64>,
    /// Column effects.
    pub col: Vec<f64>,
    /// Residuals (row-major, `rows × cols`).
    pub residuals: Vec<f64>,
    /// Number of sweeps used.
    pub iterations: usize,
    /// Whether the sum of absolute residuals stopped changing within `eps`.
    pub converged: bool,
}

/// Median polish of a `rows × cols` table (row-major, no missing values), following Tukey and
/// the scheme of R's `medpolish`: alternate row and column median sweeps, re-centring the
/// effects, until the sum of absolute residuals changes by less than `eps` (relative).
pub fn median_polish(table: &[f64], rows: usize, cols: usize, max_iter: usize, eps: f64) -> Result<MedianPolish> {
    if rows < 2 || cols < 2 || table.len() != rows * cols {
        return invalid("median polish needs a complete table with at least 2 rows and 2 columns");
    }
    if table.iter().any(|v| !v.is_finite()) {
        return invalid("median polish needs a table without missing values");
    }
    let mut z = table.to_vec();
    let (mut r, mut c, mut t) = (vec![0.0; rows], vec![0.0; cols], 0.0);
    let mut old = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    for it in 1..=max_iter.max(1) {
        iterations = it;
        for (i, ri) in r.iter_mut().enumerate() {
            let d = median_of(&z[i * cols..(i + 1) * cols]);
            z[i * cols..(i + 1) * cols].iter_mut().for_each(|v| *v -= d);
            *ri += d;
        }
        let d = median_of(&c);
        c.iter_mut().for_each(|v| *v -= d);
        t += d;
        for (j, cj) in c.iter_mut().enumerate() {
            let col: Vec<f64> = (0..rows).map(|i| z[i * cols + j]).collect();
            let d = median_of(&col);
            (0..rows).for_each(|i| z[i * cols + j] -= d);
            *cj += d;
        }
        let d = median_of(&r);
        r.iter_mut().for_each(|v| *v -= d);
        t += d;
        let new: f64 = z.iter().map(|v| v.abs()).sum();
        if new == 0.0 || (new - old).abs() < eps * new {
            converged = true;
            break;
        }
        old = new;
    }
    Ok(MedianPolish { overall: t, row: r, col: c, residuals: z, iterations, converged })
}

/// Tukey's resistant line.
#[derive(Debug, Clone, PartialEq)]
pub struct ResistantLine {
    /// Intercept.
    pub intercept: f64,
    /// Slope.
    pub slope: f64,
    /// Median x of the left, middle and right thirds.
    pub x_medians: [f64; 3],
    /// Median y of the left, middle and right thirds (of the data, before iteration).
    pub y_medians: [f64; 3],
    /// Slope iterations performed.
    pub iterations: usize,
    /// Fitted values (input order).
    pub fitted: Vec<f64>,
    /// Residuals (input order).
    pub residuals: Vec<f64>,
}

/// Resistant line by the three-group method with `max_iter` slope refinements.
///
/// The points are sorted by x and split into three groups whose sizes are as equal as possible
/// (extra points go to the outer groups); the slope is the difference of the outer groups'
/// median y over the difference of their median x, refined by adding the same ratio of the
/// outer residual medians until it changes by less than `1e-9` of the slope scale.
pub fn resistant_line(x: &[f64], y: &[f64], max_iter: usize) -> Result<ResistantLine> {
    if x.len() != y.len() {
        return invalid("x and y must have the same length");
    }
    let idx: Vec<usize> = (0..x.len()).filter(|&i| x[i].is_finite() && y[i].is_finite()).collect();
    let n = idx.len();
    if n < 6 {
        return Err(StatsError::InsufficientData { what: "resistant line", needed: 6, got: n });
    }
    let mut order = idx;
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    let (nl, nr) = (n / 3 + usize::from(n % 3 >= 1), n / 3 + usize::from(n % 3 == 2));
    let left = &order[..nl];
    let right = &order[n - nr..];
    let mid = &order[nl..n - nr];
    let mx = |g: &[usize]| median_of(&g.iter().map(|&i| x[i]).collect::<Vec<_>>());
    let my = |g: &[usize]| median_of(&g.iter().map(|&i| y[i]).collect::<Vec<_>>());
    let xm = [mx(left), mx(mid), mx(right)];
    let ym = [my(left), my(mid), my(right)];
    let dx = xm[2] - xm[0];
    if dx == 0.0 {
        return invalid("the outer thirds have the same median x; the slope is undefined");
    }
    let mut slope = (ym[2] - ym[0]) / dx;
    let mut it = 0;
    for _ in 0..max_iter {
        it += 1;
        let res = |g: &[usize]| median_of(&g.iter().map(|&i| y[i] - slope * x[i]).collect::<Vec<_>>());
        let adj = (res(right) - res(left)) / dx;
        slope += adj;
        if adj.abs() <= 1e-9 * slope.abs().max(1e-300) || adj == 0.0 {
            break;
        }
    }
    let intercept = median_of(&order.iter().map(|&i| y[i] - slope * x[i]).collect::<Vec<_>>());
    let fitted: Vec<f64> = x.iter().map(|&v| intercept + slope * v).collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, f)| a - f).collect();
    Ok(ResistantLine { intercept, slope, x_medians: xm, y_medians: ym, iterations: it, fitted, residuals })
}

/// One bin of a rootogram against a fitted normal curve.
#[derive(Debug, Clone, PartialEq)]
pub struct RootogramBin {
    /// Lower edge.
    pub lower: f64,
    /// Upper edge.
    pub upper: f64,
    /// Observed count.
    pub observed: f64,
    /// Expected count under the fitted normal.
    pub expected: f64,
    /// √observed.
    pub root_observed: f64,
    /// √expected.
    pub root_expected: f64,
    /// Hanging position √expected − √observed (the bar's bottom edge).
    pub hanging: f64,
    /// Tukey's double-root residual √(2 + 4·obs) − √(1 + 4·exp).
    pub double_root: f64,
}

/// Rootogram of `x` against the normal curve with the sample mean and standard deviation, using
/// `bins` equal-width bins (default Sturges when 0). The outer bins absorb the normal tails.
pub fn rootogram(x: &[f64], bins: usize) -> Result<Vec<RootogramBin>> {
    let s = clean(x)?;
    let n = s.len();
    if n < 5 {
        return Err(StatsError::InsufficientData { what: "rootogram", needed: 5, got: n });
    }
    let mean = s.iter().sum::<f64>() / n as f64;
    let sd = (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    if sd == 0.0 {
        return invalid("the data are constant");
    }
    let k = if bins == 0 { ((n as f64).log2().ceil() as usize + 1).max(3) } else { bins };
    if k < 2 {
        return invalid("a rootogram needs at least 2 bins");
    }
    let (lo, hi) =
        (s.iter().copied().fold(f64::INFINITY, f64::min), s.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let w = (hi - lo) / k as f64;
    let mut out = Vec::new();
    for b in 0..k {
        let (l, u) = (lo + w * b as f64, if b + 1 == k { hi } else { lo + w * (b + 1) as f64 });
        let obs = s.iter().filter(|&&v| if b + 1 == k { v >= l && v <= u } else { v >= l && v < u }).count() as f64;
        let cl = if b == 0 { 0.0 } else { norm_cdf((l - mean) / sd) };
        let cu = if b + 1 == k { 1.0 } else { norm_cdf((u - mean) / sd) };
        let exp = n as f64 * (cu - cl);
        out.push(RootogramBin {
            lower: l,
            upper: u,
            observed: obs,
            expected: exp,
            root_observed: obs.sqrt(),
            root_expected: exp.sqrt(),
            hanging: exp.sqrt() - obs.sqrt(),
            double_root: (2.0 + 4.0 * obs).sqrt() - (1.0 + 4.0 * exp).sqrt(),
        });
    }
    Ok(out)
}

/// Points of a symmetry plot: for the i-th smallest and i-th largest values, the distances
/// below and above the median.
pub fn symmetry_points(x: &[f64]) -> Result<Vec<(f64, f64)>> {
    let mut s = clean(x)?;
    let n = s.len();
    if n < 4 {
        return Err(StatsError::InsufficientData { what: "symmetry plot", needed: 4, got: n });
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let m = med(&s);
    Ok((0..n / 2).map(|i| (m - s[i], s[n - 1 - i] - m)).collect())
}
