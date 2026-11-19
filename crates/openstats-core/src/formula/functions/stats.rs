//! Column statistics (scalar results) and column transforms.
//!
//! Formulas follow Minitab's documented definitions: sample standard
//! deviation (n−1), quartiles by the (n+1)p interpolation rule, adjusted
//! Fisher–Pearson skewness, excess kurtosis with small-sample correction,
//! MSSD = Σ(xᵢ₊₁−xᵢ)² / (2(n−1)), and a 5 % trimmed mean.

use super::{def, present, scalar_int, scalar_num, Category::*, FunctionRegistry};
use crate::column::ColumnData;
use crate::error::{CoreError, Result};
use crate::formula::array::{common_len, Arr};
use crate::ops::rank::{rank, TieMethod};

/// Neumaier compensated sum.
pub fn sum(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut c = 0.0;
    for &v in x {
        let t = s + v;
        if s.abs() >= v.abs() {
            c += (s - t) + v;
        } else {
            c += (v - t) + s;
        }
        s = t;
    }
    s + c
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        // Two-step mean with a correction term for accuracy on offset data.
        let m = sum(x) / x.len() as f64;
        let corr = sum(&x.iter().map(|v| v - m).collect::<Vec<_>>()) / x.len() as f64;
        m + corr
    }
}

/// Sample variance (n−1), two-pass with compensation.
pub fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let dev: Vec<f64> = x.iter().map(|v| v - m).collect();
    let ss = sum(&dev.iter().map(|d| d * d).collect::<Vec<_>>());
    let s1 = sum(&dev);
    (ss - s1 * s1 / n as f64) / (n - 1) as f64
}

pub fn stdev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

fn sorted(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Quantile by Minitab's (n+1)p rule on sorted data.
pub fn quantile_sorted(s: &[f64], p: f64) -> f64 {
    let n = s.len();
    if n == 0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let pos = p * (n as f64 + 1.0);
    if pos <= 1.0 {
        return s[0];
    }
    if pos >= n as f64 {
        return s[n - 1];
    }
    let lo = pos.floor() as usize; // 1-based
    let frac = pos - lo as f64;
    s[lo - 1] + frac * (s[lo] - s[lo - 1])
}

pub fn median(x: &[f64]) -> f64 {
    let s = sorted(x);
    let n = s.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn skewness(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 3.0 {
        return f64::NAN;
    }
    let (m, s) = (mean(x), stdev(x));
    if s == 0.0 {
        return f64::NAN;
    }
    let s3 = sum(&x.iter().map(|v| ((v - m) / s).powi(3)).collect::<Vec<_>>());
    n / ((n - 1.0) * (n - 2.0)) * s3
}

pub fn kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 4.0 {
        return f64::NAN;
    }
    let (m, s) = (mean(x), stdev(x));
    if s == 0.0 {
        return f64::NAN;
    }
    let s4 = sum(&x.iter().map(|v| ((v - m) / s).powi(4)).collect::<Vec<_>>());
    n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * s4 - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
}

/// Mean of squared successive differences (over the non-missing sequence).
pub fn mssd(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let d: Vec<f64> = x.windows(2).map(|w| (w[1] - w[0]).powi(2)).collect();
    sum(&d) / (2.0 * (x.len() - 1) as f64)
}

/// Mean after removing round(5 % of n) values from each end.
pub fn trimmed_mean(x: &[f64]) -> f64 {
    let s = sorted(x);
    let k = (0.05 * s.len() as f64).round() as usize;
    if s.len() <= 2 * k {
        return f64::NAN;
    }
    mean(&s[k..s.len() - k])
}

/// Smallest most-frequent value.
pub fn mode(x: &[f64]) -> f64 {
    let s = sorted(x);
    let (mut best, mut best_n) = (f64::NAN, 0usize);
    let mut i = 0;
    while i < s.len() {
        let mut j = i;
        while j < s.len() && s[j] == s[i] {
            j += 1;
        }
        if j - i > best_n {
            best_n = j - i;
            best = s[i];
        }
        i = j;
    }
    best
}

fn paired(a: &Arr, b: &Arr) -> Result<(Vec<f64>, Vec<f64>)> {
    let x = a.to_nums("column")?;
    let y = b.to_nums("column")?;
    if x.len() != y.len() {
        return Err(CoreError::invalid("columns must have the same length"));
    }
    Ok(crate::missing::pairwise(&x, &y))
}

pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let (mx, my) = (mean(x), mean(y));
    sum(&x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).collect::<Vec<_>>()) / (n - 1) as f64
}

pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let c = covariance(x, y);
    let d = stdev(x) * stdev(y);
    if d == 0.0 {
        f64::NAN
    } else {
        (c / d).clamp(-1.0, 1.0)
    }
}

fn col_stat(a: &[Arr], f: fn(&[f64]) -> f64) -> Result<Arr> {
    Ok(Arr::num(f(&present(&a[0])?)))
}

/// Column transform keeping missing positions: `f` maps the whole numeric
/// vector (including NaN) to an output vector of the same length.
fn col_map(a: &Arr, f: impl Fn(&[f64]) -> Vec<f64>) -> Result<Arr> {
    let x = a.to_nums("column")?;
    Ok(Arr::nums(f(&x), a.scalar))
}

pub(super) fn register(r: &mut FunctionRegistry) {
    macro_rules! stat {
        ($name:literal, $help:literal, $f:expr) => {
            r.register(def($name, 1, Some(1), Statistics, $help, |a, _| col_stat(a, $f)));
        };
    }
    stat!("SUM", "SUM(c): sum of the column", sum);
    stat!("MEAN", "MEAN(c): arithmetic mean", mean);
    stat!("MEDIAN", "MEDIAN(c): median", median);
    stat!("STDEV", "STDEV(c): sample standard deviation", stdev);
    stat!("VARIANCE", "VARIANCE(c): sample variance", variance);
    stat!("SSQ", "SSQ(c): uncorrected sum of squares", |x| sum(&x.iter().map(|v| v * v).collect::<Vec<_>>()));
    stat!("CSSQ", "CSSQ(c): corrected sum of squares", |x| {
        let m = mean(x);
        if x.is_empty() {
            f64::NAN
        } else {
            sum(&x.iter().map(|v| (v - m) * (v - m)).collect::<Vec<_>>())
        }
    });
    stat!("MINIMUM", "MINIMUM(c): smallest value", |x| x.iter().copied().fold(f64::NAN, f64::min));
    stat!("MAXIMUM", "MAXIMUM(c): largest value", |x| x.iter().copied().fold(f64::NAN, f64::max));
    stat!("RANGE", "RANGE(c): maximum − minimum", |x| {
        let lo = x.iter().copied().fold(f64::NAN, f64::min);
        let hi = x.iter().copied().fold(f64::NAN, f64::max);
        hi - lo
    });
    stat!("PRODUCT", "PRODUCT(c): product of values", |x| if x.is_empty() { f64::NAN } else { x.iter().product() });
    stat!("SKEWNESS", "SKEWNESS(c): sample skewness", skewness);
    stat!("KURTOSIS", "KURTOSIS(c): sample excess kurtosis", kurtosis);
    stat!("MSSD", "MSSD(c): mean of squared successive differences / 2", mssd);
    stat!("Q1", "Q1(c): first quartile", |x| quantile_sorted(&sorted(x), 0.25));
    stat!("Q3", "Q3(c): third quartile", |x| quantile_sorted(&sorted(x), 0.75));
    stat!("IQR", "IQR(c): interquartile range Q3 − Q1", |x| {
        let s = sorted(x);
        quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25)
    });
    stat!("SEMEAN", "SEMEAN(c): standard error of the mean", |x| stdev(x) / (x.len() as f64).sqrt());
    stat!("TRMEAN", "TRMEAN(c): 5% trimmed mean", trimmed_mean);
    stat!("MODE", "MODE(c): most frequent value (smallest if tied)", mode);
    stat!("CV", "CV(c): coefficient of variation (stdev / mean)", |x| stdev(x) / mean(x));

    r.register(def("N", 1, Some(1), Statistics, "N(c): number of non-missing values", |a, _| {
        Ok(Arr::num((a[0].len() - a[0].data.missing_count()) as f64))
    }));
    r.register(def("NMISS", 1, Some(1), Statistics, "NMISS(c): number of missing values", |a, _| {
        Ok(Arr::num(a[0].data.missing_count() as f64))
    }));
    r.register(def("COUNT", 1, Some(1), Statistics, "COUNT(c): total number of rows", |a, _| {
        Ok(Arr::num(a[0].len() as f64))
    }));
    r.register(def("PERCENTILE", 2, Some(2), Statistics, "PERCENTILE(c, p): p-th quantile (0 ≤ p ≤ 1)", |a, _| {
        let p = scalar_num(&a[1], "p")?;
        Ok(Arr::num(quantile_sorted(&sorted(&present(&a[0])?), p)))
    }));
    r.register(def("CORRELATION", 2, Some(2), Statistics, "CORRELATION(c1, c2): Pearson correlation", |a, _| {
        let (x, y) = paired(&a[0], &a[1])?;
        Ok(Arr::num(correlation(&x, &y)))
    }));
    r.register(def("COVARIANCE", 2, Some(2), Statistics, "COVARIANCE(c1, c2): sample covariance", |a, _| {
        let (x, y) = paired(&a[0], &a[1])?;
        Ok(Arr::num(covariance(&x, &y)))
    }));

    // Column transforms.
    r.register(def("RANK", 1, Some(1), Statistics, "RANK(c): ranks, ties averaged", |a, _| {
        col_map(&a[0], |x| rank(x, TieMethod::Average, false))
    }));
    r.register(def("SORT", 1, Some(2), Statistics, "SORT(c, [descending]): sorted values, missing last", |a, _| {
        let desc = match a.get(1) {
            Some(d) => scalar_num(d, "descending")? != 0.0,
            None => false,
        };
        match &a[0].data {
            ColumnData::Text(v) => {
                let mut s: Vec<String> = v.iter().filter(|x| !x.is_empty()).cloned().collect();
                s.sort();
                if desc {
                    s.reverse();
                }
                s.resize(v.len(), String::new());
                Ok(Arr { data: ColumnData::Text(s), scalar: a[0].scalar })
            }
            ColumnData::DateTime(v) => {
                let mut s: Vec<i64> = v.iter().copied().filter(|&t| t != crate::missing::DATETIME_MISSING).collect();
                s.sort();
                if desc {
                    s.reverse();
                }
                s.resize(v.len(), crate::missing::DATETIME_MISSING);
                Ok(Arr { data: ColumnData::DateTime(s), scalar: a[0].scalar })
            }
            ColumnData::Numeric(_) => col_map(&a[0], |x| {
                let mut s = sorted(&crate::missing::drop_missing(x));
                if desc {
                    s.reverse();
                }
                s.resize(x.len(), f64::NAN);
                s
            }),
        }
    }));
    r.register(def("PARSUMS", 1, Some(1), Statistics, "PARSUMS(c): running sums (missing skipped)", |a, _| {
        col_map(&a[0], |x| {
            let mut acc = (0.0, 0.0); // Neumaier running sum
            x.iter()
                .map(|&v| {
                    if v.is_nan() {
                        return f64::NAN;
                    }
                    let t = acc.0 + v;
                    acc.1 += if acc.0.abs() >= v.abs() { (acc.0 - t) + v } else { (v - t) + acc.0 };
                    acc.0 = t;
                    acc.0 + acc.1
                })
                .collect()
        })
    }));
    r.register(def(
        "PARPRODUCTS",
        1,
        Some(1),
        Statistics,
        "PARPRODUCTS(c): running products (missing skipped)",
        |a, _| {
            col_map(&a[0], |x| {
                let mut p = 1.0;
                x.iter()
                    .map(|&v| {
                        if v.is_nan() {
                            f64::NAN
                        } else {
                            p *= v;
                            p
                        }
                    })
                    .collect()
            })
        },
    ));
    r.register(def("CUMMAX", 1, Some(1), Statistics, "CUMMAX(c): running maximum", |a, _| {
        col_map(&a[0], |x| {
            let mut m = f64::NAN;
            x.iter()
                .map(|&v| {
                    if v.is_nan() {
                        f64::NAN
                    } else {
                        m = m.max(v);
                        m
                    }
                })
                .collect()
        })
    }));
    r.register(def("CUMMIN", 1, Some(1), Statistics, "CUMMIN(c): running minimum", |a, _| {
        col_map(&a[0], |x| {
            let mut m = f64::NAN;
            x.iter()
                .map(|&v| {
                    if v.is_nan() {
                        f64::NAN
                    } else {
                        m = m.min(v);
                        m
                    }
                })
                .collect()
        })
    }));
    r.register(def("LAG", 1, Some(2), Statistics, "LAG(c, [k]): values shifted down k rows", |a, _| {
        let k = match a.get(1) {
            Some(k) => scalar_int(k, "lag")?,
            None => 1,
        };
        let n = a[0].len() as i64;
        let idx: Vec<usize> =
            (0..n).map(|i| if i - k >= 0 && i - k < n { (i - k) as usize } else { usize::MAX }).collect();
        Ok(Arr { data: a[0].data.take(&idx), scalar: false })
    }));
    r.register(def("DIFFERENCES", 1, Some(2), Statistics, "DIFFERENCES(c, [k]): x(i) − x(i−k)", |a, _| {
        let k = match a.get(1) {
            Some(k) => scalar_int(k, "lag")?,
            None => 1,
        };
        if k < 1 {
            return Err(CoreError::invalid("lag must be at least 1"));
        }
        let k = k as usize;
        col_map(&a[0], |x| (0..x.len()).map(|i| if i >= k { x[i] - x[i - k] } else { f64::NAN }).collect())
    }));
    r.register(def("STANDARDIZE", 1, Some(1), Statistics, "STANDARDIZE(c): (x − mean) / stdev", |a, _| {
        col_map(&a[0], |x| {
            let p = crate::missing::drop_missing(x);
            let (m, s) = (mean(&p), stdev(&p));
            x.iter().map(|v| (v - m) / s).map(crate::formula::array::fin).collect()
        })
    }));
    r.register(def("COUNTIF", 2, Some(2), Statistics, "COUNTIF(c, value): rows equal to value", |a, _| {
        let (n, _) = common_len(&[&a[0], &a[1]], "COUNTIF")?;
        let eq = crate::formula::eval::binary(crate::formula::ast::BinOp::Eq, &a[0], &a[1])?;
        let v = eq.data.to_f64_vec();
        Ok(Arr::num((0..n).filter(|&i| crate::formula::array::at(&v, i) == 1.0).count() as f64))
    }));

    r.alias("MIN", "MINIMUM");
    r.alias("MAX", "MAXIMUM");
    r.alias("VAR", "VARIANCE");
    r.alias("STD", "STDEV");
    r.alias("CORR", "CORRELATION");
    r.alias("COVAR", "COVARIANCE");
    r.alias("PARSUM", "PARSUMS");
    r.alias("PARPRODUCT", "PARPRODUCTS");
    r.alias("DIFF", "DIFFERENCES");
    r.alias("TRIMMEAN", "TRMEAN");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::formula::evaluate;
    use crate::worksheet::Worksheet;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn descriptive_reference_values() {
        // Reference values computed independently (and matching Minitab's definitions).
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&x), 5.0);
        assert!(close(stdev(&x), 2.138_089_935_299_395, 1e-15));
        assert_eq!(median(&x), 4.5);
        assert_eq!(quantile_sorted(&sorted(&x), 0.25), 4.0);
        assert_eq!(quantile_sorted(&sorted(&x), 0.75), 6.5);
        assert!(close(skewness(&x), 0.818_487_553_356_799_7, 1e-12));
        assert!(close(kurtosis(&x), 0.940_625_000_000_000_2, 1e-12));
        assert_eq!(mode(&x), 4.0);
        assert!(close(mssd(&x), 13.0 / 14.0, 1e-15));
        // Ill-conditioned data: large offset (NIST NumAcc-style).
        let big: Vec<f64> = [1e9 + 1.0, 1e9 + 3.0, 1e9 + 2.0, 1e9 + 3.0, 1e9 + 1.0].to_vec();
        assert!(close(variance(&big), 1.0, 1e-12));
        assert!(close(mean(&big), 1e9 + 2.0, 1e-15));
    }

    #[test]
    fn via_formulas() {
        let ws = Worksheet::with_columns(
            "s",
            vec![
                Column::numeric("x", vec![3.0, f64::NAN, 1.0, 2.0, 2.0]),
                Column::numeric("y", vec![6.0, 1.0, 2.0, 4.0, 4.0]),
                Column::text("t", vec!["b", "", "a", "c", "a"]),
            ],
        )
        .unwrap();
        let get = |f: &str| evaluate(f, &ws).unwrap().data.to_f64_vec();
        assert_eq!(get("N(x)"), [4.0]);
        assert_eq!(get("NMISS(x) + COUNT(x)"), [6.0]);
        assert_eq!(get("CORR(x, y)"), [1.0]);
        let rk = get("RANK(x)");
        assert_eq!((rk[0], rk[2], rk[3], rk[4]), (4.0, 1.0, 2.5, 2.5));
        assert!(rk[1].is_nan());
        assert_eq!(get("PARSUMS(x)")[4], 8.0);
        assert!(get("LAG(x)")[0].is_nan());
        assert_eq!(get("LAG(x, 2)")[4], 1.0);
        assert!(get("LAG(x, 2)")[3].is_nan());
        assert_eq!(get("DIFF(y)")[1], -5.0);
        assert_eq!(get("SORT(x)")[..4], [1.0, 2.0, 2.0, 3.0]);
        assert_eq!(get("SORT(x, 1)")[0], 3.0);
        assert_eq!(get("COUNTIF(t, \"a\")"), [2.0]);
        assert_eq!(get("PERCENTILE(y, 0.5)"), [4.0]);
        assert_eq!(evaluate("SORT(t)", &ws).unwrap().data.as_text().unwrap(), ["a", "a", "b", "c", ""]);
        assert!(evaluate("PERCENTILE(y, x)", &ws).is_err());
        assert_eq!(get("CUMMAX(y)"), [6.0, 6.0, 6.0, 6.0, 6.0]);
    }
}
